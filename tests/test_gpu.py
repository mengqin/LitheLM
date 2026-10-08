from copy import deepcopy
import os
import pytest
import torch

from lithelm.config import normalize
from lithelm.model import build_model
from lithelm.ops.attention import attention_op, attention_reference
from test_data import tiny_config

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")]

if torch.cuda.is_available() and "LITHELM_GPU_MEMORY_FRACTION" in os.environ:
    torch.cuda.set_per_process_memory_fraction(float(os.environ["LITHELM_GPU_MEMORY_FRACTION"]))


FLASH_IMPL = pytest.param("flash", marks=pytest.mark.skipif(
    not torch.backends.cuda.is_flash_attention_available(), reason="PyTorch Flash backend not compiled"))


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize("length,dim",[(17,16),(64,32),(129,64),(7,128)])
def test_triton_forward_backward(dtype,length,dim):
    pytest.importorskip("triton")
    torch.manual_seed(9)
    torch.set_float32_matmul_precision("highest")
    values=[torch.randn(2,2,length,dim,device="cuda",dtype=dtype,requires_grad=True) for _ in range(3)]
    mask=torch.ones(2,1,1,length,device="cuda",dtype=torch.bool)
    mask[0,...,-3:]=False
    mask[1,...,:]=False
    reference=attention_reference(*values,mask=mask)
    grad=torch.randn_like(reference)
    expected=torch.autograd.grad(reference,values,grad)
    result=attention_op(*values,impl="triton",mask=mask)
    actual=torch.autograd.grad(result,values,grad)
    tol=2e-2 if dtype==torch.bfloat16 else 2e-3 if dtype==torch.float16 else 3e-5
    torch.testing.assert_close(result,reference,atol=tol,rtol=tol)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=tol,rtol=tol)


def test_gpu_model_triton_train(tmp_path):
    cfg=tiny_config(tmp_path)
    cfg["model"]["layers"][0]["mixer"]["attention"].update(num_heads=1,num_kv_heads=1,head_dim=16,impl="triton")
    m=build_model(normalize(cfg)["model"]).cuda()
    x=torch.randint(0,280,(2,20),device="cuda")
    opt=torch.optim.AdamW(m.parameters())
    for _ in range(2):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss=m(x,labels=x).loss
        loss.backward();opt.step();opt.zero_grad()
    assert torch.isfinite(loss)


def test_triton_rejects_unsupported_inputs():
    pytest.importorskip("triton")
    q=torch.randn(1,1,4,8,device="cuda")
    with pytest.raises(ValueError,match="head_dim"):
        attention_op(q,q,q,impl="triton")
    q=torch.randn(1,1,4,16,device="cuda")
    with pytest.raises(ValueError,match="dropout"):
        attention_op(q,q,q,impl="triton",dropout=.1)
    with pytest.raises(ValueError,match="fp16/bf16/fp32"):
        attention_op(q.double(),q.double(),q.double(),impl="triton")


def test_fla_mixed_model(tmp_path):
    pytest.importorskip("fla")
    cfg=tiny_config(tmp_path)
    layer=deepcopy(cfg["model"]["layers"][0])
    layer["mixer"]["type"]="gdn"
    layer["mixer"]["gdn"]["kwargs"]=dict(num_heads=2,head_dim=16,expand_v=1.,use_short_conv=False)
    cfg["model"]["layers"].append(layer)
    model=build_model(normalize(cfg)["model"]).cuda()
    x=torch.randint(0,280,(2,24),device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        result=model(x,labels=x)
    result.loss.backward()
    assert torch.isfinite(result.loss)


@pytest.mark.parametrize("impl", ["sdpa", FLASH_IMPL, "flash2"])
def test_dense_causal_forward_backward(tmp_path, impl):
    from lithelm.data import collate
    cfg = tiny_config(tmp_path)
    cfg["model"]["layers"][0]["mixer"]["attention"].update(impl=impl)
    if impl == "flash2":
        pytest.importorskip("flash_attn")
    model = build_model(cfg["model"]).cuda()
    batch = collate([dict(input_ids=list(range(17)), labels=list(range(17)))]*2, 16, target_mode="next_token")
    assert "attention_mask" not in batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**{k:v.cuda() for k,v in batch.items()}, target_mode="next_token")
    out.loss.backward()
    assert out.token_count == 32 and torch.isfinite(out.loss)


def test_deterministic_cuda_fresh_process(tmp_path):
    import json
    import subprocess
    import sys
    cfg = tiny_config(tmp_path)
    cfg["runtime"].update(device="cuda", deterministic=True, precision="fp32")
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cfg))
    script = '''import json, os, sys, torch
from lithelm.platform_utils import build_accelerator
from lithelm.model import build_model
cfg = json.load(open(sys.argv[1]))
assert not torch.cuda.is_initialized()
a = build_accelerator(cfg)
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
assert torch.are_deterministic_algorithms_enabled()
m = build_model(cfg['model']).to(a.device)
x = torch.arange(16, device=a.device)[None]
first = m(x, labels=x).loss
first.backward()
assert torch.equal(first, m(x, labels=x).loss)
assert torch.isfinite(first)
a.end_training()
'''
    env = {key:value for key,value in os.environ.items() if key != "CUBLAS_WORKSPACE_CONFIG"}
    result = subprocess.run([sys.executable, "-c", script, str(path)], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize("impl", ["sdpa", FLASH_IMPL, "flash2", "triton"])
@pytest.mark.parametrize("position", ["rope", "learned_absolute"])
def test_cached_attention_logits(tmp_path, impl, position):
    from test_cache import cache_config, compare_cached
    if impl == "triton":
        pytest.importorskip("triton")
    if impl == "flash2":
        pytest.importorskip("flash_attn")
    torch.manual_seed(47)
    model = build_model(cache_config(tmp_path, position, impl=impl)["model"]).cuda().eval()
    ids = torch.randint(0, 280, (2, 12), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        compare_cached(model, ids, atol=3e-3, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_length,k_length,causal", [(17, 17, True), (1, 9, False), (7, 9, False)])
def test_flash2_outputs_and_gradients(dtype, q_length, k_length, causal):
    pytest.importorskip("flash_attn")
    values = [torch.randn(2, 2, length, 16, device="cuda", dtype=dtype, requires_grad=True)
              for length in (q_length, k_length, k_length)]
    out = attention_op(*values, impl="flash2", causal=causal)
    reference = attention_reference(*(value.float() for value in values), causal=causal)
    grad = torch.randn_like(out)
    expected = torch.autograd.grad(reference, values, grad.float())
    actual = torch.autograd.grad(out, values, grad)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(out, reference.to(dtype), atol=tolerance, rtol=tolerance)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=tolerance, rtol=tolerance)


def test_flash2_training_dropout():
    function = pytest.importorskip("flash_attn").flash_attn_func
    values = [torch.randn(2, 2, 17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
              for _ in range(3)]
    cuda_state = torch.cuda.get_rng_state()
    expected = function(*(value.transpose(1, 2) for value in values), causal=True, dropout_p=.2,
                        deterministic=torch.are_deterministic_algorithms_enabled()).transpose(1, 2)
    torch.cuda.set_rng_state(cuda_state)
    out = attention_op(*values, impl="flash2", dropout=.2)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    out.float().sum().backward()
    assert all(torch.isfinite(value.grad).all() for value in values)


def test_compiled_flash2():
    pytest.importorskip("flash_attn")
    values = [torch.randn(2, 2, 17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
              for _ in range(3)]
    compiled = torch.compile(lambda q, k, v:attention_op(q, k, v, impl="flash2"), backend="eager")
    out = compiled(*values)
    out.float().sum().backward()
    assert torch.isfinite(out).all() and all(torch.isfinite(value.grad).all() for value in values)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("prompt,mode,dtype,atol", [
    (4, "chunk", torch.bfloat16, 3e-3),
    (80, "chunk", torch.bfloat16, 1e-2),
    (80, "fused_recurrent", torch.bfloat16, 5e-3),
    (80, "chunk", torch.float16, 1.5e-3),
])
def test_fla_and_hybrid_cache(tmp_path, hybrid, prompt, mode, dtype, atol):
    pytest.importorskip("fla")
    from test_cache import cache_config, compare_cached
    cfg = cache_config(tmp_path)
    attention = cfg["model"]["layers"][0]
    fla = deepcopy(attention)
    fla["mixer"]["type"] = "gdn"
    fla["mixer"]["gdn"]["mode"] = mode
    fla["mixer"]["gdn"]["kwargs"] = dict(num_heads=4, head_dim=16, expand_v=1., use_short_conv=True)
    cfg["model"]["layers"] = deepcopy([attention, fla, attention, fla] if hybrid else [fla, fla])
    torch.manual_seed(47)
    model = build_model(normalize(cfg)["model"]).cuda().eval()
    ids = torch.randint(0, 280, (2, prompt+4), device="cuda")
    mask = torch.ones_like(ids)
    mask[0, :2] = 0
    mask[1, prompt-2:prompt] = 0
    # Chunk prefill and recurrent decode have different low-precision rounding.
    with torch.autocast("cuda", dtype=dtype):
        out = compare_cached(model, ids, prompt=prompt, mask=mask, atol=atol, rtol=2e-2,
                             max_context_length=ids.shape[1])
        with torch.inference_mode(), pytest.raises(ValueError, match="capacity"):
            model(ids[:, :1], cache=out.cache, use_cache=True)


def test_triton_single_query_backward():
    pytest.importorskip("triton")
    values = [torch.randn(2, 2, n, 16, device="cuda", requires_grad=True) for n in (1, 9, 9)]
    result = attention_op(*values, impl="triton", causal=False)
    expected = attention_reference(*values, causal=False)
    torch.testing.assert_close(result, expected, atol=3e-5, rtol=3e-5)
    actual_grad = torch.autograd.grad(result.sum(), values)
    expected_grad = torch.autograd.grad(expected.sum(), values)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
