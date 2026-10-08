"""Incremental logits, cache layout, position offsets and generation boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from lithelm.config import context_length, load_config, normalize, save_config
from lithelm.infer import generate
from lithelm.model import AttentionCache, ModelCache, build_model
from test_data import tiny_config


def cache_config(tmp_path, position="rope", kv=2, impl="sdpa"):
    cfg = tiny_config(tmp_path)
    cfg["model"].update(hidden_size=64, max_position_embeddings=128,
                        position_embedding=dict(type=position, max_positions=128))
    layer = cfg["model"]["layers"][0]
    layer["hidden_size"] = 64
    layer["mixer"]["attention"].update(num_heads=4, num_kv_heads=kv, head_dim=16, impl=impl)
    cfg["model"]["layers"] = [deepcopy(layer) for _ in range(3)]
    cfg["model"]["layers"][1]["norm_position"] = "post"
    return normalize(cfg)


@torch.inference_mode()
def compare_cached(model, ids, prompt=4, mask=None, atol=1e-6, rtol=1e-5, max_context_length=None):
    state_keys = set(model.state_dict())
    out = model(ids[:, :prompt], attention_mask=mask[:, :prompt] if mask is not None else None,
                use_cache=True, max_context_length=max_context_length)
    assert isinstance(out.cache, ModelCache)
    buffers = [state for state in out.cache.layers if isinstance(state, AttentionCache)]
    pointers = [(state.key.data_ptr(), state.value.data_ptr()) for state in buffers]
    max_error = 0.
    for state in buffers:
        state.key[:, :, prompt:].fill_(float("nan"))
        state.value[:, :, prompt:].fill_(float("nan"))
    for length in range(prompt, ids.shape[1]+1):
        full = model(ids[:, :length], attention_mask=mask[:, :length] if mask is not None else None)
        torch.testing.assert_close(out.logits[:, -1], full.logits[:, -1], atol=atol, rtol=rtol)
        max_error = max(max_error, (out.logits[:, -1].float()-full.logits[:, -1].float()).abs().max().item())
        assert full.cache is None and out.cache.position == length
        for i, state in enumerate(out.cache.layers):
            if isinstance(state, AttentionCache):
                cfg = model.cfg["layers"][i]["mixer"]["attention"]
                assert state.key.shape == state.value.shape == (ids.shape[0], cfg["num_kv_heads"], out.cache.max_length, cfg["head_dim"])
                assert not state.key.requires_grad and not state.value.requires_grad
                for tensor in (state.key, state.value):
                    assert tensor.is_contiguous()
                    assert torch.isnan(tensor[:, :, length:]).all()
                    assert tensor.untyped_storage().nbytes() == tensor.numel()*tensor.element_size()
            else:
                # Query FLA's public cache API without interpreting its state.
                assert state.get_seq_length(0) == length
        if length < ids.shape[1]:
            cache = out.cache
            prefix = [(s.key[:, :, :length].clone(), s.value[:, :, :length].clone()) for s in buffers]
            out = model(ids[:, length:length+1], cache=cache, use_cache=True)
            assert out.cache is cache
            assert [s for s in out.cache.layers if isinstance(s, AttentionCache)] == buffers
            assert [(s.key.data_ptr(), s.value.data_ptr()) for s in buffers] == pointers
            for state, (key, value) in zip(buffers, prefix):
                torch.testing.assert_close(state.key[:, :, :length], key, atol=0, rtol=0)
                torch.testing.assert_close(state.value[:, :, :length], value, atol=0, rtol=0)
    assert set(model.state_dict()) == state_keys
    print(f"cached_vs_full max_abs={max_error:.9g} device={ids.device} prompt={prompt}")
    return out


@pytest.mark.parametrize("position", ["none", "rope", "learned_absolute"])
@pytest.mark.parametrize("kv", [1, 2, 4])
@pytest.mark.parametrize("padded", [False, True])
def test_incremental_logits_positions_and_shapes(tmp_path, position, kv, padded):
    torch.manual_seed(47)
    model = build_model(cache_config(tmp_path, position, kv)["model"]).eval()
    ids = torch.randint(0, 280, (2, 9))
    mask = torch.ones_like(ids) if padded else None
    if padded:
        mask[0, :2] = 0
        mask[1, 2:4] = 0
    compare_cached(model, ids, mask=mask)


@torch.inference_mode()
def full_greedy(model, ids, count):
    for _ in range(count):
        ids = torch.cat((ids, model(ids).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    return ids


@pytest.mark.parametrize("prompt_length", [4, 32])
def test_generation_prefill_once_and_single_token_decode(tmp_path, prompt_length):
    model = build_model(cache_config(tmp_path)["model"]).eval()
    ids = torch.randint(0, 280, (2, prompt_length))
    expected = full_greedy(model, ids, 5)
    lengths = []
    handle = model.register_forward_pre_hook(lambda module, args:lengths.append(args[0].shape[1]))
    model.train()
    try:
        result = generate(model, ids, max_new_tokens=5)
    finally:
        handle.remove()
    assert model.training and lengths == [prompt_length, 1, 1, 1, 1]
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_generation_context_limits_and_mode_restore(tmp_path):
    cfg = cache_config(tmp_path)["model"]
    cfg["max_position_embeddings"] = 8
    model = build_model(cfg)
    ids = torch.ones(1, 6, dtype=torch.long)
    assert generate(model, ids, max_new_tokens=10).shape == (1, 8)
    assert generate(model, ids, max_new_tokens=0) is ids
    assert generate(model, torch.ones(1, 8, dtype=torch.long)).shape == (1, 8)
    for prompt in (torch.ones(1, 9, dtype=torch.long), ids[:, :0]):
        with pytest.raises(ValueError):
            generate(model, prompt)
        assert model.training
    with torch.inference_mode():
        model.eval()
        cache = model(torch.ones(1, 8, dtype=torch.long), use_cache=True).cache
        with pytest.raises(ValueError, match="max_position_embeddings"):
            model(ids[:, :1], cache=cache, use_cache=True)
        assert cache.position == 8


def test_batched_eos_and_sampling():
    class Terminates(torch.nn.Module):
        cfg = {"max_position_embeddings":12}
        def forward(self, ids, use_cache=False, cache=None, max_context_length=None):
            assert use_cache
            position = cache.position if cache else 0
            logits = torch.full((*ids.shape, 4), -torch.inf)
            logits[0, :, 3] = 0
            logits[1, :, 3 if position else 2] = 0
            return SimpleNamespace(logits=logits, cache=ModelCache(position+ids.shape[1], [], 2, max_context_length or cache.max_length))
    model = Terminates()
    ids = torch.tensor([[0,1], [0,1]])
    output = generate(model, ids, max_new_tokens=8, eos_token_id=3, do_sample=True, top_k=2, top_p=.9)
    assert output.tolist() == [[0,1,3,3], [0,1,2,3]]
    assert model.training


def test_cache_contract_and_training_unchanged(tmp_path):
    model = build_model(cache_config(tmp_path)["model"])
    ids = torch.ones(2, 4, dtype=torch.long)
    out = model(ids, labels=ids, collect_stats=True)
    assert out.cache is None and out.stats
    out.loss.backward()
    with pytest.raises(ValueError, match="eval"):
        model(ids, use_cache=True)
    model.eval()
    with pytest.raises(ValueError, match="no_grad"):
        model(ids, use_cache=True)
    with torch.inference_mode():
        cache = model(ids, use_cache=True).cache
        for inputs, kwargs, match in [
            (ids[:, :1], dict(cache=cache), "use_cache=True"),
            (ids[:, :1], dict(use_cache=True, labels=ids[:, :1]), "no labels"),
            (ids, dict(use_cache=True, cache=cache), "single-token"),
            (ids[:1, :1], dict(use_cache=True, cache=cache), "same batch"),
            (ids[:, :1], dict(use_cache=True, cache=cache, attention_mask=torch.ones_like(ids)), "mask matching"),
            (ids[:, :1], dict(use_cache=True, cache=cache, attention_mask=torch.zeros_like(ids[:, :1])), "unmasked decode"),
        ]:
            with pytest.raises(ValueError, match=match):
                model(inputs, **kwargs)
            assert cache.position == 4
        model(ids[:, :1], cache=cache, use_cache=True)
        assert cache.position == 5


def test_custom_mixer_stats_survive_and_cache_fails_clearly(tmp_path, monkeypatch):
    from test_refinement import ResearchMixer, custom_config
    model = build_model(custom_config(tmp_path, ResearchMixer, monkeypatch)["model"])
    ids = torch.ones(1, 4, dtype=torch.long)
    out = model(ids, labels=ids, collect_stats=True)
    out.loss.backward()
    assert out.cache is None and out.stats["layer/0/mixer/arbitrary_new_field"] == 7.
    with pytest.raises(NotImplementedError, match="causal attention or FLA"):
        generate(model, ids)
    assert model.training


def test_compiled_training_without_cache(tmp_path):
    model = build_model(cache_config(tmp_path)["model"])
    compiled = torch.compile(deepcopy(model), backend="eager", fullgraph=True)
    ids = torch.ones(2, 4, dtype=torch.long)
    expected, actual = model(ids, labels=ids), compiled(ids, labels=ids)
    assert actual.cache is None
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    expected.loss.backward()
    actual.loss.backward()
    for a, b in zip(model.parameters(), compiled.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@torch.inference_mode()
def test_tlm_7m_preallocated_cache():
    cfg = load_config(Path(__file__).parents[1]/"configs/examples/tlm_7m.yaml")
    torch.manual_seed(47)
    model = build_model(cfg["model"]).eval()
    assert sum(p.numel() for p in model.parameters()) == 7_432_704
    ids = torch.randint(0, 4096, (2, 36))
    out = compare_cached(model, ids, prompt=32, max_context_length=128)
    assert out.cache.max_length == 128 and out.cache.position == 36


@torch.inference_mode()
def test_requested_capacity_and_overflow(tmp_path):
    model = build_model(cache_config(tmp_path)["model"]).eval()
    ids = torch.ones(2, 8, dtype=torch.long)
    out = compare_cached(model, ids, prompt=4, max_context_length=8)
    assert out.cache.position == out.cache.max_length == 8
    with pytest.raises(ValueError, match="capacity"):
        model(ids[:, :1], cache=out.cache, use_cache=True)
    assert out.cache.position == 8
    with pytest.raises(ValueError, match="Cannot change"):
        model(ids[:, :1], cache=out.cache, use_cache=True, max_context_length=9)
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args:calls.append(args[0].shape[1]))
    try:
        assert generate(model, ids[:, :4], max_context_length=8, max_new_tokens=20).shape[1] == 8
        assert calls == [4, 1, 1, 1]
        calls.clear()
        assert generate(model, ids, max_context_length=8) is ids
        assert calls == [8]  # A full prompt gets one legal prefill and no decode.
        with pytest.raises(ValueError, match="Prompt exceeds"):
            generate(model, ids, max_context_length=7)
    finally:
        handle.remove()


@pytest.mark.parametrize("requested", [0, -1, True, 1.5, 129])
def test_invalid_capacity_config_and_api(tmp_path, requested):
    cfg = cache_config(tmp_path)
    cfg["inference"]["max_context_length"] = requested
    with pytest.raises(ValueError, match="max_context_length"):
        normalize(cfg)
    model = build_model(cfg["model"]).eval()
    ids = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="max_context_length"):
        generate(model, ids, max_context_length=requested)
    with torch.inference_mode(), pytest.raises(ValueError, match="max_context_length"):
        model(ids, use_cache=True, max_context_length=requested)


def test_position_sublimit(tmp_path):
    cfg = cache_config(tmp_path)["model"]
    cfg["position_embedding"]["max_positions"] = 64
    assert context_length(cfg) == 64
    with pytest.raises(ValueError, match="position_embedding.max_positions"):
        context_length(cfg, 65)


@pytest.mark.parametrize("field", ["key", "value"])
@pytest.mark.parametrize("mismatch", ["batch", "heads", "dim", "capacity", "dtype", "device"])
@torch.inference_mode()
def test_cache_tensor_mismatch(tmp_path, field, mismatch):
    model = build_model(cache_config(tmp_path)["model"]).eval()
    ids = torch.ones(2, 4, dtype=torch.long)
    cache = model(ids, use_cache=True, max_context_length=8).cache
    state = cache.layers[0]
    tensor = getattr(state, field)
    if mismatch == "dtype":
        wrong = tensor.double()
    elif mismatch == "device":
        wrong = torch.empty_like(tensor, device="meta")
    else:
        shape = list(tensor.shape)
        shape[{"batch":0, "heads":1, "capacity":2, "dim":3}[mismatch]] += 1
        wrong = torch.empty(shape)
    setattr(state, field, wrong)
    with pytest.raises(ValueError, match="device/dtype" if mismatch in ("dtype", "device") else "shape"):
        model(ids[:, :1], use_cache=True, cache=cache)
    assert cache.position == 4


@pytest.mark.parametrize("configured,cli,expected", [(2048, 1024, 1024), (2048, None, 2048), (None, None, 4096)])
def test_context_cli_precedence(tmp_path, monkeypatch, configured, cli, expected):
    from lithelm.cli import main
    cfg = cache_config(tmp_path)
    cfg["model"]["max_position_embeddings"] = cfg["model"]["position_embedding"]["max_positions"] = 4096
    cfg["inference"]["max_context_length"] = configured
    path = tmp_path/"config.yaml"
    save_config(cfg, path)
    seen = []
    def infer(cfg, checkpoint, prompt):
        model = build_model(cfg["model"]).eval()
        with torch.inference_mode():
            out = model(torch.ones(1, 4, dtype=torch.long), use_cache=True,
                        max_context_length=cfg["inference"]["max_context_length"])
        seen.append(out.cache.max_length)
        assert out.cache.layers[0].key.shape[2] == expected
        return "done"
    monkeypatch.setattr("lithelm.infer.infer_checkpoint", infer)
    args = ["infer", str(path), "--checkpoint", "unused", "--prompt", "test"]
    if cli is not None:
        args += ["--set", "inference.max_context_length=3072", "--max-context-length", str(cli)]
    main(args)
    assert seen == [expected]


@torch.inference_mode()
def test_fla_independent_legacy_cache_slots(tmp_path, monkeypatch):
    class LegacyCache:
        def __init__(self):
            self.states = []
        def get_seq_length(self, index):
            assert index == 0
            return self.states[0].shape[1] if self.states else 0

    class FakeFLA(torch.nn.Module):
        def __init__(self, hidden_size, mode, layer_idx, **kwargs):
            super().__init__()
            self.layer_idx = layer_idx
            self.seen = []
        def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False):
            if not use_cache:
                return x.cumsum(1), None, None
            assert self.layer_idx == 0
            self.seen.append(past_key_values)
            history = past_key_values.states
            full = torch.cat((history[0], x), dim=1) if history else x
            past_key_values.states = [full]
            return full.cumsum(1)[:, -x.shape[1]:], None, past_key_values

    monkeypatch.setitem(sys.modules, "fla.layers", SimpleNamespace(GatedDeltaNet=FakeFLA))
    monkeypatch.setitem(sys.modules, "fla.models.utils", SimpleNamespace(Cache=LegacyCache))
    cfg = cache_config(tmp_path)
    attention = cfg["model"]["layers"][0]
    fla = deepcopy(attention)
    fla["mixer"]["type"] = "gdn"
    cfg["model"]["layers"] = deepcopy([attention, fla, attention, fla])
    model = build_model(normalize(cfg)["model"]).eval()
    ids = torch.randint(0, 280, (2, 7))
    out = compare_cached(model, ids, max_context_length=7)
    a, b = out.cache.layers[1], out.cache.layers[3]
    assert isinstance(a, LegacyCache) and isinstance(b, LegacyCache) and a is not b
    for index in (1, 3):
        layer = model.layers[index].mixer.layer
        assert layer.layer_idx == 0 and len(layer.seen) == 4
        assert all(cache is out.cache.layers[index] for cache in layer.seen)
    with pytest.raises(ValueError, match="capacity"):
        model(ids[:, :1], use_cache=True, cache=out.cache)
