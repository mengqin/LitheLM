"""Cache isolation, EOS supervision, bounded retries and evaluation boundaries."""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from filelock import FileLock
import numpy as np
import pytest
import torch

from lithelm.config import normalize
from lithelm.data import cache_dataset, cache_root, collate, evaluation_batches, preprocess
from lithelm.infer import generate
from lithelm.model import build_model
from lithelm.platform_utils import build_accelerator
from lithelm.tokenizer import prepare_tokenizer, tokenizer_fingerprint
from lithelm.trainer import Trainer
from test_data import tiny_config


def test_shared_cache_identity_and_dotted_names(tmp_path, monkeypatch):
    import lithelm.data as data
    cfg = tiny_config(tmp_path)
    tok = prepare_tokenizer(cfg)
    root = cache_root(cfg)
    for name in ("wiki.en", "wiki.zh", "foo.bar.baz"):
        cache_dataset(name, cfg["datasets"]["pt"], tok, root)
        metadata, = root.glob(name+"-*.json")
        assert len(metadata.stem.removeprefix(name+"-")) == 20
    original = data.preprocess
    monkeypatch.setattr(data, "preprocess", lambda *args: pytest.fail("cache retokenized"))
    other = deepcopy(cfg)
    other["experiment"].update(name="different", output_dir=str(tmp_path/"another-run"))
    other["model"]["hidden_size"] *= 2
    other["stages"][0]["optimizer"]["lr"] *= 2
    cache_dataset("wiki.en", other["datasets"]["pt"], tok, cache_root(other))
    Path(other["experiment"]["output_dir"]).mkdir()
    shutil.rmtree(other["experiment"]["output_dir"])
    assert cache_root(other) == root and len(list(root.glob("*.json"))) == 3
    monkeypatch.setattr(data, "preprocess", original)
    other["datasets"]["pt"]["kwargs"]["max_samples"] = 2
    cache_dataset("wiki.en", other["datasets"]["pt"], tok, root)
    # The EOS role changes preprocessing identity even with unchanged vocabulary.
    tok.add_special_tokens({"eos_token": next(iter(tok.get_vocab()))})
    cache_dataset("wiki.en", cfg["datasets"]["pt"], tok, root)
    with Path(cfg["datasets"]["pt"]["path"]).open("a") as f:
        f.write("A changed source file.\n")
    cache_dataset("wiki.en", cfg["datasets"]["pt"], tok, root)
    assert len(list(root.glob("wiki.en-*.json"))) == 4
    cfg["workspace"]["copy_inputs"] = True
    assert cache_root(cfg) == Path(cfg["experiment"]["output_dir"])/"cache"


@pytest.mark.parametrize("fmt", ["prompt_response", "messages", "tinystories_instruct"])
def test_eos_roles_masks_roundtrip_and_early_stop(tmp_path, fmt):
    from transformers import AutoTokenizer
    cfg = tiny_config(tmp_path)
    cfg["tokenizer"].update(eos_token="<eos>", bos_token="<bos>", pad_token="<pad>")
    cfg = normalize(cfg)
    tok = prepare_tokenizer(cfg)
    tok.save_pretrained(tmp_path/"saved")
    restored = AutoTokenizer.from_pretrained(tmp_path/"saved", local_files_only=True)
    assert tokenizer_fingerprint(tok) == tokenizer_fingerprint(restored) == tokenizer_fingerprint(prepare_tokenizer(cfg))
    assert all(getattr(restored, role+"_id") is not None for role in ("eos_token", "bos_token", "pad_token"))
    hf = deepcopy(cfg)
    hf["tokenizer"].update(type="hf", artifact=str(tmp_path/"saved"))
    hf["workspace"]["copy_inputs"] = True
    assert tokenizer_fingerprint(prepare_tokenizer(hf)) == tokenizer_fingerprint(tok)
    assert tokenizer_fingerprint(prepare_tokenizer(hf)) == tokenizer_fingerprint(tok)
    spec = {**cfg["datasets"]["sft"], "format":fmt}
    rows = dict(prompt_response=dict(prompt="Tell me", response="A cat."),
                messages=dict(messages=[dict(role="user", content="Hi"), dict(role="assistant", content="A cat."),
                                        dict(role="user", content="More"), dict(role="assistant", content="A dog.")]),
                tinystories_instruct=dict(text="Words: cat\nStory: A cat.\nSummary: A cat."))
    old = preprocess(rows[fmt], spec, tok)
    assert tok.eos_token_id not in old["input_ids"]
    spec["append_eos"] = True
    new = preprocess(rows[fmt], spec, tok)
    eos = [i for i, value in enumerate(new["input_ids"]) if value == tok.eos_token_id]
    assert len(eos) == (2 if fmt == "messages" else 1)
    assert all(new["labels"][i] == tok.eos_token_id for i in eos)
    assert new["labels"][0] == -100
    if fmt == "tinystories_instruct":
        assert new["labels"][-1] == -100
    else:
        assert new["input_ids"][-1] == tok.eos_token_id
    pt = {**cfg["datasets"]["pt"], "append_eos":True}
    assert tok.eos_token_id not in preprocess(dict(text="A cat."), pt, tok)["input_ids"]
    class EndsImmediately(torch.nn.Module):
        cfg = {"max_position_embeddings":24}
        def forward(self, ids, use_cache=False, max_context_length=None):
            logits = torch.zeros(*ids.shape, len(tok))
            logits[..., tok.eos_token_id] = 1
            return SimpleNamespace(logits=logits)
    model = EndsImmediately()
    assert generate(model, torch.tensor([[3, 4]]), max_new_tokens=10, eos_token_id=tok.eos_token_id).tolist() == [[3, 4, tok.eos_token_id]]
    assert model.training


def test_missing_eos_fails(tmp_path):
    cfg = tiny_config(tmp_path)
    tok = prepare_tokenizer(cfg)
    cfg["datasets"]["sft"]["append_eos"] = True
    with pytest.raises(ValueError, match="append_eos requires"):
        preprocess(dict(prompt="Hi", response="Hello"), cfg["datasets"]["sft"], tok)
    cfg["tokenizer"]["eos_token"] = "<missing>"
    with pytest.raises(ValueError, match="absent from the vocabulary"):
        prepare_tokenizer(cfg)


def test_concurrent_prepare_and_crashed_lock(tmp_path):
    cfg = tiny_config(tmp_path)
    config = tmp_path/"config.json"
    config.write_text(json.dumps(cfg))
    script = '''import json, sys, time
from pathlib import Path
import lithelm.data as data
from lithelm.tokenizer import prepare_tokenizer
cfg = json.loads(Path(sys.argv[1]).read_text())
original = data.preprocess
def tracked(*args):
    with open('writers.txt', 'a') as f: f.write('row\\n')
    time.sleep(.01)
    return original(*args)
data.preprocess = tracked
tok = prepare_tokenizer(cfg)
data.cache_dataset('wiki.en', cfg['datasets']['pt'], tok, data.cache_root(cfg))
'''
    command = [sys.executable, "-c", script, str(config)]
    jobs = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        for job in jobs:
            stdout, stderr = job.communicate(timeout=90)
            assert job.returncode == 0, stdout+stderr
    finally:
        for job in jobs:
            if job.poll() is None:
                job.kill()
                job.wait()
    assert (tmp_path/"writers.txt").read_text().splitlines() == ["row"]*50
    artifact = Path(cfg["tokenizer"]["artifact"])
    stamp = artifact.stat().st_mtime_ns
    prepare_tokenizer(cfg)
    assert artifact.stat().st_mtime_ns == stamp
    lock = str(artifact)+".lock"
    result = subprocess.run([sys.executable, "-c", "import os,sys; from filelock import FileLock; lock=FileLock(sys.argv[1]); lock.acquire(); os._exit(0)", lock], timeout=20)
    assert result.returncode == 0
    with FileLock(lock, timeout=1):
        pass
    prepare_tokenizer(cfg)


@pytest.mark.parametrize("mode", ["shifted", "next_token"])
@pytest.mark.parametrize("packing", [False, True])
def test_dense_and_padded_masks(mode, packing):
    cache = dict(tokens=np.arange(17), mask=np.ones(17), offsets=np.array([0, 9, 17]))
    batches = list(evaluation_batches(cache, 4, 2, packing, target_mode=mode))
    assert "attention_mask" not in batches[0]
    padded = collate([dict(input_ids=[1,2], labels=[1,2])], 4, target_mode=mode)
    assert padded["attention_mask"].sum() == 2-int(mode == "next_token")
    assert padded["labels"][padded["attention_mask"] == 0].eq(-100).all()


@pytest.mark.parametrize("mode", ["shifted", "next_token"])
@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("world", [1, 2, 3])
@pytest.mark.parametrize("limit", [None, 1, 2])
def test_eval_shard_coverage(mode, packing, world, limit):
    cache = dict(tokens=np.arange(32), mask=np.ones(32), offsets=np.array([0, 6, 15, 32]))
    cache["mask"][[3, 11, 20]] = 0
    width = 4-int(mode == "shifted")
    documents = [list(range(1, 32))] if packing else [list(range(1,6)), list(range(7,15)), list(range(16,32))]
    windows = [doc[i:i+width] for doc in documents for i in range(0, len(doc), width)]
    selected = windows if limit is None else windows[:limit*2*world]
    expected = Counter(token for window in selected for token in window if cache["mask"][token])
    actual, batch_counts, row_counts = Counter(), [], []
    for rank in range(world):
        batches = list(evaluation_batches(cache, 4, 2, packing, rank, world, limit, mode))
        batch_counts.append(len(batches))
        row_counts.append(sum(len(b["input_ids"]) for b in batches))
        for batch in batches:
            labels = batch["labels"][:, int(mode == "shifted"):].flatten().tolist()
            actual.update(x for x in labels if x != -100)
    assert actual == expected
    assert sum(row_counts) == len(selected)
    assert max(batch_counts)-min(batch_counts) <= 1
    assert limit is None or max(batch_counts) <= limit


def test_eval_aliases_and_empty_targets(tmp_path, monkeypatch):
    import lithelm.evaluate as module
    cfg = tiny_config(tmp_path)
    cfg["stages"][0]["eval"]["datasets"] = [dict(name="pt", alias="first"), dict(name="pt", alias="second")]
    cfg = normalize(cfg)
    tok, model = prepare_tokenizer(cfg), build_model(cfg["model"])
    accelerator = SimpleNamespace(device=torch.device("cpu"), process_index=0, num_processes=1,
        unwrap_model=lambda m:m, main_process_first=nullcontext, autocast=nullcontext, reduce=lambda x, reduction:x)
    result = module.evaluate(model, cfg, cfg["stages"][0], tok, accelerator)
    assert result["eval/first/nll"] == result["eval/second/nll"]
    cache = dict(tokens=np.arange(8), mask=np.zeros(8), offsets=np.array([0,8]))
    monkeypatch.setattr(module, "cache_dataset", lambda *args:cache)
    with pytest.raises(ValueError, match="no target tokens"):
        module.evaluate(model, cfg, cfg["stages"][0], tok, accelerator)
    assert model.training


@pytest.mark.parametrize("key", ["path", "split", "streaming", "cache_dir", "download_config", "data_files"])
def test_reserved_kwargs(tmp_path, key):
    cfg = tiny_config(tmp_path)
    cfg["datasets"]["sft"]["kwargs"][key] = "collision"
    with pytest.raises(ValueError, match="reserved fields.*"+key):
        normalize(cfg)


@pytest.mark.parametrize("checkpoint", ["final", "best"])
def test_checkpoint_dependencies(tmp_path, checkpoint):
    cfg = tiny_config(tmp_path)
    cfg["stages"][1]["init_from"]["checkpoint"] = checkpoint
    normalize(cfg)
    cfg["stages"][0]["checkpoint"]["export_final" if checkpoint == "final" else "best_metric"] = False if checkpoint == "final" else None
    with pytest.raises(ValueError, match="init_from "+checkpoint):
        normalize(cfg)


def test_invalid_best_and_removed_init(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["model"]["init"]["type"] = "normal"
    with pytest.raises(ValueError, match="unknown fields.*type"):
        normalize(cfg)
    cfg["model"]["init"].pop("type")
    cfg["stages"][0]["eval"]["datasets"] = []
    with pytest.raises(ValueError, match="best_metric requires"):
        normalize(cfg)


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_cpu_fp16_rejected(tmp_path, monkeypatch, device):
    cfg = tiny_config(tmp_path)
    cfg["runtime"].update(device=device, precision="fp16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda:False)
    with pytest.raises(ValueError, match="fp16 requires CUDA"):
        build_accelerator(cfg)


def test_deterministic_config_before_cuda_probe(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    cfg["runtime"].update(device="auto", deterministic=True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    def unavailable():
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        return False
    monkeypatch.setattr(torch.cuda, "is_available", unavailable)
    a = build_accelerator(cfg)
    try:
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        a.end_training()
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("success", [False, True])
def test_optimizer_skip_cap_and_reset(tmp_path, monkeypatch, success):
    cfg = tiny_config(tmp_path)
    cfg["runtime"]["max_optimizer_step_retries"] = 3
    stage = cfg["stages"][0]
    stage.update(max_steps=2)
    stage["checkpoint"].update(best_metric=None, export_final=False)
    stage["intervals"] = dict(log=0, stats=0, eval=0, checkpoint=0)
    trainer = Trainer(cfg, None, None, None, SimpleNamespace(log=lambda row:None))
    monkeypatch.setattr(trainer, "setup_stage", lambda *args:None)
    calls, attempts = [], iter([False, False, True]*2 if success else [False]*3)
    def update():
        calls.append(trainer.state["stage_step"])
        good = next(attempts)
        if good:
            trainer.state["stage_step"] += 1
            trainer.state["global_step"] += 1
        return good
    monkeypatch.setattr(trainer, "train_update", update)
    if success:
        assert trainer.run_stage(0, stage)
        assert calls == [0,0,0,1,1,1]
    else:
        with pytest.raises(RuntimeError, match="stage=pt stage_step=0 global_step=0 retry_count=3 precision=fp32"):
            trainer.run_stage(0, stage)


@pytest.mark.parametrize("sft", [False, True])
def test_throughput_excludes_maintenance(tmp_path, monkeypatch, sft):
    import lithelm.trainer as module
    cfg = tiny_config(tmp_path)
    stage = cfg["stages"][0]
    stage.update(max_steps=2, curriculum=[])
    stage["lr_schedule"].update(warmup_steps=0, stable_steps=2, decay_steps=0)
    stage["intervals"] = dict(log=1, stats=1, eval=1, checkpoint=1)
    stage["eval"]["generate"].update(enabled=True, every=1)
    stage["checkpoint"].update(best_metric=None)
    clock, rows = [0.], []
    monkeypatch.setattr(module.time, "perf_counter", lambda:clock[0])
    a = build_accelerator(cfg)
    trainer = Trainer(cfg, build_model(cfg["model"]), None, a, SimpleNamespace(log=rows.append))
    def batch(index, size):
        clock[0] += 1
        row = dict(input_ids=list(range(8)), labels=[-100]*6+[6,7] if sft else list(range(8)))
        return collate([row]*size, 16), "pt", "sft" if sft else "pt"
    monkeypatch.setattr(trainer, "update_curriculum", lambda:setattr(trainer, "stream", SimpleNamespace(batch=batch)))
    def maintenance(*args):
        clock[0] += 100
    for name in ("save", "sample", "run_eval"):
        monkeypatch.setattr(trainer, name, maintenance)
    try:
        trainer.run_stage(0, stage)
    finally:
        a.end_training()
    logs = [row for row in rows if row.get("event") == "train"]
    assert len(logs) == 2 and clock[0] > 400
    for row in logs:
        assert row["perf/input_tokens_per_sec"] == 16
        assert row["perf/target_tokens_per_sec"] == (4 if sft else 14)
        assert row["perf/samples_per_sec"] == 2
