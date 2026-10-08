"""Correctness and recovery boundaries before training acceptance."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from lithelm.checkpoint import read_state, save_checkpoint
from lithelm.config import normalize
from lithelm.data import cache_dataset, format_record, preprocess, raw_records, source_metadata
from lithelm.infer import generate, infer_checkpoint
from lithelm.model import build_model
from lithelm.platform_utils import build_accelerator
from lithelm.tokenizer import prepare_tokenizer
from lithelm.trainer import Trainer
from lithelm.workspace import latest_checkpoint
from test_data import tiny_config


@pytest.mark.parametrize("gradient", [None, float("nan"), float("inf")])
def test_nonfinite_gradient_cannot_step(tmp_path, gradient):
    cfg = tiny_config(tmp_path)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg, build_model(cfg["model"]), prepare_tokenizer(cfg), a, SimpleNamespace(log=lambda row:None))
    try:
        trainer.setup_stage(0, cfg["stages"][0])
        parameters = [p.detach().clone() for p in trainer.model.parameters()]
        calls = []
        trainer.optimizer.optimizer.register_step_pre_hook(lambda *args:calls.append(True))
        if gradient is None:
            assert trainer.train_update()
            assert calls == [True] and trainer.state["global_step"] == 1
        else:
            handle = next(trainer.model.parameters()).register_hook(lambda grad:torch.full_like(grad, gradient))
            with pytest.raises(FloatingPointError, match="stage=pt stage_step=0 global_step=0 precision=fp32 grad_norm="):
                trainer.train_update()
            handle.remove()
            assert calls == [] and trainer.optimizer.optimizer.state == {}
            assert trainer.state["global_step"] == trainer.state["stage_step"] == 0
            for before, after in zip(parameters, trainer.model.parameters()):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
    finally:
        a.free_memory()
        a.end_training()


def test_checkpoint_post_save_barrier(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    tok = prepare_tokenizer(cfg)
    events = []
    path = tmp_path/"checkpoint-1"
    def save_state(directory, **kwargs):
        events.append("save_state")
        assert not (Path(directory)/"COMPLETE").exists()
    replace = Path.replace
    def publish(source, target):
        assert (source/"COMPLETE").is_file()
        events.append("publish")
        return replace(source, target)
    monkeypatch.setattr(Path, "replace", publish)
    a = SimpleNamespace(is_main_process=True, num_processes=2, save_state=save_state,
                        wait_for_everyone=lambda:events.append("wait"))
    save_checkpoint(path, a, dict(global_step=1), cfg, tok, {}, cfg["stages"][0]["checkpoint"])
    assert events == ["wait", "save_state", "wait", "publish", "wait"]


def test_directory_source_invalidates_shared_cache(tmp_path, monkeypatch):
    cfg = tiny_config(tmp_path)
    tok = prepare_tokenizer(cfg)
    directory = tmp_path/"dataset"
    directory.mkdir()
    a = directory/"a.txt"
    a.write_text("one")
    (directory/"nested").mkdir()
    (directory/"nested/b.txt").write_text("two")
    spec = {**cfg["datasets"]["pt"], "path":str(directory)}
    monkeypatch.setattr("lithelm.data.raw_records", lambda spec:iter([{"text":"A little cat went home."}]))
    identities = []
    def snapshot():
        # Metadata discovery must never read corpus contents.
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", lambda *args, **kwargs:pytest.fail("metadata read file contents"))
            metadata = source_metadata(directory)
        assert list(metadata) == sorted(metadata)
        identities.append(hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest())
        cache_dataset("directory", spec, tok, tmp_path/"cache")
        assert len(list((tmp_path/"cache").glob("*.json"))) == len(identities)
        return metadata
    first = snapshot()
    assert list(first) == ["a.txt", "nested/b.txt"]
    assert first == source_metadata(directory)
    a.write_text("longer contents")
    snapshot()
    stat = a.stat()
    a.write_text("changed content")  # Same size; only mtime distinguishes it.
    os.utime(a, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
    snapshot()
    (directory/"c.txt").write_text("new file")
    snapshot()
    (directory/"nested/b.txt").unlink()
    snapshot()
    assert len(set(identities)) == 5


@pytest.mark.parametrize("fmt", ["text", "prompt_response", "messages", "tinystories_instruct"])
def test_tokenizer_training_uses_formatted_records(tmp_path, monkeypatch, fmt):
    cfg = tiny_config(tmp_path)
    story = "A little cat and a friendly dog found a bright garden. They shared a picnic and walked home together."
    rows = dict(text={"text":story}, prompt_response={"prompt":"Tell a story", "response":story},
                messages={"messages":[{"role":"user", "content":"Tell a story"}, {"role":"assistant", "content":story}]},
                tinystories_instruct={"text":"Words: picnic\nStory: "+story+"\nSummary: Friends share."})
    spec = cfg["datasets"]["pt" if fmt == "text" else "sft"]
    spec["format"] = "prompt_response" if fmt == "text" else fmt
    if fmt in ("text", "tinystories_instruct"):
        spec.update(source="text", path=str(tmp_path/"source.txt"))
        Path(spec["path"]).write_text(rows[fmt]["text"])
    else:
        spec.update(source="json", path=str(tmp_path/"source.jsonl"))
        Path(spec["path"]).write_text(json.dumps(rows[fmt])+"\n")
    cfg["tokenizer"]["train_from"].update(dataset="pt" if fmt == "text" else "sft", sample_bytes=10000)
    expected = [format_record(row, spec)[0] for row in raw_records(spec)]
    seen = []
    def format_and_record(row, spec):
        result = format_record(row, spec)
        seen.append(result[0])
        return result
    monkeypatch.setattr("lithelm.data.format_record", format_and_record)
    tok = prepare_tokenizer(cfg)
    assert seen == expected and all(expected)
    for row, text in zip(raw_records(spec), expected):
        assert preprocess(row, spec, tok)["input_ids"] == tok(text, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("external", [False, True])
def test_checkpoint_infer_eval_inside_and_outside_run(tmp_path, external):
    from lithelm.train import train
    from lithelm.evaluate import evaluate_checkpoint
    cfg = tiny_config(tmp_path)
    cfg["inference"]["max_new_tokens"] = 2
    train(cfg, stop_after=1)
    root = Path(cfg["experiment"]["output_dir"])
    path = root/"pt/checkpoint-1"
    if external:
        target = tmp_path/"exported-checkpoint"
        shutil.copytree(path, target)
        path = target
    assert infer_checkpoint(cfg, path, "A cat").startswith("A cat")
    record = json.loads((root/"inference/samples.jsonl").read_text())
    assert record["checkpoint"] == (str(path.resolve()) if external else "pt/checkpoint-1")
    assert evaluate_checkpoint(cfg, path)["eval/nll"] > 0
    assert list((root/"evaluations").glob("*.json"))


@pytest.mark.parametrize("lookup", ["latest", "resume"])
@pytest.mark.parametrize("normal,writing,previous,expected", [
    (None, True, None, 3), (None, True, True, 3), (None, False, True, 1),
    (True, None, True, 2), (True, False, None, 2), (None, False, None, None),
])
def test_checkpoint_crash_recovery(tmp_path, lookup, normal, writing, previous, expected):
    path = tmp_path/"pt/checkpoint-2"
    for suffix, complete, step in [("", normal, 2), (".writing", writing, 3), (".previous", previous, 1)]:
        if complete is not None:
            directory = path.with_name(path.name+suffix)
            directory.mkdir(parents=True)
            (directory/"trainer_state.json").write_text(json.dumps(dict(global_step=step, stage_index=0)))
            if complete:
                (directory/"COMPLETE").touch()
    if expected is None:
        assert latest_checkpoint(tmp_path) is None
        with pytest.raises(ValueError, match="Incomplete checkpoint"):
            read_state(path)
        assert not path.exists()
    else:
        if lookup == "latest":
            assert latest_checkpoint(tmp_path) == str(path)
        assert read_state(path)["global_step"] == expected
        assert not path.with_name(path.name+".writing").exists()
        assert not path.with_name(path.name+".previous").exists()


def test_checkpoint_recovery_mutates_only_main_rank(tmp_path):
    path = tmp_path/"final"
    writing = tmp_path/"final.writing"
    writing.mkdir()
    (writing/"COMPLETE").touch()
    (writing/"trainer_state.json").write_text('{"global_step": 3}')
    waits = []
    a = SimpleNamespace(is_main_process=False, wait_for_everyone=lambda:waits.append(path.exists()))
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        read_state(path, a)
    assert waits == [False] and writing.exists()
    a.is_main_process = True
    assert read_state(path, a)["global_step"] == 3
    assert waits == [False, True] and not writing.exists()


def test_compiled_cached_generation(tmp_path):
    if not hasattr(torch, "compile") or not torch._dynamo.is_dynamo_supported():
        pytest.skip("torch.compile is unavailable on this platform")
    from test_cache import cache_config
    model = build_model(cache_config(tmp_path)["model"])
    ids = torch.ones(2, 4, dtype=torch.long)
    expected = generate(model, ids, max_new_tokens=4)
    compiled = torch.compile(deepcopy(model), backend="aot_eager")
    lengths = []
    handle = compiled.register_forward_pre_hook(lambda module, args:lengths.append(args[0].shape[1]))
    try:
        result = generate(compiled, ids, max_new_tokens=4)
    finally:
        handle.remove()
    assert compiled.training and lengths == [4, 1, 1, 1]
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_cross_stage_accumulation_counts(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["stages"][1]["batch"]["grad_accum_steps"] = 3
    for stage in cfg["stages"]:
        stage["init_from"]["stage"] = None
        stage["checkpoint"].update(best_metric=None, export_final=False)
        stage["intervals"] = dict(log=0, stats=0, eval=0, checkpoint=0)
    cfg = normalize(cfg)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg, build_model(cfg["model"]), prepare_tokenizer(cfg), a, SimpleNamespace(log=lambda row:None))
    updates, microbatches = [], []
    try:
        for index, stage in enumerate(cfg["stages"]):
            trainer.setup_stage(index, stage)
            hook = trainer.model.register_forward_pre_hook(lambda *args:microbatches.append(1))
            def record_update(*args, **kwargs):
                updates.append((stage["name"], len(microbatches)))
                microbatches.clear()
            step_hook = trainer.optimizer.optimizer.register_step_pre_hook(record_update)
            for expected_step in range(1, stage["max_steps"]+1):
                assert trainer.train_update()
                assert trainer.state["stage_step"] == trainer.state["scheduler_position"] == expected_step
                assert trainer.state["global_step"] == len(updates)
            hook.remove()
            step_hook.remove()
        assert updates == [("pt", 2)]*4+[("sft", 3)]*3
        assert not microbatches
    finally:
        a.free_memory()
        a.end_training()


def test_stage_optimizer_only_source_recovers(tmp_path):
    cfg = tiny_config(tmp_path)
    stage = cfg["stages"][1]
    stage["init_from"].update(checkpoint="checkpoint-1", load_model=False, load_optimizer=True)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg, build_model(cfg["model"]), prepare_tokenizer(cfg), a, SimpleNamespace(log=lambda row:None))
    try:
        trainer.setup_stage(0, cfg["stages"][0])
        assert trainer.train_update()
        trainer.save("checkpoint-1")
        path = Path(cfg["experiment"]["output_dir"])/"pt/checkpoint-1"
        path.rename(path.with_name(path.name+".previous"))
        trainer.setup_stage(1, stage)
        assert path.is_dir() and trainer.optimizer.optimizer.state
        assert trainer.train_update()
    finally:
        a.free_memory()
        a.end_training()
