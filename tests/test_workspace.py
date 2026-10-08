"""The full-course runner must resume published checkpoints only."""
import json
from pathlib import Path


def test_latest_full_course_checkpoint(tmp_path):
    from lithelm.workspace import latest_checkpoint
    assert latest_checkpoint(tmp_path) is None
    for name, step, complete in [("pt/best", 59000, True), ("pt/final", 60000, True),
                                 ("sft/checkpoint-500", 60500, True),
                                 ("sft/checkpoint-1000.writing", 61000, True),
                                 ("sft/checkpoint-1500", 61500, False)]:
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "trainer_state.json").write_text(json.dumps(dict(global_step=step)))
        if complete:
            (path / "COMPLETE").touch()
    assert latest_checkpoint(tmp_path) == str(tmp_path / "sft/checkpoint-1000")


def test_portable_run_and_fresh_allocation(tmp_path):
    import shutil
    from omegaconf import OmegaConf
    from test_data import tiny_config
    from test_smoke_train import run
    from lithelm.workspace import run_directory

    cfg = tiny_config(tmp_path / "input")
    cfg["workspace"]["copy_inputs"] = True
    cfg["experiment"].update(output_dir=None, model_size="6K")
    # An absent generated tokenizer is run-relative, not an external input.
    cfg["tokenizer"]["artifact"] = "tokenizer/tokenizer.json"
    source = tmp_path / "source.yaml"
    OmegaConf.save(OmegaConf.create(cfg), source)
    original = source.read_bytes()
    run("train", source, "--stop-after", 2)
    root, = (tmp_path / "outputs").iterdir()
    assert root.name.startswith("tiny_6K_")
    config = root / "resolved_config.yaml"
    snapshot = OmegaConf.to_container(OmegaConf.load(config))
    assert snapshot["experiment"]["output_dir"] == "."
    assert all(not Path(d["path"]).is_absolute() for d in snapshot["datasets"].values())
    # Move the bundle, remove external source files and change working directory.
    moved = tmp_path / "archive" / root.name
    moved.parent.mkdir()
    shutil.move(root, moved)
    shutil.rmtree(tmp_path / "input")
    run("train", moved/"resolved_config.yaml", "--resume", "latest")
    final = moved / "sft/final"
    state = json.loads((final/"trainer_state.json").read_text())
    assert state["global_step"] == 7
    assert json.loads(run("eval", moved/"resolved_config.yaml", "--checkpoint", final))["eval/nll"] > 0
    assert "cat" in run("infer", moved/"resolved_config.yaml", "--checkpoint", final, "--prompt", "cat", "--max-new-tokens", 2)
    assert (moved/"inference/samples.jsonl").exists()
    assert (moved/"evaluations/sft-final-on-sft.json").exists()
    assert source.read_bytes() == original
    # Fresh runs on the same date must not share a directory or overwrite one another.
    other = tiny_config(tmp_path/"second")
    other["experiment"].update(output_dir=None, model_size="6K")
    other["tokenizer"]["artifact"] = "tokenizer/tokenizer.json"
    with run_directory(other) as (_, first):
        pass
    with run_directory(other) as (_, second):
        pass
    assert first != second and second.name == first.name + "-02"


def test_output_escape_rejected(tmp_path):
    import pytest
    from test_data import tiny_config
    from lithelm.workspace import run_directory
    cfg = tiny_config(tmp_path)
    for value in ("../outside.jsonl", "\\outside.jsonl", "C:\\outside.jsonl"):
        cfg["logging"]["jsonl_file"] = value
        with pytest.raises(ValueError, match="inside the run directory"):
            with run_directory(cfg):
                pass
    assert not (tmp_path/"outside.jsonl").exists()


def test_hf_tokenizer_snapshot_survives_move(tmp_path):
    import shutil
    from lithelm.config import load_config, save_config
    from lithelm.workspace import run_directory
    from lithelm.tokenizer import prepare_tokenizer, tokenizer_fingerprint
    from test_data import tiny_config
    cfg = tiny_config(tmp_path/"source")
    cfg["workspace"]["copy_inputs"] = True
    tokenizer = prepare_tokenizer(cfg)
    external = tmp_path/"external-tokenizer"
    tokenizer.save_pretrained(external)
    cfg["experiment"].update(output_dir=None, model_size="6K")
    cfg["tokenizer"].update(type="hf", artifact=str(external))
    with run_directory(cfg) as (bound, root):
        assert tokenizer_fingerprint(prepare_tokenizer(bound)) == tokenizer_fingerprint(tokenizer)
        save_config(bound, "resolved_config.yaml")
    moved = tmp_path/"moved"/root.name
    moved.parent.mkdir()
    shutil.move(root, moved)
    shutil.rmtree(external)
    shutil.rmtree(tmp_path/"source")
    with run_directory(load_config(moved/"resolved_config.yaml")) as (bound, root):
        assert tokenizer_fingerprint(prepare_tokenizer(bound)) == tokenizer_fingerprint(tokenizer)
