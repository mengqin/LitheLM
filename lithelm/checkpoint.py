"""Atomic checkpoint publication, Accelerate state, compatibility and retention."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import torch
from .config import save_config
from .tokenizer import tokenizer_fingerprint


def signature(cfg, tokenizer):
    relevant = deepcopy({k:cfg[k] for k in ("model","datasets","stages","runtime")})
    # Default shifted semantics retain the pre-targets checkpoint signature.
    for stage in relevant["stages"]:
        if stage["lr_schedule"].get("type", "wsd") == "wsd":
            for key in ("path","name","kwargs"):
                stage["lr_schedule"].pop(key,None)
        if stage.get("loss", {}).get("type", "default") == "default":
            stage.pop("loss", None)
        if stage["data"].get("strategy", "default") == "default":
            stage["data"].pop("strategy", None)
            stage["data"].pop("step_plan", None)
        if stage.get("targets", {}).get("mode", "shifted") == "shifted":
            stage.pop("targets", None)
    relevant["tokenizer_hash"] = tokenizer_fingerprint(tokenizer)
    relevant["seed"] = cfg["experiment"]["seed"]
    return hashlib.sha256(json.dumps(relevant,sort_keys=True).encode()).hexdigest()


def recover_checkpoint(path, accelerator=None):
    """Recover an interrupted publication; only rank zero mutates directories."""
    path = Path(path)
    if accelerator is None or accelerator.is_main_process:
        writing = path.with_name(path.name + ".writing")
        previous = path.with_name(path.name + ".previous")
        if not path.exists():
            for candidate in (writing, previous):
                if (candidate / "COMPLETE").is_file():
                    candidate.replace(path)
                    break
        if (path / "COMPLETE").is_file():
            for stale in (writing, previous):
                if stale.is_dir():
                    shutil.rmtree(stale)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return path


def read_state(path, accelerator=None):
    path = recover_checkpoint(path, accelerator)
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Incomplete checkpoint: {path}")
    return json.loads((path / "trainer_state.json").read_text())


def save_checkpoint(path, accelerator, state, cfg, tokenizer, metrics, options):
    """All ranks participate; rank 0 atomically publishes only complete snapshots."""
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    if accelerator.is_main_process:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(temporary), safe_serialization=False)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        saved = {**state, "signature":signature(cfg,tokenizer), "world_size":accelerator.num_processes,
                 "save_optimizer":options["save_optimizer"], "save_scheduler":options["save_scheduler"],
                 "tokenizer_hash":tokenizer_fingerprint(tokenizer)}
        (temporary / "trainer_state.json").write_text(json.dumps(saved,indent=2),encoding="utf-8")
        if options["save_metrics"]:
            (temporary / "metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
        save_config(cfg,temporary / "resolved_config.yaml")
        if cfg["workspace"]["copy_inputs"]:
            tokenizer.save_pretrained(temporary / "tokenizer")
        for keep, pattern in [(options["save_optimizer"],"optimizer*.bin"),(options["save_scheduler"],"scheduler*.bin")]:
            if not keep:
                for file in temporary.glob(pattern):
                    file.unlink()
        (temporary / "COMPLETE").touch()
        backup = path.with_name(path.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if path.exists():
            path.replace(backup)
        temporary.replace(path)
        if backup.exists():
            shutil.rmtree(backup)
    accelerator.wait_for_everyone()


def load_checkpoint(path, accelerator, cfg, tokenizer):
    state = read_state(path, accelerator)
    if state["signature"] != signature(cfg,tokenizer):
        raise ValueError("Resume config/tokenizer differs from checkpoint (model, data, stages, runtime, seed)")
    if state["world_size"] != accelerator.num_processes:
        raise ValueError("Exact resume requires unchanged world_size")
    if not state["save_optimizer"] or not state["save_scheduler"]:
        raise ValueError("Weights-only checkpoint cannot resume; optimizer and scheduler required")
    accelerator.load_state(str(path))
    return state


def load_model_weights(model, path, accelerator=None):
    read_state(path, accelerator)
    weights = torch.load(Path(path)/"pytorch_model.bin",map_location="cpu",weights_only=True)
    # torch.compile is a runtime wrapper, not part of the saved model schema.
    weights = {k.removeprefix("_orig_mod."):v for k,v in weights.items()}
    model = getattr(model,"_orig_mod",model)
    model.load_state_dict(weights,strict=True)


def rotate_checkpoints(root, keep_last):
    if keep_last is None:
        return
    paths = sorted((p for p in Path(root).glob("checkpoint-*") if p.name.split("-")[-1].isdigit() and (p/"COMPLETE").exists()),
                   key=lambda p:int(p.name.split("-")[-1]))
    for path in paths[:-keep_last]:
        shutil.rmtree(path)
