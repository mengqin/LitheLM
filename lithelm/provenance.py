"""Best-effort environment facts; never a dependency compatibility gate."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess

import torch


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def command(args, cwd=None):
    try:
        return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def input_metadata(cfg, root):
    def describe(value):
        if isinstance(value, dict):
            return {k:describe(v) for k,v in value.items()}
        if isinstance(value, list):
            return [describe(v) for v in value]
        path = (root/str(value)).resolve()
        result = dict(original_path=str(value), resolved_path=str(path) if path.exists() else None)
        if path.is_file():
            result.update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        return result
    return dict(corpora={name:dict(source=spec["source"], path=describe(spec["path"]),
                                  data_files=describe(spec["kwargs"].get("data_files", {})))
                         for name,spec in cfg["datasets"].items()},
                tokenizer=describe(cfg["tokenizer"]["artifact"]))


def record_environment(cfg, tokenizer=None, accelerator=None):
    if accelerator is not None and not accelerator.is_main_process:
        return
    from .tokenizer import tokenizer_fingerprint
    source = Path(__file__).resolve().parents[1]
    commit = command(["git", "rev-parse", "HEAD"], source)
    status = command(["git", "status", "--porcelain"], source)
    versions = {name:package_version(name) for name in ["lithelm", "torch", "transformers", "accelerate", "datasets",
                                                       "triton", "triton-windows", "flash-linear-attention",
                                                       "flash-attn", "flash-attn-3"]}
    from . import __version__
    versions["lithelm"] = versions["lithelm"] or __version__
    data = dict(recorded_at=datetime.now(timezone.utc).isoformat(), versions=versions, git_commit=commit,
                git_dirty=bool(status) if status is not None else None, python=platform.python_version(),
                os=platform.platform(), cuda_runtime=torch.version.cuda,
                nvidia_driver=command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                world_size=accelerator.num_processes if accelerator else 1, training_dtype=cfg["runtime"]["precision"],
                inputs=input_metadata(cfg, Path.cwd()),
                tokenizer_fingerprint=tokenizer_fingerprint(tokenizer) if tokenizer is not None else None)
    # Keep each invocation (including resume) without overwriting its provenance.
    path = Path("environment.json")
    if path.exists():
        path = Path("environments") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json")
        path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
