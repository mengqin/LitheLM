"""Run outputs with shared input references, or explicit self-contained copies."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone
import json
import filecmp
import os
from pathlib import Path, PureWindowsPath
import re
import shutil


def find_run(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / "run.json").is_file():
            return parent
    return None


def relative_path(value):
    path = Path(value)
    windows = PureWindowsPath(value)
    if path.is_absolute() or windows.drive or windows.root or ".." in path.parts or ".." in windows.parts:
        raise ValueError(f"Output path must stay inside the run directory: {value}")
    return path


def model_size(cfg):
    size = cfg["experiment"]["model_size"]
    if size is None:
        import torch
        from .model import build_model
        with torch.device("meta"):
            count = sum(p.numel() for p in build_model(cfg["model"]).parameters())
        unit, divisor = next((u, d) for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3), ("", 1)) if count >= d)
        size = f"{count/divisor:.3f}".rstrip("0").rstrip(".") + unit
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[KMB]?", str(size)):
        raise ValueError("experiment.model_size must be a parameter size such as 7.4M")
    return str(size)


def import_inputs(cfg, root):
    """Keep shared inputs by reference unless the user requests a portable bundle."""
    def asset(value, destination):
        source = Path(value)
        if not source.is_absolute():
            relative_path(value)
            source = (root / source).resolve()
        try:
            return source.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
        if not cfg["workspace"]["copy_inputs"]:
            return str(source.resolve())
        target = root / destination / source.name
        if not source.exists():
            return target.relative_to(root).as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            originals = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file()) if source.is_dir() else [Path(".")]
            copies = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file()) if target.is_dir() else [Path(".")]
            pairs = [(source/p, target/p) for p in originals] if source.is_dir() else [(source, target)]
            if originals != copies or any(not a.is_file() or not b.is_file() or not filecmp.cmp(a, b, shallow=False) for a, b in pairs):
                raise FileExistsError(f"Imported input differs; refusing to overwrite: {target}")
            return target.relative_to(root).as_posix()
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return target.relative_to(root).as_posix()

    for name, spec in cfg["datasets"].items():
        if spec["source"] != "hf" or Path(spec["path"]).is_absolute():
            spec["path"] = asset(spec["path"], Path("data/raw") / name)
        if spec["cache_dir"] is not None and cfg["workspace"]["copy_inputs"]:
            spec["cache_dir"] = "cache/hf/datasets"
        def data_files(value, location):
            if isinstance(value, dict):
                return {k: data_files(v, location / str(i)) for i, (k, v) in enumerate(value.items())}
            if isinstance(value, list):
                return [data_files(v, location / str(i)) for i, v in enumerate(value)]
            return value if "://" in str(value) else asset(value, location)
        if "data_files" in spec["kwargs"]:
            spec["kwargs"]["data_files"] = data_files(spec["kwargs"]["data_files"], Path("data/raw") / name)
    t = cfg["tokenizer"]
    if t["type"] == "bpe" or Path(t["artifact"]).is_absolute():
        t["artifact"] = asset(t["artifact"], Path("tokenizer"))
    if cfg["workspace"]["copy_inputs"]:
        cfg["paths"].update(corpus_root="data/raw", tokenizer_root="tokenizer", run_root=".")
    relative_path(cfg["logging"]["jsonl_file"])
    relative_path(cfg["logging"]["csv_file"])
    relative_path(cfg["logging"]["tensorboard"]["dir"])


@contextmanager
def run_directory(cfg, accelerator=None):
    """Allocate on rank zero, then enter the same run directory on every rank.

    The cwd and environment are process-wide; run one experiment per process.
    """
    previous = Path.cwd()
    bound = deepcopy(cfg)
    bound["paths"] = {key:str((previous/value).resolve()) for key,value in bound["paths"].items()}
    payload = [None]
    main = accelerator is None or accelerator.is_main_process
    if main:
        try:
            exp = bound["experiment"]
            if exp["output_dir"] is not None:
                root = Path(exp["output_dir"]).resolve()
                if not (root / "run.json").is_file():
                    raise ValueError("experiment.output_dir must reference an existing run.json; omit it for a new run")
                metadata = json.loads((root / "run.json").read_text())
                exp.update({k: metadata[k] for k in ("name", "model_size", "start_date")})
            else:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", exp["name"]):
                    raise ValueError("experiment.name must be a safe model name")
                exp["model_size"] = model_size(bound)
                exp["start_date"] = exp["start_date"] or datetime.now(timezone.utc).date().isoformat()
                date.fromisoformat(exp["start_date"])
                stem = f"{exp['name']}_{exp['model_size']}_{exp['start_date']}"
                outputs = Path(bound["paths"]["run_root"])
                outputs.mkdir(parents=True, exist_ok=True)
                for index in range(1, 10000):
                    root = outputs / (stem if index == 1 else f"{stem}-{index:02d}")
                    try:
                        root.mkdir()
                        break
                    except FileExistsError:
                        continue
                else:
                    raise FileExistsError(f"Too many runs named {stem}")
                metadata = {k: exp[k] for k in ("name", "model_size", "start_date")}
                metadata.update(run_id=root.name, created_at=datetime.now(timezone.utc).isoformat())
                (root / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            from .provenance import input_metadata
            inputs = input_metadata(bound, root)
            import_inputs(bound, root)
            if not (root/"inputs.json").exists():
                (root/"inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
            if exp["resume_from"]:
                source = Path(exp["resume_from"])
                exp["resume_from"] = str((source if source.is_absolute() else root/source).resolve().relative_to(root))
            exp["output_dir"] = "."
            payload[0] = (str(root), bound, None)
        except Exception as exc:
            payload[0] = (None, None, f"{type(exc).__name__}: {exc}")
    if accelerator is not None and accelerator.num_processes > 1:
        from accelerate.utils import broadcast_object_list
        broadcast_object_list(payload)
    directory, bound, error = payload[0]
    if error:
        raise ValueError(error)
    root = Path(directory)
    hf = root/"cache/hf" if bound["workspace"]["copy_inputs"] else Path(bound["paths"]["corpus_root"])/"hf"
    env = {"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf/"hub"),
           "HF_DATASETS_CACHE": str(hf/"datasets"), "HF_XET_CACHE": str(hf/"xet"), "TRITON_CACHE_DIR": str(root/"cache/triton"),
           "TORCHINDUCTOR_CACHE_DIR": str(root/"cache/inductor"), "CUDA_CACHE_PATH": str(root/"cache/cuda"),
           "MPLCONFIGDIR": str(root/"cache/matplotlib"), "WANDB_DIR": str(root/"logs/wandb")}
    env.update(TORCH_HOME=str(root/"cache/torch"), WANDB_CACHE_DIR=str(root/"cache/wandb"),
               WANDB_CONFIG_DIR=str(root/"cache/wandb-config"))
    old_env = {k: os.environ.get(k) for k in env}
    # Accelerate may have imported Hub constants before the per-run environment.
    # Route implicit dataset downloads to the selected shared/copied asset root.
    from huggingface_hub import constants as hub
    hub_paths = {"HF_HOME": hf, "HF_HUB_CACHE": hf/"hub", "HUGGINGFACE_HUB_CACHE": hf/"hub",
                 "HF_ASSETS_CACHE": hf/"assets", "HF_XET_CACHE": hf/"xet"}
    old_hub = {k: getattr(hub, k) for k in hub_paths if hasattr(hub, k)}
    try:
        os.environ.update(env)
        for key in old_hub:
            setattr(hub, key, str(hub_paths[key]))
        os.chdir(root)
        yield bound, root
    finally:
        os.chdir(previous)
        for key, value in old_hub.items():
            setattr(hub, key, value)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def latest_checkpoint(root, accelerator=None):
    from .checkpoint import recover_checkpoint
    if accelerator is None or accelerator.is_main_process:
        for path in sorted(Path(root).glob("*/*")):
            name = path.name.removesuffix(".writing").removesuffix(".previous")
            if name in ("best", "final") or (name.startswith("checkpoint-") and name.removeprefix("checkpoint-").isdigit()):
                recover_checkpoint(path.with_name(name))
    if accelerator is not None:
        accelerator.wait_for_everyone()
    candidates = []
    for complete in Path(root).glob("*/*/COMPLETE"):
        path = complete.parent
        numbered = path.name.startswith("checkpoint-") and path.name.removeprefix("checkpoint-").isdigit()
        if path.name in ("best", "final") or numbered:
            state = json.loads((path / "trainer_state.json").read_text())
            candidates.append((state["global_step"], path.name == "final", str(path)))
    return max(candidates)[2] if candidates else None
