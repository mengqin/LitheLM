"""Training composition and no other training logic."""
from pathlib import Path

from .config import save_config
from .logging_utils import Logger
from .provenance import record_environment
from .model import build_model
from .platform_utils import build_accelerator
from .stages import run_stages
from .tokenizer import prepare_tokenizer
from .workspace import run_directory, latest_checkpoint


def train(cfg,resume=None,stop_after=None):
    a = build_accelerator(cfg)
    resume = str(Path(resume).resolve()) if resume and resume != "latest" else resume
    try:
        with run_directory(cfg, a) as (cfg, root):
            if resume == "latest":
                resume = latest_checkpoint(root, a)
                if resume is None:
                    raise ValueError("No published checkpoint in this run")
            if not (resume or cfg["experiment"]["resume_from"]) and latest_checkpoint(root, a):
                raise ValueError("This run already has checkpoints; use --resume to continue")
            return _train(cfg, a, resume, stop_after)
    finally:
        a.end_training()


def _train(cfg,a,resume,stop_after):
    root = Path(cfg["experiment"]["output_dir"])
    if a.is_main_process:
        root.mkdir(parents=True,exist_ok=True)
        save_config(cfg,root/"resolved_config.yaml")
    with a.main_process_first():
        tokenizer = prepare_tokenizer(cfg)
    if a.is_main_process:
        save_config(cfg,root/"resolved_config.yaml")
    record_environment(cfg, tokenizer, a)
    model = build_model(cfg["model"])
    if cfg["runtime"]["compile"]:
        import torch
        model = torch.compile(model)
    logger = Logger(cfg["logging"],root,a.is_main_process, {k:s.format for k,s in model.stat_specs().items() if s.format})
    try:
        return run_stages(cfg,model,tokenizer,a,logger,resume,stop_after)
    finally:
        logger.close()
