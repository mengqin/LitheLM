"""Real rank-local validation with expected failures caught by every worker."""
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_extension_hardening import SelectiveLoss
from test_extensions import GlobalPlan, MultiLoss, one_composed_update
from lithelm.config import load_config
from lithelm.extensions import LOSS_COMPOSERS, STEP_PLANS, file_identity
from lithelm.platform_utils import build_accelerator
from lithelm.model import build_model
from lithelm.tokenizer import prepare_tokenizer
from lithelm.trainer import Trainer
from types import SimpleNamespace

kind,case,config,output = sys.argv[1:]
rank = int(os.environ["RANK"])
cfg = load_config(config)
if kind == "loss":
    STEP_PLANS["global"] = GlobalPlan
    LOSS_COMPOSERS["multi"] = SelectiveLoss
    try:
        result = one_composed_update(cfg)
    except ValueError as exc:
        assert case == "empty" and "No active weighted loss component" in str(exc)
    else:
        assert case == "rank1_only"
        if rank == 0:
            torch.save(result,output)
elif kind == "file":
    a = build_accelerator(cfg)
    path = Path(config).parent/f"rank-{rank}.bin"
    original_open = Path.open
    if case == "unreadable":
        def denied(self,*args,**kwargs):
            if rank == 1 and self == path:
                raise PermissionError("rank-local frozen file is unreadable")
            return original_open(self,*args,**kwargs)
        Path.open = denied
    try:
        try:
            identity = file_identity(path,a)
        except ValueError as exc:
            message = "sha256/size mismatch" if case == "different" else "Frozen extension file"
            assert case != "same" and message in str(exc)
        else:
            assert case == "same", f"rank {rank}: local file divergence was accepted"
            assert identity["path"] == str(Path(config).parent/"rank-0.bin")
            assert identity["size"] == len(b"same payload")
    finally:
        Path.open = original_open
        a.end_training()
else:
    STEP_PLANS["global"] = GlobalPlan
    LOSS_COMPOSERS["multi"] = MultiLoss
    a = build_accelerator(cfg)
    path = Path(config).parent/f"rank-{rank}.bin"
    configured = Path(cfg["stages"][0]["data"]["step_plan"]["path"])
    original_open = Path.open
    def local_read(self,*args,**kwargs):
        return original_open(path if self == configured else self,*args,**kwargs)
    Path.open = local_read
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=lambda row:None))
    calls = []
    handle = trainer.raw.register_forward_pre_hook(lambda *args:calls.append("forward"))
    try:
        if case == "resume_changed":
            trainer.setup_stage(0,cfg["stages"][0])
            assert trainer.train_update()
            trainer.save("checkpoint-1")
            calls.clear()
            if rank == 1:
                path.write_bytes(b"evil payload")
            a.wait_for_everyone()
            trainer = Trainer(cfg,trainer.raw,trainer.tokenizer,a,SimpleNamespace(log=lambda row:None))
            resume = Path(cfg["experiment"]["output_dir"])/"pt/checkpoint-1"
        else:
            resume = None
        try:
            trainer.setup_stage(0,cfg["stages"][0],resume)
        except ValueError as exc:
            assert "StepPlan" in str(exc) and "sha256/size mismatch" in str(exc)
        else:
            raise AssertionError("rank-local plan change was accepted")
        assert calls == [] and trainer.state["stage_step"] == trainer.state["global_step"] == 0
    finally:
        handle.remove()
        Path.open = original_open
        a.free_memory()
        a.end_training()
