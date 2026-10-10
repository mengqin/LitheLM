"""Empty objectives fail before updates; frozen file contents agree per rank."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lithelm.config import save_config
from lithelm.extensions import ComponentSpec, StepContext, file_identity
from lithelm.loss_composition import ComposedUpdate
from lithelm.model import build_model
from lithelm.platform_utils import build_accelerator
from lithelm.tokenizer import prepare_tokenizer
from lithelm.trainer import Trainer
from test_extensions import MultiLoss, composition_config, one_composed_update


class SelectiveLoss(MultiLoss):
    def __init__(self, stage, scope="rank1_only"):
        super().__init__(stage,optional=True)
        self.scope = scope

    def components(self,context):
        weights = (0.,0.) if self.scope == "zero_weights" else (
            (0.,1.7) if self.scope == "inactive_mix" else (.3,1.7))
        return {name:ComponentSpec(weight,False) for name,weight in zip(("token","example"),weights)}

    def masks(self,batch,context):
        masks = super().masks(batch,context)
        masks["example"] = torch.zeros_like(masks["example"])
        if self.scope == "empty":
            masks["token"] = torch.zeros_like(masks["token"])
        elif self.scope == "rank1_only":
            masks["token"] = masks["token"] & (batch.input_ids[:, :1] >= 6*11)
        return masks

    def numerators(self,output,batch,context):
        if not self.masks(batch,context)["token"].any():
            return {}
        return super().numerators(output,batch,context)


def assert_state_equal(actual,expected):
    if isinstance(actual,torch.Tensor):
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    elif isinstance(actual,dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_state_equal(actual[key],expected[key])
    elif isinstance(actual,(list,tuple)):
        assert len(actual) == len(expected)
        for left,right in zip(actual,expected):
            assert_state_equal(left,right)
    else:
        assert actual == expected


@pytest.mark.parametrize("scope", ["empty","zero_weights","inactive_mix"])
def test_inactive_composition_preserves_model_optimizer_and_steps(tmp_path,scope,monkeypatch):
    from lithelm.extensions import LOSS_COMPOSERS, STEP_PLANS
    from test_extensions import GlobalPlan
    monkeypatch.setitem(STEP_PLANS,"global",GlobalPlan)
    monkeypatch.setitem(LOSS_COMPOSERS,"multi",MultiLoss)
    cfg = composition_config(tmp_path,2)
    cfg["stages"][0]["optimizer"]["weight_decay"] = .1
    a = build_accelerator(cfg)
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=lambda row:None))
    calls = []
    try:
        trainer.setup_stage(0,cfg["stages"][0])
        assert trainer.train_update()  # Populate Adam moments before testing failure.
        trainer.composer = SelectiveLoss(trainer.stage,scope)
        expected = deepcopy((trainer.raw.state_dict(),trainer.optimizer.state_dict(),
                             trainer.scheduler.state_dict(),trainer.state))
        forward_hook = trainer.model.register_forward_pre_hook(lambda *args:calls.append("forward"))
        step_hook = trainer.optimizer.optimizer.register_step_pre_hook(lambda *args:calls.append("step"))
        try:
            with pytest.raises(ValueError,match="No active weighted loss component"):
                trainer.train_update()
        finally:
            forward_hook.remove()
            step_hook.remove()
        assert calls == []
        assert_state_equal((trainer.raw.state_dict(),trainer.optimizer.state_dict(),
                            trainer.scheduler.state_dict(),trainer.state),expected)
    finally:
        a.free_memory()
        a.end_training()


def test_active_negative_weight_and_optional_absence(tmp_path):
    a = build_accelerator(composition_config(tmp_path))
    value = torch.ones(1,requires_grad=True)
    composer = SimpleNamespace(components=lambda context:{"active":ComponentSpec(-2.),"absent":ComponentSpec(1.,False)},
        denominators=lambda *args:{"active":4.,"absent":0.},
        numerators=lambda output,*args:{"active":output.logits.sum()*3})
    try:
        update = ComposedUpdate(composer,[None],StepContext("pt",0,0,"shifted"),a,None)
        update.loss(SimpleNamespace(logits=value),None,0).backward()
        torch.testing.assert_close(value.grad,torch.full_like(value,-1.5),atol=0,rtol=0)
        assert update.metrics()["train/loss/absent"] == 0.
    finally:
        a.end_training()


def test_file_identity_single_rank_keeps_provenance(tmp_path):
    path = tmp_path/"frozen.bin"
    content = b"frozen contents"*100000  # More than one streaming hash chunk.
    path.write_bytes(content)
    a = build_accelerator(composition_config(tmp_path/"config"))
    try:
        assert file_identity(path,a) == dict(path=str(path.resolve()),size=len(content),sha256=hashlib.sha256(content).hexdigest())
    finally:
        a.end_training()


def test_non_main_rank_cannot_accept_missing_local_file(tmp_path,monkeypatch):
    from accelerate import utils
    path = tmp_path/"rank-1-missing.bin"
    main_record = dict(path=str(tmp_path/"rank-0.bin"),size=1,sha256=hashlib.sha256(b"x").hexdigest())
    monkeypatch.setattr(utils,"broadcast_object_list",lambda payload:payload.__setitem__(0,main_record))
    def gather(values,value):
        values[:] = [value]*len(values)
    monkeypatch.setattr(torch.distributed,"all_gather_object",gather)
    with pytest.raises(ValueError,match="Frozen extension file"):
        file_identity(path,SimpleNamespace(is_main_process=False,num_processes=2))


@pytest.mark.integration
@pytest.mark.parametrize("case", ["rank1_only","empty"])
def test_inactive_or_rank_local_objectives_two_ranks(tmp_path,case,monkeypatch):
    from lithelm.extensions import LOSS_COMPOSERS, STEP_PLANS
    from test_distributed import run_distributed
    from test_extensions import GlobalPlan
    monkeypatch.setitem(STEP_PLANS,"global",GlobalPlan)
    monkeypatch.setitem(LOSS_COMPOSERS,"multi",SelectiveLoss)
    cfg = composition_config(tmp_path/"distributed",3,2)
    cfg["runtime"]["device"] = os.environ.get("LITHELM_TEST_DEVICE","cpu")
    if cfg["runtime"]["device"] == "cuda" and torch.cuda.device_count()<2:
        pytest.skip("two CUDA devices required")
    cfg["stages"][0]["loss"]["kwargs"] = {"scope":case}
    if case == "rank1_only":
        single = composition_config(tmp_path/"single",3)
        single["runtime"]["device"] = cfg["runtime"]["device"]
        single["stages"][0]["loss"]["kwargs"] = {"scope":case}
        expected = one_composed_update(single)
    path,output = tmp_path/"config.yaml",tmp_path/"result.pt"
    save_config(cfg,path)
    run_distributed(Path(__file__).parent/"fixtures/hardening_rank.py","loss",case,path,output)
    if case == "rank1_only":
        actual = torch.load(output,weights_only=True)
        tolerance = 3e-5 if cfg["runtime"]["device"] == "cuda" else 2e-6
        for group in (0,1):
            for name in expected[group]:
                torch.testing.assert_close(actual[group][name],expected[group][name],atol=tolerance,rtol=5e-5)
        assert actual[2]["train/loss_total"] == pytest.approx(expected[2]["train/loss_total"],rel=2e-5)


@pytest.mark.integration
@pytest.mark.parametrize("case", ["same","different","missing","unreadable"])
def test_rank_local_frozen_files_agree_or_fail_together(tmp_path,case):
    from test_distributed import run_distributed
    cfg = composition_config(tmp_path/"config",3,2)
    cfg["runtime"]["device"] = os.environ.get("LITHELM_TEST_DEVICE","cpu")
    if cfg["runtime"]["device"] == "cuda" and torch.cuda.device_count()<2:
        pytest.skip("two CUDA devices required")
    for rank in range(2):
        if case == "missing" and rank == 1:
            continue
        (tmp_path/f"rank-{rank}.bin").write_bytes(b"evil payload" if case == "different" and rank == 1 else b"same payload")
    path = tmp_path/"config.yaml"
    save_config(cfg,path)
    run_distributed(Path(__file__).parent/"fixtures/hardening_rank.py","file",case,path,tmp_path/"identity.pt")


@pytest.mark.integration
@pytest.mark.parametrize("case", ["different","resume_changed"])
def test_step_plan_preflight_rejects_rank_local_content_before_update(tmp_path,case):
    from test_distributed import run_distributed
    cfg = composition_config(tmp_path/"config",3,2)
    cfg["runtime"]["device"] = os.environ.get("LITHELM_TEST_DEVICE","cpu")
    if cfg["runtime"]["device"] == "cuda" and torch.cuda.device_count()<2:
        pytest.skip("two CUDA devices required")
    shared = tmp_path/"configured-plan.bin"
    shared.write_bytes(b"same payload")
    cfg["stages"][0]["data"]["step_plan"]["path"] = str(shared)
    for rank in range(2):
        (tmp_path/f"rank-{rank}.bin").write_bytes(b"evil payload" if case == "different" and rank == 1 else b"same payload")
    path = tmp_path/"config.yaml"
    save_config(cfg,path)
    run_distributed(Path(__file__).parent/"fixtures/hardening_rank.py","preflight",case,path,tmp_path/"unused.pt")
