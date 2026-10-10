"""Step-indexed batches and extension resume without changing the default loop."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lithelm.config import normalize
from lithelm.extensions import ComponentSpec, LOSS_COMPOSERS, PlannedBatch, STEP_PLANS, validate_batch
from lithelm.metrics import loss_inputs
from lithelm.model import build_model
from lithelm.platform_utils import build_accelerator
from lithelm.tokenizer import prepare_tokenizer
from lithelm.train import train
from lithelm.trainer import Trainer
from test_data import tiny_config


class FixedPlan:
    calls = []

    def __init__(self, stage, tokenizer, path=None):
        self.stage, self.path = stage, path

    def identity(self):
        return {"format":"synthetic-v1"}

    def batch_at(self, stage_step, rank, microbatch_index):
        self.calls.append((self.stage["name"],stage_step,rank,microbatch_index))
        size = self.stage["batch"]["micro_batch_size"]
        start = stage_step*31+rank*17+microbatch_index*7
        ids = (torch.arange(size*8).reshape(size,8)+start) % 270
        labels = ids.clone()
        labels[:, :2] = -100
        return PlannedBatch(ids,labels=labels,loss_masks={"target":labels != -100},
                            source_name="synthetic",objective=self.stage["type"],metadata={"sample_id":start})


def plan_config(root):
    cfg = tiny_config(root)
    for stage in cfg["stages"]:
        stage.update(curriculum=[])
        stage["data"].update(strategy="step_plan",packing=False,datasets=[],
                             step_plan=dict(name="fixed",path=None,kwargs={}))
    return normalize(cfg)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setitem(STEP_PLANS,"fixed",FixedPlan)
    FixedPlan.calls = []
    monkeypatch.setitem(STEP_PLANS,"global",GlobalPlan)
    monkeypatch.setitem(LOSS_COMPOSERS,"multi",MultiLoss)


def test_plan_keys_and_shapes(tmp_path):
    cfg = plan_config(tmp_path)
    plans = [FixedPlan(cfg["stages"][0],None) for _ in range(2)]
    first = plans[0].batch_at(2,0,0)
    second = plans[1].batch_at(2,0,0)
    assert torch.equal(first.input_ids,second.input_ids)
    assert not torch.equal(first.input_ids,plans[0].batch_at(2,1,0).input_ids)
    assert not torch.equal(first.input_ids,plans[0].batch_at(2,0,1).input_ids)
    assert first.metadata == second.metadata
    validate_batch(first,cfg["stages"][0],280)
    first.input_ids = first.input_ids.repeat(2,1)
    with pytest.raises(ValueError,match="micro_batch_size"):
        validate_batch(first,cfg["stages"][0],280)


def test_plan_ce_retry_keeps_keys(tmp_path, monkeypatch):
    cfg = plan_config(tmp_path)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=lambda row:None))
    try:
        trainer.setup_stage(0,cfg["stages"][0])
        optimizer = trainer.optimizer
        original_step = optimizer.step
        def skip(*args,**kwargs):
            if a.sync_gradients:
                optimizer._is_overflow = True
        monkeypatch.setattr(optimizer,"step",skip)
        assert not trainer.train_update()
        assert trainer.state["stage_step"] == 0
        optimizer._is_overflow = False
        monkeypatch.setattr(optimizer,"step",original_step)
        assert trainer.train_update()
        assert FixedPlan.calls == [("pt",0,0,0),("pt",0,0,1)]*2
    finally:
        a.free_memory()
        a.end_training()


def test_plan_resume_matches_continuous_and_checks_hash(tmp_path):
    cfg = plan_config(tmp_path/"continuous")
    train(cfg)
    expected_calls = list(FixedPlan.calls)
    expected = torch.load(Path(cfg["experiment"]["output_dir"])/"sft/final/pytorch_model.bin",weights_only=True)
    FixedPlan.calls.clear()
    resumed = plan_config(tmp_path/"resumed")
    asset = tmp_path/"plan.bin"
    asset.write_bytes(b"frozen plan identity")
    for stage in resumed["stages"]:
        stage["data"]["step_plan"]["path"] = str(asset)
    train(resumed,stop_after=2)
    root = Path(resumed["experiment"]["output_dir"])
    train(resumed,resume="latest")
    assert FixedPlan.calls == expected_calls
    actual = torch.load(root/"sft/final/pytorch_model.bin",weights_only=True)
    for key in expected:
        torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
    state = json.loads((root/"sft/final/trainer_state.json").read_text())
    assert state["extensions"]["pt"]["step_plan"]["file"]["sha256"]
    asset.write_bytes(b"changed plan identity")
    with pytest.raises(ValueError,match="identity differs"):
        train(resumed,resume="latest")


def test_plan_configuration_conflicts(tmp_path):
    cfg = plan_config(tmp_path)
    for change in (dict(curriculum=[dict(until_step=4)]), dict(data={**cfg["stages"][0]["data"],"packing":True})):
        bad = deepcopy(cfg)
        bad["stages"][0].update(change)
        with pytest.raises(ValueError,match="empty curriculum/datasets"):
            normalize(bad)
    cfg["stages"][0]["data"]["step_plan"]["name"] = "missing"
    with pytest.raises(ValueError,match="Unregistered training extension"):
        train(cfg)


class GlobalPlan(FixedPlan):
    def batch_at(self, stage_step, rank, microbatch_index):
        size, accum = self.stage["batch"]["micro_batch_size"], self.stage["batch"]["grad_accum_steps"]
        index = torch.arange(size)+size*(rank*accum+microbatch_index)
        ids = (index[:,None]*11+torch.arange(8)[None,:]+stage_step) % 270
        labels = ids.clone()
        masks = torch.arange(8)[None,:] >= (index[:,None] % 5+1)
        labels[~masks] = -100
        return PlannedBatch(ids,labels=labels,loss_masks={"token":masks,"example":masks & (index[:,None]<3)},
                            source_name="synthetic",objective="pt")


class MultiLoss:
    def __init__(self, stage, optional=False):
        self.optional = optional

    def identity(self):
        return {"format":"token-and-example-v1"}

    def components(self, context):
        return {"token":ComponentSpec(.3),"example":ComponentSpec(1.7,required=not self.optional)}

    def masks(self,batch,context):
        offset = int(context.target_mode == "shifted")
        valid = batch.labels[:,offset:] != -100
        return {name:batch.loss_masks.get(name,torch.ones_like(batch.labels,dtype=torch.bool))[:,offset:] & valid
                for name in ("token","example")}

    def denominators(self,batch,context):
        masks = self.masks(batch,context)
        return {"token":masks["token"].sum(),"example":masks["example"].any(1).sum()}

    def numerators(self,output,batch,context):
        predictions,labels = loss_inputs(output.logits,batch.labels,context.target_mode)
        losses = F.cross_entropy(predictions.float().transpose(1,2),labels,ignore_index=-100,reduction="none")
        masks = self.masks(batch,context)
        result = {"token":(losses*masks["token"]).sum()}
        if masks["example"].any():
            result["example"] = ((losses*masks["example"]).sum(1)/masks["example"].sum(1).clamp_min(1)).sum()
        return result


def composition_config(root, accumulation=1, world_size=1, mode="shifted"):
    cfg = plan_config(root)
    cfg["stages"] = cfg["stages"][:1]
    stage = cfg["stages"][0]
    stage["data"]["step_plan"]["name"] = "global"
    stage["batch"].update(micro_batch_size=12//(accumulation*world_size),grad_accum_steps=accumulation,max_grad_norm=0.)
    stage["loss"].update(type="custom",name="multi")
    stage["targets"]["mode"] = mode
    stage["intervals"].update(log=1,stats=0)
    return normalize(cfg)


def one_composed_update(cfg):
    a = build_accelerator(cfg)
    rows, gradients = [], {}
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=rows.append))
    try:
        trainer.setup_stage(0,cfg["stages"][0])
        def capture(*args):
            gradients.update({name:p.grad.detach().cpu().clone() for name,p in trainer.raw.named_parameters()})
        handle = trainer.optimizer.optimizer.register_step_pre_hook(capture)
        assert trainer.train_update()
        handle.remove()
        weights = {name:p.detach().cpu().clone() for name,p in trainer.raw.named_parameters()}
        return gradients,weights,[row for row in rows if row.get("event") == "train"][-1]
    finally:
        a.free_memory()
        a.end_training()


@pytest.mark.parametrize("mode", ["shifted","next_token"])
def test_component_gradients_match_accumulation_and_global_reference(tmp_path,mode):
    baseline = one_composed_update(composition_config(tmp_path/"one",mode=mode))
    for accumulation in (2,3):
        actual = one_composed_update(composition_config(tmp_path/str(accumulation),accumulation,mode=mode))
        for group in (0,1):
            for name in baseline[group]:
                torch.testing.assert_close(actual[group][name],baseline[group][name],atol=2e-6,rtol=2e-5)
        for key,value in baseline[2].items():
            if key.startswith("train/loss"):
                assert actual[2][key] == pytest.approx(value,rel=2e-6)
    cfg = composition_config(tmp_path/"reference",mode=mode)
    a = build_accelerator(cfg)
    try:
        model = build_model(cfg["model"])
        batch = GlobalPlan(cfg["stages"][0],None).batch_at(0,0,0)
        output = model(**batch.model_inputs(labels=False))
        prediction,labels = loss_inputs(output.logits,batch.labels,mode)
        ce = F.cross_entropy(prediction.float().transpose(1,2),labels,ignore_index=-100,reduction="none")
        token = batch.loss_masks["token"][:,int(mode == "shifted"):]
        example = batch.loss_masks["example"][:,int(mode == "shifted"):]
        means = (ce*example).sum(1)/example.sum(1).clamp_min(1)
        expected = .3*(ce*token).sum()/token.sum()+1.7*means.sum()/example.any(1).sum()
        incorrect = .3*(ce*token).sum()/token.sum()+1.7*(ce*example).sum()/example.sum()
        assert abs(expected.item()-incorrect.item()) > 1e-5
        expected.backward()
        for name,p in model.named_parameters():
            torch.testing.assert_close(p.grad,baseline[0][name],atol=2e-6,rtol=2e-5)
        assert baseline[2]["train/loss_total"] == pytest.approx(expected.item(),rel=2e-6)
        assert "train/nll" not in baseline[2] and "train/ppl" not in baseline[2]
        assert baseline[2]["data/source_samples/synthetic"] == 12
        assert baseline[2]["data/source_targets/synthetic"] == token.sum().item()
        assert baseline[2]["data/source_objective/synthetic"] == "pt"
    finally:
        a.end_training()


def test_composer_with_default_sampling(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["stages"] = cfg["stages"][:1]
    cfg["stages"][0]["loss"].update(type="custom",name="multi")
    cfg["stages"][0]["intervals"]["log"] = 1
    _,_,row = one_composed_update(normalize(cfg))
    assert row["train/loss_total"] > 0 and "train/nll" not in row


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["normal","invalid"])
def test_composer_two_ranks_and_rank_local_failure(tmp_path,kind):
    import os
    from lithelm.config import save_config
    from test_distributed import run_distributed
    cfg = composition_config(tmp_path/"distributed",3,2)
    cfg["runtime"]["device"] = os.environ.get("LITHELM_TEST_DEVICE","cpu")
    if cfg["runtime"]["device"] == "cuda":
        if torch.cuda.device_count()<2:
            pytest.skip("two CUDA devices required")
        # Keep the mathematical equivalence check in fp32 across CPU/CUDA.
        cfg["runtime"]["precision"] = "fp32"
    if kind == "normal":
        single = composition_config(tmp_path/"single",3)
        single["runtime"].update(device=cfg["runtime"]["device"],precision="fp32")
        baseline = one_composed_update(single)
    path,output = tmp_path/"distributed.yaml",tmp_path/"result.pt"
    save_config(cfg,path)
    script = Path(__file__).parent/"fixtures/extensions_rank.py"
    run_distributed(str(script),path,output,kind)
    if kind == "normal":
        actual = torch.load(output,weights_only=True)
        tolerance = 3e-5 if cfg["runtime"]["device"] == "cuda" else 2e-6
        for group in (0,1):
            for name in baseline[group]:
                torch.testing.assert_close(actual[group][name],baseline[group][name],atol=tolerance,rtol=5e-5)
        for key,value in baseline[2].items():
            if key.startswith("train/loss"):
                assert actual[2][key] == pytest.approx(value,rel=2e-5)


@pytest.mark.parametrize("case", ["required_zero","optional_zero","negative","nan","grad_denominator","nan_numerator","vector_numerator","detached_numerator","nan_weight"])
def test_composer_validation_and_optional_zero(tmp_path,monkeypatch,case):
    from lithelm.extensions import StepContext
    from lithelm.loss_composition import ComposedUpdate
    cfg = composition_config(tmp_path)
    a = build_accelerator(cfg)
    batch = GlobalPlan(cfg["stages"][0],None).batch_at(0,0,0)
    composer = MultiLoss(cfg["stages"][0],optional=case == "optional_zero")
    context = StepContext("pt",0,0,"shifted")
    output = build_model(cfg["model"])(**batch.model_inputs(labels=False))
    if case.endswith("zero"):
        batch.loss_masks["example"].zero_()
    if case in ("negative","nan","grad_denominator"):
        value = {"negative":-1.,"nan":float("nan"),"grad_denominator":torch.tensor(1.,requires_grad=True)}[case]
        monkeypatch.setattr(composer,"denominators",lambda *args:{"token":value,"example":1.})
    if case == "nan_weight":
        monkeypatch.setattr(composer,"components",lambda *args:{"token":ComponentSpec(float("nan"))})
    if case.endswith("numerator"):
        value = {"nan_numerator":output.logits.sum()*float("nan"),"vector_numerator":output.logits.flatten()[:2],
                 "detached_numerator":torch.tensor(1.,requires_grad=True)}[case]
        monkeypatch.setattr(composer,"numerators",lambda *args:{"token":value,"example":output.logits.sum()})
    try:
        if case == "optional_zero":
            update = ComposedUpdate(composer,[batch],context,a,None)
            update.loss(output,batch,0).backward()
            assert update.metrics()["train/loss/example"] == 0
        else:
            with pytest.raises(ValueError):
                update = ComposedUpdate(composer,[batch],context,a,None)
                update.loss(output,batch,0)
    finally:
        a.end_training()


def test_registered_recipe_two_stages_resume_and_independent_validation(tmp_path):
    from examples.extensions.run import make_example
    from lithelm.config import load_config
    uninterrupted = load_config(make_example(tmp_path/"uninterrupted"))
    interrupted = load_config(make_example(tmp_path/"interrupted"))
    train(uninterrupted)
    train(interrupted,stop_after=2)
    state = train(interrupted,resume="latest")
    assert state["global_step"] == 7 and state["stage_step"] == 3
    root = Path(interrupted["experiment"]["output_dir"])
    expected = torch.load(Path(uninterrupted["experiment"]["output_dir"])/"sft/final/pytorch_model.bin",weights_only=True)
    actual = torch.load(root/"sft/final/pytorch_model.bin",weights_only=True)
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],atol=0,rtol=0)
    rows = [json.loads(row) for row in (root/"metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows if row.get("event") == "train"]
    assert [row["optim/lr"] for row in updates] == [.0002,.0005,.0001,0.,.0007,.0003,0.]
    assert all("train/loss_total" in row and "train/nll" not in row for row in updates)
    assert any("eval/nll" in row and "eval/ppl" in row for row in rows)
    assert (root/"extensions.json").is_file()


def test_old_checkpoint_default_signature_exact_resume(tmp_path):
    import hashlib
    from lithelm.tokenizer import tokenizer_fingerprint
    from lithelm.config import load_config
    cfg = tiny_config(tmp_path)
    train(cfg,stop_after=2)
    path = Path(cfg["experiment"]["output_dir"])/"pt/checkpoint-2/trainer_state.json"
    saved_config = load_config(path.parent/"resolved_config.yaml")
    legacy = deepcopy({k:saved_config[k] for k in ("model","datasets","stages","runtime")})
    for stage in legacy["stages"]:
        stage.pop("targets")
        stage.pop("loss")
        stage["data"].pop("strategy")
        stage["data"].pop("step_plan")
        for key in ("path","name","kwargs"):
            stage["lr_schedule"].pop(key)
    legacy.update(seed=cfg["experiment"]["seed"],tokenizer_hash=tokenizer_fingerprint(prepare_tokenizer(cfg)))
    state = json.loads(path.read_text())
    assert "extensions" not in state
    state["signature"] = hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(state),encoding="utf-8")
    assert train(cfg,resume=str(path.parent))["global_step"] == 7


def test_planned_payload_and_tensor_metadata_stay_separate():
    ids = torch.ones(2,4,dtype=torch.long)
    batch = PlannedBatch(ids,metadata={"sample_ids":torch.arange(2),"tag":"hello","nested":[torch.ones(1)]})
    moved = batch.to(torch.device("meta"))
    assert moved.metadata["sample_ids"].device.type == moved.metadata["nested"][0].device.type == "meta"
    assert moved.metadata["tag"] == "hello"
    assert set(moved.model_inputs()) == {"input_ids"}


@pytest.mark.parametrize("factor,denominator,weight,count", [(1e20,1e20,1e20,1),(1e308,1.,.5,3)])
def test_component_scaling_avoids_intermediate_overflow(tmp_path,factor,denominator,weight,count):
    from lithelm.extensions import StepContext
    from lithelm.loss_composition import ComposedUpdate
    a = build_accelerator(composition_config(tmp_path))
    value = torch.ones(1,dtype=torch.float64 if count == 3 else torch.float32,requires_grad=True)
    output = SimpleNamespace(logits=value)
    composer = SimpleNamespace(components=lambda context:{"component":ComponentSpec(weight)},
        denominators=lambda batch,context:{"component":denominator},
        numerators=lambda output,batch,context:{"component":output.logits.sum()*factor})
    try:
        update = ComposedUpdate(composer,[None]*count,StepContext("pt",0,0,"shifted"),a,None)
        for index in range(count):
            update.loss(output,None,index).backward()
        assert torch.isfinite(value.grad).all()
        assert update.metrics()["train/loss_total"] == pytest.approx(weight*(factor/denominator))
    finally:
        a.end_training()
