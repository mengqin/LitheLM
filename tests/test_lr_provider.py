"""Absolute LR indexing, stage inheritance, and frozen-content resume checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lithelm.config import load_config, normalize, save_config
from lithelm.extensions import LR_PROVIDERS
from lithelm.lr_provider import ValuesFileLR
from lithelm.model import build_model
from lithelm.platform_utils import build_accelerator
from lithelm.scheduler import build_scheduler
from lithelm.tokenizer import prepare_tokenizer
from lithelm.train import train
from lithelm.trainer import Trainer
from test_data import tiny_config


VALUES = [0.0002,0.0005,0.0001,0.]


def lr_config(root, stages=1):
    cfg = tiny_config(root)
    cfg["stages"] = cfg["stages"][:stages]
    for i,stage in enumerate(cfg["stages"]):
        path = Path(root)/f"lr-{i}.bin"
        values = VALUES if i == 0 else [.0007,.0003,0.]
        np.asarray(values,dtype="<f8").tofile(path)
        stage["lr_schedule"] = dict(type="values_file",path=str(path))
        stage["intervals"].update(log=1)
    return normalize(cfg)


def test_values_file_absolute_groups_first_last_and_bounds(tmp_path):
    path = tmp_path/"lr.bin"
    np.asarray(VALUES,dtype="<f8").tofile(path)
    provider = ValuesFileLR(path,4)
    params = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.AdamW([dict(params=[params[0]],lr=.3),dict(params=[params[1]],lr=.7)])
    scheduler = build_scheduler(optimizer,{},provider=provider,max_steps=4)
    actual = []
    for expected in VALUES:
        actual.append(optimizer.param_groups[0]["lr"])
        assert [g["lr"] for g in optimizer.param_groups] == [expected]*2
        optimizer.step()
        scheduler.step()
    assert actual == VALUES and scheduler.last_epoch == 4
    assert "provider" not in scheduler.state_dict()
    for index in (-1,4,True):
        with pytest.raises(IndexError):
            provider.value_at(index)


@pytest.mark.parametrize("values", [[1.]*3,[1.]*5,[float("nan"),1.,1.,1.],[float("inf"),1.,1.,1.],[-1.,1.,1.,1.]])
def test_values_file_invalid_content(tmp_path,values):
    path = tmp_path/"bad.bin"
    np.asarray(values,dtype="<f8").tofile(path)
    with pytest.raises(ValueError):
        ValuesFileLR(path,4)


def test_lr_resume_logs_and_content_identity(tmp_path):
    cfg = lr_config(tmp_path)
    train(cfg,stop_after=2)
    train(cfg,resume="latest")
    root = Path(cfg["experiment"]["output_dir"])
    rows = [json.loads(row) for row in (root/"metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows if row.get("event") == "train"]
    assert [row["optim/lr"] for row in updates] == VALUES
    assert all(row["optim/lr_phase"] == "values_file" and "optim/wsd_phase" not in row for row in updates)
    # Resuming a completed stage must not query an out-of-range LR index.
    assert train(cfg,resume="latest")["stage_step"] == 4
    path = cfg["stages"][0]["lr_schedule"]["path"]
    np.asarray([.0009,*VALUES[1:]],dtype="<f8").tofile(path)
    with pytest.raises(ValueError,match="identity differs"):
        train(cfg,resume="latest")


@pytest.mark.parametrize("inherit", [False,True])
def test_new_stage_preserves_moments_only_and_applies_new_first_lr(tmp_path,inherit):
    cfg = lr_config(tmp_path,2)
    cfg["stages"][1]["init_from"].update(checkpoint="checkpoint-1",load_optimizer=inherit)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=lambda row:None))
    try:
        trainer.setup_stage(0,cfg["stages"][0])
        assert trainer.train_update()
        moments = {p:state["exp_avg"].clone() for p,state in trainer.optimizer.optimizer.state.items()}
        trainer.save("checkpoint-1")
        trainer.setup_stage(1,cfg["stages"][1])
        assert [g["lr"] for g in trainer.optimizer.param_groups] == [.0007]*2
        states = trainer.optimizer.optimizer.state
        if inherit:
            assert states
            for p,state in states.items():
                torch.testing.assert_close(state["exp_avg"],moments[p],atol=0,rtol=0)
        else:
            assert not states
        used = []
        handle = trainer.optimizer.optimizer.register_step_pre_hook(lambda *args:used.append(trainer.optimizer.param_groups[0]["lr"]))
        assert trainer.train_update()
        handle.remove()
        assert used == [.0007]
    finally:
        a.free_memory()
        a.end_training()


def test_continued_provider_requires_matching_schedule_and_index(tmp_path):
    cfg = lr_config(tmp_path,2)
    target = cfg["stages"][1]
    target["init_from"].update(checkpoint="checkpoint-2",load_optimizer=True,load_scheduler=True,reset_stage_step=False)
    target["max_steps"] = 4
    target["lr_schedule"] = deepcopy(cfg["stages"][0]["lr_schedule"])
    cfg = normalize(cfg)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=lambda row:None))
    try:
        trainer.setup_stage(0,cfg["stages"][0])
        for _ in range(2):
            assert trainer.train_update()
        trainer.save("checkpoint-2")
        trainer.setup_stage(1,cfg["stages"][1])
        assert trainer.state["stage_step"] == trainer.scheduler.scheduler.last_epoch == 2
        assert [g["lr"] for g in trainer.optimizer.param_groups] == [VALUES[2]]*2
        assert trainer.train_update()
    finally:
        a.free_memory()
        a.end_training()
    cfg["stages"][1]["init_from"]["reset_stage_step"] = True
    with pytest.raises(ValueError,match="reset_stage_step=false"):
        normalize(cfg)


class CountingLR:
    calls = []

    def __init__(self,stage):
        self.stage = stage

    def identity(self):
        return {"values":VALUES}

    def value_at(self,index):
        self.calls.append(index)
        return VALUES[index]


def test_custom_provider_skip_does_not_advance(tmp_path,monkeypatch):
    monkeypatch.setitem(LR_PROVIDERS,"counting",CountingLR)
    CountingLR.calls.clear()
    cfg = tiny_config(tmp_path)
    cfg["stages"] = cfg["stages"][:1]
    cfg["stages"][0]["lr_schedule"] = dict(type="custom",name="counting")
    cfg = normalize(cfg)
    a = build_accelerator(cfg)
    trainer = Trainer(cfg,build_model(cfg["model"]),prepare_tokenizer(cfg),a,SimpleNamespace(log=lambda row:None))
    try:
        trainer.setup_stage(0,cfg["stages"][0])
        assert CountingLR.calls == [0]
        optimizer, original = trainer.optimizer, trainer.optimizer.step
        def skip(*args,**kwargs):
            if a.sync_gradients:
                optimizer._is_overflow = True
        monkeypatch.setattr(optimizer,"step",skip)
        assert not trainer.train_update()
        assert CountingLR.calls == [0] and optimizer.param_groups[0]["lr"] == VALUES[0]
        optimizer._is_overflow = False
        monkeypatch.setattr(optimizer,"step",original)
        assert trainer.train_update()
        assert CountingLR.calls == [0,1] and optimizer.param_groups[0]["lr"] == VALUES[1]
    finally:
        a.free_memory()
        a.end_training()


def test_yaml_asset_paths_anchor_to_config_directory(tmp_path,monkeypatch):
    cfg = lr_config(tmp_path)
    cfg["stages"][0]["lr_schedule"]["path"] = "lr-0.bin"
    path = tmp_path/"config.yaml"
    save_config(cfg,path)
    elsewhere = tmp_path/"elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert load_config(path)["stages"][0]["lr_schedule"]["path"] == str(tmp_path/"lr-0.bin")
