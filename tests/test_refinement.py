"""Boundary, scalar aggregation, target exposure and shared-asset contracts."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lithelm.config import load_config, normalize
from lithelm.data import collate, evaluation_batches
from lithelm.logging_utils import Logger
from lithelm.metrics import loss_inputs, token_accuracy
from lithelm.model import build_model
from lithelm.model.builders import CUSTOM_LAYERS, FLAMixer
from lithelm.statistics import ScalarStats, StatSpec
from lithelm.tokenizer import prepare_tokenizer
from lithelm.workspace import run_directory
from test_data import tiny_config


class FixedMixer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.out = nn.Linear(hidden_size, hidden_size)
        nn.init.constant_(self.out.weight, .125)
        nn.init.constant_(self.out.bias, .25)

    def forward(self, x, attention_mask=None):
        return self.out(x)


class ResearchMixer(FixedMixer):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.calls = 0

    def init_weights(self):
        self.calls += 1
        nn.init.constant_(self.out.weight, .5)

    def stat_specs(self):
        return {"anything": StatSpec(aggregation="sum", format=".3f")}

    def forward(self, x, attention_mask=None, collect_stats=False):
        y = super().forward(x, attention_mask)
        return (y, {"anything": y.square().mean(), "arbitrary_new_field": 7.0}) if collect_stats else y


def custom_config(tmp_path, cls, monkeypatch):
    cfg = tiny_config(tmp_path)
    monkeypatch.setitem(CUSTOM_LAYERS, "research", cls)
    cfg["model"]["layers"][0]["mixer"].update(type="custom", custom=dict(name="research", kwargs={}))
    return cfg


@pytest.mark.parametrize("cls,weight,calls", [(FixedMixer,.125,0), (ResearchMixer,.5,1)])
def test_custom_initialization_owned(tmp_path, monkeypatch, cls, weight, calls):
    cfg = custom_config(tmp_path, cls, monkeypatch)
    model = build_model(cfg["model"])
    mixer = model.layers[0].mixer
    assert torch.equal(mixer.out.weight, torch.full_like(mixer.out.weight, weight))
    assert torch.equal(mixer.out.bias, torch.full_like(mixer.out.bias, .25))
    assert getattr(mixer, "calls", 0) == calls
    x = torch.ones(2, 8, dtype=torch.long)
    model(x, labels=x).loss.backward()
    assert getattr(mixer, "calls", 0) == calls


def test_external_passthrough(monkeypatch):
    import sys
    seen = {}
    class External(FixedMixer):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(kwargs["hidden_size"])
        def forward(self, x, **kwargs):
            seen.update(kwargs)
            return super().forward(x), None
    monkeypatch.setitem(sys.modules, "fla.layers", SimpleNamespace(GatedDeltaNet=External))
    mixer = FLAMixer(4, dict(mode="future-mode", kwargs={"arbitrary":123}), 2)
    assert mixer(torch.ones(1,2,4)).shape == (1,2,4)
    assert seen["mode"] == "future-mode" and seen["arbitrary"] == 123 and seen["layer_idx"] == 0


def test_scalar_aggregation_and_logger(tmp_path, monkeypatch, capsys):
    cfg = custom_config(tmp_path, ResearchMixer, monkeypatch)
    model = build_model(cfg["model"])
    x = torch.ones(1, 6, dtype=torch.long)
    out = model(x, collect_stats=True)
    assert "layer/0/mixer/anything" in out.stats
    assert out.stats["layer/0/mixer/arbitrary_new_field"] == 7
    a = SimpleNamespace(device=torch.device("cpu"), num_processes=1, reduce=lambda x,reduction:x)
    for mode,expected in [("last",6), ("mean",4), ("sum",12), ("min",2), ("max",6)]:
        collector = ScalarStats(dict(microbatch_aggregation=mode, overrides={}), {}, a)
        for value in [2.,4.,6.]:
            collector.add({"test":torch.tensor(value, requires_grad=True)})
        assert collector.finish() == {"test":expected}
    name = "layer/0/mixer/anything"
    options = dict(microbatch_aggregation="last", overrides={"*/anything":"mean", name:"max"})
    collector = ScalarStats(options, model.stat_specs(), a)
    for value in [2., 8., 5.]:
        collector.add({name:value})
    assert collector.finish()[name] == 8 # exact override > wildcard > spec > global
    options["overrides"].pop(name)
    assert collector.finish()[name] == 5
    options["overrides"].clear()
    assert collector.finish()[name] == 15
    with pytest.raises(ValueError, match="finite scalars"):
        collector.add({name:torch.ones(2)})
    buffer = torch.zeros((), dtype=torch.float64)
    collector = ScalarStats(dict(microbatch_aggregation="mean", overrides={}), {}, a)
    for value in [2.,8.,5.]:
        collector.add({"reused_buffer":buffer.fill_(value), "number":1.23456789012345})
    assert collector.finish() == pytest.approx({"reused_buffer":5., "number":1.23456789012345}, rel=1e-14)
    cfg["logging"].update(csv=True, stdout=True)
    logger = Logger(cfg["logging"], tmp_path, formats={name:".3f"})
    logger.log({name:1.23456, "global_step":1})
    logger.close()
    assert json.loads(capsys.readouterr().out)[name] == "1.235"
    assert json.loads((tmp_path/"metrics.jsonl").read_text())[name] == 1.23456
    assert any(row["metric"] == name for row in csv.DictReader((tmp_path/"metrics.csv").open()))


@pytest.mark.parametrize("mode,count,labels", [("shifted",3,[10,11,12,13]), ("next_token",4,[11,12,13,14])])
def test_target_protocol_toy(tmp_path, mode, count, labels):
    row = dict(input_ids=[10,11,12,13,14], labels=[10,11,12,13,14])
    batch = collate([row],4,target_mode=mode)
    assert batch["input_ids"].tolist() == [[10,11,12,13]]
    assert batch["labels"].tolist() == [labels]
    model = build_model(tiny_config(tmp_path)["model"])
    out = model(**batch, target_mode=mode)
    assert out.token_count.item() == count
    predictions, targets = loss_inputs(out.logits,batch["labels"],mode)
    expected = torch.nn.functional.cross_entropy(predictions.reshape(-1,280),targets.reshape(-1),reduction="sum")
    torch.testing.assert_close(out.loss_sum,expected)
    assert token_accuracy(out.logits,batch["labels"],mode)[1] == count
    row["labels"] = [-100,-100,-100,13,14]
    masked = collate([row],4,target_mode=mode)
    assert model(**masked,target_mode=mode).token_count.item() == (1 if mode == "shifted" else 2)
    cache = dict(tokens=np.arange(10,19), mask=np.ones(9), offsets=np.array([0,5,9]))
    for packing,expected in [(True,8),(False,7)]:
        batches = list(evaluation_batches(cache,4,2,packing,target_mode=mode))
        assert sum(model(**b,target_mode=mode).token_count.item() for b in batches) == expected


def test_shared_inputs_paths_and_provenance(tmp_path, monkeypatch):
    import os
    from lithelm.data import raw_records
    from lithelm.provenance import record_environment
    import lithelm.provenance as provenance
    cfg = tiny_config(tmp_path/"source")
    tok = prepare_tokenizer(cfg)
    artifact = Path(cfg["tokenizer"]["artifact"])
    timestamp = artifact.stat().st_mtime_ns
    cfg["experiment"].update(output_dir=None, model_size="6K")
    cfg["paths"]["run_root"] = str(tmp_path/"runs")
    cfg["paths"]["corpus_root"] = str(tmp_path/"corpora")
    roots = []
    for _ in range(2):
        with run_directory(cfg) as (bound,root):
            roots.append(root)
            assert bound["tokenizer"]["artifact"] == str(artifact)
            assert bound["datasets"]["pt"]["path"] == cfg["datasets"]["pt"]["path"]
            assert not (root/"data").exists() and not (root/"tokenizer").exists()
            assert Path(os.environ["HF_DATASETS_CACHE"]) == tmp_path/"corpora/hf/datasets"
            assert len(list(raw_records(bound["datasets"]["sft"]))) == 20
            assert not (root/"cache/hf").exists()
            assert prepare_tokenizer(bound) is not None
            monkeypatch.setattr(provenance, "package_version", lambda name:None)
            record_environment(bound,tok)
            env = json.loads((root/"environment.json").read_text())
            assert env["versions"]["flash-linear-attention"] is None
            assert env["versions"]["flash-attn"] is None and env["versions"]["flash-attn-3"] is None
            assert env["tokenizer_fingerprint"] and env["inputs"]["corpora"]["pt"]["path"]["bytes"] > 0
    assert roots[0] != roots[1] and roots[0].parent == tmp_path/"runs"
    assert artifact.stat().st_mtime_ns == timestamp


def test_targets_accounting_stats_resume(tmp_path, monkeypatch):
    from lithelm.train import train
    cfg = custom_config(tmp_path, ResearchMixer, monkeypatch)
    for stage in cfg["stages"]:
        stage["targets"]["mode"] = "next_token"
        stage["data"]["datasets"] = [dict(name="pt",objective="pt",weight=1.)]
        stage["curriculum"] = []
    cfg["statistics"].update(every=1, overrides={"*/arbitrary_new_field":"sum"})
    cfg["logging"]["csv"] = True
    train(cfg, stop_after=2)
    root = Path(cfg["experiment"]["output_dir"])
    environment = root/"environment.json"
    data = json.loads(environment.read_text())
    data["versions"]["torch"] = "different-installed-version"
    environment.write_text(json.dumps(data))
    final = train(load_config(root/"resolved_config.yaml"), resume=root/"pt/checkpoint-2")
    expected = sum(s["max_steps"]*s["batch"]["micro_batch_size"]*s["batch"]["grad_accum_steps"]*s["data"]["max_length"] for s in cfg["stages"])
    assert final["tokens_seen"] == expected
    rows = [json.loads(line) for line in (root/"metrics.jsonl").read_text().splitlines()]
    values = [r["layer/0/mixer/arbitrary_new_field"] for r in rows if r.get("event") == "model_stats"]
    assert len(values) == 7 and values == [14.] * 7
    assert list((root/"environments").glob("*.json"))


def test_target_config_defaults_and_fla_mode(tmp_path):
    cfg = tiny_config(tmp_path)
    assert all(s["targets"]["mode"] == "shifted" for s in cfg["stages"])
    cfg["model"]["layers"][0]["mixer"].update(type="gdn")
    cfg["model"]["layers"][0]["mixer"]["gdn"]["mode"] = "new-external-mode"
    normalize(cfg)
    cfg["stages"][0]["targets"]["mode"] = "misspelling"
    with pytest.raises(ValueError,match="targets.mode"):
        normalize(cfg)


def test_legacy_signature_and_default_tokenizer_root(tmp_path, monkeypatch):
    from lithelm.checkpoint import signature
    from lithelm.tokenizer import tokenizer_fingerprint
    monkeypatch.chdir(tmp_path)
    cfg = tiny_config(tmp_path/"source")
    cfg["tokenizer"]["artifact"] = None
    cfg = normalize(cfg)
    assert cfg["tokenizer"]["artifact"] == str(tmp_path/"tokenizers/bpe-280.json")
    tok = prepare_tokenizer(cfg)
    current = signature(cfg, tok)
    for stage in cfg["stages"]:
        stage.pop("targets")
    legacy = {key:cfg[key] for key in ("model","datasets","stages","runtime")}
    legacy.update(tokenizer_hash=tokenizer_fingerprint(tok), seed=cfg["experiment"]["seed"])
    assert current == hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()
    cfg["targets"]["mode"] = "next_token"
    cfg["stages"][1]["targets"] = dict(mode="shifted")
    cfg = normalize(cfg)
    assert [s["targets"]["mode"] for s in cfg["stages"]] == ["next_token","shifted"]
    assert signature(cfg, tok) != current
