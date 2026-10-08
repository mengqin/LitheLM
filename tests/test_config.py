from copy import deepcopy
from pathlib import Path
import pytest

from lithelm.config import load_config, normalize

ROOT = Path(__file__).resolve().parents[1]


def test_full_course():
    c = load_config(ROOT / "configs/examples/tlm_7m.yaml")
    assert len(c["model"]["layers"]) == 8
    assert c["distributed"]["strategy"] == "single"
    assert [s["max_steps"] for s in c["stages"]] == [60000, 4000]
    assert all(s["batch"]["micro_batch_size"] == 32 for s in c["stages"])
    assert c["experiment"]["output_dir"] is None
    assert c["model"]["layers"][0]["norm1"]["type"] == "layernorm"
    assert c["stages"][1]["data"]["datasets"][1]["weight"] == .1


def test_override_defaults_and_independent_layers():
    c = load_config(ROOT / "configs/examples/tlm_7m.yaml", ["stages.0.batch.micro_batch_size=2"])
    assert c["stages"][0]["batch"]["micro_batch_size"] == 2
    assert c["model"]["layers"][0]["mixer"]["attention"]["rope"]["enabled"] is False
    c["model"]["layers"][0]["hidden_size"] = 4
    assert c["model"]["layers"][1]["hidden_size"] == 256


@pytest.mark.parametrize("override", [
    "model.layer_defaults.mixer.attention.num_heads=7",
    "model.layer_defaults.mixer.attention.num_kv_heads=3",
    "stages.0.data.datasets.0.weight=-1",
    "stages.0.lr_schedule.stable_steps=10",
    "tokenizer.vocab_size=64",
    "model.position_embedding.max_positions=8",
    "runtime.precison=fp32",
    "stages.0.curriculum.0.until_step=3",
    "model.layer_defaults.ffn.hidden_size=0",
    "model.layer_defaults.mixer.attention.num_kv_heads=0",
    "datasets.tinystories_pt_train.kwargs.sampel_mode=random_window",
    "inference.temperature=0",
    "stages.0.checkpoint.keep_last=0",
])
def test_invalid(override):
    with pytest.raises(ValueError):
        load_config(ROOT / "configs/examples/tlm_7m.yaml", [override])


@pytest.mark.parametrize("path", ["configs/base.yaml", "configs/examples/tlm_7m.yaml"])
def test_canonical_config_reload(path):
    c = load_config(ROOT / path)
    assert normalize(deepcopy(c)) == c
