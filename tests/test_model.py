from copy import deepcopy
import torch
from lithelm.config import load_config, normalize
from lithelm.model import build_model
from test_config import ROOT


def test_tlm_shape_loss_count_tied():
    c = load_config(ROOT / "configs/examples/tlm_7m.yaml")
    m = build_model(c["model"])
    assert sum(p.numel() for p in m.parameters()) == 7_432_704
    assert m.position.weight.numel() == 65_536
    assert m.lm_head.weight is m.embedding.weight
    x = torch.randint(0,4096,(2,9))
    o = m(x, labels=x)
    assert o.logits.shape == (2,9,4096) and torch.isfinite(o.loss)
    o.loss.backward()


def test_width_rms_gated_gqa_rope():
    c = load_config(ROOT / "configs/examples/tlm_7m.yaml")
    c["model"]["auto_project_width_changes"] = True
    c["model"]["position_embedding"]["type"] = "rope"
    s = deepcopy(c["model"]["layers"][0])
    s["hidden_size"] = 128
    s["mixer"]["attention"].update(num_heads=4, num_kv_heads=2)
    s["ffn"].update(type="gated", gated=True)
    s["norm1"]["type"] = s["norm2"]["type"] = "rmsnorm"
    c["model"]["layers"] = [s]
    m = build_model(normalize(c)["model"])
    x = torch.randint(0,4096,(2,8))
    m(x, labels=x).loss.backward()
    assert m(x).logits.shape == (2,8,4096)


def test_disabled_ffn_and_ignored_labels():
    c = load_config(ROOT / "configs/examples/tlm_7m.yaml")
    c["model"]["layers"] = c["model"]["layers"][:1]
    c["model"]["layers"][0]["ffn"]["enabled"] = False
    m = build_model(c["model"])
    assert m.layers[0].ffn is None
    x = torch.ones(1,4,dtype=torch.long)
    o = m(x, labels=torch.full_like(x,-100))
    assert o.loss.item() == 0
