"""Explicit external backends: layout, validation, and errors without fallback."""
import sys
from types import SimpleNamespace

import pytest
import torch

from lithelm.config import normalize
from lithelm.model import build_model
from lithelm.ops.attention import attention_op
from test_data import tiny_config


@pytest.mark.parametrize("impl,module,dropout", [
    ("flash2", "flash_attn", .1), ("flash3", "flash_attn_interface", 0.),
])
def test_external_config_layout_and_gradients(tmp_path, monkeypatch, impl, module, dropout):
    cfg = tiny_config(tmp_path)
    cfg["model"]["layers"][0]["mixer"]["attention"].update(impl=impl)
    model = build_model(normalize(cfg)["model"])
    assert model.layers[0].supports_cache
    values = [torch.randn(2, 3, 5, 16, requires_grad=True) for _ in range(3)]
    calls = []
    def function(q, k, v, **kwargs):
        assert q.shape == k.shape == v.shape == (2, 5, 3, 16)
        calls.append(kwargs)
        return q+2*k+3*v
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(flash_attn_func=function))
    out = attention_op(*values, impl=impl, causal=False, dropout=dropout)
    torch.testing.assert_close(out, values[0]+2*values[1]+3*values[2])
    out.sum().backward()
    for factor, value in enumerate(values, 1):
        torch.testing.assert_close(value.grad, torch.full_like(value, factor))
    expected = dict(causal=False, deterministic=torch.are_deterministic_algorithms_enabled())
    if impl == "flash2":
        expected["dropout_p"] = dropout
    assert calls == [expected]


@pytest.mark.parametrize("impl,module", [("flash2", "flash_attn"), ("flash3", "flash_attn_interface")])
def test_missing_external_library_raises(monkeypatch, impl, module):
    monkeypatch.setitem(sys.modules, module, None)
    q = torch.randn(1, 1, 4, 16)
    with pytest.raises(ModuleNotFoundError, match=module):
        attention_op(q, q, q, impl=impl)


@pytest.mark.parametrize("impl,module", [("flash2", "flash_attn"), ("flash3", "flash_attn_interface")])
def test_external_execution_error_propagates(monkeypatch, impl, module):
    error = RuntimeError("kernel does not support this GPU")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(flash_attn_func=fail))
    q = torch.randn(1, 1, 4, 16)
    with pytest.raises(RuntimeError) as raised:
        attention_op(q, q, q, impl=impl)
    assert raised.value is error


@pytest.mark.parametrize("impl", ["flash2", "flash3"])
def test_external_rejects_masks_and_non_square_causality(impl):
    q = torch.randn(1, 1, 4, 16)
    with pytest.raises(ValueError, match="attention masks"):
        attention_op(q, q, q, impl=impl, mask=torch.ones(1, 1, 1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="equal query/key lengths"):
        attention_op(q[:, :, :1], q, q, impl=impl)


def test_flash3_rejects_dropout():
    q = torch.randn(1, 1, 4, 16)
    with pytest.raises(ValueError, match="attention dropout"):
        attention_op(q, q, q, impl="flash3", dropout=.1)
