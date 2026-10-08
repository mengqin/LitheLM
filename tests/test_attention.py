import pytest
import torch
from lithelm.ops.attention import attention_op, attention_reference


def test_sdpa_reference_causality_padding():
    torch.manual_seed(3)
    q,k,v = [torch.randn(2,3,7,16,requires_grad=True) for _ in range(3)]
    mask = torch.ones(2,1,1,7,dtype=torch.bool)
    mask[0,...,-2:] = False
    y = attention_op(q,k,v,mask=mask)
    torch.testing.assert_close(y,attention_reference(q,k,v,mask=mask),atol=1e-6,rtol=1e-5)
    changed = v.detach().clone()
    changed[:,:,4:] += 100
    torch.testing.assert_close(y[:,:,:4], attention_op(q,k,changed,mask=mask)[:,:,:4])
    y.sum().backward()
    assert torch.isfinite(q.grad).all()
    with pytest.raises(ValueError, match="Unknown"):
        attention_op(q,k,v,impl="missing")
