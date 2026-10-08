"""Attention dispatch over [batch, heads, length, dim] tensors."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F

CUSTOM_OPS = {}


def attention_reference(q, k, v, causal=True, scale=None, mask=None):
    scores = q.float() @ k.float().transpose(-2, -1) * (scale or q.shape[-1] ** -.5)
    allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device)
    if causal:
        allowed = allowed.tril()
    if mask is not None:
        allowed = allowed & mask.bool()
    scores = scores.masked_fill(~allowed, -torch.inf)
    probs = torch.softmax(scores, -1).nan_to_num(0.)
    return (probs @ v.float()).to(q.dtype)


def attention_op(q, k, v, *, impl="sdpa", op=None, causal=True, mask=None, dropout=0.):
    if impl in ("sdpa", "flash"):
        if mask is not None and causal:
            mask = mask.bool() & torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
        context = nullcontext()
        if impl == "flash":
            from torch.nn.attention import sdpa_kernel, SDPBackend
            context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        with context:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout,
                                                  is_causal=causal and mask is None)
    if impl == "triton":
        from .attention_triton import triton_attention
        if op not in (None, "attention_v1", "my_attention_v1"):
            raise ValueError(f"Unregistered Triton op: {op}")
        if dropout:
            raise ValueError("Triton attention_v1 does not support dropout")
        return triton_attention(q, k, v, causal=causal, mask=mask)
    if impl == "custom" and op in CUSTOM_OPS:
        return CUSTOM_OPS[op](q, k, v, causal=causal, mask=mask, dropout=dropout)
    raise ValueError(f"Unknown attention backend/op: {impl}/{op}")
