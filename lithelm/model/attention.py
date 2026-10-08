"""Shared projections, head reshaping and RoPE for all attention operators."""
import torch
import torch.nn as nn
from .cache import AttentionCache
from ..ops.attention import attention_op


def rope(x, theta, position_offset=0):
    dim, length = x.shape[-1], x.shape[-2]
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = torch.arange(position_offset, position_offset+length, device=x.device)[:, None] * inv[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        h, kv, d = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
        self.qkv = nn.Linear(dim, (h + 2 * kv) * d, bias=cfg["qkv_bias"])
        self.out = nn.Linear(h * d, dim, bias=cfg["out_bias"])
        self.dropout = nn.Dropout(cfg["output_dropout"])

    def forward(self, x, attention_mask=None, cache=None, use_cache=False, position_offset=0, max_length=None):
        b, t, _ = x.shape
        c = self.cfg
        h, kv, d = c["num_heads"], c["num_kv_heads"], c["head_dim"]
        q, k, v = self.qkv(x).split((h * d, kv * d, kv * d), dim=-1)
        q, k, v = [a.view(b, t, heads, d).transpose(1, 2) for a, heads in [(q,h),(k,kv),(v,kv)]]
        if c["rope"]["enabled"]:
            q, k = rope(q, c["rope"]["theta"], position_offset), rope(k, c["rope"]["theta"], position_offset)
        prefill = cache is None
        if cache is not None and (not use_cache or t != 1):
            raise ValueError("Attention cache requires use_cache=True and single-token decode")
        if use_cache:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Attention cache requires eval() and no_grad/inference_mode")
            end = position_offset+t
            if type(max_length) is not int or not 0 <= position_offset < end <= max_length:
                raise ValueError("Attention cache write exceeds max_length capacity or has an invalid position")
            shape = (b, kv, max_length, d)
            if prefill:
                if position_offset != 0:
                    raise ValueError("Attention prefill must start at position zero")
                cache = AttentionCache(torch.empty(shape, device=k.device, dtype=k.dtype),
                                       torch.empty(shape, device=v.device, dtype=v.dtype))
            for buffer, current in ((cache.key, k), (cache.value, v)):
                if buffer.shape != shape:
                    raise ValueError("Attention cache shape must match batch, KV heads, capacity and head_dim")
                if buffer.device != current.device or buffer.dtype != current.dtype:
                    raise ValueError("Attention cache device/dtype must match the current forward")
            # Store rotated, unexpanded KV heads; only read the initialized prefix.
            cache.key[:, :, position_offset:end].copy_(k)
            cache.value[:, :, position_offset:end].copy_(v)
            k, v = cache.key[:, :, :end], cache.value[:, :, :end]
        if kv != h:
            k, v = k.repeat_interleave(h // kv, 1), v.repeat_interleave(h // kv, 1)
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        y = attention_op(q, k, v, impl=c["impl"], op=c["op"], causal=c["causal"] and prefill, mask=mask,
                         dropout=c["attention_dropout"] if self.training else 0.)
        y = self.dropout(self.out(y.transpose(1, 2).reshape(b, t, h * d)))
        return (y, cache) if use_cache else y
