"""Ephemeral inference state; never part of model or training checkpoints."""
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AttentionCache:
    """Contiguous [batch, KV heads, capacity, head dim] buffers; ModelCache tracks length."""
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class ModelCache:
    position: int
    layers: list[Any]
    batch_size: int
    max_length: int
    attention_mask: torch.Tensor | None = None
