import torch.nn as nn
import torch.nn.functional as F
from .builders import build_norm

ACTIVATIONS = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}


class FFN(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        mid, bias = cfg["hidden_size"], cfg["bias"]
        self.up = nn.Linear(dim, mid, bias=bias)
        self.gate = nn.Linear(dim, mid, bias=bias) if cfg["type"] == "gated" else None
        self.down = nn.Linear(mid, dim, bias=bias)
        self.norm = build_norm(cfg["internal_norm"], mid)
        self.act = ACTIVATIONS[cfg["activation"]]
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        y = self.act(self.up(x)) if self.gate is None else self.up(x) * self.act(self.gate(x))
        return self.dropout(self.down(self.norm(y)))
