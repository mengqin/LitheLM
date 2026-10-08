import torch.nn as nn
from .builders import build_mixer, build_norm
from .ffn import FFN


class Block(nn.Module):
    def __init__(self, cfg, in_dim, index):
        super().__init__()
        dim = cfg["hidden_size"]
        self.projection = nn.Linear(in_dim, dim, bias=False) if in_dim != dim else nn.Identity()
        self.norm1 = build_norm(cfg["norm1"], dim)
        self.norm2 = build_norm(cfg["norm2"], dim) if cfg["ffn"]["enabled"] else nn.Identity()
        self.mixer = build_mixer(dim, cfg["mixer"], index)
        mixer = cfg["mixer"]
        self.supports_cache = mixer["type"] == "gdn" or (mixer["type"] == "attention" and
            mixer["attention"]["causal"] and mixer["attention"]["impl"] in ("sdpa", "flash", "flash2", "flash3", "triton"))
        self.ffn = FFN(dim, cfg["ffn"]) if cfg["ffn"]["enabled"] else None
        self.residual, self.pre = cfg["residual"], cfg["norm_position"] == "pre"

    def forward(self, x, attention_mask=None, stats=None, cache=None, use_cache=False, position_offset=0, max_length=None):
        x = self.projection(x)
        kwargs = {"collect_stats": stats is not None} if getattr(self.mixer, "_lithelm_collect_stats", False) else {}
        if use_cache:
            if not self.supports_cache:
                raise NotImplementedError("Cached inference requires built-in causal attention or FLA Mixers")
            kwargs.update(cache=cache, use_cache=True, position_offset=position_offset, max_length=max_length)
        y = self.mixer(self.norm1(x) if self.pre else x, attention_mask=attention_mask, **kwargs)
        if use_cache:
            y, cache = y
        elif isinstance(y, tuple):
            y, values = y
            if stats is not None:
                stats.update({f"mixer/{key}": value for key, value in values.items()})
        x = x + y if self.residual else y
        if not self.pre:
            x = self.norm1(x)
        if self.ffn is not None:
            y = self.ffn(self.norm2(x) if self.pre else x)
            x = x + y if self.residual else y
            if not self.pre:
                x = self.norm2(x)
        if stats is not None:
            stats["layer_output_rms"] = x.detach().float().square().mean().sqrt()
            if self.ffn is not None:
                stats["ffn_output_rms"] = y.detach().float().square().mean().sqrt()
        return (x, cache) if use_cache else x
