"""Small module builders and the single FLA adaptation boundary."""
import inspect
import torch.nn as nn

CUSTOM_LAYERS = {}


def build_norm(cfg, dim):
    if cfg["type"] == "none":
        return nn.Identity()
    if cfg["type"] == "layernorm":
        return nn.LayerNorm(dim, eps=cfg["eps"])
    if cfg["type"] == "rmsnorm":
        return nn.RMSNorm(dim, eps=cfg["eps"])
    raise ValueError(f"Unknown norm: {cfg['type']}")


class FLAMixer(nn.Module):
    _lithelm_external = True

    def __init__(self, dim, cfg, index):
        super().__init__()
        try:
            from fla.layers import GatedDeltaNet
        except ImportError as exc:
            raise ImportError("GDN selected: install lithelm[fla] with compatible CUDA/Triton") from exc
        # Each wrapper owns an independent, single-layer FLA cache.
        self.layer = GatedDeltaNet(hidden_size=dim, mode=cfg["mode"], layer_idx=0, **cfg["kwargs"])

    def forward(self, x, attention_mask=None, cache=None, use_cache=False, position_offset=0, max_length=None):
        if use_cache:
            if cache is None:
                from fla.models.utils import Cache
                cache = Cache()
            result = self.layer(x, attention_mask=attention_mask, past_key_values=cache, use_cache=True)
            return result[0], result[2]
        result = self.layer(x, attention_mask=attention_mask, use_cache=False)
        return result[0] if isinstance(result, tuple) else result


def build_mixer(dim, cfg, index):
    if cfg["type"] == "attention":
        from .attention import Attention
        return Attention(dim, cfg["attention"])
    if cfg["type"] == "gdn":
        return FLAMixer(dim, cfg["gdn"], index)
    spec = cfg["custom"]
    if spec["name"] not in CUSTOM_LAYERS:
        raise ValueError(f"Unregistered custom layer: {spec['name']}")
    mixer = CUSTOM_LAYERS[spec["name"]](hidden_size=dim, **spec["kwargs"])
    mixer._lithelm_external = True
    parameters = inspect.signature(mixer.forward).parameters
    mixer._lithelm_collect_stats = "collect_stats" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())
    if callable(getattr(mixer, "init_weights", None)):
        mixer.init_weights()
    return mixer


def native_modules(module):
    """External subtrees own their initialization, regardless of attribute names."""
    if not getattr(module, "_lithelm_external", False):
        yield module
        for child in module.children():
            yield from native_modules(child)
