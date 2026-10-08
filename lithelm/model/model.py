"""One configurable causal LM. Labels are aligned; shifting occurs exactly once."""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block
from .builders import build_norm, native_modules
from .cache import ModelCache
from ..config import context_length
from ..metrics import loss_inputs


@dataclass
class LMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    token_count: torch.Tensor | None = None
    stats: dict | None = None
    cache: ModelCache | None = None


class CausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg["hidden_size"]
        self.embedding = nn.Embedding(cfg["vocab_size"], dim)
        pos = cfg["position_embedding"]
        self.position = nn.Embedding(pos["max_positions"], dim) if pos["type"] == "learned_absolute" else None
        self.dropout = nn.Dropout(cfg["embeddings"]["dropout"])
        self.layers = nn.ModuleList()
        for i, spec in enumerate(cfg["layers"]):
            self.layers.append(Block(spec, dim, i))
            dim = spec["hidden_size"]
        self.final_norm = build_norm(cfg["final_norm"], dim)
        out_dim = cfg["hidden_size"] if cfg["tie_word_embeddings"] else dim
        self.head_projection = nn.Linear(dim, out_dim, bias=False) if dim != out_dim else nn.Identity()
        self.lm_head = nn.Linear(out_dim, cfg["vocab_size"], bias=cfg["lm_head_bias"])
        if cfg["tie_word_embeddings"]:
            self.lm_head.weight = self.embedding.weight
        # Initialize shared tensors only once, preserving third-party GDN initialization.
        seen = set()
        for module in native_modules(self):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if id(module.weight) not in seen:
                    nn.init.normal_(module.weight, std=cfg["init"]["std"])
                    seen.add(id(module.weight))
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        if cfg["init"]["scale_residual_projections"]:
            for layer in self.layers:
                projections = []
                if not getattr(layer.mixer, "_lithelm_external", False):
                    projections.append(layer.mixer.out)
                if layer.ffn is not None:
                    projections.append(layer.ffn.down)
                for projection in projections:
                    nn.init.normal_(projection.weight, std=cfg["init"]["std"] / math.sqrt(2 * len(self.layers)))

    def stat_specs(self):
        specs = {}
        for i, layer in enumerate(self.layers):
            method = getattr(layer.mixer, "stat_specs", None)
            if callable(method):
                specs.update({f"layer/{i}/mixer/{key}": value for key, value in method().items()})
        return specs

    def forward(self, input_ids, attention_mask=None, labels=None, collect_stats=False, target_mode="shifted",
                cache=None, use_cache=False, max_context_length=None):
        if cache is not None and not use_cache:
            raise ValueError("Passing cache requires use_cache=True")
        if max_context_length is not None and not use_cache:
            raise ValueError("max_context_length requires use_cache=True")
        start = cache.position if cache is not None else 0
        limit = min(self.cfg["max_position_embeddings"], self.cfg["position_embedding"]["max_positions"])
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or start+input_ids.shape[1] > limit:
            raise ValueError("input_ids must be [batch, length] within max_position_embeddings")
        if use_cache:
            if self.training or torch.is_grad_enabled() or labels is not None:
                raise ValueError("Generation cache requires eval(), no_grad/inference_mode, and no labels")
            if not all(layer.supports_cache for layer in self.layers):
                raise NotImplementedError("Cached inference requires built-in causal attention or FLA Mixers")
            batch, length = input_ids.shape
            capacity = context_length(self.cfg, cache.max_length if cache is not None else max_context_length)
            if cache is not None and max_context_length is not None:
                if context_length(self.cfg, max_context_length) != capacity:
                    raise ValueError("Cannot change max_context_length of an existing cache")
            if start+length > capacity:
                raise ValueError("Input exceeds cache max_context_length capacity")
            if cache is None:
                cache = ModelCache(0, [None]*len(self.layers), batch, capacity)
            elif (start < 1 or length != 1 or len(cache.layers) != len(self.layers) or
                  cache.batch_size != batch or any(state is None for state in cache.layers)):
                raise ValueError("Cache requires single-token decode with the same batch and model layers")
            if attention_mask is not None:
                if (attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device or
                    not ((attention_mask == 0) | (attention_mask == 1)).all()):
                    raise ValueError("Cached attention_mask must be a 0/1 mask matching the new input shape/device")
                if not attention_mask.any(dim=1).all():
                    raise ValueError("Cached inference requires a nonempty prompt and unmasked decode tokens")
                attention_mask = attention_mask.bool()
            if start and cache.attention_mask is not None:
                attention_mask = torch.cat((cache.attention_mask, torch.ones_like(input_ids, dtype=torch.bool)), dim=1)
            if attention_mask is not None and attention_mask.all():
                attention_mask = None
        x = self.embedding(input_ids)
        if self.position is not None:
            x = x + self.position(torch.arange(start, start+input_ids.shape[1], device=x.device))
        x = self.dropout(x)
        stats = {} if collect_stats else None
        for i, layer in enumerate(self.layers):
            local = {} if collect_stats else None
            if use_cache:
                x, cache.layers[i] = layer(x, attention_mask=attention_mask, stats=local,
                    cache=cache.layers[i], use_cache=True, position_offset=start, max_length=cache.max_length)
            else:
                x = layer(x, attention_mask=attention_mask, stats=local)
            if stats is not None:
                stats.update({f"layer/{i}/{k}": v for k,v in local.items()})
        logits = self.lm_head(self.head_projection(self.final_norm(x)))
        if use_cache:
            cache.position += input_ids.shape[1]
            cache.attention_mask = attention_mask
        result = LMOutput(logits=logits, stats=stats, cache=cache)
        if labels is not None:
            predictions, targets = loss_inputs(logits, labels, target_mode)
            result.loss_sum = F.cross_entropy(predictions.float().reshape(-1, logits.shape[-1]),
                                               targets.reshape(-1), ignore_index=-100, reduction="sum")
            result.token_count = targets.ne(-100).sum()
            result.loss = result.loss_sum / result.token_count.clamp_min(1)
        return result


def build_model(cfg):
    """Build from the canonical model configuration returned by load_config."""
    return CausalLM(cfg)
