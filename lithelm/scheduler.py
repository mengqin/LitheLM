"""AdamW groups and WSD indexed by zero-based optimizer update."""
import math
import torch
import torch.nn as nn


def build_optimizer(model, cfg):
    excluded = set()
    rules = cfg["exclude_from_weight_decay"]
    for name, module in model.named_modules():
        if ("norm" in rules and isinstance(module, (nn.LayerNorm, nn.RMSNorm))) or ("embedding" in rules and isinstance(module, nn.Embedding)):
            excluded.update(id(p) for p in module.parameters(recurse=False))
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        target = no_decay if getattr(param,"_no_weight_decay",False) or id(param) in excluded or ("bias" in rules and name.endswith("bias")) else decay
        target.append(param)
    return torch.optim.AdamW([dict(params=decay, weight_decay=cfg["weight_decay"]),
                             dict(params=no_decay, weight_decay=0.)],
                            lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"])


def wsd_phase(step, cfg):
    if step < cfg["warmup_steps"]:
        return "warmup"
    if step < cfg["warmup_steps"] + cfg["stable_steps"]:
        return "stable"
    return "decay"


def wsd_scale(step, cfg):
    warmup, stable, decay = (cfg[k] for k in ("warmup_steps", "stable_steps", "decay_steps"))
    if step < warmup:
        return (step + 1) / warmup
    if step < warmup + stable or not decay:
        return 1.
    progress = min(1., (step-warmup-stable+1) / decay)
    factor = 1.-progress if cfg["decay_style"] == "linear" else .5*(1+math.cos(math.pi*progress))
    minimum = cfg["min_lr"] / cfg["peak_lr"]
    return minimum + (1.-minimum)*factor


def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: wsd_scale(step,cfg))
