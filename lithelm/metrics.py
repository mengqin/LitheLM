"""Scalar statistics and additive sufficient statistics for exact NLL reduction."""
import math
import torch


def tensor_norm(tensors):
    tensors = list(tensors)
    if not tensors:
        return 0.
    return torch.stack([t.detach().float().square().sum() for t in tensors]).sum().sqrt().item()


def parameter_norm(model, **kwargs):
    return tensor_norm(model.parameters())


def gradient_norm(model, **kwargs):
    return tensor_norm(p.grad for p in model.parameters() if p.grad is not None)


def update_norm(model, before=None, **kwargs):
    return tensor_norm(p.detach()-b for p,b in zip(model.parameters(),before))


def update_ratio(model, before=None, **kwargs):
    return update_norm(model,before) / max(tensor_norm(before),1e-12)


def gpu_memory(model, **kwargs):
    device = next(model.parameters()).device
    return torch.cuda.memory_allocated(device)/1e9 if device.type == "cuda" else 0.


STATS = dict(grad_norm=gradient_norm, param_norm=parameter_norm, update_norm=update_norm,
             update_ratio=update_ratio, gpu_memory=gpu_memory)


def perplexity(nll):
    return math.exp(min(nll, 80.))


def loss_inputs(logits, labels, target_mode="shifted"):
    if target_mode == "shifted":
        return logits[:, :-1], labels[:, 1:]
    if target_mode != "next_token":
        raise ValueError(f"Unknown target mode: {target_mode}")
    return logits, labels


def token_accuracy(logits, labels, target_mode="shifted"):
    logits, labels = loss_inputs(logits, labels, target_mode)
    valid = labels.ne(-100)
    return ((logits.argmax(-1) == labels) & valid).sum(), valid.sum()


def exact_match(logits, labels, target_mode="shifted"):
    """Teacher-forced full-target sequence match; generation samples are separate."""
    logits, labels = loss_inputs(logits, labels, target_mode)
    valid = labels.ne(-100)
    match = (logits.argmax(-1) == labels) | ~valid
    return (match.all(-1) & valid.any(-1)).sum(), valid.any(-1).sum()


METRICS = dict(accuracy=token_accuracy, exact_match=exact_match)


def meter_metrics(meters):
    """meters: dataset -> [NLL sum, target tokens, samples, objective]."""
    out, objectives = {}, {}
    loss_sum = sum(v[0] for v in meters.values())
    tokens = sum(v[1] for v in meters.values())
    samples = sum(v[2] for v in meters.values())
    if tokens:
        out.update({"train/nll":loss_sum/tokens,"train/loss":loss_sum/tokens,"train/ppl":perplexity(loss_sum/tokens)})
    for name,(loss,count,n,obj) in meters.items():
        if count:
            out[f"train/{name}/nll"] = loss/count
        bucket = objectives.setdefault(obj,[0.,0,0])
        for i,v in enumerate((loss,count,n)):
            bucket[i] += v
    for obj,(loss,count,n) in objectives.items():
        out[f"data/{obj}_ratio"] = n/max(samples,1)
        if count:
            out[f"train/{obj}_nll"] = loss/count
    return out
