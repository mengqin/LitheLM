"""Global component denominators with one microbatch graph at a time."""
import math

import torch

from .extensions import ComponentSpec, agree, checked


class ComposedUpdate:
    def __init__(self, composer, batches, context, accelerator, component_names):
        self.composer, self.context, self.a = composer, context, accelerator
        def components():
            specs = composer.components(context)
            if not isinstance(specs,dict) or not specs or any(not isinstance(k,str) or not k for k in specs):
                raise ValueError("LossComposer.components must return a nonempty named mapping")
            for spec in specs.values():
                if (not isinstance(spec,ComponentSpec) or not isinstance(spec.required,bool)
                        or not isinstance(spec.weight,(int,float)) or not math.isfinite(spec.weight)):
                    raise ValueError("Loss components require finite weights and ComponentSpec")
            return {name:specs[name] for name in sorted(specs)}
        self.specs = checked(accelerator, components)
        agree({k:[v.weight,v.required] for k,v in self.specs.items()},accelerator)
        self.names = tuple(self.specs)
        if component_names is not None and component_names != self.names:
            raise ValueError("LossComposer component names must remain fixed within a stage")
        def denominators():
            rows = []
            for batch in batches:
                values = composer.denominators(batch,context)
                if not isinstance(values,dict) or set(values)-set(self.names):
                    raise ValueError("LossComposer denominators contain unknown component names")
                row = []
                for name in self.names:
                    value = torch.as_tensor(values.get(name,0.))
                    if value.ndim or value.requires_grad or not torch.isfinite(value) or value < 0:
                        raise ValueError(f"Loss denominator {name!r} must be a finite nonnegative scalar without gradients")
                    row.append(value.item())
                rows.append(row)
            return torch.tensor(rows,device=accelerator.device,dtype=torch.float64)
        self.local_denominators = checked(accelerator,denominators)
        self.denominators = accelerator.reduce(self.local_denominators.sum(0),reduction="sum")
        if not torch.isfinite(self.denominators).all():
            raise ValueError("Non-finite global loss denominators")
        for name,denom in zip(self.names,self.denominators):
            if denom == 0 and self.specs[name].required:
                raise ValueError(f"Required loss component {name!r} has zero global denominator")
        if not any(denom > 0 and self.specs[name].weight != 0 for name,denom in zip(self.names,self.denominators)):
            raise ValueError("No active weighted loss component in optimizer update")
        self.means = torch.zeros_like(self.denominators)

    def loss(self, output, batch, microbatch_index):
        def compute():
            values = self.composer.numerators(output,batch,self.context)
            if not isinstance(values,dict) or set(values)-set(self.names):
                raise ValueError("LossComposer numerators contain unknown component names")
            loss = output.logits.float().sum()*0.
            for i,name in enumerate(self.names):
                numerator = values.get(name)
                local_denominator = self.local_denominators[microbatch_index,i]
                if numerator is None:
                    if local_denominator != 0:
                        raise ValueError(f"Loss component {name!r} is missing its numerator")
                    continue
                if (not isinstance(numerator,torch.Tensor) or numerator.ndim
                        or not numerator.is_floating_point()
                        or numerator.device != output.logits.device or not torch.isfinite(numerator)
                        or (local_denominator != 0 and (not numerator.requires_grad or numerator.grad_fn is None))):
                    raise ValueError(f"Loss numerator {name!r} must be a finite differentiable scalar on the model device")
                if local_denominator == 0 and numerator.detach() != 0:
                    raise ValueError(f"Loss component {name!r} has nonzero numerator with zero local denominator")
                if self.denominators[i] != 0:
                    normalized = numerator/self.denominators[i]
                    self.means[i] += numerator.detach().double()/self.denominators[i]
                    loss = loss + self.specs[name].weight*normalized
            if not torch.isfinite(loss) or not torch.isfinite(self.means).all():
                raise ValueError("Composed loss normalization produced a non-finite value")
            return loss
        return checked(self.a,compute)

    def metrics(self):
        means = self.a.reduce(self.means,reduction="sum")
        values, total = {}, 0.
        for i,name in enumerate(self.names):
            denom, weight = self.denominators[i].item(), self.specs[name].weight
            mean = means[i].item() if denom else 0.
            total += weight*mean
            values.update({f"train/loss/{name}":mean, f"train/loss_weight/{name}":weight,
                           f"train/loss_denominator/{name}":denom})
        if not math.isfinite(total):
            raise FloatingPointError("Non-finite composed loss total")
        return {"train/loss_total":total, **values}


def target_count(batch, mode):
    mask = None
    for value in batch.loss_masks.values():
        mask = value.ne(0) if mask is None else mask | value.ne(0)
    if batch.labels is not None:
        valid = batch.labels.ne(-100)
        mask = valid if mask is None else mask & valid
    return 0 if mask is None else mask[:,int(mode == "shifted"):].sum().item()
