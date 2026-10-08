"""Optional scalar specs and two explicit reductions: microbatches, then ranks."""
from dataclasses import dataclass
from fnmatch import fnmatchcase

import torch


@dataclass(frozen=True)
class StatSpec:
    aggregation: str | None = None
    distributed: str = "mean"
    format: str | None = None

    def __post_init__(self):
        if self.aggregation not in (None, "last", "mean", "sum", "min", "max"):
            raise ValueError(f"Unknown microbatch aggregation: {self.aggregation}")
        if self.distributed not in ("mean", "sum", "min", "max", "local"):
            raise ValueError(f"Unknown distributed aggregation: {self.distributed}")
        if self.format is not None:
            format(1.0, self.format)


class ScalarStats:
    def __init__(self, config, specs, accelerator):
        self.config, self.specs, self.a = config, specs, accelerator
        self.values = {}
        if any(not isinstance(spec, StatSpec) for spec in specs.values()):
            raise TypeError("stat_specs() values must be StatSpec")

    def add(self, values):
        for name, value in values.items():
            value = torch.as_tensor(value, device=self.a.device, dtype=torch.float64).detach().clone()
            if value.ndim != 0 or not torch.isfinite(value):
                raise ValueError(f"{name}: statistics must be finite scalars")
            self.values.setdefault(name, []).append(value)

    def finish(self):
        # Detect rank-dependent fields/policies before issuing scalar collectives.
        keys = [(key, self.specs.get(key, StatSpec())) for key in sorted(self.values)]
        if self.a.num_processes > 1:
            from accelerate.utils import gather_object
            if any(other != keys for other in gather_object([keys])):
                raise ValueError("Collected statistic fields/specs must agree across ranks")
        result = {}
        overrides = self.config["overrides"]
        for name, spec in keys:
            fallback = spec.aggregation or self.config["microbatch_aggregation"]
            mode = overrides.get(name, next((value for pattern, value in overrides.items()
                                             if fnmatchcase(name, pattern)), fallback))
            values = torch.stack(self.values[name])
            value = values[-1] if mode == "last" else getattr(values, mode)()
            if spec.distributed in ("sum", "mean"):
                value = self.a.reduce(value, reduction=spec.distributed)
            elif spec.distributed != "local" and self.a.num_processes > 1:
                op = getattr(torch.distributed.ReduceOp, spec.distributed.upper())
                torch.distributed.all_reduce(value, op=op)
            result[name] = value.item()
        return result
