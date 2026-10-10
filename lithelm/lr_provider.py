"""Absolute learning rates indexed by successful stage updates."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch


class ValuesFileLR:
    def __init__(self, path, max_steps):
        path = Path(path).resolve()
        data = path.read_bytes()
        if len(data) != max_steps*8:
            raise ValueError("LR values_file must contain exactly max_steps little-endian float64 values")
        self.values = np.frombuffer(data,dtype="<f8")
        if not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("LR values_file must contain finite nonnegative values")
        self._identity = dict(path=str(path),size=len(data),sha256=hashlib.sha256(data).hexdigest(),
                              length=max_steps,dtype="<f8")

    def identity(self):
        return dict(self._identity)

    def value_at(self, step):
        if type(step) is not int or not 0 <= step < len(self.values):
            raise IndexError(f"LR index {step} is outside the frozen schedule")
        return float(self.values[step])


class ProviderScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, provider, max_steps, position=0):
        self.provider, self.max_steps = provider, max_steps
        super().__init__(optimizer,last_epoch=position-1)

    def get_lr(self):
        if self.last_epoch == self.max_steps and hasattr(self,"_last_lr"):
            return self._last_lr
        if self.last_epoch > self.max_steps:
            raise IndexError("Scheduler advanced beyond stage max_steps")
        value = float(self.provider.value_at(min(self.last_epoch,self.max_steps-1)))
        if not math.isfinite(value) or value < 0:
            raise ValueError("LRProvider.value_at must return a finite nonnegative learning rate")
        return [value]*len(self.optimizer.param_groups)

    def state_dict(self):
        return {k:v for k,v in self.__dict__.items() if k not in ("optimizer","provider")}
