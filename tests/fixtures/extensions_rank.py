"""Every torchrun rank explicitly imports and registers the synthetic hooks."""
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_extensions import GlobalPlan, MultiLoss, one_composed_update
from lithelm.config import load_config
from lithelm.extensions import LOSS_COMPOSERS, STEP_PLANS

STEP_PLANS["global"] = GlobalPlan


class InvalidDenominator(MultiLoss):
    def denominators(self,batch,context):
        values = super().denominators(batch,context)
        if context.rank == 1:
            values["token"] = -1.
        return values


kind = sys.argv[3] if len(sys.argv)>3 else "normal"
LOSS_COMPOSERS["multi"] = InvalidDenominator if kind == "invalid" else MultiLoss
try:
    result = one_composed_update(load_config(sys.argv[1]))
except ValueError as exc:
    if kind != "invalid" or "finite nonnegative scalar" not in str(exc):
        raise
else:
    assert kind == "normal", "invalid denominator did not fail"
    if int(os.environ["RANK"]) == 0:
        torch.save(result,sys.argv[2])
