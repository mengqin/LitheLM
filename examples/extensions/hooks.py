"""Synthetic external hooks; no training control or distributed operations."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lithelm.extensions import ComponentSpec, PlannedBatch
from lithelm.metrics import loss_inputs


class SyntheticPlan:
    def __init__(self, stage, tokenizer, path=None):
        self.stage, self.vocab = stage, len(tokenizer)
        self.offset = json.loads(Path(path).read_text())["offset"] if path else 0

    def identity(self):
        return {"format":"synthetic-ids-v1","offset":self.offset}

    def batch_at(self, stage_step, rank, microbatch_index):
        size, length = self.stage["batch"]["micro_batch_size"], self.stage["data"]["max_length"]
        index = torch.arange(size)[:,None]
        positions = torch.arange(length)[None,:]
        ids = (index*length+positions+self.offset+stage_step*31+rank*17+microbatch_index*7) % self.vocab
        labels = ids.clone()
        mask = positions >= index % 4+1
        labels[~mask] = -100
        return PlannedBatch(ids,labels=labels,loss_masks={"target":mask},source_name="synthetic",
                            objective=self.stage["type"],metadata={"sample_ids":index.flatten()})


class MaskedCrossEntropy:
    def __init__(self, stage, token_weight=.3, example_weight=1.7):
        self.token_weight, self.example_weight = token_weight, example_weight

    def identity(self):
        return {"format":"masked-ce-v1"}

    def components(self, context):
        return {"token_ce":ComponentSpec(self.token_weight),"example_ce":ComponentSpec(self.example_weight)}

    def mask(self,batch,context):
        offset = int(context.target_mode == "shifted")
        valid = batch.labels[:,offset:] != -100
        return valid & batch.loss_masks["target"][:,offset:] if "target" in batch.loss_masks else valid

    def denominators(self,batch,context):
        mask = self.mask(batch,context)
        return {"token_ce":mask.sum(),"example_ce":mask.any(1).sum()}

    def numerators(self,output,batch,context):
        predictions,labels = loss_inputs(output.logits,batch.labels,context.target_mode)
        losses = F.cross_entropy(predictions.float().transpose(1,2),labels,ignore_index=-100,reduction="none")
        mask = self.mask(batch,context)
        losses = losses*mask
        return {"token_ce":losses.sum(),"example_ce":(losses.sum(1)/mask.sum(1).clamp_min(1)).sum()}


class SequenceLR:
    def __init__(self, stage, values):
        self.values = np.asarray(values,dtype=np.float64)
        if len(self.values) != stage["max_steps"] or not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("SequenceLR requires max_steps finite nonnegative values")
        self.values.setflags(write=False)

    def identity(self):
        return {"values":self.values.tolist()}

    def value_at(self, step):
        if not 0 <= step < len(self.values):
            raise IndexError("SequenceLR index out of bounds")
        return float(self.values[step])
