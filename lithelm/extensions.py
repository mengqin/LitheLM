"""Explicit training registries and small, serializable extension contracts."""
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path

import torch

STEP_PLANS = {}
LOSS_COMPOSERS = {}
LR_PROVIDERS = {}


@dataclass
class PlannedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    loss_masks: dict = field(default_factory=dict)
    source_name: str | None = None
    objective: str | None = None
    metadata: dict = field(default_factory=dict)

    def to(self, device):
        def move(value):
            if isinstance(value,torch.Tensor):
                return value.to(device)
            if isinstance(value,dict):
                return {k:move(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):
                return type(value)(move(v) for v in value)
            return value
        return replace(self, input_ids=move(self.input_ids), labels=move(self.labels),
                       attention_mask=move(self.attention_mask),
                       loss_masks=move(self.loss_masks),metadata=move(self.metadata))

    def model_inputs(self, labels=True):
        result = {"input_ids":self.input_ids}
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if labels and self.labels is not None:
            result["labels"] = self.labels
        return result


@dataclass(frozen=True)
class StepContext:
    stage_name: str
    stage_step: int
    rank: int
    target_mode: str


@dataclass(frozen=True)
class ComponentSpec:
    weight: float = 1.
    required: bool = True


def checked(accelerator, function):
    """Propagate extension validation failures before the next DDP collective."""
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = exc
    if accelerator.num_processes == 1:
        if error is not None:
            raise error
    else:
        errors = [None]*accelerator.num_processes
        torch.distributed.all_gather_object(errors, None if error is None else f"{type(error).__name__}: {error}")
        if any(errors):
            raise ValueError(f"Training extension failed across ranks: {errors}")
    return result


def agree(value, accelerator):
    encoded = checked(accelerator, lambda:json.dumps(value, sort_keys=True, allow_nan=False))
    if accelerator.num_processes > 1:
        values = [None]*accelerator.num_processes
        torch.distributed.all_gather_object(values, encoded)
        if any(x != values[0] for x in values):
            raise ValueError("Training extension identities/components differ across ranks")
    return value


def file_identity(path, accelerator, description="Frozen extension file"):
    from accelerate.utils import broadcast_object_list
    def read():
        try:
            local_path = Path(path).resolve()
            digest = hashlib.sha256()
            size = 0
            with local_path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(1024*1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            return dict(path=str(local_path),size=size,sha256=digest.hexdigest())
        except Exception as exc:
            raise ValueError(f"{description}: {type(exc).__name__}: {exc}") from exc
    local = checked(accelerator,read)
    try:
        agree({k:local[k] for k in ("size","sha256")},accelerator)
    except ValueError as exc:
        raise ValueError(f"{description}: sha256/size mismatch across ranks") from exc
    payload = [local if accelerator.is_main_process else None]
    if accelerator.num_processes > 1:
        broadcast_object_list(payload)
    return payload[0]


def validate_batch(batch, stage, vocab_size, *, require_labels=True):
    if not isinstance(batch, PlannedBatch):
        raise ValueError("StepPlan.batch_at must return PlannedBatch")
    ids = batch.input_ids
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2
            or ids.shape[0] != stage["batch"]["micro_batch_size"]
            or not 1 <= ids.shape[1] <= stage["data"]["max_length"]
            or (ids < 0).any() or (ids >= vocab_size).any()):
        raise ValueError("Planned input_ids must be int64 [micro_batch_size, length] within context/vocabulary")
    if require_labels and batch.labels is None:
        raise ValueError("StepPlan with default CE requires labels aligned to targets.mode")
    for name, value in (("labels",batch.labels), ("attention_mask",batch.attention_mask)):
        if value is not None and (not isinstance(value,torch.Tensor) or value.shape != ids.shape):
            raise ValueError(f"Planned {name} must match input_ids shape")
    labels = batch.labels
    if labels is not None and (labels.dtype != torch.long or ((labels != -100) & ((labels < 0) | (labels >= vocab_size))).any()):
        raise ValueError("Planned labels must be int64 vocabulary IDs or -100")
    mask = batch.attention_mask
    if mask is not None and not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Planned attention_mask must contain only 0/1")
    if not isinstance(batch.loss_masks, dict) or any(
            not isinstance(v,torch.Tensor) or v.shape != ids.shape or not torch.isfinite(v).all() for v in batch.loss_masks.values()):
        raise ValueError("Planned loss_masks must map names to tensors matching input_ids")
    if not isinstance(batch.metadata,dict):
        raise ValueError("Planned metadata must be a mapping")
    if any(x is not None and (not isinstance(x,str) or not x) for x in (batch.source_name,batch.objective)):
        raise ValueError("Planned source_name/objective must be nonempty strings or None")
    return batch


def registered(registry, spec, **context):
    if spec["name"] not in registry:
        raise ValueError(f"Unregistered training extension {spec['name']!r}; import/register it on every rank")
    return registry[spec["name"]](**context, **spec["kwargs"])


def prepare_extensions(cfg, tokenizer, accelerator):
    """Freeze all stage identities once, before any optimizer update."""
    bundles, identities = [], {}
    for stage in cfg["stages"]:
        plan = composer = provider = None
        identity = {}
        if stage["data"]["strategy"] == "step_plan":
            spec = stage["data"]["step_plan"]
            frozen = file_identity(spec["path"],accelerator,f"StepPlan {spec['name']!r} frozen file") if spec["path"] else None
            plan = checked(accelerator, lambda:registered(STEP_PLANS, spec, stage=stage, tokenizer=tokenizer, path=spec["path"]))
            identity["step_plan"] = dict(config=spec, identity=agree(checked(accelerator, plan.identity), accelerator))
            if spec["path"]:
                identity["step_plan"]["file"] = frozen
        if stage["loss"]["type"] == "custom":
            spec = stage["loss"]
            composer = checked(accelerator, lambda:registered(LOSS_COMPOSERS,spec,stage=stage))
            identity["loss_composer"] = dict(config=spec, identity=agree(checked(accelerator,composer.identity),accelerator))
            context = StepContext(stage["name"],0,accelerator.process_index,stage["targets"]["mode"])
            names = checked(accelerator, lambda:sorted(composer.components(context)))
            identity["loss_composer"]["components"] = agree(names,accelerator)
        if stage["lr_schedule"]["type"] != "wsd":
            from .lr_provider import ValuesFileLR
            spec = stage["lr_schedule"]
            frozen = file_identity(spec["path"],accelerator,f"LRProvider {spec['type']!r} frozen file") if spec["path"] else None
            if spec["type"] == "values_file":
                provider = checked(accelerator,lambda:ValuesFileLR(spec["path"],stage["max_steps"]))
            else:
                provider = checked(accelerator,lambda:registered(LR_PROVIDERS,spec,stage=stage))
            identity["lr_provider"] = dict(config=spec,identity=agree(checked(accelerator,provider.identity),accelerator))
            if spec["path"]:
                identity["lr_provider"]["file"] = frozen
                if spec["type"] == "values_file" and frozen["sha256"] != provider.identity()["sha256"]:
                    raise ValueError("LR values_file changed during loading")
        bundles.append((plan, composer, provider))
        if identity:
            identities[stage["name"]] = identity
    if identities:
        agree(identities,accelerator)
    return bundles, identities
