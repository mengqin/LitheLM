"""Token-weighted, sharded evaluation without duplicated tail samples."""
from collections import defaultdict
import torch

from .data import cache_dataset, cache_root, evaluation_batches
from .metrics import METRICS, perplexity


@torch.no_grad()
def evaluate(model, cfg, stage, tokenizer, accelerator):
    was_training = model.training
    model.eval()
    # Uneven evaluation shard lengths must not execute DDP buffer collectives.
    raw = accelerator.unwrap_model(model)
    device = accelerator.device
    results, aggregate = {}, defaultdict(lambda: [0.,0.])
    mode = stage["targets"]["mode"]
    try:
        for ref in stage["eval"]["datasets"]:
            spec = cfg["datasets"][ref["name"]]
            with accelerator.main_process_first():
                cache = cache_dataset(ref["name"],spec,tokenizer,cache_root(cfg))
            requested = [m for m in stage["eval"]["metrics"] if m not in ("nll","ppl")]
            for name in requested:
                if name not in METRICS:
                    raise ValueError(f"Unknown evaluation metric {name}; register in METRICS")
            sums = torch.zeros(2+2*len(requested),device=device,dtype=torch.float64)
            batches = evaluation_batches(cache,stage["data"]["max_length"],stage["eval"]["batch_size"],
                stage["data"]["packing"],accelerator.process_index,accelerator.num_processes,stage["eval"]["max_batches"], target_mode=mode)
            for batch in batches:
                batch = {k:v.to(device) for k,v in batch.items()}
                with accelerator.autocast():
                    out = raw(**batch, target_mode=mode)
                sums[0] += out.loss_sum.double()
                sums[1] += out.token_count
                for i,name in enumerate(requested):
                    numerator,denominator = METRICS[name](out.logits,batch["labels"], **({"target_mode":mode} if mode != "shifted" else {}))
                    sums[2+2*i] += numerator
                    sums[3+2*i] += denominator
            values = accelerator.reduce(sums,reduction="sum").tolist()
            if not values[1]:
                raise ValueError(f"Eval dataset {ref['name']} has no target tokens")
            nll = values[0]/values[1]
            prefix = f"eval/{ref['alias']}"
            results.update({prefix+"/nll":nll,prefix+"/ppl":perplexity(nll),prefix+"/tokens":values[1]})
            for i,name in enumerate(requested):
                results[prefix+"/"+name] = values[2+2*i]/max(1,values[3+2*i])
            for group in ("all",spec["objective"]):
                aggregate[group][0] += values[0]
                aggregate[group][1] += values[1]
        for group,(loss,tokens) in aggregate.items():
            prefix = "eval" if group == "all" else f"eval/{group}"
            results.update({prefix+"/nll":loss/tokens,prefix+"/ppl":perplexity(loss/tokens)})
        return results
    finally:
        model.train(was_training)


def evaluate_checkpoint(cfg, path, stage_name=None):
    from pathlib import Path
    from .workspace import run_directory, find_run
    path = Path(path).resolve()
    cfg = {**cfg, "experiment": {**cfg["experiment"], "output_dir": find_run(path) or cfg["experiment"]["output_dir"]}}
    from .platform_utils import build_accelerator
    a = build_accelerator(cfg)
    try:
        with run_directory(cfg, a) as (cfg, root):
            result = _evaluate_checkpoint(cfg, path, stage_name, a)
            if a.is_main_process:
                import json
                target = root / "evaluations"
                target.mkdir(exist_ok=True)
                (target/f"{path.parent.name}-{path.name}-on-{stage_name or path.parent.name}.json").write_text(json.dumps(result,indent=2))
            return result
    finally:
        a.end_training()


def _evaluate_checkpoint(cfg, path, stage_name, a):
    from .checkpoint import read_state, load_model_weights
    from .model import build_model
    from .infer import checkpoint_tokenizer
    state = read_state(path,a)
    stage_name = stage_name or state["stage_name"]
    stage = next((s for s in cfg["stages"] if s["name"] == stage_name),None)
    if stage is None:
        raise ValueError(f"Unknown stage {stage_name}")
    tok = checkpoint_tokenizer(path,cfg,a)
    from .provenance import record_environment
    record_environment(cfg, tok, a)
    model = build_model(cfg["model"])
    load_model_weights(model,path,a)
    model = a.prepare_model(model)
    return evaluate(model,cfg,stage,tok,a)
