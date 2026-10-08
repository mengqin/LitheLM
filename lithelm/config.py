"""OmegaConf defaults, canonicalization and strict, path-oriented validation."""
from copy import deepcopy
import math
from pathlib import Path
import re

from omegaconf import OmegaConf


NORM = {"type": "rmsnorm", "eps": 1e-6}
ATTENTION = dict(impl="sdpa", op=None, num_heads=8, num_kv_heads=None,
                 head_dim="auto", causal=True, qkv_bias=False, out_bias=False,
                 dropout=0., attention_dropout=None, output_dropout=0.,
                 rope=dict(enabled=False, theta=10000.))
FFN = dict(enabled=True, type="gated", hidden_size=None, activation="silu",
           gated=None, bias=False, dropout=0., internal_norm=dict(type="none", eps=1e-6))
LAYER = dict(type="transformer", name=None, hidden_size=None, residual=True,
             norm_position="pre", norm=NORM, norm1=None, norm2=None,
             mixer=dict(type="attention", attention=ATTENTION,
                        gdn=dict(implementation="fla", mode="chunk", kwargs={}),
                        custom=dict(name=None, kwargs={})), ffn=FFN)
DATASET = dict(source="json", path=None, split="train", objective="pt", shuffle=True, append_eos=False,
               text_field="text", format="prompt_response", messages_field="messages",
               prompt_field="prompt", response_field="response", system_field=None,
               streaming=False, cache_dir=None, kwargs={})
STAGE = dict(name=None, type="pt", max_steps=None, targets=dict(mode=None),
             init_from=dict(stage=None, checkpoint="final", load_model=True,
                            load_optimizer=False, load_scheduler=False, reset_stage_step=True),
             batch=dict(micro_batch_size=1, grad_accum_steps=1, max_grad_norm=1.),
             data=dict(max_length=256, packing=False, datasets=[]), curriculum=[],
             optimizer=dict(type="adamw", lr=3e-4, betas=[.9,.95], eps=1e-8,
                            weight_decay=0., exclude_from_weight_decay=["bias","norm","embedding"]),
             lr_schedule=dict(type="wsd", warmup_steps=0, stable_steps=None,
                              decay_steps=0, peak_lr=None, min_lr=0., decay_style="linear"),
             intervals=dict(log=10, stats=100, eval=1000, checkpoint=5000),
             stats=dict(items=["grad_norm","param_norm","gpu_memory"],
                        model_internal=dict(enabled=False, items=[])),
             eval=dict(datasets=[], metrics=["nll","ppl"], max_batches=None, batch_size=None,
                       generate=dict(enabled=False, every=1000, prompts=[], max_new_tokens=64,
                                     do_sample=True, temperature=.8, top_k=50, top_p=1.)),
             checkpoint=dict(keep_last=3, save_optimizer=True, save_scheduler=True,
                             save_metrics=True, best_metric=None, best_mode="min",
                             eval_before_save=False, export_final=True))
DEFAULTS = dict(version=1,
    paths=dict(corpus_root="./corpora", tokenizer_root="./tokenizers", run_root="./outputs"),
    workspace=dict(copy_inputs=False), targets=dict(mode="shifted"),
    statistics=dict(every=None, microbatch_aggregation="mean", overrides={}),
    experiment=dict(name="model", model_size=None, start_date=None, output_dir=None, seed=42,
                    resume_from=None, tags=[]),
    runtime=dict(device="auto", precision="bf16", deterministic=False, max_optimizer_step_retries=8,
                 matmul_precision="high", compile=False, detect_anomaly=False),
    distributed=dict(strategy="auto", backend="auto", find_unused_parameters=False, timeout_minutes=30),
    tokenizer=dict(type="bpe", vocab_size=None, artifact=None, eos_token=None, bos_token=None, pad_token=None,
                   train_from=dict(dataset=None, sample_bytes=10485760),
                   special_tokens=[], reuse_across_stages=True),
    model=dict(vocab_size=None, hidden_size=None, max_position_embeddings=4096,
               tie_word_embeddings=True, lm_head_bias=False, auto_project_width_changes=True,
               position_embedding=dict(type="none", max_positions=None), embeddings=dict(dropout=0.),
               init=dict(std=.02, scale_residual_projections=True),
               final_norm=NORM, layer_defaults={}, layers=[]),
    datasets={}, stages=[],
    logging=dict(rank0_only=True, stdout=True, jsonl=True, jsonl_file="metrics.jsonl", csv=False, csv_file="metrics.csv", flush_every=1,
                 wandb=dict(enabled=False, project="lithelm"), tensorboard=dict(enabled=False, dir="tb")),
    inference=dict(max_context_length=None, max_new_tokens=256, do_sample=False, temperature=1., top_p=1., top_k=0))


def merge(defaults, value, path):
    """Merge a small schema dict, rejecting misspelled keys (kwargs are open)."""
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
    result = deepcopy(defaults)
    for k, v in value.items():
        if isinstance(defaults[k],bool) and not isinstance(v,bool):
            raise ValueError(f"{path}.{k}: expected boolean")
        if isinstance(defaults[k], dict) and k not in {"kwargs", "weights", "datasets", "layer_defaults", "overrides"}:
            result[k] = merge(defaults[k], v, f"{path}.{k}")
        else:
            result[k] = deepcopy(v)
    return result


def choice(value, values, path):
    if value not in values:
        raise ValueError(f"{path}: {value!r}; expected one of {list(values)}")


def positive(value, path, zero=False):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{path}: expected {'nonnegative' if zero else 'positive'} number")


def integer(value, path, zero=False):
    positive(value, path, zero)
    if not isinstance(value, int):
        raise ValueError(f"{path}: expected integer")


def context_length(model, requested=None):
    """Resolve a fixed inference capacity within the model's position limits."""
    limit = min(model["max_position_embeddings"],
                model.get("position_embedding", {}).get("max_positions") or model["max_position_embeddings"])
    if requested is None:
        return limit
    integer(requested, "inference.max_context_length")
    if requested > limit:
        raise ValueError("inference.max_context_length exceeds max_position_embeddings or position_embedding.max_positions")
    return requested


def weights(values, path):
    for v in values:
        positive(v, path, zero=True)
    if not values or sum(values) <= 0:
        raise ValueError(f"{path}: at least one positive weight required")


def norm(value, path):
    result = merge(NORM, value, path)
    choice(result["type"], ["none", "rmsnorm", "layernorm"], path)
    positive(result["eps"], path + ".eps")
    return result


def canonical_layer(spec):
    spec = deepcopy(spec)
    spec.pop("repeat", None)
    for kind in ("attention", "gdn", "custom"):
        if kind in spec:
            spec.setdefault("mixer", {})[kind] = spec.pop(kind)
    if spec.get("type") in ("attention", "gdn", "custom"):
        spec.setdefault("mixer", {})["type"] = spec["type"]
    return spec


def validate_model(m, precision):
    for k in ("vocab_size", "hidden_size", "max_position_embeddings"):
        integer(m[k], "model." + k)
    pos = m["position_embedding"]
    choice(pos["type"], ["learned_absolute", "rope", "none"], "model.position_embedding")
    pos["max_positions"] = m["max_position_embeddings"] if pos["max_positions"] is None else pos["max_positions"]
    integer(pos["max_positions"], "model.position_embedding.max_positions")
    positive(m["init"]["std"], "model.init.std")
    m["final_norm"] = norm(m["final_norm"], "model.final_norm")
    base = canonical_layer(m["layer_defaults"])
    merge(LAYER, base, "model.layer_defaults")
    expanded = []
    previous = m["hidden_size"]
    for index, spec in enumerate(m["layers"]):
        p = f"model.layers.{index}"
        integer(spec.get("repeat", 1), p + ".repeat")
        raw = OmegaConf.to_container(OmegaConf.merge(base, canonical_layer(spec)))
        layer = merge(LAYER, raw, p)
        choice(layer["type"], ["transformer", "attention", "gdn", "custom"], p + ".type")
        dim = layer["hidden_size"] = m["hidden_size"] if layer["hidden_size"] is None else layer["hidden_size"]
        integer(dim, p + ".hidden_size")
        if previous != dim and not m["auto_project_width_changes"]:
            raise ValueError(p + ": width changes require auto_project_width_changes")
        previous = dim
        choice(layer["norm_position"], ["pre", "post"], p + ".norm_position")
        for k in ("norm", "norm1", "norm2"):
            layer[k] = norm(layer[k] or layer["norm"], p + "." + k)
        f = layer["ffn"]
        f["hidden_size"] = 4 * dim if f["hidden_size"] is None else f["hidden_size"]
        integer(f["hidden_size"], p + ".ffn.hidden_size")
        choice(f["type"], ["standard", "mlp", "gated"], p + ".ffn.type")
        choice(f["activation"], ["gelu", "silu", "relu"], p + ".ffn.activation")
        if f["gated"] is not None and f["gated"] != (f["type"] == "gated"):
            raise ValueError(p + ": ffn.gated conflicts with ffn.type")
        f["internal_norm"] = norm(f["internal_norm"], p + ".ffn.internal_norm")
        mix = layer["mixer"]
        choice(mix["type"], ["attention", "gdn", "custom"], p + ".mixer.type")
        a = mix["attention"]
        if mix["type"] == "attention":
            integer(a["num_heads"], p + ".num_heads")
            a["num_kv_heads"] = a["num_heads"] if a["num_kv_heads"] is None else a["num_kv_heads"]
            integer(a["num_kv_heads"], p + ".num_kv_heads")
            if dim % a["num_heads"] or a["num_heads"] % a["num_kv_heads"]:
                raise ValueError(p + ": hidden_size / num_heads / num_kv_heads divisibility")
            if a["head_dim"] == "auto":
                a["head_dim"] = dim // a["num_heads"]
            if a["head_dim"] != dim // a["num_heads"]:
                raise ValueError(p + ": head_dim must equal hidden_size / num_heads")
            a["attention_dropout"] = a["dropout"] if a["attention_dropout"] is None else a["attention_dropout"]
            choice(a["impl"], ["sdpa", "flash", "triton", "custom"], p + ".attention.impl")
            if not a["causal"]:
                raise ValueError(p + ": causal LM requires causal attention")
            if pos["type"] == "rope":
                a["rope"]["enabled"] = True
            if a["rope"]["enabled"] and a["head_dim"] % 2:
                raise ValueError(p + ": RoPE requires even head_dim")
            positive(a["rope"]["theta"], p + ".rope.theta")
            if a["impl"] == "triton":
                a["op"] = a["op"] or "attention_v1"
                choice(a["op"], ["attention_v1", "my_attention_v1"], p + ".attention.op")
                if a["head_dim"] not in (16, 32, 64, 128) or a["attention_dropout"]:
                    raise ValueError(p + ": Triton supports head_dim 16/32/64/128, dropout=0")
            if a["impl"] == "custom" and not a["op"]:
                raise ValueError(p + ": custom attention requires op")
        if mix["type"] == "gdn":
            choice(mix["gdn"]["implementation"], ["fla"], p + ".gdn.implementation")
        if mix["type"] == "custom" and not mix["custom"]["name"]:
            raise ValueError(p + ": custom mixer requires name")
        for d in [m["embeddings"]["dropout"], f["dropout"], a["dropout"], a["output_dropout"], a["attention_dropout"] or 0]:
            if not isinstance(d, (int, float)) or not 0 <= d < 1:
                raise ValueError(p + ": dropout must be in [0, 1)")
        expanded.extend(deepcopy(layer) for _ in range(spec.get("repeat", 1)))
    if not expanded:
        raise ValueError("model.layers: at least one layer required")
    if previous != m["hidden_size"] and m["tie_word_embeddings"] and not m["auto_project_width_changes"]:
        raise ValueError("model: tied head width mismatch")
    m["layers"] = expanded


def normalize(raw):
    """Return a fully resolved, validated ordinary dict; no dependency checks here."""
    cfg = merge(DEFAULTS, raw, "config")
    choice(cfg["version"], [1], "version")
    choice(cfg["targets"]["mode"], ["shifted", "next_token"], "targets.mode")
    statistics = cfg["statistics"]
    if statistics["every"] is not None:
        integer(statistics["every"], "statistics.every", zero=True)
    for mode in [statistics["microbatch_aggregation"], *statistics["overrides"].values()]:
        choice(mode, ["last", "mean", "sum", "min", "max"], "statistics aggregation")
    r = cfg["runtime"]
    integer(r["max_optimizer_step_retries"], "runtime.max_optimizer_step_retries")
    choice(r["device"], ["auto", "cpu", "cuda"], "runtime.device")
    choice(r["precision"], ["fp32", "fp16", "bf16"], "runtime.precision")
    choice(r["matmul_precision"], ["highest", "high", "medium"], "runtime.matmul_precision")
    choice(cfg["distributed"]["strategy"], ["auto", "single", "ddp"], "distributed.strategy")
    choice(cfg["distributed"]["backend"], ["auto", "nccl", "gloo"], "distributed.backend")
    validate_model(cfg["model"], r["precision"])
    t = cfg["tokenizer"]
    choice(t["type"], ["bpe", "hf"], "tokenizer.type")
    t["vocab_size"] = cfg["model"]["vocab_size"] if t["vocab_size"] is None else t["vocab_size"]
    if t["vocab_size"] != cfg["model"]["vocab_size"] or not t["reuse_across_stages"]:
        raise ValueError("tokenizer: model vocab must match; reuse_across_stages must be true")
    if t["artifact"] is None:
        if t["type"] == "hf":
            raise ValueError("HF tokenizer requires an artifact path or model ID")
        t["artifact"] = str(Path(cfg["paths"]["tokenizer_root"]).resolve()/f"bpe-{t['vocab_size']}.json")
    integer(t["train_from"]["sample_bytes"], "tokenizer.train_from.sample_bytes")
    for key in ("eos_token", "bos_token", "pad_token"):
        if t[key] is not None and (not isinstance(t[key], str) or not t[key]):
            raise ValueError(f"tokenizer.{key}: expected nonempty string or null")
        if t["type"] == "bpe" and t[key] is not None and t[key] not in t["special_tokens"]:
            t["special_tokens"].append(t[key])
    if t["type"] == "bpe" and t["vocab_size"] < 256 + len(t["special_tokens"]):
        raise ValueError("Byte-level BPE vocab must cover 256 bytes plus special tokens")
    for name, d in cfg["datasets"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Dataset names must be safe directory names")
        d = cfg["datasets"][name] = merge(DATASET, d, f"datasets.{name}")
        choice(d["source"], ["text", "json", "hf"], f"datasets.{name}.source")
        choice(d["objective"], ["pt", "sft"], f"datasets.{name}.objective")
        if not d["path"]:
            raise ValueError(f"datasets.{name}.path is required")
        validate_dataset_kwargs(d, f"datasets.{name}")
        choice(d["kwargs"].get("loss_mask", "response_only"), ["all_tokens", "response_only"], f"datasets.{name}.loss_mask")
        choice(d["kwargs"].get("format", d["format"]), ["prompt_response", "messages", "tinystories_instruct"], f"datasets.{name}.format")
        for key in ("max_chars","max_samples"):
            if key in d["kwargs"]:
                integer(d["kwargs"][key],f"datasets.{name}.kwargs.{key}")
        choice(d["kwargs"].get("stream_mode","documents"),["continuous","documents"],f"datasets.{name}.stream_mode")
        choice(d["kwargs"].get("sample_mode","random_window"),["random_window"],f"datasets.{name}.sample_mode")
        if d["source"] == "text":
            unknown = set(d["kwargs"]) - {"stream_mode","sample_mode","max_chars","max_samples","format","response_marker","loss_mask","separator"}
            if unknown:
                raise ValueError(f"datasets.{name}.kwargs: unsupported text preprocessing fields {sorted(unknown)}")
    names = []
    for i, raw_stage in enumerate(cfg["stages"]):
        p = f"stages.{i}"
        raw_stage = deepcopy(raw_stage)
        ck = raw_stage.setdefault("checkpoint", {})
        if "save_metrics_snapshot" in ck:
            ck["save_metrics"] = ck.pop("save_metrics_snapshot")
        ev = raw_stage.setdefault("eval", {})
        if isinstance(ev.get("generate"), bool):
            ev["generate"] = {"enabled": ev["generate"]}
        s = cfg["stages"][i] = merge(STAGE, raw_stage, p)
        if not isinstance(s["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", s["name"]) or s["name"] in names:
            raise ValueError(p + ": stage name must be unique, safe directory name")
        choice(s["type"], ["pt", "sft"], p + ".type")
        s["targets"]["mode"] = s["targets"]["mode"] or cfg["targets"]["mode"]
        choice(s["targets"]["mode"], ["shifted", "next_token"], p + ".targets.mode")
        integer(s["max_steps"], p + ".max_steps")
        init = s["init_from"]
        if init["stage"] and init["stage"] not in names:
            raise ValueError(p + ": init_from.stage must precede current stage")
        if not init["stage"] and (init["load_optimizer"] or init["load_scheduler"] or not init["reset_stage_step"]):
            raise ValueError(p + ": init_from state loading requires a source stage")
        if init["load_scheduler"] and not init["load_optimizer"]:
            raise ValueError(p + ": loading scheduler requires loading its optimizer")
        if init["checkpoint"] not in ("best", "final") and not re.fullmatch(r"checkpoint-\d+", init["checkpoint"]):
            raise ValueError(p + ": invalid init_from.checkpoint")
        names.append(s["name"])
        for k in ("micro_batch_size", "grad_accum_steps"):
            integer(s["batch"][k], p + ".batch." + k)
        positive(s["batch"]["max_grad_norm"], p + ".batch.max_grad_norm", zero=True)
        refs = s["data"]["datasets"]
        for j, ref in enumerate(refs):
            refs[j] = merge(dict(name=None, weight=1., objective=None), ref, p + ".data.datasets")
            if refs[j]["name"] not in cfg["datasets"]:
                raise ValueError(p + ": unknown dataset " + str(refs[j]["name"]))
            if refs[j]["objective"] is not None:
                choice(refs[j]["objective"], ["pt", "sft"], p + ".objective")
        if len({x["name"] for x in refs}) != len(refs):
            raise ValueError(p + ": duplicate dataset reference")
        weights([x["weight"] for x in refs], p + ".data.weights")
        lengths = [s["data"]["max_length"]]
        last = 0
        for j, phase in enumerate(s["curriculum"]):
            phase = s["curriculum"][j] = merge(dict(until_step=None, max_length=None, weights={}), phase, p + ".curriculum")
            integer(phase["until_step"], p + ".curriculum.until_step")
            if phase["until_step"] <= last:
                raise ValueError(p + ": curriculum steps must increase")
            last = phase["until_step"]
            if phase["max_length"] is not None:
                lengths.append(phase["max_length"])
            if phase["weights"]:
                if set(phase["weights"]) - {x["name"] for x in refs}:
                    raise ValueError(p + ": curriculum weights reference non-stage dataset")
                weights(list(phase["weights"].values()), p + ".curriculum.weights")
        if s["curriculum"] and last < s["max_steps"]:
            raise ValueError(p + ": curriculum must cover max_steps")
        for length in lengths:
            integer(length, p + ".max_length")
            if length < 2 or length > min(cfg["model"]["max_position_embeddings"], cfg["model"]["position_embedding"]["max_positions"]):
                raise ValueError(p + ": max_length exceeds position limit or is < 2")
            if length > 1024 and any(x["mixer"]["type"] == "attention" and x["mixer"]["attention"]["impl"] == "triton" for x in cfg["model"]["layers"]):
                raise ValueError(p + ": Triton attention_v1 supports max_length <=1024")
        o, w = s["optimizer"], s["lr_schedule"]
        choice(o["type"], ["adamw"], p + ".optimizer.type")
        positive(o["lr"], p + ".optimizer.lr")
        positive(o["eps"], p + ".optimizer.eps")
        positive(o["weight_decay"], p + ".optimizer.weight_decay", zero=True)
        if len(o["betas"]) != 2 or any(not 0 <= b < 1 for b in o["betas"]):
            raise ValueError(p + ": Adam betas must be in [0, 1)")
        choice(w["type"], ["wsd"], p + ".lr_schedule.type")
        choice(w["decay_style"], ["linear", "cosine"], p + ".decay_style")
        w["peak_lr"] = o["lr"] if w["peak_lr"] is None else w["peak_lr"]
        positive(w["peak_lr"],p + ".lr_schedule.peak_lr")
        o["lr"] = w["peak_lr"]
        if w["stable_steps"] is None:
            w["stable_steps"] = s["max_steps"] - w["warmup_steps"] - w["decay_steps"]
        for k in ("warmup_steps", "stable_steps", "decay_steps"):
            integer(w[k], p + ".lr_schedule." + k, zero=True)
        if sum(w[k] for k in ("warmup_steps", "stable_steps", "decay_steps")) != s["max_steps"]:
            raise ValueError(p + ": WSD steps must sum to max_steps")
        if not 0 <= w["min_lr"] <= o["lr"]:
            raise ValueError(p + ": min_lr must be between 0 and peak_lr")
        for k, v in s["intervals"].items():
            integer(v or 0, p + ".intervals." + k, zero=True)
        if s["checkpoint"]["keep_last"] is not None:
            integer(s["checkpoint"]["keep_last"], p + ".checkpoint.keep_last")
        choice(s["checkpoint"]["best_mode"], ["min", "max"], p + ".checkpoint.best_mode")
        ev = s["eval"]
        ev["batch_size"] = s["batch"]["micro_batch_size"] if ev["batch_size"] is None else ev["batch_size"]
        integer(ev["batch_size"], p + ".eval.batch_size")
        if ev["max_batches"] is not None:
            integer(ev["max_batches"], p + ".eval.max_batches")
        for j, ref in enumerate(ev["datasets"]):
            ref = {"name": ref} if isinstance(ref, str) else ref
            ref = ev["datasets"][j] = merge(dict(name=None, alias=None), ref, p + ".eval.datasets")
            if ref["name"] not in cfg["datasets"]:
                raise ValueError(p + ": unknown eval dataset")
            ref["alias"] = ref["alias"] or ref["name"]
        if ev["generate"]["enabled"] and not ev["generate"]["prompts"]:
            raise ValueError(p + ": generation requires explicit prompts")
        integer(ev["generate"]["every"],p + ".eval.generate.every")
        if len({ref["alias"] for ref in ev["datasets"]}) != len(ev["datasets"]):
            raise ValueError(p + ": evaluation aliases must be unique")
        best = s["checkpoint"]["best_metric"]
        metrics = {f"eval/{ref['alias']}/{metric}" for ref in ev["datasets"]
                   for metric in {"nll", "ppl", "tokens", *ev["metrics"]}}
        if ev["datasets"]:
            metrics |= {f"eval{group}/{metric}" for metric in ("nll", "ppl")
                        for group in {"", *("/"+cfg["datasets"][ref["name"]]["objective"] for ref in ev["datasets"])}}
        if best and best not in metrics:
            raise ValueError(p + ": checkpoint.best_metric requires a matching evaluation metric/dataset")
        if init["stage"]:
            source = next(x for x in cfg["stages"][:i] if x["name"] == init["stage"])["checkpoint"]
            if init["checkpoint"] == "final" and not source["export_final"]:
                raise ValueError(p + ": init_from final requires source checkpoint.export_final=true")
            if init["checkpoint"] == "best" and not source["best_metric"]:
                raise ValueError(p + ": init_from best requires source checkpoint.best_metric")
            for kind in ("optimizer", "scheduler"):
                if init["load_"+kind] and not source["save_"+kind]:
                    raise ValueError(p + ": init_from requires source checkpoint.save_"+kind)
        if init["load_scheduler"]:
            previous = next(x for x in cfg["stages"][:i] if x["name"] == init["stage"])
            if previous["lr_schedule"] != s["lr_schedule"]:
                raise ValueError(p + ": continued scheduler requires identical WSD configuration")
    if not cfg["stages"]:
        raise ValueError("stages: at least one stage required")
    if not cfg["logging"]["rank0_only"]:
        raise ValueError("logging.rank0_only=false is unsupported; shared logs require rank 0")
    integer(cfg["logging"]["flush_every"], "logging.flush_every")
    context_length(cfg["model"], cfg["inference"]["max_context_length"])
    for generation in [cfg["inference"]] + [s["eval"]["generate"] for s in cfg["stages"]]:
        integer(generation["max_new_tokens"],"generation.max_new_tokens",zero=True)
        integer(generation["top_k"],"generation.top_k",zero=True)
        positive(generation["temperature"],"generation.temperature")
        if not 0 < generation["top_p"] <= 1:
            raise ValueError("generation.top_p must be in (0,1]")
    return cfg


def validate_dataset_kwargs(spec, path="dataset"):
    reserved = {"path", "split", "streaming", "cache_dir", "download_config"}
    if spec["source"] == "json":
        reserved.add("data_files")
    conflicts = reserved & spec["kwargs"].keys()
    if conflicts:
        raise ValueError(f"{path}.kwargs: reserved fields {sorted(conflicts)}; use top-level dataset fields")


def load_config(path, overrides=()):
    """Load YAML with numeric list-index dot overrides, then resolve interpolations."""
    path = Path(path).resolve()
    cfg = OmegaConf.merge({"paths": DEFAULTS["paths"]}, OmegaConf.load(path))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        key, value = item.split("=", 1)
        value = OmegaConf.to_container(OmegaConf.from_dotlist(["value=" + value]))["value"]
        OmegaConf.update(cfg, key, value, merge=False)
    from .workspace import find_run
    root = find_run(path)
    for key in cfg.paths:
        cfg.paths[key] = str(((root or Path.cwd()) / cfg.paths[key]).resolve())
    cfg = normalize(OmegaConf.to_container(cfg, resolve=True))
    if root is not None:
        cfg["experiment"]["output_dir"] = str(root)
    else:
        # Existing inputs beside a source YAML are imported into a new run.
        for spec in cfg["datasets"].values():
            candidate = path.parent / spec["path"]
            if spec["source"] != "hf" and candidate.exists():
                spec["path"] = str(candidate.resolve())
            def anchor(value):
                if isinstance(value, dict):
                    return {k: anchor(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [anchor(v) for v in value]
                candidate = path.parent / value
                return str(candidate.resolve()) if candidate.exists() else value
            if "data_files" in spec["kwargs"]:
                spec["kwargs"]["data_files"] = anchor(spec["kwargs"]["data_files"])
        candidate = path.parent / cfg["tokenizer"]["artifact"]
        if candidate.exists():
            cfg["tokenizer"]["artifact"] = str(candidate.resolve())
    return cfg


def save_config(cfg, path):
    OmegaConf.save(OmegaConf.create(cfg), Path(path))
