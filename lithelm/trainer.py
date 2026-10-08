"""One explicit PT/SFT loop; objectives only appear in metric provenance."""
import json
from pathlib import Path
import time

import torch

from .checkpoint import save_checkpoint, load_checkpoint, load_model_weights, read_state, rotate_checkpoints
from .data import BatchStream, curriculum
from .evaluate import evaluate
from .infer import generate
from .metrics import STATS, meter_metrics
from .statistics import ScalarStats
from .scheduler import build_optimizer, build_scheduler, wsd_phase


def due(step,every):
    return bool(every and step % every == 0)


class Trainer:
    def __init__(self,cfg,model,tokenizer,accelerator,logger):
        self.cfg, self.raw, self.tokenizer = cfg,model,tokenizer
        self.a, self.logger = accelerator,logger
        self.root = Path(cfg["experiment"]["output_dir"])
        self.state = dict(stage_index=0,stage_name="",stage_step=0,global_step=0,curriculum_index=-1,
                          lr_phase="warmup",scheduler_position=0,meters={},samples_seen=0,tokens_seen=0,best_value=None)
        self.latest = {}
        self.stream = None
        self.start = time.perf_counter()

    def emit(self,metrics):
        s = self.state
        row = dict(stage=s["stage_name"],stage_step=s["stage_step"],global_step=s["global_step"],
                   **{"state/stage_step":s["stage_step"],"state/global_step":s["global_step"],
                      "state/curriculum_stage":s["curriculum_index"],"state/samples_seen":s["samples_seen"],
                      "state/tokens_seen":s["tokens_seen"],"perf/elapsed_seconds":time.perf_counter()-self.start},**metrics)
        self.latest.update(row)
        self.logger.log(row)

    def setup_stage(self,index,stage,resume=None):
        # Accelerate owns the wrappers/state lists; free_memory publicly resets them.
        if hasattr(self,"model"):
            self.raw = self.a.unwrap_model(self.model)
            self.model = self.optimizer = self.scheduler = None
            self.a.free_memory()
        self.stage = stage
        if not resume:
            self.state.update(stage_index=index,stage_name=stage["name"],stage_step=0,curriculum_index=-1,
                              lr_phase="warmup",scheduler_position=0,meters={},best_value=None)
            self.latest = {}
        self.a.gradient_accumulation_steps = stage["batch"]["grad_accum_steps"]
        init = stage["init_from"]
        source = self.root / init["stage"] / init["checkpoint"] if init["stage"] else None
        source_state = read_state(source,self.a) if source and not resume else None
        if source and not resume and init["load_model"]:
            load_model_weights(self.raw,source,self.a)
        optimizer = build_optimizer(self.raw,stage["optimizer"])
        if source and not resume and init["load_optimizer"]:
            optimizer.load_state_dict(torch.load(source/"optimizer.bin",map_location="cpu",weights_only=True))
        if not (source and not resume and init["load_scheduler"]):
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] = stage["optimizer"]["lr"]
        scheduler = build_scheduler(optimizer,stage["lr_schedule"])
        if source and not resume:
            if init["load_scheduler"]:
                scheduler.load_state_dict(torch.load(source/"scheduler.bin",map_location="cpu",weights_only=True))
                for group,lr in zip(optimizer.param_groups,scheduler.get_last_lr()):
                    group["lr"] = lr
            if not init["reset_stage_step"]:
                self.state["stage_step"] = source_state["stage_step"]
                if self.state["stage_step"] >= stage["max_steps"]:
                    raise ValueError("init_from.reset_stage_step=false starts beyond this stage max_steps")
        self.model,self.optimizer,self.scheduler = self.a.prepare(self.raw,optimizer,scheduler)
        if resume:
            self.state = load_checkpoint(resume,self.a,self.cfg,self.tokenizer)
            metrics = Path(resume)/"metrics.json"
            self.latest = json.loads(metrics.read_text()) if metrics.exists() else {}
        self.stream = None
        self.window_input_tokens = self.window_target_tokens = self.window_samples = self.window_seconds = 0
        self.last_eval_step = -1
        self.optimizer.zero_grad(set_to_none=True)
        unknown = set(stage["stats"]["items"]) - set(STATS) - {"input_tokens_per_sec","target_tokens_per_sec","samples_per_sec","objective_ratio","objective_nll"}
        if unknown:
            raise ValueError(f"Unregistered stats: {unknown}")
        self.stat_specs = self.raw.stat_specs()
        global_batch = stage["batch"]["micro_batch_size"]*stage["batch"]["grad_accum_steps"]*self.a.num_processes
        self.emit({"event":"stage_start","data/global_batch_size":global_batch,
                   "data/max_targets_per_update":global_batch*(stage["data"]["max_length"]-int(stage["targets"]["mode"] == "shifted")),
                   "optim/reset":not resume and not init["load_optimizer"],
                   "optim/scheduler_reset":not resume and not init["load_scheduler"]})

    def update_curriculum(self):
        index,data = curriculum(self.stage,self.state["stage_step"])
        if self.stream is None or index != self.state["curriculum_index"]:
            with self.a.main_process_first():
                self.stream = BatchStream(self.cfg,self.tokenizer,data,
                    self.cfg["experiment"]["seed"]+1009*self.state["stage_index"],self.a.process_index, self.stage["targets"]["mode"])
            self.state["curriculum_index"] = index
            self.emit({"event":"curriculum","data/max_length":data["max_length"]})

    def save(self,name):
        path = self.root/self.stage["name"]/name
        save_checkpoint(path,self.a,self.state,self.cfg,self.tokenizer,self.latest,self.stage["checkpoint"])
        if self.a.is_main_process and name.startswith("checkpoint-"):
            rotate_checkpoints(path.parent,self.stage["checkpoint"]["keep_last"])

    def run_eval(self):
        # Evaluation and sampling must not change the training dropout RNG stream.
        devices = [torch.cuda.current_device()] if self.a.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            metrics = evaluate(self.model,self.cfg,self.stage,self.tokenizer,self.a)
        self.last_eval_step = self.state["stage_step"]
        if metrics:
            self.emit(metrics)
        key = self.stage["checkpoint"]["best_metric"]
        if key:
            if key not in metrics:
                raise ValueError(f"best_metric {key!r} absent from evaluation")
            best,value = self.state["best_value"],metrics[key]
            improved = best is None or (value < best if self.stage["checkpoint"]["best_mode"] == "min" else value > best)
            if improved:
                self.state["best_value"] = value
                self.save("best")

    def sample(self):
        spec = self.stage["eval"]["generate"]
        if self.a.is_main_process:
            raw = self.a.unwrap_model(self.model)
            options = {k:v for k,v in spec.items() if k not in ("enabled","every","prompts")}
            devices = [torch.cuda.current_device()] if self.a.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), self.a.autocast():
                for prompt in spec["prompts"]:
                    ids = self.tokenizer(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.a.device)
                    tokens = generate(raw,ids,**options,eos_token_id=self.tokenizer.eos_token_id)
                    row = dict(stage=self.state["stage_name"],global_step=self.state["global_step"],prompt=prompt,
                               text=self.tokenizer.decode(tokens[0],skip_special_tokens=True))
                    with (self.root/"samples.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(row,ensure_ascii=False)+"\n")
        self.a.wait_for_everyone()

    def train_update(self):
        self.update_curriculum()
        stage,a = self.stage,self.a
        if a.device.type == "cuda":
            torch.cuda.synchronize(a.device)
        started = time.perf_counter()
        step = self.state["stage_step"]
        accumulation,size = stage["batch"]["grad_accum_steps"],stage["batch"]["micro_batch_size"]
        batches = [self.stream.batch(step*accumulation+i,size) for i in range(accumulation)]
        mode = stage["targets"]["mode"]
        local_count = sum(batch["labels"][:,int(mode == "shifted"):].ne(-100).sum().item() for batch,_,_ in batches)
        input_count = sum(batch["attention_mask"].sum().item() if "attention_mask" in batch else batch["input_ids"].numel() for batch,_,_ in batches)
        counts = a.reduce(torch.tensor([input_count,local_count,size*accumulation],device=a.device,dtype=torch.float64),reduction="sum")
        total_count = counts[1]
        if total_count.item() <= 0:
            raise ValueError("No valid targets in optimizer update")
        names = [r["name"] for r in stage["data"]["datasets"]]
        meters = torch.zeros(len(names),3,device=a.device,dtype=torch.float64)
        objectives = {r["name"]:r["objective"] or self.cfg["datasets"][r["name"]]["objective"] for r in stage["data"]["datasets"]}
        stats_due = due(step+1,stage["intervals"]["stats"])
        items = stage["stats"]["items"]
        statistics = self.cfg["statistics"]
        collect = due(step+1, statistics["every"]) if statistics["every"] is not None else stats_due and stage["stats"]["model_internal"]["enabled"]
        internals = ScalarStats(statistics, self.stat_specs, a)
        before = [p.detach().clone() for p in self.model.parameters()] if stats_due and {"update_norm","update_ratio"} & set(items) else None
        lr = self.optimizer.param_groups[0]["lr"]
        phase = wsd_phase(self.scheduler.scheduler.last_epoch,stage["lr_schedule"])
        self.model.train()
        grad_norm = 0.
        for batch,name,objective in batches:
            batch = {k:v.to(a.device) for k,v in batch.items()}
            with a.accumulate(self.model):
                with a.autocast():
                    out = self.model(**batch,collect_stats=collect,target_mode=mode)
                # DDP averages ranks, Accelerate divides accumulation. Undo both
                # normalizations, then divide by actual global target-token count.
                a.backward(out.loss_sum*(a.num_processes*accumulation/total_count))
                row = names.index(name)
                meters[row,0] += out.loss_sum.detach().double()
                meters[row,1] += out.token_count
                meters[row,2] += size
                if a.sync_gradients:
                    grad_norm = a.clip_grad_norm_(self.model.parameters(),stage["batch"]["max_grad_norm"] or float("inf")).item()
                    if self.cfg["runtime"]["precision"] != "fp16":
                        invalid = (~torch.isfinite(torch.tensor(grad_norm, device=a.device))).int()
                        if a.reduce(invalid, reduction="sum").item():
                            raise FloatingPointError(f"Non-finite gradient on one or more ranks: stage={stage['name']} "
                                f"stage_step={step} global_step={self.state['global_step']} "
                                f"precision={self.cfg['runtime']['precision']} grad_norm={grad_norm}")
                self.optimizer.step()
                if a.sync_gradients and not a.optimizer_step_was_skipped:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                if collect:
                    internals.add(out.stats)
        if a.device.type == "cuda":
            torch.cuda.synchronize(a.device)
        elapsed = torch.tensor(time.perf_counter()-started, device=a.device, dtype=torch.float64)
        if a.num_processes > 1:
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        self.window_seconds += elapsed.item()
        inputs, targets, samples = counts.tolist()
        self.window_input_tokens += inputs
        self.window_target_tokens += targets
        self.window_samples += samples
        throughput = {"perf/"+key+"_per_sec":value/max(self.window_seconds,1e-9)
                      for key,value in (("input_tokens",self.window_input_tokens),
                                        ("target_tokens",self.window_target_tokens),("samples",self.window_samples))}
        if a.optimizer_step_was_skipped:
            self.emit({"event":"optimizer_step_skipped"})
            return False
        if not torch.isfinite(meters).all() or not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError("Non-finite training loss or gradient norm")
        values = a.reduce(meters,reduction="sum").tolist()
        for name,values in zip(names,values):
            bucket = self.state["meters"].setdefault(name,[0.,0.,0.,objectives[name]])
            for i,value in enumerate(values):
                bucket[i] += value
            self.state["tokens_seen"] += int(values[1])
            self.state["samples_seen"] += int(values[2])
        self.state["stage_step"] += 1
        self.state["global_step"] += 1
        self.state["lr_phase"] = phase
        self.state["scheduler_position"] = self.scheduler.scheduler.last_epoch
        common = {"optim/lr":lr,"optim/wsd_phase":phase,"optim/grad_norm":grad_norm}
        if stats_due:
            stats = {}
            stats.update({key:value for key,value in throughput.items() if key.removeprefix("perf/") in items})
            for name in items:
                if name in STATS and name != "grad_norm":
                    key = "perf/gpu_memory_gb" if name == "gpu_memory" else "optim/"+name
                    stats[key] = STATS[name](self.model,before=before)
            self.emit({**common,**stats,**meter_metrics(self.state["meters"]),"event":"stats"})
        if collect:
            selected = stage["stats"]["model_internal"]["items"]
            values = {k:v for k,v in internals.finish().items()
                      if statistics["every"] is not None or not selected or k.rsplit("/",1)[-1] in selected}
            self.emit({**values, "event":"model_stats"})
        if due(self.state["stage_step"],stage["intervals"]["log"]):
            self.emit({**common,**meter_metrics(self.state["meters"]),**throughput,"event":"train",
                       "perf/gpu_memory_gb":STATS["gpu_memory"](self.model)})
            self.state["meters"] = {}
            self.window_input_tokens = self.window_target_tokens = self.window_samples = self.window_seconds = 0
        return True

    def run_stage(self,index,stage,resume=None,stop_after=None):
        self.setup_stage(index,stage,resume)
        retries = 0
        while self.state["stage_step"] < stage["max_steps"]:
            if not self.train_update():
                retries += 1
                if retries >= self.cfg["runtime"]["max_optimizer_step_retries"]:
                    raise RuntimeError(f"Optimizer step skipped: stage={stage['name']} "
                        f"stage_step={self.state['stage_step']} global_step={self.state['global_step']} "
                        f"retry_count={retries} precision={self.cfg['runtime']['precision']}")
                continue
            retries = 0
            step = self.state["stage_step"]
            checkpoint_due = due(step,stage["intervals"]["checkpoint"])
            if due(step,stage["intervals"]["eval"]) or (checkpoint_due and stage["checkpoint"]["eval_before_save"]):
                self.run_eval()
            gen = stage["eval"]["generate"]
            if gen["enabled"] and due(step,gen["every"]):
                self.sample()
            if checkpoint_due:
                self.save(f"checkpoint-{step}")
            if stop_after is not None and self.state["global_step"] >= stop_after:
                if not checkpoint_due:
                    self.save(f"checkpoint-{step}")
                return False
        need_best = stage["checkpoint"]["best_metric"] and self.state["best_value"] is None
        need_final_eval = stage["checkpoint"]["export_final"] and stage["checkpoint"]["eval_before_save"] and self.last_eval_step != self.state["stage_step"]
        if need_best or need_final_eval:
            self.run_eval()
        if stage["checkpoint"]["export_final"]:
            self.save("final")
        self.emit({"event":"stage_end"})
        return True
