"""Scalar dicts to stdout/JSONL and optional external sinks."""
from datetime import datetime, timezone
import json
import csv
from pathlib import Path


class Logger:
    def __init__(self, cfg, output_dir, enabled=True, formats=None):
        self.cfg, self.enabled, self.count = cfg, enabled, 0
        self.file = self.tb = self.wb = self.csv_file = None
        self.formats = formats or {}
        if not enabled:
            return
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if cfg["jsonl"]:
            (root / cfg["jsonl_file"]).parent.mkdir(parents=True, exist_ok=True)
            self.file = (root / cfg["jsonl_file"]).open("a", encoding="utf-8")
        if cfg["csv"]:
            path = root / cfg["csv_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            header = not path.exists() or path.stat().st_size == 0
            self.csv_file = path.open("a", newline="", encoding="utf-8")
            self.csv = csv.writer(self.csv_file)
            if header:
                self.csv.writerow(["timestamp", "global_step", "stage", "event", "metric", "value"])
        if cfg["tensorboard"]["enabled"]:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError("Install lithelm[tensorboard]") from exc
            self.tb = SummaryWriter(str(root / cfg["tensorboard"]["dir"]))
        if cfg["wandb"]["enabled"]:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("Install lithelm[wandb]") from exc
            self.wb = wandb.init(project=cfg["wandb"]["project"], dir=str(root))

    def log(self, metrics):
        if not self.enabled:
            return
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        if self.cfg["stdout"]:
            display = {k:format(v, self.formats[k]) if k in self.formats else v for k,v in row.items()}
            print(json.dumps(display, ensure_ascii=False, allow_nan=False), flush=True)
        if self.file:
            self.file.write(line + "\n")
            self.count += 1
            if self.count % self.cfg["flush_every"] == 0:
                self.file.flush()
        if self.csv_file:
            for key, value in metrics.items():
                self.csv.writerow([row["timestamp"], row.get("global_step", 0), row.get("stage", ""), row.get("event", ""), key, value])
            self.csv_file.flush()
        step = row.get("global_step", 0)
        if self.tb:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb.add_scalar(key, value, step)
        if self.wb:
            self.wb.log(metrics, step=step)

    def close(self):
        for sink in (self.file, self.tb, self.csv_file):
            if sink:
                sink.close()
        if self.wb:
            self.wb.finish()
