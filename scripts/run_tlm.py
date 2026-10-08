#!/usr/bin/env python3
"""Single-GPU TLM: prepare, PT, checkpoint resume, SFT, evaluate, infer, plot."""
import argparse
import json
import subprocess
import sys

from prepare_tlm import arguments, prepare
from lithelm.config import load_config
from lithelm.workspace import latest_checkpoint, run_directory


def plot(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, keys in zip(axes, [("train/nll", "eval/pt_val/nll", "eval/sft_val/nll"), ("optim/lr",)]):
        for key in keys:
            points = [(r["global_step"], r[key]) for r in rows if key in r]
            if points:
                axis.plot(*zip(*points), label=key)
        axis.legend()
        axis.grid(alpha=.2)
    axes[0].set_ylabel("NLL")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_xlabel("Global optimizer step")
    fig.tight_layout()
    (root / "charts").mkdir(exist_ok=True)
    fig.savefig(root / "charts/training.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    args = parser.parse_args()
    root = prepare(args)
    config = root / "resolved_config.yaml"
    cfg = load_config(config)
    with run_directory(cfg) as (cfg, root):
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        complete = root / "WORKFLOW_COMPLETE"
        if complete.exists():
            print(f"Workflow already complete: {root}", flush=True)
            return
        def launch(*arguments):
            print("lithelm " + " ".join(map(str, arguments)), flush=True)
            with (logs / "workflow.log").open("a", encoding="utf-8") as output:
                subprocess.run([sys.executable, "-m", "lithelm", *map(str, arguments)],
                               stdout=output, stderr=subprocess.STDOUT, check=True)
        final = root / "sft/final"
        if not (final / "COMPLETE").exists():
            resume = latest_checkpoint(root)
            if resume is None:
                stop = 10 if "smoke" in cfg["experiment"]["tags"] else 1000
                launch("train", config, "--stop-after", stop)
                resume = latest_checkpoint(root)
                if resume is None:
                    raise RuntimeError("Training did not publish a resumable checkpoint")
            launch("train", config, "--resume", resume)
        state = json.loads((final / "trainer_state.json").read_text())
        assert state["global_step"] == sum(s["max_steps"] for s in cfg["stages"])
        assert state["stage_step"] == cfg["stages"][-1]["max_steps"] and state["world_size"] == 1
        for name in ("pt/best", "pt/final", "sft/best", "sft/final"):
            overrides = [] if "smoke" in cfg["experiment"]["tags"] else ["stages.1.eval.max_batches=null"]
            launch("eval", config, "--checkpoint", root/name, "--stage", "sft", *overrides)
            launch("infer", config, "--checkpoint", root/name,
                   "--prompt", "Once upon a time", "--max-new-tokens", 40)
        plot(root)
        complete.write_text(json.dumps(dict(global_step=state["global_step"], world_size=1,
                                           smoke="smoke" in cfg["experiment"]["tags"]), indent=2))
        print(f"Workflow complete: {root}", flush=True)


if __name__ == "__main__":
    main()
