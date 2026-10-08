"""Dependency-light command dispatch; training imports only when selected."""
import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="lithelm")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    for name in ("train", "eval", "infer", "prepare"):
        c = commands.add_parser(name)
        c.add_argument("config")
        c.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        c.add_argument("--run-dir", help="reuse an existing run directory (including a moved run)")
        if name in ("eval", "infer"):
            c.add_argument("--checkpoint", required=True)
        if name == "eval":
            c.add_argument("--stage", default=None)
        if name == "train":
            c.add_argument("--resume")
            c.add_argument("--stop-after", type=int, help="stop at global optimizer step, save resumable checkpoint")
        if name == "infer":
            c.add_argument("--prompt", required=True)
            c.add_argument("--max-new-tokens", type=int)
            c.add_argument("--max-context-length", type=int, help="total prompt and generation capacity")
    args, overrides = p.parse_known_args(argv)
    if args.command == "doctor":
        if overrides:
            p.error("doctor takes no overrides")
        from .platform_utils import environment_report
        print(json.dumps(environment_report(), indent=2))
        return
    overrides = args.set + overrides
    if args.command == "infer" and args.max_context_length is not None:
        overrides.append(f"inference.max_context_length={args.max_context_length}")
    cfg = load_config(args.config, overrides)
    from .workspace import find_run, run_directory
    root = args.run_dir or (find_run(args.resume) if args.command == "train" and args.resume and args.resume != "latest" else None)
    if root:
        cfg["experiment"]["output_dir"] = str(Path(root).resolve())
    if args.command == "train":
        from .train import train
        if args.resume == "latest":
            if not cfg["experiment"]["output_dir"]:
                p.error("--resume latest requires a saved run config or --run-dir")
        train(cfg, args.resume, args.stop_after)
    elif args.command == "prepare":
        from .tokenizer import prepare_tokenizer
        from .data import prepare_data
        from .config import save_config
        with run_directory(cfg) as (cfg, root):
            tokenizer = prepare_tokenizer(cfg)
            prepare_data(cfg, tokenizer)
            from .provenance import record_environment
            record_environment(cfg, tokenizer)
            save_config(cfg, "resolved_config.yaml")
            print(root)
    elif args.command == "eval":
        from .evaluate import evaluate_checkpoint
        print(json.dumps(evaluate_checkpoint(cfg, args.checkpoint, args.stage), indent=2))
    else:
        from .infer import infer_checkpoint
        if args.max_new_tokens is not None:
            cfg["inference"]["max_new_tokens"] = args.max_new_tokens
        print(infer_checkpoint(cfg, args.checkpoint, args.prompt))
