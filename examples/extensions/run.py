"""Explicit registration on every rank before invoking the public train API."""
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from lithelm.config import load_config, save_config
from lithelm.extensions import LOSS_COMPOSERS, LR_PROVIDERS, STEP_PLANS
from lithelm.train import train
from lithelm.workspace import run_directory
from .hooks import MaskedCrossEntropy, SequenceLR, SyntheticPlan

STEP_PLANS["synthetic"] = SyntheticPlan
LOSS_COMPOSERS["masked_ce"] = MaskedCrossEntropy
LR_PROVIDERS["sequence"] = SequenceLR


def make_example(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    (directory/"bootstrap.txt").write_text("A small synthetic sentence for tokenizer preparation.\n"*40,encoding="utf-8")
    (directory/"plan.json").write_text('{"offset":3}',encoding="utf-8")
    np.asarray([.0002,.0005,.0001,0.],dtype="<f8").tofile(directory/"lr-pt.bin")
    np.asarray([.0007,.0003,0.],dtype="<f8").tofile(directory/"lr-sft.bin")
    config = OmegaConf.load(Path(__file__).with_name("config.yaml"))
    config.paths = {"run_root":str(directory/"outputs"),"corpus_root":str(directory/"cache"),
                    "tokenizer_root":str(directory/"tokenizers")}
    config.tokenizer.artifact = str(directory/"tokenizer.json")
    path = directory/"config.yaml"
    OmegaConf.save(config,path)
    with run_directory(load_config(path)) as (bound,root):
        bound["experiment"]["output_dir"] = str(root)
        save_config(bound,path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",nargs="?")
    parser.add_argument("--init",type=Path)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after",type=int)
    parser.add_argument("--set",action="append",default=[])
    args = parser.parse_args()
    if args.init:
        print(make_example(args.init))
    elif args.config:
        train(load_config(args.config,args.set),resume=args.resume,stop_after=args.stop_after)
    else:
        parser.error("provide a config or --init directory")


if __name__ == "__main__":
    main()
