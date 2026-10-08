#!/usr/bin/env python3
"""Download official corpora (or bounded smoke slices) and write a runnable YAML."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import shutil

from lithelm.config import load_config, save_config
from lithelm.workspace import run_directory


def download(repo,filename,path,limit=None):
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}",timeout=30))
    revision = meta["sha"]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        headers = {"Range":f"bytes=0-{limit-1}"} if limit else {}
        request = urllib.request.Request(url,headers=headers)
        temp = path.with_suffix(path.suffix+".partial")
        with urllib.request.urlopen(request,timeout=120) as response, temp.open("wb") as out:
            count = 0
            while limit is None or count < limit:
                chunk = response.read(min(1024*1024,limit-count) if limit else 1024*1024)
                if not chunk:
                    break
                out.write(chunk)
                count += len(chunk)
        if limit:
            text = temp.read_text(encoding="utf-8",errors="ignore")
            boundary = text.rfind("<|endoftext|>")
            if boundary < 0:
                raise ValueError(f"No complete record in {filename}")
            temp.write_text(text[:boundary+len("<|endoftext|>")],encoding="utf-8")
        temp.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    row=dict(path=str(path),url=url,revision=revision,bytes=path.stat().st_size,sha256=digest.hexdigest(),subset=bool(limit))
    print(json.dumps(row),flush=True)
    return row


TEMPLATE = Path(__file__).resolve().parents[1] / "configs/examples/tlm_7m.yaml"


def arguments(parser):
    parser.add_argument("--config", type=Path, default=TEMPLATE)
    parser.add_argument("--run-dir", type=Path, help="continue an existing prepared run")
    parser.add_argument("--smoke", action="store_true", help="20 PT + 20 SFT steps on bounded corpus slices")
    parser.add_argument("--data-dir", type=Path, help="copy existing official raw corpora instead of downloading")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def prepare(args):
    if args.run_dir:
        root = args.run_dir.resolve()
        cfg = load_config(root / "resolved_config.yaml", args.set)
    else:
        cfg = load_config(args.config, args.set)
        if args.smoke:
            cfg["experiment"]["name"] += "-smoke"
            cfg["experiment"]["tags"].append("smoke")
            corpus = Path(cfg["paths"]["corpus_root"])
            for spec in cfg["datasets"].values():
                spec["path"] = str(corpus/"tlm-smoke"/Path(spec["path"]).relative_to(corpus))
            cfg["tokenizer"]["artifact"] = str(Path(cfg["paths"]["tokenizer_root"])/"tlm-smoke/tokenizer.json")
            for stage in cfg["stages"]:
                stage["max_steps"] = 20
                stage["batch"].update(micro_batch_size=4, grad_accum_steps=2)
                stage["lr_schedule"].update(warmup_steps=2, stable_steps=14, decay_steps=4)
                stage["curriculum"] = [dict(until_step=10, max_length=128), dict(until_step=20, max_length=256)]
                stage["intervals"].update(log=2, stats=4, eval=10, checkpoint=10)
                stage["eval"].update(max_batches=2, batch_size=4)
                stage["eval"]["generate"].update(every=20, max_new_tokens=20)
    if cfg["distributed"]["strategy"] != "single":
        raise ValueError("The TLM example requires distributed.strategy=single")
    smoke = "smoke" in cfg["experiment"]["tags"]
    data_dir = args.data_dir.resolve() if args.data_dir else None
    template = args.config.resolve()
    with run_directory(cfg) as (cfg, root):
        if not args.run_dir:
            target = root / "configs/examples/tlm_7m.yaml"
            target.parent.mkdir(parents=True)
            shutil.copy2(template, target)
            (root / "scripts").mkdir()
            for script in ("prepare_tlm.py", "run_tlm.py"):
                shutil.copy2(Path(__file__).with_name(script), root / "scripts" / script)
        save_config(cfg, "resolved_config.yaml")
        sources = [("tinystories_pt_train", "roneneldan/TinyStories", "tinystories", "TinyStories-train.txt", 12*1024**2),
                   ("tinystories_pt_val", "roneneldan/TinyStories", "tinystories", "TinyStories-valid.txt", 1024**2),
                   ("tinystories_instruct_train", "roneneldan/TinyStoriesInstruct", "tinystories_instruct", "TinyStories-Instruct-train.txt", 4*1024**2),
                   ("tinystories_instruct_val", "roneneldan/TinyStoriesInstruct", "tinystories_instruct", "TinyStories-Instruct-valid.txt", 1024**2)]
        manifest = []
        for key, repo, folder, filename, limit in sources:
            path = Path(cfg["datasets"][key]["path"])
            if not path.exists() and data_dir:
                source = data_dir / folder / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                if smoke:
                    with source.open("rb") as f:
                        text = f.read(limit).decode("utf-8", errors="ignore")
                    end = text.rfind("<|endoftext|>")
                    if end < 0:
                        raise ValueError(f"No complete record in {source}")
                    path.write_text(text[:end+len("<|endoftext|>")], encoding="utf-8")
                else:
                    shutil.copy2(source, path)
            if path.exists():
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1024**2), b""):
                        digest.update(block)
                manifest.append(dict(path=str(path), bytes=path.stat().st_size,
                                     sha256=digest.hexdigest(), subset=smoke, repository=repo))
            else:
                manifest.append(download(repo, filename, path, limit if smoke else None))
        manifest_path = Path("corpus-manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        from lithelm.tokenizer import prepare_tokenizer
        from lithelm.data import prepare_data
        tokenizer = prepare_tokenizer(cfg)
        prepare_data(cfg, tokenizer)
        save_config(cfg, "resolved_config.yaml")
        from lithelm.provenance import record_environment
        record_environment(cfg, tokenizer)
        print(root, flush=True)
        return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
