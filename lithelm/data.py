"""HF loading, offset-safe SFT masks, mmap caches and deterministic batch sampling."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import warnings

from filelock import FileLock
import numpy as np
import torch

PREPROCESS_KEYS = {"stream_mode", "sample_mode", "max_chars", "max_samples", "format", "response_marker", "loss_mask", "separator"}


def raw_records(cfg):
    """Stream bounded records; TinyStories text records end in <|endoftext|>."""
    from .config import validate_dataset_kwargs
    validate_dataset_kwargs(cfg)
    kw = cfg["kwargs"]
    max_chars, max_samples = kw.get("max_chars"), kw.get("max_samples")
    if cfg["source"] == "text":
        path = Path(cfg["path"])
        sep = kw.get("separator", "<|endoftext|>" if kw.get("stream_mode") == "continuous" or kw.get("format", cfg["format"]) == "tinystories_instruct" else "\n")
        if not sep:
            raise ValueError("dataset separator must be nonempty")
        def records():
            remaining, pending = max_chars, ""
            with path.open(encoding="utf-8") as f:
                while remaining is None or remaining > 0:
                    chunk = f.read(min(65536, remaining) if remaining is not None else 65536)
                    if not chunk:
                        break
                    if remaining is not None:
                        remaining -= len(chunk)
                    pending += chunk
                    while sep in pending:
                        record, pending = pending.split(sep, 1)
                        if record.strip():
                            yield {cfg["text_field"]: record + sep}
                if pending.strip():
                    yield {cfg["text_field"]: pending}
        source = records()
    else:
        from datasets import load_dataset, DownloadConfig
        kwargs = {k:v for k,v in kw.items() if k not in PREPROCESS_KEYS}
        cache = cfg["cache_dir"] or os.environ.get("HF_DATASETS_CACHE", "cache/hf/datasets")
        kwargs["download_config"] = DownloadConfig(cache_dir=str(Path(cache)/"downloads"))
        if cfg["source"] == "json":
            source = load_dataset("json", data_files={cfg["split"]: cfg["path"]}, split=cfg["split"],
                                  streaming=cfg["streaming"], cache_dir=cache, **kwargs)
        else:
            source = load_dataset(cfg["path"], split=cfg["split"], streaming=cfg["streaming"], cache_dir=cache, **kwargs)
    seen_chars = 0
    for i, row in enumerate(source):
        if max_samples is not None and i >= max_samples:
            break
        if i == 0 and kw.get("format",cfg["format"]) == "tinystories_instruct" and kw.get("loss_mask","response_only") == "response_only":
            marker = kw.get("response_marker","Story:\n").rstrip()
            if not re.search(r"(?m)^" + re.escape(marker),row[cfg["text_field"]]):
                warnings.warn(f"{cfg['path']}: skipped initial partial Instruct record with no Story target boundary",stacklevel=2)
                continue
        if cfg["source"] != "text" and max_chars is not None:
            text = row.get(cfg["text_field"])
            if text is None:
                raise ValueError("max_chars on structured data requires text_field; use max_samples")
            row = {**row, cfg["text_field"]: text[:max_chars-seen_chars]}
            seen_chars += len(row[cfg["text_field"]])
            if not row[cfg["text_field"]]:
                break
        yield row


def format_record(row, cfg):
    """Return rendered text and character ranges whose tokens are training targets."""
    if cfg["objective"] == "pt":
        text = row[cfg["text_field"]]
        return text, [(0, len(text))]
    kw = cfg["kwargs"]
    fmt = kw.get("format", cfg["format"])
    if fmt == "tinystories_instruct":
        text = row[cfg["text_field"]]
        if kw.get("loss_mask","response_only") == "all_tokens":
            return text,[(0,len(text))]
        marker = kw.get("response_marker", "Story:\n")
        # Official records use both "Story: <text>" and "Story: \n\n<text>".
        match = re.search(r"(?m)^" + re.escape(marker.rstrip()) + r"[ \t]*(?:\r?\n[ \t]*)*", text) if marker.endswith("\n") else None
        if marker in text:
            start = text.index(marker) + len(marker)
        elif match:
            start = match.end()
        else:
            raise ValueError(f"TinyStoriesInstruct record missing response_marker {marker!r}")
        suffix = re.search(r"(?m)^(?:Summary|Features|Words|Random sentence):",text[start:])
        ranges = [(start, start + suffix.start() if suffix else len(text))]
    elif fmt == "prompt_response":
        system = str(row.get(cfg["system_field"], "")) + "\n" if cfg["system_field"] else ""
        prompt = system + str(row[cfg["prompt_field"]]) + "\n"
        text = prompt + str(row[cfg["response_field"]])
        ranges = [(len(prompt), len(text))]
    elif fmt == "messages":
        text, ranges = "", []
        for message in row[cfg["messages_field"]]:
            role, content = message["role"], message["content"]
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("messages require system/user/assistant roles and string content")
            text += f"{role}:\n"
            start = len(text)
            text += content + "\n"
            if role == "assistant":
                ranges.append((start, len(text)))
    else:
        raise ValueError(f"Unknown SFT format {fmt}")
    if kw.get("loss_mask", "response_only") == "all_tokens":
        ranges = [(0, len(text))]
    return text, ranges


def preprocess(row, cfg, tokenizer):
    text, ranges = format_record(row, cfg)
    if cfg["objective"] == "sft" and cfg["append_eos"]:
        eos = tokenizer.eos_token
        if eos is None or tokenizer.eos_token_id is None:
            raise ValueError("SFT append_eos requires tokenizer.eos_token")
        # Insert after each response, before any later prompt or metadata.
        parts, adjusted, cursor, shift = [], [], 0, 0
        for lo, hi in ranges:
            parts.extend((text[cursor:hi], eos))
            adjusted.append((lo+shift, hi+shift+len(eos)))
            cursor, shift = hi, shift+len(eos)
        text, ranges = "".join(parts)+text[cursor:], adjusted
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False,
                        verbose=False)
    ids = encoded["input_ids"]
    # Token must lie entirely inside a target range; a boundary-spanning token
    # remains masked so prompt characters cannot accidentally become targets.
    labels = [token if end > start and any(lo <= start and end <= hi for lo,hi in ranges) else -100
              for token, (start,end) in zip(ids, encoded["offset_mapping"])]
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels}


def collate(examples, length, pad_id=0, target_mode="shifted"):
    if target_mode not in ("shifted", "next_token"):
        raise ValueError(f"Unknown target mode: {target_mode}")
    offset = int(target_mode == "next_token")
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        n = min(length, max(0, len(row["input_ids"])-offset))
        ids[i,:n] = torch.as_tensor(np.asarray(row["input_ids"][:n]).copy(), dtype=torch.long)
        labels[i,:n] = torch.as_tensor(np.asarray(row["labels"][offset:n+offset]).copy(), dtype=torch.long)
        mask[i,:n] = 1
    batch = dict(input_ids=ids, labels=labels)
    if not mask.all():
        batch["attention_mask"] = mask
    return batch


def source_metadata(value):
    if isinstance(value, dict):
        return {k: source_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [source_metadata(v) for v in value]
    path = Path(value)
    if path.is_file():
        stat = path.stat()
        return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if path.is_dir():
        return {file.relative_to(path).as_posix(): source_metadata(file)
                for file in sorted(path.rglob("*")) if file.is_file()}
    return None


def cache_dataset(name, cfg, tokenizer, root):
    """Persist tokens/masks and record boundaries atomically, opening them via mmap."""
    from .tokenizer import tokenizer_fingerprint
    sources = [cfg["path"], cfg["kwargs"].get("data_files", [])]
    identity = dict(schema=3, config=cfg, source=source_metadata(sources), tokenizer=tokenizer_fingerprint(tokenizer))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    base = Path(root) / f"{name}-{digest}"
    def file(suffix):
        return base.with_name(base.name + suffix)
    metadata = file(".json")
    dtype = np.dtype("<u2" if len(tokenizer) <= 65536 else "<u4")
    base.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(file(".lock")), timeout=600):
        if metadata.exists():
            if json.loads(metadata.read_text())["identity"] != identity:
                raise ValueError(f"Cache identity mismatch: {metadata}")
        else:
            _write_cache(name, cfg, tokenizer, file, dtype, identity)
    return dict(tokens=np.memmap(file(".tokens.bin"), dtype=dtype, mode="r"),
                mask=np.memmap(file(".mask.bin"), dtype=np.uint8, mode="r"),
                offsets=np.load(file(".offsets.npy"), mmap_mode="r"), cfg=cfg)


def _write_cache(name, cfg, tokenizer, file, dtype, identity):
    offsets, total = [0], 0
    with file(".tokens.tmp").open("wb") as tokens, file(".mask.tmp").open("wb") as masks:
        for row in raw_records(cfg):
            record = preprocess(row, cfg, tokenizer)
            ids, labels = record["input_ids"], record["labels"]
            if len(ids) < 2 or not any(x != -100 for x in labels[1:]):
                continue
            np.asarray(ids, dtype=dtype).tofile(tokens)
            np.asarray([x != -100 for x in labels], dtype=np.uint8).tofile(masks)
            total += len(ids)
            offsets.append(total)
    if not total:
        raise ValueError(f"Dataset {name} has no usable target tokens")
    file(".tokens.tmp").replace(file(".tokens.bin"))
    file(".mask.tmp").replace(file(".mask.bin"))
    np.save(file(".offsets.npy"), np.asarray(offsets, dtype=np.int64))
    tmp = file(".json.tmp")
    tmp.write_text(json.dumps(dict(identity=identity, tokens=total, records=len(offsets)-1)), encoding="utf-8")
    tmp.replace(file(".json"))


def cache_root(cfg):
    return (Path(cfg["experiment"]["output_dir"]) / "cache" if cfg["workspace"]["copy_inputs"]
            else Path(cfg["paths"]["corpus_root"]) / ".lithelm-cache")


def prepare_data(cfg, tokenizer):
    for name, spec in cfg["datasets"].items():
        cache_dataset(name, spec, tokenizer, cache_root(cfg))


def curriculum(stage, step):
    data = deepcopy(stage["data"])
    for i, phase in enumerate(stage["curriculum"]):
        if step < phase["until_step"]:
            if phase["max_length"] is not None:
                data["max_length"] = phase["max_length"]
            if phase["weights"]:
                for ref in data["datasets"]:
                    ref["weight"] = phase["weights"].get(ref["name"], 0.)
            return i, data
    return 0, data


def slice_record(cache, start, end):
    ids = np.asarray(cache["tokens"][start:end], dtype=np.int64)
    labels = np.where(cache["mask"][start:end], ids, -100)
    return dict(input_ids=ids, labels=labels)


class BatchStream:
    """Counter-seeded sampling makes resume O(1), independent of model RNG.

    A batch has one source, so provenance and replay statistics need no token tags.
    Mmap batches are read synchronously.
    """
    def __init__(self, cfg, tokenizer, data, seed, rank=0, target_mode="shifted"):
        self.data, self.seed, self.rank = data, seed, rank
        self.target_mode = target_mode
        self.refs = [r for r in data["datasets"] if r["weight"] > 0]
        self.probabilities = np.asarray([r["weight"] for r in self.refs], dtype=float)
        self.probabilities /= self.probabilities.sum()
        self.caches = []
        for ref in self.refs:
            spec = deepcopy(cfg["datasets"][ref["name"]])
            spec["objective"] = ref.get("objective") or spec["objective"]
            self.caches.append(cache_dataset(ref["name"], spec, tokenizer, cache_root(cfg)))

    def batch(self, index, size):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.rank, index]))
        source = int(rng.choice(len(self.refs), p=self.probabilities))
        cache, ref = self.caches[source], self.refs[source]
        length = self.data["max_length"] + int(self.target_mode == "next_token")
        examples = []
        for sample in range(size):
            for attempt in range(100):
                if self.data["packing"]:
                    high = max(1, len(cache["tokens"]) - length + 1)
                    start = int(rng.integers(high)) if cache["cfg"]["shuffle"] else (index*size+sample+attempt)*length % high
                    end = min(start + length, len(cache["tokens"]))
                else:
                    count = len(cache["offsets"])-1
                    doc = int(rng.integers(count)) if cache["cfg"]["shuffle"] else (index*size+sample+attempt) % count
                    start, end = map(int, cache["offsets"][doc:doc+2])
                    # Long SFT prompts must not silently produce zero-loss batches.
                    if end-start > length:
                        start += int(rng.integers(end-start-length+1))
                    end = min(end, start+length)
                row = slice_record(cache, start, end)
                if np.any(row["labels"][1:] != -100):
                    examples.append(row)
                    break
            else:
                raise ValueError(f"{ref['name']}: no supervised window in 100 draws; increase max_length")
        batch = collate(examples, self.data["max_length"], target_mode=self.target_mode)
        return batch, ref["name"], cache["cfg"]["objective"]


def evaluation_batches(cache, length, batch_size, packing, rank=0, world_size=1, max_batches=None, target_mode="shifted"):
    context = length
    length += int(target_mode == "next_token")
    if packing:
        # One overlapping context token preserves all next-token targets in a stream.
        windows = range(0, max(0, len(cache["tokens"])-1), length-1)
        total = len(windows)
        get = lambda i: slice_record(cache, windows[i], min(windows[i]+length, len(cache["tokens"])))
    else:
        boundaries = cache["offsets"]
        windows = [(int(start), min(int(start)+length, int(end)))
                   for lo, end in zip(boundaries[:-1], boundaries[1:])
                   for start in range(int(lo), int(end)-1, length-1)]
        total = len(windows)
        get = lambda i: slice_record(cache, *windows[i])
    if max_batches is not None:
        total = min(total, max_batches*batch_size*world_size)
    rows = []
    for i in range(rank, total, world_size):
        rows.append(get(i))
        if len(rows) == batch_size:
            yield collate(rows, context, target_mode=target_mode)
            rows = []
    if rows:
        yield collate(rows, context, target_mode=target_mode)
