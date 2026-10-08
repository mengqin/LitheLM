"""Idempotent BPE preparation using Tokenizers, and HF artifact loading."""
import hashlib
import json
from pathlib import Path

from filelock import FileLock


def tokenizer_fingerprint(tokenizer):
    content = tokenizer.backend_tokenizer.to_str()
    if tokenizer.special_tokens_map:
        content += json.dumps(tokenizer.special_tokens_map, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def prepare_tokenizer(cfg, allow_train=True):
    c = cfg["tokenizer"]
    path = Path(c["artifact"])
    target = Path("tokenizer") if cfg["workspace"]["copy_inputs"] else Path(cfg["paths"]["tokenizer_root"])/("hf-"+hashlib.sha256(c["artifact"].encode()).hexdigest()[:12])
    asset = path if c["type"] == "bpe" or path.is_dir() else target
    asset.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(asset.with_name(asset.name+".lock")), timeout=600):
        return _prepare_tokenizer(cfg, allow_train, path, target)


def _prepare_tokenizer(cfg, allow_train, path, target):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    c = cfg["tokenizer"]
    if c["type"] == "hf":
        source = path if path.is_dir() else target if target.is_dir() else c["artifact"]
        tok = AutoTokenizer.from_pretrained(source, use_fast=True, cache_dir=str(Path(cfg["paths"]["tokenizer_root"])/"hub"))
        if not tok.is_fast:
            raise ValueError("A fast tokenizer is required for response offset masking")
    else:
        if not path.is_file():
            if not allow_train:
                raise FileNotFoundError(f"Tokenizer artifact missing: {path}; run lithelm prepare")
            from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
            from .data import raw_records, format_record
            name = c["train_from"]["dataset"]
            if name not in cfg["datasets"]:
                raise ValueError("tokenizer.train_from.dataset must name a catalog entry")
            d = cfg["datasets"][name]
            budget = c["train_from"]["sample_bytes"]
            def texts():
                remaining = budget
                for row in raw_records(d):
                    text, _ = format_record(row, d)
                    b = text.encode("utf-8")[:remaining]
                    yield b.decode("utf-8", errors="ignore")
                    remaining -= len(b)
                    if remaining <= 0:
                        break
            backend = Tokenizer(models.BPE())
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(vocab_size=c["vocab_size"], special_tokens=c["special_tokens"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
            backend.train_from_iterator(texts(), trainer=trainer)
            if backend.get_vocab_size() != c["vocab_size"]:
                raise ValueError("Tokenizer training corpus too small for requested exact vocab_size")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            backend.save(str(temporary))
            temporary.replace(path)
        tok = PreTrainedTokenizerFast(tokenizer_file=str(path), model_max_length=cfg["model"]["max_position_embeddings"])
    for role in ("eos_token", "bos_token", "pad_token"):
        token = c[role]
        if token is not None:
            if token not in tok.get_vocab():
                raise ValueError(f"tokenizer.{role} {token!r} is absent from the vocabulary; prepare a new tokenizer artifact")
            tok.add_special_tokens({role: token})
            if tok.encode(token, add_special_tokens=False) != [getattr(tok, role+"_id")]:
                raise ValueError(f"tokenizer.{role} must encode as one token")
    if len(tok) != cfg["model"]["vocab_size"]:
        raise ValueError(f"Tokenizer artifact vocab {len(tok)} != model vocab {cfg['model']['vocab_size']}")
    if c["type"] == "hf" and (cfg["workspace"]["copy_inputs"] or not path.is_dir()):
        tok.save_pretrained(target)
        c["artifact"] = str(target)
    return tok
