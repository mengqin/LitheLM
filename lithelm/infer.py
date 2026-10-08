"""Reference generation with one prompt prefill and cached single-token decode."""
from pathlib import Path
import torch
from .config import context_length


@torch.inference_mode()
def generate(model, input_ids, *, max_new_tokens=256, do_sample=False, temperature=1., top_p=1., top_k=0,
             eos_token_id=None, max_context_length=None):
    if max_new_tokens < 0 or temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("Invalid generation length/temperature/top_p/top_k")
    ids = input_ids
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("Generation requires a nonempty tokenized prompt")
    limit = context_length(model.cfg, max_context_length)
    if ids.shape[1] > limit:
        raise ValueError("Prompt exceeds requested max_context_length capacity; shorten it before generation")
    budget = min(max_new_tokens, limit-ids.shape[1])
    if max_new_tokens == 0:
        return ids
    was_training = model.training
    model.eval()
    done = torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device)
    try:
        out = model(ids, use_cache=True, max_context_length=limit)
        for step in range(budget):
            logits = out.logits[:, -1].float()
            if do_sample:
                logits = logits / temperature
                if top_k:
                    logits = logits.masked_fill(logits < logits.topk(min(top_k,logits.shape[-1])).values[:,-1:],-torch.inf)
                if top_p < 1:
                    sorted_logits, order = logits.sort(descending=True)
                    remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
                    remove[:,1:] = remove[:,:-1].clone()
                    remove[:,0] = False
                    logits = logits.masked_fill(torch.zeros_like(remove).scatter(1,order,remove),-torch.inf)
                nxt = torch.multinomial(logits.softmax(-1),1)
            else:
                nxt = logits.argmax(-1,keepdim=True)
            if eos_token_id is not None:
                nxt[done] = eos_token_id
                done |= nxt[:,0].eq(eos_token_id)
            ids = torch.cat((ids,nxt),dim=1)
            if done.all() or step+1 == budget:
                break
            out = model(nxt, cache=out.cache, use_cache=True)
        return ids
    finally:
        model.train(was_training)


def checkpoint_tokenizer(path,cfg,accelerator=None):
    from transformers import AutoTokenizer
    from .tokenizer import prepare_tokenizer, tokenizer_fingerprint
    from .checkpoint import read_state
    state = read_state(path, accelerator)
    if (Path(path)/"tokenizer").is_dir():
        tok = AutoTokenizer.from_pretrained(Path(path)/"tokenizer",local_files_only=True)
    else:
        tok = prepare_tokenizer(cfg, allow_train=False)
    expected = state.get("tokenizer_hash")
    if expected and tokenizer_fingerprint(tok) != expected:
        raise ValueError("Checkpoint tokenizer fingerprint mismatch")
    if len(tok) != cfg["model"]["vocab_size"]:
        raise ValueError("Checkpoint tokenizer vocabulary mismatch")
    return tok


def infer_checkpoint(cfg,path,prompt):
    from pathlib import Path
    from .workspace import run_directory, find_run
    path = Path(path).resolve()
    cfg = {**cfg, "experiment": {**cfg["experiment"], "output_dir": find_run(path) or cfg["experiment"]["output_dir"]}}
    from .platform_utils import build_accelerator
    a = build_accelerator(cfg)
    try:
        with run_directory(cfg, a) as (cfg, root):
            text = _infer_checkpoint(cfg, path, prompt, a)
            if a.is_main_process:
                import json
                output = root / "inference"
                output.mkdir(exist_ok=True)
                try:
                    checkpoint = path.relative_to(root).as_posix()
                except ValueError:
                    checkpoint = str(path)
                with (output/"samples.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(dict(checkpoint=checkpoint, prompt=prompt,text=text),ensure_ascii=False)+"\n")
            return text
    finally:
        a.end_training()


def _infer_checkpoint(cfg,path,prompt,a):
    from .model import build_model
    from .checkpoint import load_model_weights
    tok = checkpoint_tokenizer(path,cfg,a)
    from .provenance import record_environment
    record_environment(cfg, tok, a)
    model = build_model(cfg["model"])
    load_model_weights(model,path,a)
    model.to(a.device)
    ids = tok(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"].to(a.device)
    with a.autocast():
        result = generate(model,ids,**cfg["inference"],eos_token_id=tok.eos_token_id)
    return tok.decode(result[0],skip_special_tokens=True)
