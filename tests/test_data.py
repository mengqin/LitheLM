import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch

from lithelm.config import normalize
from lithelm.data import preprocess, collate, raw_records, curriculum, cache_dataset, BatchStream
from lithelm.tokenizer import prepare_tokenizer, tokenizer_fingerprint


def tiny_config(root):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    output = root / "out"
    output.mkdir(exist_ok=True)
    (output / "run.json").write_text(json.dumps(dict(name="tiny", model_size="6K", start_date="2026-10-06")))
    pt = output / "pt.txt"
    pt.write_text("One small cat went home. Another dog found a happy friend.\n"*50)
    sft = output / "sft.jsonl"
    sft.write_text("\n".join(json.dumps(dict(prompt="Tell a story",response="A little cat found a friend and went home.")) for _ in range(20)))
    return normalize(dict(
        experiment=dict(name="tiny",output_dir=str(root / "out"),seed=17),
        runtime=dict(device="cpu",precision="fp32"),
        tokenizer=dict(vocab_size=280,artifact=str(output/"tokenizer.json"),train_from=dict(dataset="pt",sample_bytes=2000)),
        model=dict(vocab_size=280,hidden_size=16,max_position_embeddings=24,
                   position_embedding=dict(type="learned_absolute",max_positions=24),
                   layer_defaults=dict(norm=dict(type="layernorm"),mixer=dict(attention=dict(num_heads=2)),
                                       ffn=dict(type="standard",hidden_size=32)),layers=[dict(type="transformer")]),
        datasets=dict(pt=dict(source="text",path=str(pt)),sft=dict(source="json",path=str(sft),objective="sft")),
        stages=[dict(name="pt",max_steps=4,data=dict(max_length=16,packing=True,datasets=[dict(name="pt")]),
                     batch=dict(micro_batch_size=2,grad_accum_steps=2),
                     curriculum=[dict(until_step=2,max_length=12),dict(until_step=4,max_length=16)],
                     lr_schedule=dict(warmup_steps=1,decay_steps=1),intervals=dict(log=1,stats=2,eval=2,checkpoint=2),
                     stats=dict(items=["grad_norm","param_norm","update_ratio"]),
                     eval=dict(datasets=["pt"],max_batches=2),checkpoint=dict(best_metric="eval/pt/nll")),
                dict(name="sft",type="sft",max_steps=3,
                     init_from=dict(stage="pt",checkpoint="best"),
                     data=dict(max_length=24,packing=True,datasets=[dict(name="sft",weight=.9),dict(name="pt",weight=.1)]),
                     batch=dict(micro_batch_size=2,grad_accum_steps=2),
                     intervals=dict(log=1,stats=2,eval=2,checkpoint=2),
                     eval=dict(datasets=["pt","sft"],max_batches=1))],
        logging=dict(stdout=False)))


@pytest.fixture
def example(tmp_path):
    c=tiny_config(tmp_path)
    return c,prepare_tokenizer(c)


def test_tokenizer_idempotent(example):
    cfg,tok=example
    path=Path(cfg["tokenizer"]["artifact"])
    before=path.stat().st_mtime_ns
    assert len(tok)==280
    assert tokenizer_fingerprint(prepare_tokenizer(cfg))==tokenizer_fingerprint(tok)
    assert before==path.stat().st_mtime_ns


def test_labels_padding_boundary(example):
    c,tok=example
    pt=preprocess(dict(text="A happy cat."),c["datasets"]["pt"],tok)
    assert pt["input_ids"]==pt["labels"]
    sft=preprocess(dict(prompt="Tell me",response="A happy cat."),c["datasets"]["sft"],tok)
    assert sft["labels"][0]==-100 and sft["labels"][-1]!=-100
    b=collate([pt,sft],64)
    assert (b["labels"][b["attention_mask"]==0]==-100).all()
    assert (b["attention_mask"][:,-1]==0).all()


def test_instruct_and_messages(example):
    c,tok=example
    d=deepcopy(c["datasets"]["sft"])
    d["kwargs"]=dict(format="tinystories_instruct",response_marker="Story:\n")
    row=dict(text="Summary: A cat.\nStory:\nA cat was happy.\n<|endoftext|>")
    out=preprocess(row,d,tok)
    assert -100 in out["labels"] and out["labels"][-1]>=0
    actual=preprocess(dict(text=row["text"].replace("Story:\n","Story: \n\n")),d,tok)
    assert actual["labels"][0]==-100 and actual["labels"][-1]>=0
    inline=preprocess(dict(text="Words: cat\nStory: A happy cat.\nSummary: A cat."),d,tok)
    assert inline["labels"][0]==-100 and inline["labels"][-1]==-100
    assert any(x>=0 for x in inline["labels"])
    with pytest.raises(ValueError,match="response_marker"):
        preprocess(dict(text="no marker"),d,tok)
    d["kwargs"]["loss_mask"]="all_tokens"
    all_tokens=preprocess(row,d,tok)
    assert all_tokens["labels"]==all_tokens["input_ids"]
    d["kwargs"]={}
    d["format"]="messages"
    out=preprocess(dict(messages=[dict(role="user",content="Hi"),dict(role="assistant",content="Hello")]),d,tok)
    assert out["labels"][0]==-100 and out["labels"][-1]>=0


def test_cache_mixing_curriculum_resume(example):
    c,tok=example
    phase,data=curriculum(c["stages"][0],2)
    assert phase==1 and data["max_length"]==16
    stream=BatchStream(c,tok,c["stages"][1]["data"],17)
    counts={"sft":0,"pt":0}
    for i in range(500):
        batch,name,obj=stream.batch(i,1)
        counts[obj]+=1
        assert batch["labels"][:,1:].ne(-100).any()
    assert .84 < counts["sft"]/500 < .95
    restored=BatchStream(c,tok,c["stages"][1]["data"],17)
    torch.testing.assert_close(stream.batch(35,2)[0]["input_ids"],restored.batch(35,2)[0]["input_ids"])
    cache=cache_dataset("pt",c["datasets"]["pt"],tok,Path(c["experiment"]["output_dir"])/"cache")
    assert isinstance(cache["tokens"],np.memmap) and cache["tokens"].dtype.itemsize==2


def test_hf_streaming_and_partial_record(example,tmp_path):
    c,tok=example
    spec=deepcopy(c["datasets"]["sft"])
    spec.update(source="hf",path="json",streaming=True,
                kwargs=dict(data_files={"train":c["datasets"]["sft"]["path"]}))
    assert len(list(raw_records(spec)))==20
    partial=tmp_path/"partial.txt"
    partial.write_text("truncated story<|endoftext|>\nWords: cat\nStory: A cat.<|endoftext|>")
    spec.update(source="text",path=str(partial),kwargs=dict(format="tinystories_instruct"))
    with pytest.warns(UserWarning,match="initial partial"):
        rows=list(raw_records(spec))
    assert len(rows)==1
    spec["kwargs"]["loss_mask"]="all_tokens"
    assert len(list(raw_records(spec)))==2


def retry_stream(monkeypatch, packing, mode="shifted", shuffle=False, lengths=(4, 4)):
    """Use in-memory caches to isolate candidate selection from tokenization."""
    spec = dict(objective="sft", shuffle=shuffle)
    cache = dict(tokens=np.arange(1, sum(lengths)+1), mask=np.ones(sum(lengths)),
                 offsets=np.array([0, *np.cumsum(lengths)]), cfg=spec)
    monkeypatch.setattr("lithelm.data.cache_dataset", lambda *args:cache)
    cfg = dict(datasets=dict(sft=spec), workspace=dict(copy_inputs=False), paths=dict(corpus_root="."))
    data = dict(max_length=4, packing=packing, datasets=[dict(name="sft", weight=1.)])
    return BatchStream(cfg, None, data, seed=17, rank=3, target_mode=mode), cache


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("mode", ["shifted", "next_token"])
def test_sequential_retry_advances_to_supervised_candidate(monkeypatch, packing, mode):
    length = 4+int(mode == "next_token")
    stream, cache = retry_stream(monkeypatch, packing, mode, lengths=(length, length))
    cache["mask"][:] = 0
    cache["mask"][-1] = 1
    batch, name, objective = stream.batch(index=0, size=1)
    assert (name, objective) == ("sft", "sft")
    assert batch["input_ids"].tolist() == [list(range(length+1, length+5))]
    assert batch["labels"].tolist() == [[-100, -100, -100, 2*length]]


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_retry_batch_identity_survives_recreation(monkeypatch, packing, shuffle):
    stream, cache = retry_stream(monkeypatch, packing, "next_token", shuffle, lengths=(19, 23))
    cache["mask"][:19] = 0
    expected = stream.batch(index=7, size=2)
    stream.batch(index=2, size=3)  # Out-of-order access must not advance hidden state.
    repeated = stream.batch(index=7, size=2)
    restored, restored_cache = retry_stream(monkeypatch, packing, "next_token", shuffle, lengths=(19, 23))
    restored_cache["mask"][:19] = 0
    for actual in (repeated, restored.batch(index=7, size=2)):
        assert actual[1:] == expected[1:] == ("sft", "sft")
        assert actual[0].keys() == expected[0].keys()
        for key in expected[0]:
            torch.testing.assert_close(actual[0][key], expected[0][key], atol=0, rtol=0)
        assert actual[0]["labels"].ne(-100).any(dim=1).all()


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("documents", [2, 101])
def test_sequential_retry_limit_and_wraparound(monkeypatch, packing, documents):
    from lithelm.data import slice_record
    stream, cache = retry_stream(monkeypatch, packing, lengths=(4,)*documents)
    cache["mask"][:] = 0
    candidates = []
    def record_candidate(cache, start, end):
        candidates.append((start, end))
        assert 0 <= start < end <= len(cache["tokens"])
        return slice_record(cache, start, end)
    monkeypatch.setattr("lithelm.data.slice_record", record_candidate)
    with pytest.raises(ValueError, match="sft: no supervised window in 100 draws"):
        stream.batch(index=0, size=1)
    assert len(candidates) == 100
    if documents == 101:
        assert len(set(candidates)) == 100
    else:
        assert 1 < len(set(candidates)) < 100
