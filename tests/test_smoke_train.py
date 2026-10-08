import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from omegaconf import OmegaConf

from test_data import tiny_config


def run(*args):
    result=subprocess.run([sys.executable,"-m","lithelm",*map(str,args)],capture_output=True,encoding="utf-8",
                          env={**os.environ,"OMP_NUM_THREADS":"1"},timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_pt_checkpoint_resume_sft_eval_infer(tmp_path):
    cfg=tiny_config(tmp_path)
    cfg["model"]["embeddings"]["dropout"] = .1
    cfg["model"]["layers"][0]["mixer"]["attention"]["attention_dropout"] = .1
    path=tmp_path/"config.yaml"
    OmegaConf.save(OmegaConf.create(cfg),path)
    run("train",path)
    output=Path(cfg["experiment"]["output_dir"])
    final=output/"sft/final"
    state=json.loads((final/"trainer_state.json").read_text())
    assert state["global_step"]==7 and state["stage_step"]==3
    original=torch.load(final/"pytorch_model.bin",weights_only=True)
    resumed=tmp_path/"resumed"
    resumed.mkdir()
    (resumed/"run.json").write_text((output/"run.json").read_text())
    override=f"experiment.output_dir={resumed}"
    run("train",path,"--stop-after",2,override)
    checkpoint=resumed/"pt/checkpoint-2"
    state=json.loads((checkpoint/"trainer_state.json").read_text())
    assert state["scheduler_position"]==2 and state["stage_step"]==2
    run("train",path,"--resume",checkpoint,override)
    restored=torch.load(resumed/"sft/final/pytorch_model.bin",weights_only=True)
    for key,value in original.items():
        torch.testing.assert_close(value,restored[key],rtol=0,atol=0)
    rows=[json.loads(line) for line in (resumed/"metrics.jsonl").read_text().splitlines()]
    starts=[r for r in rows if r.get("event")=="stage_start" and r["stage"]=="sft"]
    assert starts[0]["stage_step"]==0 and starts[0]["global_step"]==4
    assert starts[0]["optim/reset"] and starts[0]["optim/scheduler_reset"]
    assert any("data/sft_ratio" in r for r in rows)
    assert any("eval/sft/nll" in r for r in rows)
    result=json.loads(run("eval",path,"--checkpoint",final))
    assert result["eval/nll"]>0 and result["eval/pt/tokens"]>0
    text=run("infer",path,"--checkpoint",final,"--prompt","A happy cat","--max-new-tokens",3)
    assert "A happy cat" in text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows legacy stdout encodings")
@pytest.mark.parametrize("encoding", ["cp1252", "gbk"])
def test_infer_unicode_stdout(tmp_path, monkeypatch, encoding):
    cfg = tiny_config(tmp_path)
    path = tmp_path/"config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    run("train", path, "--stop-after", 1)
    checkpoint = Path(cfg["experiment"]["output_dir"])/"pt/checkpoint-1"
    monkeypatch.setenv("PYTHONIOENCODING", encoding)
    prompt = "\u732b \u2603 \U0001f408"
    assert run("infer", path, "--checkpoint", checkpoint, "--prompt", prompt,
               "--max-new-tokens", 0).strip() == prompt
    record = json.loads((checkpoint.parent.parent/"inference/samples.jsonl").read_text(encoding="utf-8"))
    assert record["prompt"] == record["text"] == prompt


def test_incompatible_resume_rejected(tmp_path):
    cfg=tiny_config(tmp_path)
    path=tmp_path/"config.yaml"
    OmegaConf.save(OmegaConf.create(cfg),path)
    run("train",path,"--stop-after",2)
    from lithelm.checkpoint import read_state
    ck=Path(cfg["experiment"]["output_dir"])/"pt/checkpoint-2"
    assert read_state(ck)["stage_step"]==2
    result=subprocess.run([sys.executable,"-m","lithelm","train",str(path),"--resume",str(ck),"experiment.seed=1"],capture_output=True,text=True)
    assert result.returncode!=0 and "differs" in result.stderr


@pytest.mark.parametrize("keep_scheduler",[False,True])
def test_keep_optimizer_scheduler_policy(tmp_path,keep_scheduler):
    cfg=tiny_config(tmp_path)
    cfg["stages"][1]["init_from"].update(checkpoint="final",load_optimizer=True)
    cfg["stages"][1]["lr_schedule"].update(warmup_steps=1,stable_steps=1,decay_steps=1)
    if keep_scheduler:
        cfg["stages"][0]["lr_schedule"].update(warmup_steps=2,stable_steps=1,decay_steps=1)
        cfg["stages"][1]["max_steps"]=4
        cfg["stages"][1]["lr_schedule"]=cfg["stages"][0]["lr_schedule"].copy()
        cfg["stages"][1]["init_from"].update(checkpoint="checkpoint-2",load_scheduler=True)
    path=tmp_path/"config.yaml"
    OmegaConf.save(OmegaConf.create(cfg),path)
    run("train",path)
    rows=[json.loads(x) for x in (Path(cfg["experiment"]["output_dir"])/"metrics.jsonl").read_text().splitlines()]
    first=next(r for r in rows if r["stage"]=="sft" and r.get("event")=="train")
    assert first["optim/lr"]==cfg["stages"][1]["optimizer"]["lr"]
    assert first["optim/wsd_phase"]==("stable" if keep_scheduler else "warmup")
