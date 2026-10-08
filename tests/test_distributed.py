"""Real two-process Accelerate integration (CPU by default, CUDA via env)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch
from omegaconf import OmegaConf
from test_data import tiny_config


def run_distributed(*args):
    env = {**os.environ, "OMP_NUM_THREADS":"1", "PYTHONIOENCODING":"utf-8"}
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2"]
    with tempfile.TemporaryDirectory() as rendezvous:
        if sys.platform == "win32":
            # torchrun's TCP rendezvous ignores USE_LIBUV in some supported
            # PyTorch versions. Keep its agent on FileStore and let the workers
            # create the legacy TCPStore through env://.
            env.pop("USE_LIBUV", None)
            env["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
            command += ["--rdzv_backend=c10d", "--rdzv_conf=store_type=file",
                        f"--rdzv_endpoint={Path(rendezvous)/'store'}", "--local_addr=127.0.0.1"]
        else:
            command += ["--standalone"]
        result = subprocess.run(command+list(map(str, args)), capture_output=True,
                                encoding="utf-8", errors="replace", env=env, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    return result


@pytest.mark.integration
@pytest.mark.parametrize("target_mode", ["shifted", "next_token"])
def test_two_rank_resume_and_eval(tmp_path,target_mode):
    cfg=tiny_config(tmp_path)
    for stage in cfg["stages"]:
        stage["targets"]["mode"] = target_mode
    cfg["runtime"]["device"]=os.environ.get("LITHELM_TEST_DEVICE","cpu")
    if cfg["runtime"]["device"]=="cuda":
        if torch.cuda.device_count()<2:
            pytest.skip("two CUDA devices required")
        cfg["runtime"]["precision"]="bf16"
    cfg["datasets"]["pt"]["kwargs"]["max_samples"]=4
    cfg["datasets"]["sft"]["kwargs"]["max_samples"]=4
    for stage in cfg["stages"]:
        stage["eval"]["max_batches"]=None
        stage["intervals"]["eval"]=1
    config=tmp_path/"config.yaml"
    OmegaConf.save(OmegaConf.create(cfg),config)
    root=Path(cfg["experiment"]["output_dir"])
    resume = "latest" if target_mode == "next_token" else str(root/"pt/checkpoint-2")
    for args in (["--stop-after","2"],["--resume",resume]):
        run_distributed("-m", "lithelm", "train", config, *args)
        if args[0] == "--stop-after":
            path = root/"pt/checkpoint-2"
            assert all((path/f"random_states_{rank}.pkl").is_file() for rank in range(2))
            path.rename(path.with_name(path.name+".writing"))
    final=root/"sft/final"
    state=json.loads((final/"trainer_state.json").read_text())
    assert state["world_size"]==2 and state["scheduler_position"]==3 and state["global_step"]==7
    rows=[json.loads(x) for x in (root/"metrics.jsonl").read_text().splitlines()]
    distributed=[r for r in rows if r["global_step"]==7 and "eval/nll" in r][-1]
    out=subprocess.run([sys.executable,"-m","lithelm","eval",str(config),"--checkpoint",str(final)],encoding="utf-8",capture_output=True,env={**os.environ,"OMP_NUM_THREADS":"1"},timeout=60)
    assert out.returncode==0,out.stdout+out.stderr
    single=json.loads(out.stdout)
    for key,value in single.items():
        if key.endswith("/nll"):
            assert distributed[key]==pytest.approx(value,rel=2e-4,abs=2e-4)


@pytest.mark.integration
def test_custom_stats_two_ranks(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg['runtime']['device'] = os.environ.get('LITHELM_TEST_DEVICE','cpu')
    cfg['model']['layers'][0]['mixer'].update(type='custom',custom=dict(name='probe',kwargs={}))
    cfg['stages'] = cfg['stages'][:1]
    cfg['statistics'].update(every=1)
    config = tmp_path/'stats.yaml'
    OmegaConf.save(OmegaConf.create(cfg),config)
    script = tmp_path/'stats_rank.py'
    script.write_text('''import os,sys,torch
from torch import nn
from lithelm.config import load_config
from lithelm.model.builders import CUSTOM_LAYERS
from lithelm.statistics import StatSpec
from lithelm.train import train
class Probe(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.projection=nn.Linear(hidden_size,hidden_size)
        self.calls=0
    def stat_specs(self):
        return {'average':StatSpec('mean','mean'), 'total':StatSpec('sum','sum'),
                'minimum':StatSpec('min','min'), 'maximum':StatSpec('max','max'),
                'last_local':StatSpec('last','local')}
    def forward(self,x,attention_mask=None,collect_stats=False):
        y=self.projection(x)
        if not collect_stats: return y
        value=1+2*(self.calls%2)+10*int(os.environ['RANK'])
        self.calls+=1
        return y,{key:value for key in self.stat_specs()}
CUSTOM_LAYERS['probe']=Probe
train(load_config(sys.argv[1]))
''')
    run_distributed(script, config)
    rows=[json.loads(line) for line in (Path(cfg['experiment']['output_dir'])/'metrics.jsonl').read_text().splitlines()]
    stats=[row for row in rows if row.get('event')=='model_stats']
    assert len(stats)==4
    for row in stats:
        for key,value in dict(average=7.,total=28.,minimum=1.,maximum=13.,last_local=3.).items():
            assert row['layer/0/mixer/'+key]==value


@pytest.mark.integration
@pytest.mark.parametrize("invalid", ["nan", "inf"])
def test_nonfinite_decision_across_ranks(tmp_path, invalid):
    cfg = tiny_config(tmp_path)
    cfg["runtime"]["device"] = os.environ.get("LITHELM_TEST_DEVICE", "cpu")
    if cfg["runtime"]["device"] == "cuda":
        if torch.cuda.device_count() < 2:
            pytest.skip("two CUDA devices required")
        cfg["runtime"]["precision"] = "bf16"
    config = tmp_path/"nonfinite.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config)
    script = tmp_path/"nonfinite_rank.py"
    script.write_text('''import json, sys, torch
from pathlib import Path
from types import SimpleNamespace
from lithelm.config import load_config
from lithelm.model import build_model
from lithelm.platform_utils import build_accelerator
from lithelm.tokenizer import prepare_tokenizer
from lithelm.trainer import Trainer
cfg = load_config(sys.argv[1])
a = build_accelerator(cfg)
try:
    with a.main_process_first():
        tok = prepare_tokenizer(cfg)
    trainer = Trainer(cfg, build_model(cfg['model']), tok, a, SimpleNamespace(log=lambda row:None))
    trainer.setup_stage(0, cfg['stages'][0])
    original = [p.detach().clone() for p in trainer.model.parameters()]
    clip = a.clip_grad_norm_
    def rank_local_norm(*args, **kwargs):
        norm = clip(*args, **kwargs)
        return torch.full_like(norm, float(sys.argv[2])) if a.process_index == 1 else norm
    a.clip_grad_norm_ = rank_local_norm
    def forbidden_step(*args, **kwargs):
        raise AssertionError('optimizer must not step on either rank')
    trainer.optimizer.optimizer.step = forbidden_step
    try:
        trainer.train_update()
    except FloatingPointError as exc:
        assert 'stage=pt stage_step=0 global_step=0' in str(exc)
        for before, after in zip(original, trainer.model.parameters()):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        Path(sys.argv[1]).with_name(f'rank-{a.process_index}.json').write_text(json.dumps(str(exc)))
    else:
        raise AssertionError('non-finite gradient must fail on both ranks')
    a.wait_for_everyone()
finally:
    a.end_training()
''')
    run_distributed(script, config, invalid)
    for rank in range(2):
        message = json.loads((tmp_path/f"rank-{rank}.json").read_text())
        assert f"precision={cfg['runtime']['precision']} grad_norm=" in message
