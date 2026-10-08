import pytest
import torch
from lithelm.scheduler import wsd_scale, wsd_phase, build_scheduler


@pytest.mark.parametrize("style",["linear","cosine"])
def test_wsd_boundaries(style):
    c=dict(warmup_steps=2,stable_steps=3,decay_steps=2,min_lr=.1,peak_lr=1.,decay_style=style)
    assert [wsd_phase(i,c) for i in (0,1,2,4,5,6)]==["warmup","warmup","stable","stable","decay","decay"]
    assert wsd_scale(0,c)==.5 and wsd_scale(1,c)==1 and wsd_scale(3,c)==1
    assert wsd_scale(5,c)==pytest.approx(.55)
    assert wsd_scale(6,c)==pytest.approx(.1) and wsd_scale(7,c)==pytest.approx(.1)
    opt=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=1.)
    sch=build_scheduler(opt,c)
    used=[]
    for _ in range(7):
        used.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    assert used==pytest.approx([wsd_scale(i,c) for i in range(7)])
