import torch
from lithelm.metrics import meter_metrics, token_accuracy, exact_match
from lithelm.model import build_model
from test_data import tiny_config


def test_weighted_nll_and_masked_accuracy():
    m=meter_metrics(dict(a=[2.,1,1,"pt"],b=[40.,10,2,"sft"]))
    assert m["train/nll"]==42/11
    assert m["data/sft_ratio"]==2/3
    labels=torch.tensor([[-100,-100,2,1]])
    logits=torch.zeros(1,4,3)
    logits[0,1,2]=10
    logits[0,2,1]=10
    assert tuple(x.item() for x in token_accuracy(logits,labels))==(2,2)
    assert tuple(x.item() for x in exact_match(logits,labels))==(1,1)


def test_accumulation_token_weighting(tmp_path):
    cfg=tiny_config(tmp_path)
    torch.manual_seed(7)
    model=build_model(cfg["model"])
    ids=torch.randint(0,280,(2,12))
    labels=ids.clone()
    labels[0,:8]=-100
    model(ids,labels=labels).loss.backward()
    gradients=[p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    denominator=labels[:,1:].ne(-100).sum()
    for i in range(2):
        (model(ids[i:i+1],labels=labels[i:i+1]).loss_sum/denominator).backward()
    for actual,expected in zip(model.parameters(),gradients):
        torch.testing.assert_close(actual.grad,expected,atol=2e-7,rtol=1e-5)
