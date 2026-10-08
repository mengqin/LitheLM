"""Stage sequencing; defaults continue weights and reset optimizer/scheduler."""
from .checkpoint import read_state
from .trainer import Trainer


def run_stages(cfg,model,tokenizer,accelerator,logger,resume=None,stop_after=None):
    trainer = Trainer(cfg,model,tokenizer,accelerator,logger)
    resume = resume or cfg["experiment"]["resume_from"]
    start = read_state(resume,accelerator)["stage_index"] if resume else 0
    for index in range(start,len(cfg["stages"])):
        if not trainer.run_stage(index,cfg["stages"][index],resume if index == start else None,stop_after):
            break
    return trainer.state
