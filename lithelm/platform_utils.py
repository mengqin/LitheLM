"""Keep platform and optional dependency policy out of model/training code."""
from datetime import timedelta
import importlib.util
import os
import platform

import torch


def available(name):
    return importlib.util.find_spec(name) is not None


def default_backend(cuda=True):
    return "nccl" if cuda and platform.system() != "Windows" and torch.distributed.is_nccl_available() else "gloo"


def build_accelerator(cfg):
    r, d = cfg["runtime"], cfg["distributed"]
    if platform.system() == "Windows":
        os.environ.setdefault("USE_LIBUV", "0")
    if r["deterministic"] and r["device"] != "cpu":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
    if r["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is unavailable")
    cuda = r["device"] != "cpu" and torch.cuda.is_available()
    if not cuda and r["precision"] == "fp16":
        raise ValueError("runtime.precision=fp16 requires CUDA; use fp32 or bf16 on CPU")
    backend = default_backend(cuda) if d["backend"] == "auto" else d["backend"]
    if backend == "nccl" and (not cuda or not torch.distributed.is_nccl_available()):
        raise RuntimeError("NCCL requires a supported CUDA/Linux environment; choose gloo")
    # Explicit backend on a non-distributed launch incorrectly implies a process
    # group to Accelerate. Let it detect single-process execution itself.
    launched = any(int(os.environ.get(key,"1")) > 1 for key in ("WORLD_SIZE","OMPI_COMM_WORLD_SIZE","PMI_SIZE"))
    handlers = [DistributedDataParallelKwargs(find_unused_parameters=d["find_unused_parameters"])]
    if launched:
        handlers.append(InitProcessGroupKwargs(backend=backend, timeout=timedelta(minutes=d["timeout_minutes"])))
    a = Accelerator(cpu=not cuda, mixed_precision={"fp32":"no", "fp16":"fp16", "bf16":"bf16"}[r["precision"]],
        step_scheduler_with_optimizer=False,
        kwargs_handlers=handlers)
    if d["strategy"] == "single" and a.num_processes != 1:
        raise ValueError("distributed.strategy=single conflicts with multi-process launch")
    if d["strategy"] == "ddp" and a.num_processes == 1:
        raise ValueError("ddp requires accelerate launch --multi_gpu --num_processes N")
    torch.set_float32_matmul_precision(r["matmul_precision"])
    torch.use_deterministic_algorithms(r["deterministic"])
    torch.autograd.set_detect_anomaly(r["detect_anomaly"])
    set_seed(cfg["experiment"]["seed"])
    if cuda and "LITHELM_GPU_MEMORY_FRACTION" in os.environ:
        fraction = float(os.environ["LITHELM_GPU_MEMORY_FRACTION"])
        if not 0 < fraction <= 1:
            raise ValueError("LITHELM_GPU_MEMORY_FRACTION must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(fraction, torch.cuda.current_device())
    return a


def environment_report():
    cuda = torch.cuda.is_available()
    return dict(os=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, cuda_available=cuda,
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                bf16=cuda and torch.cuda.is_bf16_supported(), triton=available("triton"), fla=available("fla"),
                distributed_backend=default_backend(cuda), custom_kernel_available=cuda and available("triton"))
