"""Small exact attention: PyTorch GEMMs + Triton masked softmax and its backward.

This is deliberately a materialized attention research path, not FlashAttention.
O(B*H*T^2) storage, T<=1024; longer contexts require tiled attention.
"""
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Triton attention selected: install lithelm[triton-linux] or a compatible Windows Triton distribution") from exc


@triton.jit
def _softmax_fwd(Scores, Mask, Prob, N:tl.constexpr, CAUSAL:tl.constexpr, HAS_MASK:tl.constexpr, BLOCK:tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0,BLOCK)
    offset = row*N+col
    valid = col < N
    if CAUSAL:
        valid = valid & (col <= row % N)
    if HAS_MASK:
        valid = valid & tl.load(Mask+offset,col<N,other=0)
    values = tl.load(Scores+offset,col<N,other=-float("inf"))
    values = tl.where(valid,values,-float("inf"))
    maximum = tl.max(values,0)
    # Fully masked rows return zeros instead of NaN.
    maximum = tl.where(maximum == -float("inf"),0.,maximum)
    exps = tl.exp(values-maximum)
    denom = tl.sum(exps,0)
    probs = exps / tl.maximum(denom,1.e-30)
    tl.store(Prob+offset,probs,col<N)


@triton.jit
def _softmax_bwd(Prob, DP, DS, N:tl.constexpr, BLOCK:tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0,BLOCK)
    offset = row*N+col
    p = tl.load(Prob+offset,col<N,other=0.)
    dp = tl.load(DP+offset,col<N,other=0.)
    ds = p*(dp-tl.sum(p*dp,0))
    tl.store(DS+offset,ds,col<N)


class _Attention(torch.autograd.Function):
    @staticmethod
    @torch.autocast("cuda",enabled=False)
    def forward(ctx,q,k,v,mask,causal,scale):
        scores = (q.float() @ k.float().transpose(-2,-1) * scale).contiguous()
        probs = torch.empty_like(scores)
        n = k.shape[-2]
        expanded = torch.broadcast_to(mask,scores.shape).contiguous() if mask is not None else scores
        _softmax_fwd[(scores.numel()//n,)](scores,expanded,probs,n,causal,mask is not None,triton.next_power_of_2(n))
        ctx.save_for_backward(q,k,v,probs)
        ctx.scale = scale
        return (probs @ v.float()).to(q.dtype)

    @staticmethod
    @once_differentiable
    @torch.autocast("cuda",enabled=False)
    def backward(ctx,grad):
        q,k,v,p = ctx.saved_tensors
        grad = grad.float()
        dv = p.transpose(-2,-1) @ grad
        dp = (grad @ v.float().transpose(-2,-1)).contiguous()
        ds = torch.empty_like(dp)
        n = k.shape[-2]
        _softmax_bwd[(p.numel()//n,)](p,dp,ds,n,triton.next_power_of_2(n))
        dq = ds @ k.float() * ctx.scale
        dk = ds.transpose(-2,-1) @ q.float() * ctx.scale
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None


def triton_attention(q,k,v,causal=True,scale=None,mask=None):
    """Exact forward/backward; reject unsupported inputs instead of silent fallback."""
    if not q.is_cuda or any(t.device != q.device for t in (k,v)):
        raise ValueError("Triton attention requires Q/K/V on the same CUDA device")
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or
        q.shape[:2]+q.shape[3:] != k.shape[:2]+k.shape[3:] or
        (q.shape[-2] != k.shape[-2] and (q.shape[-2] != 1 or causal))):
        raise ValueError("Triton attention expects matching Q/K/V shapes or noncausal single-query decode (expand GQA first)")
    if q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Triton attention supports matching fp16/bf16/fp32")
    if q.shape[-1] not in (16,32,64,128) or not 1 <= k.shape[-2] <= 1024:
        raise ValueError("Triton attention supports head_dim 16/32/64/128 and sequence length <=1024")
    if mask is not None and (mask.dtype != torch.bool or mask.device != q.device):
        raise ValueError("Triton attention mask must be bool on the Q device")
    return _Attention.apply(q,k,v,mask,causal,scale if scale is not None else q.shape[-1]**-.5)
