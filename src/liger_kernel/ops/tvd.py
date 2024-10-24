import torch
import triton
import triton.language as tl

from typing import Literal
from liger_kernel.ops.utils import ensure_contiguous

MAX_FUSED_SIZE = 65536 // 4

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

_REDUCTION_MODE_NONE = tl.constexpr(0)
_REDUCTION_MODE_SUM = tl.constexpr(1)
_REDUCTION_MODE_MEAN = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}

def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps

@triton.jit
def _tv_distance_kernel_forward(
    p_ptr, 
    p_stride,  
    q_ptr, 
    q_stride, 
    loss_ptr,  
    loss_stride, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    p_ptr += pid * p_stride
    q_ptr += pid * q_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)

        # TVD(P || Q) = 0.5 * |P - Q|
        tv_loss = 0.5 * tl.abs(p - q)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, tv_loss, mask=mask)
        else:
            loss_sum += tl.sum(tv_loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)

@triton.jit
def _tvd_kernel_backward(p_ptr, p_stride, q_ptr, q_stride, new_grads_ptr, new_grads_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)

    p_ptr += pid * p_stride
    q_ptr += pid * q_stride
    new_grads_ptr += pid * new_grads_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)  # Load P
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)  # Load Q

        grad_res = tl.where(p > q, 0.5, -0.5)  # 0.5 if P > Q, -0.5 if P < Q

        tl.store(new_grads_ptr + offsets, grad_res, mask=mask)

def tv_distance_forward_triton(p, q, reduction):  
    BT, V = p.shape  
    
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (BT,)  
    
    reduction = _str_to_reduction_mode[reduction]

    out_size = (BT, V) if reduction == _REDUCTION_MODE_NONE.value else (BT,)
    output_tensor = torch.zeros(out_size, device=p.device, dtype=torch.float32)
    
    _tv_distance_kernel_forward[grid](
        p,
        p.stride(0),
        q,
        q.stride(0),
        output_tensor,
        output_tensor.stride(0),
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        reduction=reduction,
    )
    
    if reduction == _REDUCTION_MODE_BATCHMEAN.value:
        return output_tensor.sum() / BT
    elif reduction == _REDUCTION_MODE_SUM.value:
        return output_tensor.sum(dim=0)
    elif reduction == _REDUCTION_MODE_MEAN.value:
        return output_tensor.sum() / (BT * V)
    else:
        return output_tensor

def tvd_backward_triton(p, q, grad_output, new_grads):
    BT, V = p.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))  # Set a maximum block size
    grid = (BT,)

    _tvd_kernel_backward[grid](
        p,
        p.stride(0),
        q,
        q.stride(0),
        new_grads,
        new_grads.stride(0),
        V,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul then.
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return new_grads

    return new_grads * grad_output

class LigerTVDLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the Total Variation Distance Loss using Triton.
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx, p: torch.Tensor, q: torch.Tensor, reduction: REDUCTION_LITERAL = "batchmean"
    ) -> torch.Tensor:
        """A forward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            p (torch.Tensor): A tensor of shape (BT, V) containing the first distribution.
            q (torch.Tensor): A tensor of shape (BT, V) containing the second distribution.
            reduction (REDUCTION_LITERAL, optional): The reduction method to be applied. Defaults to "batchmean".

        Returns:
            torch.Tensor: The computed Total Variation Distance Loss.
        """
        ctx.save_for_backward(p, q)
        ctx.reduction = reduction
        return tv_distance_forward_triton(p, q, reduction)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """A backward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, None]: The gradient of the loss with respect to the inputs.
        """
        p, q = ctx.saved_tensors
        BT, V = p.shape

        new_grads_p = torch.empty_like(p)

        new_grads_p = tvd_backward_triton(p, q, grad_output, new_grads_p)

        if ctx.reduction == "batchmean":
            new_grads_p /= BT
        elif ctx.reduction == "mean":
            new_grads_p /= (BT * V)

        return new_grads_p, None
