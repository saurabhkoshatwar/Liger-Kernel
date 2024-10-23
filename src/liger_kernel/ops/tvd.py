import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous


MAX_FUSED_SIZE = 65536 // 4 

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
):
    pid = tl.program_id(0).to(tl.int64)  # Get the program ID for batching
    p_ptr += pid * p_stride
    q_ptr += pid * q_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)

        # TVD(P || Q) = 0.5 * |P - Q|
        tv_loss = 0.5 * tl.abs(p - q)
        
        tl.store(loss_ptr + offsets, tv_loss, mask=mask)


def tv_distance_forward_triton(p, q):
    BT, V = p.shape  
    
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    grid = (BT,)  
    
    output_tensor = torch.zeros_like(p, device=p.device, dtype=torch.float32)
    
    _tv_distance_kernel_forward[grid](
        p,
        p.stride(0),
        q,
        q.stride(0),
        output_tensor,
        output_tensor.stride(0),
        V,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor.sum(dim=-1) 


class LigerTVDLossFunction(torch.autograd.Function):
    """
    Class implementing the forward and backward pass for the Total Variation Distance Loss using Triton.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """A forward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            p (torch.Tensor): A tensor of shape (BT, V) containing the first distribution.
            q (torch.Tensor): A tensor of shape (BT, V) containing the second distribution.

        Returns:
            torch.Tensor: The computed Total Variation Distance Loss.
        """
        ctx.save_for_backward(p, q)
        return tv_distance_forward_triton(p, q)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        """A backward pass for the Total Variation Distance Loss.

        Args:
            ctx: Torch autograd context
            grad_output (torch.Tensor): The gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The gradients of the loss with respect to the inputs p and q.
        """
        p, q = ctx.saved_tensors
        # Derivative of |p - q| is 1 where p > q and -1 where p < q
        grad_p = torch.where(p > q, 0.5, -0.5)
        grad_q = -grad_p
        
        return grad_p * grad_output.unsqueeze(-1), grad_q * grad_output.unsqueeze(-1)