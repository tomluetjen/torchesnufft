import torch

from torchesnufft.utils import helpers


class SpreadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, alpha, beta, d, h, n):
        b = torch.zeros(
            (c.shape[0], c.shape[1]) + tuple(n.to(torch.int).tolist()),
            device=c.device,
            dtype=c.dtype,
        )
        b_flat = b.reshape(c.shape[0], c.shape[1], -1)
        linear_idx, psi_kernel = helpers.compute_local_kernel(x, alpha, beta, h, n)
        src = c.unsqueeze(-1) * psi_kernel.unsqueeze(0).unsqueeze(0)
        idx = linear_idx.unsqueeze(0).unsqueeze(0).expand(c.shape[0], c.shape[1], -1, -1)
        b_flat.scatter_add_(
            2, idx.reshape(c.shape[0], c.shape[1], -1), src.reshape(c.shape[0], c.shape[1], -1)
        )
        ctx.save_for_backward(x, c, alpha, h, n, linear_idx, psi_kernel)
        ctx.beta = beta
        return b

    @staticmethod
    def backward(ctx, grad_b):
        x, c, alpha, h, n, linear_idx, psi_kernel = ctx.saved_tensors
        beta = ctx.beta
        grad_c = torch.zeros_like(c)
        grad_x = torch.zeros_like(x)
        d = int(n.numel())
        grad_b_flat = grad_b.reshape(grad_b.shape[0], grad_b.shape[1], -1)
        grad_b_local = grad_b_flat[..., linear_idx]
        grad_c = (grad_b_local * psi_kernel.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        for dim in range(d):
            _, dpsi_kernel = helpers.compute_local_kernel(x, alpha, beta, h, n, ddim=dim)
            contracted = (grad_b_local.conj() * dpsi_kernel.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            grad_x[dim] = (-(c * contracted).sum(dim=(0, 1))).real

        return grad_x, grad_c, None, None, None, None, None


def spread(x, c, alpha, beta, d, h, n):
    b = SpreadFunction.apply(x, c, alpha, beta, d, h, n)
    return b


class InterpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b, alpha, beta, d, h, n):
        c = torch.zeros((b.shape[0], b.shape[1], x.shape[-1]), device=b.device, dtype=b.dtype)
        b_flat = b.reshape(b.shape[0], b.shape[1], -1)
        linear_idx, psi_kernel = helpers.compute_local_kernel(x, alpha, beta, h, n)
        b_local = b_flat[..., linear_idx]
        c = (b_local * psi_kernel.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        ctx.save_for_backward(x, b, alpha, h, n, linear_idx, psi_kernel)
        ctx.beta = beta
        return c

    @staticmethod
    def backward(ctx, grad_c):
        x, b, alpha, h, n, linear_idx, psi_kernel = ctx.saved_tensors
        beta = ctx.beta
        grad_b = torch.zeros_like(b)
        grad_x = torch.zeros_like(x)
        d = int(n.numel())
        b_flat = b.reshape(b.shape[0], b.shape[1], -1)
        grad_b_flat = grad_b.reshape(grad_b.shape[0], grad_b.shape[1], -1)
        src = grad_c.unsqueeze(-1) * psi_kernel.unsqueeze(0).unsqueeze(0)
        idx = linear_idx.unsqueeze(0).unsqueeze(0).expand(grad_c.shape[0], grad_c.shape[1], -1, -1)
        grad_b_flat.scatter_add_(
            2,
            idx.reshape(grad_c.shape[0], grad_c.shape[1], -1),
            src.reshape(grad_c.shape[0], grad_c.shape[1], -1),
        )

        b_local = b_flat[..., linear_idx]
        for dim in range(d):
            _, dpsi_kernel = helpers.compute_local_kernel(x, alpha, beta, h, n, ddim=dim)
            contracted = (b_local * dpsi_kernel.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            grad_x[dim] = (-(grad_c.conj() * contracted).sum(dim=(0, 1))).real

        return grad_x, grad_b, None, None, None, None, None


def interp(x, b, alpha, beta, d, h, n):
    c = InterpFunction.apply(x, b, alpha, beta, d, h, n)
    return c
