import numpy as np
import torch


def next235even(n):
    # https://github.com/flatironinstitute/finufft/blob/master/src/utils.cpp
    if n <= 2:
        return 2
    if n % 2 == 1:
        n += 1
    nplus = n - 2
    numdiv = 2
    while numdiv > 1:
        nplus += 2
        numdiv = nplus
        while numdiv % 2 == 0:
            numdiv //= 2
        while numdiv % 3 == 0:
            numdiv //= 3
        while numdiv % 5 == 0:
            numdiv //= 5
    return nplus


def outer(vectors):
    out = vectors[0]
    for v in vectors[1:]:
        out = out.unsqueeze(-1) * v
    return out


def phi(z, beta):
    mask = torch.abs(z) <= 1
    val = torch.exp(beta * (torch.sqrt(1.0 - z**2) - 1.0))
    return torch.where(mask, val, torch.zeros_like(z))


def psi(x, alpha, beta):
    mask = torch.abs(x) <= alpha
    val = torch.exp(beta * (torch.sqrt(1.0 - (x / alpha) ** 2) - 1.0))
    return torch.where(mask, val, torch.zeros_like(x))


def psi_tilde(x, alpha, beta):
    # This is a neat trick to wrap x into the range (-pi, pi]
    x_wrapped = torch.atan2(torch.sin(x), torch.cos(x))
    return psi(x_wrapped, alpha, beta)


def psi_hat(p, qj, wj, k, w, alpha, phi_qj):
    cos_vals = torch.cos(alpha * qj * k[..., None])
    result = torch.sum(wj[: int(p)] * phi_qj[: int(p)] * cos_vals[..., : int(p)], dim=-1)
    return w * result


def dpsi_tilde(x, alpha, beta):
    x_wrapped = torch.atan2(torch.sin(x), torch.cos(x))
    return (
        -(beta / alpha**2)
        * x_wrapped
        / torch.sqrt(
            torch.clamp(
                1.0 - (x_wrapped / alpha) ** 2, min=torch.finfo(x_wrapped.dtype).eps
            )  # Avoid division by zero
        )
        * psi(x_wrapped, alpha, beta)
    )


def setup(d, N, eps, dtype, device, x=None, s=None):
    w = np.ceil(np.log10(1 / eps)) + 1
    beta = 2.30 * w
    sigma = 2.0
    p = np.ceil(1.5 * w + 2)
    qj, wj = np.polynomial.legendre.leggauss(int(2 * p))
    qj = torch.from_numpy(qj).to(device=device, dtype=dtype)
    wj = torch.from_numpy(wj).to(device=device, dtype=dtype)
    phi_qj = phi(qj, beta)
    if x is None and s is None:
        n = torch.zeros(d, device=device, dtype=torch.int64)
        for dim in range(d):
            n[dim] = next235even(int(max((sigma * N[dim], 2 * w))))
        alpha = torch.pi * w / n
        h = 2 * torch.pi / n
        return alpha, beta, h, n, p, phi_qj, qj, w, wj
    else:
        X = torch.amax(torch.abs(x), dim=-1)
        S = torch.amax(torch.abs(s), dim=-1)
        n = torch.ceil(2 * sigma / torch.pi * X * S + w).to(dtype=torch.int64)
        gamma = n / (2 * sigma * S)
        scale = gamma.unsqueeze(-1)
        x_prime = x / scale
        alpha = torch.pi * w / n
        h = 2 * torch.pi / n
        s_prime = scale * s
        s_prime_prime = (h * gamma).unsqueeze(-1) * s
        return alpha, beta, h, n, p, phi_qj, qj, w, wj, x_prime, s_prime, s_prime_prime


def compute_strides(n):
    strides = torch.cumprod(n.flip(0), dim=0).flip(0)
    return torch.cat((strides[1:], torch.ones(1, device=n.device, dtype=n.dtype)))


def compute_local_kernel(x, alpha, beta, h, n, ddim=None):
    lead_shape = x.shape[:-2]
    d = x.shape[0]
    k = x.shape[-1]

    strides = compute_strides(n)

    idxs = list()
    psi_tildes = list()
    for dim in range(d):
        n_dim = int(n[dim].item())
        kernel_half_width = int(torch.ceil(alpha[dim] / h[dim]).item())
        offsets = torch.arange(
            -kernel_half_width, kernel_half_width + 1, device=x.device, dtype=torch.int64
        )
        center = torch.floor(x[dim, :] / h[dim]).to(torch.int64)
        idx_dim = (center.unsqueeze(-1) + offsets.view(*((1,) * center.ndim), -1)) % n_dim
        dist = idx_dim.to(dtype=x.dtype) * h[dim] - x[dim, :].unsqueeze(-1)
        idxs.append(idx_dim)
        if ddim is not None and dim == ddim:
            # Compute the derivative of the kernel for the specified dimension
            psi_tildes.append(dpsi_tilde(dist, alpha[dim], beta))
        else:
            psi_tildes.append(psi_tilde(dist, alpha[dim], beta))

    linear_idx = torch.zeros((*lead_shape, k, 1), device=x.device, dtype=torch.int64)
    psi_kernel = torch.ones((*lead_shape, k, 1), device=x.device, dtype=x.dtype)
    for dim in range(d):
        linear_idx = linear_idx.unsqueeze(-1) + idxs[dim].unsqueeze(-2) * strides[dim]
        linear_idx = linear_idx.reshape(*lead_shape, k, -1)
        psi_kernel = (psi_kernel.unsqueeze(-1) * psi_tildes[dim].unsqueeze(-2)).reshape(
            *lead_shape, k, -1
        )

    return linear_idx, psi_kernel
