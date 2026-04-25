import torch

from torchesnufft.utils import helpers, spreadinterp


def nufft1(x, c, N, eps=1e-6):
    # Setup
    d = len(N)
    device = c.device
    alpha, beta, h, n, p, phi_qj, qj, w, wj = helpers.setup(d, N, eps, x.dtype, device)
    # Spreading
    b = spreadinterp.spread(x, c, alpha, beta, d, h, n)
    # FFT
    b_hat = torch.fft.fftshift(
        torch.fft.ifftn(b, dim=tuple(range(-d, 0)), norm="forward"), dim=tuple(range(-d, 0))
    )
    # Correction
    p_list = list()
    for dim in range(d):
        mode_idx = torch.arange(
            -int((N[dim] - (N[dim] % 2)) / 2),
            int((N[dim] - (N[dim] % 2)) / 2 + N[dim] % 2),
            device=device,
        )
        p_list.append(1 / helpers.psi_hat(p, qj, wj, mode_idx, w, alpha[dim], phi_qj))
    p_block = helpers.outer(p_list)
    return (
        p_block[None, None, ...]
        * b_hat[
            (slice(None), slice(None))
            + tuple(
                slice(
                    int(n[i].item()) // 2 - int(N[i]) // 2,
                    int(n[i].item()) // 2 - int(N[i]) // 2 + int(N[i]),
                )
                for i in range(len(N))
            )
        ]
    )


def nufft2(x, f, eps=1e-6):
    # Setup
    d = x.shape[-2]
    N = f.shape[-d:]
    d = len(N)
    device = f.device
    cplx_dtype = f.dtype
    real_dtype = x.dtype
    alpha, beta, h, n, p, phi_qj, qj, w, wj = helpers.setup(d, N, eps, real_dtype, device)
    # Correction
    p_list = list()
    for dim in range(d):
        mode_idx = torch.arange(
            -int((N[dim] - (N[dim] % 2)) / 2),
            int((N[dim] - (N[dim] % 2)) / 2 + N[dim] % 2),
            device=device,
        )
        p_list.append(1 / helpers.psi_hat(p, qj, wj, mode_idx, w, alpha[dim], phi_qj))
    p_block = helpers.outer(p_list)
    b_hat = torch.zeros(
        f.shape[:-d] + tuple(n.to(torch.int).tolist()), device=device, dtype=cplx_dtype
    )
    b_hat[
        (Ellipsis,)
        + tuple(
            slice(
                int(n[i].item()) // 2 - int(N[i]) // 2,
                int(n[i].item()) // 2 - int(N[i]) // 2 + int(N[i]),
            )
            for i in range(len(N))
        )
    ] = p_block * f
    # FFT
    b = torch.fft.ifftn(
        torch.fft.fftshift(b_hat, dim=tuple(range(-d, 0))), dim=tuple(range(-d, 0)), norm="forward"
    )
    # Interpolation
    c = spreadinterp.interp(x, b, alpha, beta, d, h, n)

    return c


def nufft3(x, c, s, eps=1e-6):
    d = x.shape[-2]
    real_dtype = x.dtype
    device = c.device
    alpha, beta, h, n, p, phi_qj, qj, w, wj, x_prime, s_prime, s_prime_prime = helpers.setup(
        d, None, eps, real_dtype, device, x, s
    )

    b_hat = spreadinterp.spread(x_prime, c, alpha, beta, d, h, n)
    b = nufft2(s_prime_prime, torch.fft.fftshift(b_hat, dim=tuple(range(-d, 0))), eps=eps)

    p_block = torch.ones_like(s_prime[0, :])
    for dim in range(d):
        p_block *= helpers.psi_hat(p, qj, wj, s_prime[dim, :], w, alpha[dim], phi_qj)
    f = b / p_block
    return f


def nufft_inv(x, c, N, eps=1e-6):
    # Setup
    d = len(N)
    device = c.device
    alpha, beta, h, n, p, phi_qj, qj, w, wj = helpers.setup(d, N, eps, x.dtype, device)
    # Density compensation
    density = get_density(x, c, N, eps)
    density = density.reshape(c.shape)
    # Spreading
    b = spreadinterp.spread(x, c * density, alpha, beta, d, h, n)
    # FFT
    b_hat = torch.fft.fftshift(
        torch.fft.ifftn(b, dim=tuple(range(-d, 0)), norm="forward"), dim=tuple(range(-d, 0))
    )
    # Correction
    p_list = list()
    for dim in range(d):
        mode_idx = torch.arange(
            -int((N[dim] - (N[dim] % 2)) / 2),
            int((N[dim] - (N[dim] % 2)) / 2 + N[dim] % 2),
            device=device,
        )
        p_list.append(1 / helpers.psi_hat(p, qj, wj, mode_idx, w, alpha[dim], phi_qj))
    p_block = helpers.outer(p_list)
    return (
        p_block[None, None, ...]
        * b_hat[
            (slice(None), slice(None))
            + tuple(
                slice(
                    int(n[i].item()) // 2 - int(N[i]) // 2,
                    int(n[i].item()) // 2 - int(N[i]) // 2 + int(N[i]),
                )
                for i in range(len(N))
            )
        ]
    )


def get_density(x, c, N, eps=1e-6, n_iter=10):
    # Pipe JG, Menon P. Sampling density compensation in MRI: rationale and an iterative numerical solution. Magn Reson Med. 1999 Jan;41(1):179-86. doi: 10.1002/(sici)1522-2594(199901)41:1<179::aid-mrm25>3.0.co;2-v. PMID: 10025627.
    d = len(N)
    device = c.device
    alpha, beta, h, n, _, _, _, _, _ = helpers.setup(d, N, eps, x.dtype, device)
    N_total = torch.prod(torch.tensor(N)).item()
    density = torch.abs(torch.ones_like(c))
    for _ in range(n_iter):
        b = spreadinterp.spread(x, density, alpha, beta, d, h, n)
        b = spreadinterp.interp(x, b, alpha, beta, d, h, n)
        density = density / torch.clamp(b, min=torch.finfo(x.dtype).eps)
    # Normalize so mass of density equals total number of grid points, resulting in a constant density of 1 for uniform sampling
    return N_total * density / density.sum(dim=-1, keepdim=True)
