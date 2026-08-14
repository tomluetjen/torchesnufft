import torch

from torchesnufft.utils import helpers, spreadinterp


def nufft1(x, c, N, eps=1e-6):
    """Compute the multi-dimensional type 1 (non-uniform to uniform) NUFFT.

    Evaluates f_k = sum_j c_j * exp(1j * dot(k, x_j)).

    Parameters
    ----------
    x : torch.Tensor, shape (d, M)
        Non-uniform source points.
    c : torch.Tensor, shape (B, C, M)
        Batched multi-channel complex source strengths at the non-uniform source points.
    N : tuple[int, ...], shape (d,)
        Number of Fourier modes along each dimension.
    eps : float, optional
        Precision. Defaults to ``1e-6``.

    Returns
    -------
    torch.Tensor, shape (B, C, *N)
        Batched multi-channel complex Fourier transform values at uniform points.
    """
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
    """Compute the multi-dimensional type 2 (uniform to non-uniform) NUFFT.

    Evaluates c_j = sum_k f_k * exp(1j * dot(k, x_j)).

    Parameters
    ----------
    x : torch.Tensor, shape (d, M)
        Non-uniform target points.
    f : torch.Tensor, shape (B, C, *N)
        Batched multi-channel complex Fourier mode coefficients at uniform points.
    eps : float, optional
        Precision. Defaults to ``1e-6``.

    Returns
    -------
    torch.Tensor, shape (B, C, M)
        Batched multi-channel complex Fourier transform values at the non-uniform target points.
    """
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
    """Compute the multi-dimensional type-3 (non-uniform to non-uniform) NUFFT.

    Evaluates f_k = sum_j c_j * exp(1j * dot(s_k, x_j)).

    Parameters
    ----------
    x : torch.Tensor, shape (d, M)
        Non-uniform source points.
    c : torch.Tensor, shape (B, C, M)
        Batched multi-channel complex source strengths at the non-uniform source points.
    s : torch.Tensor, shape (d, N)
        Non-uniform target points.
    eps : float, optional
        Precision. Defaults to ``1e-6``.

    Returns
    -------
    torch.Tensor, shape (B, C, N)
        Batched multi-channel complex Fourier transform values at the non-uniform target points.
    """
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


def get_density(x, c, N, eps=1e-6, n_iter=10):
    """Estimate multi-dimensional density-compensation weights for non-uniform source points.

    The weights are computed using the iterative method described in [1].
    With ``n_iter=1``, this corresponds to the approximation described in
    [2].

    Parameters
    ----------
    x : torch.Tensor, shape (d, M)
        Non-uniform source points.
    c : torch.Tensor, shape (B, C, M)
        Batched multi-channel complex source strengths at the non-uniform source points.
    N : tuple[int, ...], shape (d,)
        Number of Fourier modes along each dimension.
    eps : float, optional
        Precision. Defaults to ``1e-6``.
    n_iter : int, optional
        Number of density-compensation iterations. Defaults to ``10``.

    Returns
    -------
    torch.Tensor, shape (B, C, M)
        Density-compensation weights.

    References
    ----------
    [1] Pipe JG, Menon P.
        Sampling density compensation in MRI: rationale and an iterative numerical solution.
        Magn Reson Med. 1999 Jan;41(1):179-86. doi: 10.1002/(sici)1522-2594(199901)41:1<179::aid-mrm25>3.0.co;2-v. PMID: 10025627.

    [2] Jackson JI, Meyer CH, Nishimura DG,  Macovski A.
        Selection of a convolution function for Fourier inversion using gridding (computerised tomography application).
        IEEE Transactions on Medical Imaging. 1991 Sept;10(3):473-78. doi: 10.1109/42.97598
    [3] Comby et al.
        MRI-NUFFT: Doing non-Cartesian MRI has never been easier.
        Journal of Open Source Software. 2025;10(108):7743. doi: 10.21105/joss.07743
    """
    d = len(N)
    device = c.device
    alpha, beta, h, n, _, _, _, _, _ = helpers.setup(d, N, eps, x.dtype, device)
    density = torch.abs(torch.ones_like(c))
    for _ in range(n_iter):
        b = spreadinterp.spread(x, density, alpha, beta, d, h, n)
        b = spreadinterp.interp(x, b, alpha, beta, d, h, n)
        density = density / torch.clamp(b, min=torch.finfo(x.dtype).eps)
    # Normalize to preserve the pure-adjoint 1 / M scaling needed for correctly scaled reconstruction.
    return density / density.mean(dim=-1, keepdim=True)
