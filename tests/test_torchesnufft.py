import pytest
import torch

from torchesnufft.functional import nufft1, nufft2, nufft3

device = "cuda" if torch.cuda.is_available() else "cpu"


def _generate_random_data_nufft1(batch_size, channel_size, M, N):
    d = len(N)
    x = 2 * torch.pi * torch.rand(size=(M,))
    y = 2 * torch.pi * torch.rand(size=(M,))
    z = 2 * torch.pi * torch.rand(size=(M,))
    if d == 1:
        xyz = x.unsqueeze(0)
    elif d == 2:
        xyz = torch.stack((x, y))
    elif d == 3:
        xyz = torch.stack((x, y, z))
    c = torch.randn(size=(batch_size, channel_size, M)) + 1j * torch.randn(
        size=(batch_size, channel_size, M)
    )
    return xyz, c, N, M


def _generate_random_data_nufft2(batch_size, channel_size, M, N):
    d = len(N)
    x = 2 * torch.pi * torch.rand(size=(M,))
    y = 2 * torch.pi * torch.rand(size=(M,))
    z = 2 * torch.pi * torch.rand(size=(M,))
    if d == 1:
        xyz = x.unsqueeze(0)
    elif d == 2:
        xyz = torch.stack((x, y))
    elif d == 3:
        xyz = torch.stack((x, y, z))
    f = torch.randn(size=(batch_size, channel_size, *N)) + 1j * torch.randn(
        size=(batch_size, channel_size, *N)
    )
    return xyz, f, N, M


def _generate_random_data_nufft3(batch_size, channel_size, M, N, d):
    x = 2 * torch.pi * torch.rand(size=(M,))
    y = 2 * torch.pi * torch.rand(size=(M,))
    z = 2 * torch.pi * torch.rand(size=(M,))
    if d == 1:
        xyz = x.unsqueeze(0)
    elif d == 2:
        xyz = torch.stack((x, y))
    elif d == 3:
        xyz = torch.stack((x, y, z))
    s = 2 * torch.pi * torch.rand(size=(N,))
    t = 2 * torch.pi * torch.rand(size=(N,))
    u = 2 * torch.pi * torch.rand(size=(N,))
    if d == 1:
        stu = s.unsqueeze(0)
    elif d == 2:
        stu = torch.stack((s, t))
    elif d == 3:
        stu = torch.stack((s, t, u))
    c = torch.randn(
        size=(
            batch_size,
            channel_size,
            M,
        )
    ) + 1j * torch.randn(
        size=(
            batch_size,
            channel_size,
            M,
        )
    )
    return xyz, c, stu, M, N


def _manual_nufft1(xyz, c, N):
    d = len(N)
    device = c.device
    dtype = xyz.dtype

    mode_idxs = [
        torch.arange(
            -((N[dim] - N[dim] % 2) // 2),
            (N[dim] - N[dim] % 2) // 2 + N[dim] % 2,
            device=device,
            dtype=dtype,
        )
        for dim in range(d)
    ]

    mode_grid = torch.meshgrid(*mode_idxs, indexing="ij")
    k = torch.stack(mode_grid, dim=0)
    phase = torch.sum(k[..., None] * xyz[:, *([None] * d), :], dim=0)
    kernel = torch.exp(1j * phase)

    return torch.einsum("bcm,...m->bc...", c, kernel)


def _manual_nufft2(xyz, f):
    d = len(f.shape) - 2
    device = f.device
    dtype = xyz.dtype

    mode_idxs = [
        torch.arange(
            -((f.shape[2 + dim] - f.shape[2 + dim] % 2) // 2),
            (f.shape[2 + dim] - f.shape[2 + dim] % 2) // 2 + f.shape[2 + dim] % 2,
            device=device,
            dtype=dtype,
        )
        for dim in range(d)
    ]

    mode_grid = torch.meshgrid(*mode_idxs, indexing="ij")
    k = torch.stack(mode_grid, dim=0)
    phase = torch.sum(k[..., None] * xyz[:, *([None] * d), :], dim=0)
    kernel = torch.exp(1j * phase)

    return torch.einsum("bc...,...m->bcm", f, kernel)


def _manual_nufft3(xyz, c, stu):
    phase = torch.einsum("dm,dn->mn", xyz, stu)
    kernel = torch.exp(1j * phase)  # (M, N)

    return torch.einsum("bcm,mn->bcn", c, kernel)


def check_nufft1(batch_size, channel_size, M, N):
    xyz, c, N, M = _generate_random_data_nufft1(batch_size, channel_size, M, N)
    manual_result = _manual_nufft1(xyz.to(device), c.to(device), N)
    nufft_result = nufft1(xyz.to(device), c.to(device), N)
    # TODO: investigate the influence of eps on accuracy
    assert torch.allclose(nufft_result, manual_result, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channel_size", [1, 3])
@pytest.mark.parametrize("M", [8, 11])
@pytest.mark.parametrize("N", [(3,), (2, 5), (4, 3, 4)])
def test_nufft1(batch_size, channel_size, M, N):
    check_nufft1(batch_size, channel_size, M, N)


def check_nufft2(batch_size, channel_size, M, N):
    xyz, f, N, M = _generate_random_data_nufft2(batch_size, channel_size, M, N)
    manual_result = _manual_nufft2(xyz.to(device), f.to(device))
    nufft_result = nufft2(xyz.to(device), f.to(device))
    # TODO: investigate the influence of eps on accuracy
    assert torch.allclose(nufft_result, manual_result, atol=1e-4)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channel_size", [1, 3])
@pytest.mark.parametrize("M", [8, 11])
@pytest.mark.parametrize("N", [(3,), (2, 5), (4, 3, 4)])
def test_nufft2(batch_size, channel_size, M, N):
    check_nufft2(batch_size, channel_size, M, N)


def check_nufft3(batch_size, channel_size, M, N, d):
    xyz, c, stu, M, N = _generate_random_data_nufft3(batch_size, channel_size, M, N, d)
    manual_result = _manual_nufft3(xyz.to(device), c.to(device), stu.to(device))
    nufft_result = nufft3(xyz.to(device), c.to(device), stu.to(device))
    # TODO: investigate the influence of eps on accuracy
    assert torch.allclose(nufft_result, manual_result, atol=1e-4)


def check_nufft1_autograd(batch_size, channel_size, M, N):
    xyz, c, N, M = _generate_random_data_nufft1(batch_size, channel_size, M, N)
    xyz = xyz.to(dtype=torch.float64, device="cpu").requires_grad_()
    c = c.to(dtype=torch.complex128, device="cpu").requires_grad_()
    assert torch.autograd.gradcheck(
        nufft1,
        (xyz, c, N),
        nondet_tol=1e-8,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channel_size", [1, 3])
@pytest.mark.parametrize("M", [8, 11])
@pytest.mark.parametrize("N", [(3,), (2, 5), (4, 3, 4)])
def test_nufft1_autograd(batch_size, channel_size, M, N):
    check_nufft1_autograd(batch_size, channel_size, M, N)


def check_nufft2_autograd(batch_size, channel_size, M, N):
    xyz, f, N, M = _generate_random_data_nufft2(batch_size, channel_size, M, N)
    xyz = xyz.to(dtype=torch.float64, device="cpu").requires_grad_()
    f = f.to(dtype=torch.complex128, device="cpu").requires_grad_()
    assert torch.autograd.gradcheck(
        nufft2,
        (xyz, f),
        nondet_tol=1e-8,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channel_size", [1, 3])
@pytest.mark.parametrize("M", [8, 11])
@pytest.mark.parametrize("N", [(3,), (2, 5), (4, 3, 4)])
def test_nufft2_autograd(batch_size, channel_size, M, N):
    check_nufft2_autograd(batch_size, channel_size, M, N)


def check_nufft3_autograd(batch_size, channel_size, M, N, d):
    xyz, c, stu, M, N = _generate_random_data_nufft3(batch_size, channel_size, M, N, d)
    xyz = xyz.to(dtype=torch.float64, device="cpu").requires_grad_()
    c = c.to(dtype=torch.complex128, device="cpu").requires_grad_()
    stu = stu.to(dtype=torch.float64, device="cpu").requires_grad_()
    assert torch.autograd.gradcheck(
        nufft3,
        (xyz, c, stu),
        nondet_tol=1e-8,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channel_size", [1, 3])
@pytest.mark.parametrize("M", [8, 11])
@pytest.mark.parametrize("N", [3, 6])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_nufft3_autograd(batch_size, channel_size, M, N, d):
    check_nufft3_autograd(batch_size, channel_size, M, N, d)
