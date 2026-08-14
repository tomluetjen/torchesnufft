import pytest
import torch
import torchkbnufft

from torchesnufft.functional import nufft1, nufft2

# Load MNIST once per module and reuse
torch.manual_seed(0)


@pytest.fixture(scope="module")
def random_data():
    M = (64, 64, 64)
    x = 2 * torch.pi * torch.rand(size=M)
    y = 2 * torch.pi * torch.rand(size=M)
    z = 2 * torch.pi * torch.rand(size=M)
    c = torch.randn((4, 1, *M)) + 1j * torch.randn((4, 1, *M))
    N = (128, 128, 128)
    xyz = torch.stack((x, y, z))
    f = torch.randn((4, 1, *N)) + 1j * torch.randn((4, 1, *N))
    return xyz, c, N, M, f


# Parameters
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchesnufft", "torchkbnufft"], ids=lambda n: f"{n}")
def benchmark_nufft1(benchmark, name, device, random_data):
    xyz, c, N, _, _ = random_data
    benchmark.group = "NUFFT (Type 1) on random data"
    device_name = "CPU" if device == "cpu" else "GPU"
    benchmark.name = f"{name} ({device_name})"

    if name == "torchesnufft":
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        c = torch.reshape(c, (c.shape[0], c.shape[1], -1)).to(device)

        # Warm-up
        with torch.inference_mode():
            _ = nufft1(xyz, c, N)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                f = nufft1(xyz, c, N)
            if device == "cuda":
                torch.cuda.synchronize()
            return f
    else:  # torchkbnufft
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        c = torch.reshape(c, (c.shape[0], c.shape[1], -1)).to(device)

        # Warm-up
        with torch.inference_mode():
            nufft_ob = torchkbnufft.KbNufftAdjoint(im_size=N).to(device)
            _ = nufft_ob(c, xyz)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                f = nufft_ob(c, xyz)
            if device == "cuda":
                torch.cuda.synchronize()
            return f

    benchmark(run)


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchesnufft", "torchkbnufft"], ids=lambda n: f"{n}")
def benchmark_nufft2(benchmark, name, device, random_data):
    xyz, c, N, _, f = random_data
    benchmark.group = "NUFFT (Type 2) on random data"
    device_name = "CPU" if device == "cpu" else "GPU"
    benchmark.name = f"{name} ({device_name})"

    if name == "torchesnufft":
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        f = f.to(device)

        # Warm-up
        with torch.inference_mode():
            _ = nufft2(-xyz, f)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                c = nufft2(-xyz, f)
            if device == "cuda":
                torch.cuda.synchronize()
            return c
    else:  # torchkbnufft
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        f = f.to(device)

        # Warm-up
        with torch.inference_mode():
            nufft_ob = torchkbnufft.KbNufft(im_size=N).to(device)
            _ = nufft_ob(f, xyz)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                c = nufft_ob(f, xyz)
            if device == "cuda":
                torch.cuda.synchronize()
            return c

    benchmark(run)
