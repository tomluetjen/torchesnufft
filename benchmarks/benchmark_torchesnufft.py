import pytest
import torch
import torchkbnufft

import torchesnufft

# Load MNIST once per module and reuse
torch.manual_seed(0)


@pytest.fixture(scope="module")
def random_data():
    M = (64, 64, 64)
    x = 2 * torch.pi * torch.rand(size=M)
    y = 2 * torch.pi * torch.rand(size=M)
    z = 2 * torch.pi * torch.rand(size=M)
    c = torch.randn((4, 1, *M)) + 1j * torch.randn((4, 1, *M))
    N = (64, 64, 64)
    xyz = torch.stack((x, y, z))
    return xyz, c, N, M


# Parameters
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES, ids=lambda d: f"device={d}")
@pytest.mark.parametrize("name", ["torchesnufft", "torchkbnufft"], ids=lambda n: f"{n}")
def benchmark_nufft1(benchmark, name, device, random_data):
    xyz, c, N, M = random_data
    benchmark.group = "NUFFT (Type 1) on random data"
    device_name = "CPU" if device == "cpu" else "GPU"
    benchmark.name = f"{name} ({device_name})"

    if name == "torchesnufft":
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        c = torch.reshape(c, (c.shape[0], c.shape[1], -1)).to(device)

        # Warm-up
        with torch.inference_mode():
            _ = torchesnufft.functional.nufft1(xyz, c, N)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                f = torchesnufft.functional.nufft1(xyz, c, N)
            if device == "cuda":
                torch.cuda.synchronize()
            return f
    else:  # torchkbnufft
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        c = c.to(device)

        # Warm-up
        with torch.inference_mode():
            nufft_ob = torchkbnufft.KbNufft(im_size=M).to(device)
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
    xyz, c, N, M = random_data
    benchmark.group = "NUFFT (Type 2) on random data"
    device_name = "CPU" if device == "cpu" else "GPU"
    benchmark.name = f"{name} ({device_name})"

    if name == "torchesnufft":
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        c = c.to(device)

        # Warm-up
        with torch.inference_mode():
            _ = torchesnufft.functional.nufft2(xyz, c)
        if device == "cuda":
            torch.cuda.synchronize()

        def run():
            with torch.inference_mode():
                f = torchesnufft.functional.nufft2(xyz, c)
            if device == "cuda":
                torch.cuda.synchronize()
            return f
    else:  # torchkbnufft
        xyz = torch.reshape(xyz, (xyz.shape[0], -1)).to(device)
        c = torch.reshape(c, (c.shape[0], c.shape[1], -1)).to(device)

        # Warm-up
        with torch.inference_mode():
            nufft_ob = torchkbnufft.KbNufftAdjoint(im_size=M).to(device)
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
