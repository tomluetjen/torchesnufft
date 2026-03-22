# https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft3d2
import torch

from torchesnufft.functional import nufft2

# number of nonuniform points
M = 100

# the nonuniform points
x = 2 * torch.pi * torch.rand(size=(M,))
y = 2 * torch.pi * torch.rand(size=(M,))
z = 2 * torch.pi * torch.rand(size=(M,))
xyz = torch.stack((x, y, z))
# number of Fourier modes
N1, N2, N3 = 50, 75, 100

# the Fourier mode coefficients
f = torch.randn(size=(1, 1, N1, N2, N3)) + 1j * torch.randn(size=(1, 1, N1, N2, N3))

# calculate the type-2 NUFFT
c = nufft2(xyz, f)
