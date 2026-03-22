# https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft3d1
import torch

from torchesnufft.functional import nufft1

# number of nonuniform points
M = 100

# the nonuniform points
x = 2 * torch.pi * torch.rand(size=(M,))
y = 2 * torch.pi * torch.rand(size=(M,))
z = 2 * torch.pi * torch.rand(size=(M,))
xyz = torch.stack((x, y, z))
# their complex strengths
c = torch.randn(size=(1, 1, M)) + 1j * torch.randn(size=(1, 1, M))
# desired number of Fourier modes
N1, N2, N3 = 50, 75, 100

# calculate the type-1 NUFFT
f = nufft1(xyz, c, (N1, N2, N3))
