import torch

from torchesnufft.functional import get_density, nufft1, nufft2

# number of nonuniform points
M = 1000

# the nonuniform points
x = 2 * torch.pi * torch.rand(size=(M,))
y = 2 * torch.pi * torch.rand(size=(M,))
z = 2 * torch.pi * torch.rand(size=(M,))
xyz = torch.stack((x, y, z))
# number of Fourier modes (M >> N necessary for accurate reconstruction)
N1, N2, N3 = 3, 4, 5

# the Fourier mode coefficients
f = torch.randn(size=(1, 1, N1, N2, N3)) + 1j * torch.randn(size=(1, 1, N1, N2, N3))

# calculate the type-2 NUFFT
c = nufft2(-xyz, f)

# calculate the type-2 NUFFT inverse via density compensation
density = get_density(xyz, c, (N1, N2, N3))
f_reco = nufft1(xyz, c * density, (N1, N2, N3)) / M
