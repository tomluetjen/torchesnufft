import torch

from torchesnufft.functional import nufft2, nufft_inv

# number of nonuniform points
M = 10000

# the nonuniform points
x = 2 * torch.pi * torch.rand(size=(M,))
y = 2 * torch.pi * torch.rand(size=(M,))
z = 2 * torch.pi * torch.rand(size=(M,))
xyz = torch.stack((x, y, z))
# number of Fourier modes
N1, N2, N3 = 5, 7, 2

# the Fourier mode coefficients
f = torch.randn(size=(1, 1, N1, N2, N3)) + 1j * torch.randn(size=(1, 1, N1, N2, N3))

# calculate the type-2 NUFFT (forward)
c = nufft2(-xyz, f)

# calculate the type-2 NUFFT (inverse)
f_reco = nufft_inv(xyz, c, (N1, N2, N3)) / (N1 * N2 * N3)
