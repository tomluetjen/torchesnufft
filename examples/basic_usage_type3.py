# https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft3d3
import torch

from torchesnufft.functional import nufft3

# number of source points
M = 100

# number of target points
N = 200

# the source points
x = 2 * torch.pi * torch.rand(size=(M,))
y = 2 * torch.pi * torch.rand(size=(M,))
z = 2 * torch.pi * torch.rand(size=(M,))
xyz = torch.stack((x, y, z))

# the target points
s = 2 * torch.pi * torch.rand(size=(N,))
t = 2 * torch.pi * torch.rand(size=(N,))
u = 2 * torch.pi * torch.rand(size=(N,))
stu = torch.stack((s, t, u))

# their complex strengths
c = torch.randn(
    size=(
        1,
        1,
        M,
    )
) + 1j * torch.randn(
    size=(
        1,
        1,
        M,
    )
)

# calcuate the type-3 NUFFT
f = nufft3(xyz, c, stu)
