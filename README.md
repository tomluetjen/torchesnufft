# torchesnufft

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](#) [![CI](https://github.com/tomluetjen/torchesnufft/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/tomluetjen/torchesnufft/actions/workflows/python-app.yml) [![Coverage](https://codecov.io/gh/tomluetjen/torchesnufft/branch/main/graph/badge.svg)](https://codecov.io/gh/tomluetjen/torchesnufft)

## About
WORK IN PROGESS: `torchesnufft` implements the non-uniform fast Fourier transforms (Type 1, Type 2 & Type 3) with an exponential of semicircle kernel [1, 2] in PyTorch. All transforms work with batched multi-channel data and are fully differentiable. This allows backpropagation through `torchesnufft` transforms to train neural networks or to solve optimization problems with [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html).


## Installation
```console
pip install .
```

## Basic Usage
### Type 1
```python
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
```
### Type 2
```python
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
```
### Type 3
```python
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

```

## Examples
For more detailed examples and use cases, see the `examples` directory:

- [`examples/uniform.py`](examples/uniform.py) - Standard DFT using `torchesnufft` NUFFTs

## Performance compared to torchkbnufft
```console
---------------------------------------------------------------------- benchmark 'NUFFT (Type 1) on random data': 4 tests ----------------------------------------------------------------------
Name (time in ms)           Min                   Max                Mean              StdDev              Median                 IQR            Outliers      OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torchesnufft (GPU)      17.3531 (1.0)         22.4222 (1.0)       18.9538 (1.0)        1.6382 (1.0)       18.0799 (1.0)        3.2203 (1.0)          15;0  52.7600 (1.0)          56           1
torchkbnufft (GPU)      55.6630 (3.21)        65.6254 (2.93)      63.1279 (3.33)       4.2089 (2.57)      64.9827 (3.59)       3.3200 (1.03)          1;1  15.8409 (0.30)          5           1
torchkbnufft (CPU)     387.4568 (22.33)      545.6971 (24.34)    452.1273 (23.85)     63.4495 (38.73)    423.9652 (23.45)     91.2222 (28.33)         2;0   2.2118 (0.04)          5           1
torchesnufft (CPU)     864.3891 (49.81)    1,104.7883 (49.27)    984.4262 (51.94)    107.7794 (65.79)    934.5187 (51.69)    184.9233 (57.42)         3;0   1.0158 (0.02)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------ benchmark 'NUFFT (Type 2) on random data': 4 tests ------------------------------------------------------------------------
Name (time in ms)             Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torchesnufft (GPU)        24.5631 (1.0)         31.7494 (1.0)         26.8916 (1.0)       2.4724 (1.0)         25.3644 (1.0)       4.6969 (1.0)          11;0  37.1864 (1.0)          40           1
torchkbnufft (GPU)       148.6083 (6.05)       225.9004 (7.12)       170.7348 (6.35)     26.5956 (10.76)      162.7521 (6.42)     24.2778 (5.17)          1;1   5.8570 (0.16)          7           1
torchkbnufft (CPU)       766.9130 (31.22)      939.8924 (29.60)      859.7188 (31.97)    64.5078 (26.09)      854.6073 (33.69)    83.2242 (17.72)         2;0   1.1632 (0.03)          5           1
torchesnufft (CPU)     1,385.0010 (56.39)    1,491.5670 (46.98)    1,457.6868 (54.21)    43.5951 (17.63)    1,466.2686 (57.81)    53.4877 (11.39)         1;0   0.6860 (0.02)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
```
## Accuracy compared to finufft
```console
--------------------------- Relative Error of NUFFT Type 1 on random data: 2 tests ---------------------------
Name                             Min           Max          Mean        StdDev        Median           IQR      Outliers
------------------------------------------------------------------------------------------------------------------------
torchesnufft (CPU)        0.0000e+00    8.1995e-02    1.5683e-05    3.8490e-05    9.8549e-06    1.1968e-05 1622552;3635254
torchesnufft (GPU)        0.0000e+00    7.4902e-02    1.5687e-05    3.8069e-05    9.8640e-06    1.1991e-05 1646515;3625938
------------------------------------------------------------------------------------------------------------------------

--------------------------- Relative Error of NUFFT Type 2 on random data: 2 tests ---------------------------
Name                             Min           Max          Mean        StdDev        Median           IQR      Outliers
------------------------------------------------------------------------------------------------------------------------
torchesnufft (CPU)        4.2678e-08    7.9130e-04    1.3886e-05    2.4288e-05    7.8234e-06    1.1039e-05      789;1115
torchesnufft (GPU)        0.0000e+00    7.5902e-04    1.3887e-05    2.4177e-05    7.8571e-06    1.1133e-05      800;1096
------------------------------------------------------------------------------------------------------------------------

--------------------------- Relative Error of NUFFT Type 3 on random data: 2 tests ---------------------------
Name                             Min           Max          Mean        StdDev        Median           IQR      Outliers
------------------------------------------------------------------------------------------------------------------------
torchesnufft (CPU)        3.1846e-08    2.3867e-04    5.9046e-06    6.6178e-06    4.6324e-06    3.7061e-06     1537;1586
torchesnufft (GPU)        5.3433e-08    2.8241e-04    5.7703e-06    6.7408e-06    4.5119e-06    3.6439e-06     1498;1616
------------------------------------------------------------------------------------------------------------------------
Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
```

## Other Packages
This package is inspired by

1. [`finfufft`](https://github.com/flatironinstitute/finufft)

2. [`torchkbnufft`](https://github.com/mmuckley/torchkbnufft)

3. [`pytorch-finufft`](https://github.com/flatironinstitute/pytorch-finufft)


## References
1. Barnett AH, Magland J, af Klinteberg L, ["A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel"](https://epubs.siam.org/doi/10.1137/18M120885X), Software and High-Performance Computing, 2019
2. Shih YH, Wright G, Anden J, Blaschke J, Barnett AH, ["cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs"](https://arxiv.org/abs/2102.08463), PDSEC2021 workshop of the IPDPS2021 conference, 2021
