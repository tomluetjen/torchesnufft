# torchesnufft

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](#) [![PyPI](https://img.shields.io/pypi/v/torchesnufft.svg?label=PyPI&logo=pypi)](https://pypi.org/project/torchesnufft/) [![CI](https://github.com/tomluetjen/torchesnufft/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/tomluetjen/torchesnufft/actions/workflows/python-app.yml) [![Coverage](https://codecov.io/gh/tomluetjen/torchesnufft/branch/main/graph/badge.svg)](https://codecov.io/gh/tomluetjen/torchesnufft)

## About
`torchesnufft` implements the non-uniform fast Fourier transform (Type 1, Type 2 & Type 3) with an exponential of semicircle kernel [1, 2] and a simple density compensation routine [3] in PyTorch. All transforms work with batched multi-channel data and are fully differentiable. This allows backpropagation through `torchesnufft` transforms to train neural networks or to solve optimization problems with [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html).


## Installation
```console
pip install torchesnufft
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

### Inverse
```python
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
```

## Examples
For more detailed examples and use cases, see the `examples` directory:

- [`examples/uniform.py`](examples/uniform.py) - Standard DFT & iDFT using `torchesnufft` NUFFTs
- [`examples/radial.py`](examples/radial.py) - `torchesnufft` 2D NUFFT functions on radial data
- [`examples/rand.py`](examples/rand.py) - `torchesnufft` 1D NUFFT functions on randomly sampled data

## Performance compared to torchkbnufft
```console
------------------------------------------------------------------------- benchmark 'NUFFT (Type 1) on random data': 4 tests -------------------------------------------------------------------------
Name (time in ms)             Min                   Max                  Mean              StdDev                Median                 IQR            Outliers      OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torchesnufft (GPU)        38.5997 (1.0)         50.0770 (1.0)         41.3866 (1.0)        3.4476 (1.0)         39.7005 (1.0)        4.1097 (1.0)           1;1  24.1624 (1.0)          11           1
torchkbnufft (GPU)       154.3641 (4.00)       207.5648 (4.14)       176.4468 (4.26)      18.5911 (5.39)       173.4083 (4.37)      22.3300 (5.43)          2;0   5.6674 (0.23)          6           1
torchkbnufft (CPU)     1,361.2283 (35.27)    1,601.6301 (31.98)    1,482.7317 (35.83)    111.3348 (32.29)    1,466.6016 (36.94)    211.0206 (51.35)         2;0   0.6744 (0.03)          5           1
torchesnufft (CPU)     1,671.5433 (43.30)    1,795.4848 (35.85)    1,742.4148 (42.10)     56.6646 (16.44)    1,758.9578 (44.31)    103.9310 (25.29)         1;0   0.5739 (0.02)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------- benchmark 'NUFFT (Type 2) on random data': 4 tests -------------------------------------------------------------------------
Name (time in ms)             Min                   Max                  Mean              StdDev                Median                 IQR            Outliers      OPS            Rounds  Iterations
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torchesnufft (GPU)        32.4498 (1.0)         41.1495 (1.0)         35.7500 (1.0)        2.4407 (1.0)         35.9135 (1.0)        4.6097 (1.0)          11;0  27.9720 (1.0)          28           1
torchkbnufft (GPU)        53.8941 (1.66)        72.1182 (1.75)        59.3074 (1.66)       7.9965 (3.28)        54.6892 (1.52)      11.6341 (2.52)          2;0  16.8613 (0.60)          7           1
torchkbnufft (CPU)       658.2220 (20.28)      720.0952 (17.50)      700.8981 (19.61)     25.2738 (10.36)      712.6485 (19.84)     28.8190 (6.25)          1;0   1.4267 (0.05)          5           1
torchesnufft (CPU)     1,925.4414 (59.34)    2,153.0908 (52.32)    2,028.5190 (56.74)    107.8065 (44.17)    1,979.9341 (55.13)    197.6982 (42.89)         1;0   0.4930 (0.02)          5           1
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

4. [`mri-nufft`](https://github.com/mind-inria/mri-nufft)

## References
1. Barnett AH, Magland J, af Klinteberg L, ["A Parallel Nonuniform Fast Fourier Transform Library Based on an “Exponential of Semicircle" Kernel"](https://epubs.siam.org/doi/10.1137/18M120885X), Software and High-Performance Computing, 2019
2. Shih YH, Wright G, Anden J, Blaschke J, Barnett AH, ["cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs"](https://arxiv.org/abs/2102.08463), PDSEC2021 workshop of the IPDPS2021 conference, 2021
3. Pipe JG, Menon P. ["Sampling density compensation in MRI: rationale and an iterative numerical solution"](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V). Magn Reson Med. 1999 Jan;41(1):179-86. doi: 10.1002/(sici)1522-2594(199901)41:1<179::aid-mrm25>3.0.co;2-v. PMID: 10025627.