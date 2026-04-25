# torchesnufft

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](#) [![CI](https://github.com/tomluetjen/torchesnufft/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/tomluetjen/torchesnufft/actions/workflows/python-app.yml) [![Coverage](https://codecov.io/gh/tomluetjen/torchesnufft/branch/main/graph/badge.svg)](https://codecov.io/gh/tomluetjen/torchesnufft)

## About
`torchesnufft` implements the non-uniform fast Fourier transform (Type 1, Type 2 & Type 3) with an exponential of semicircle kernel [1, 2] and its' inverse [3] in PyTorch. All transforms work with batched multi-channel data and are fully differentiable. This allows backpropagation through `torchesnufft` transforms to train neural networks or to solve optimization problems with [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html).


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

### Inverse
```python
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
```

## Examples
For more detailed examples and use cases, see the `examples` directory:

- [`examples/uniform.py`](examples/uniform.py) - Standard DFT & iDFT using `torchesnufft` NUFFTs
- [`examples/radial.py`](examples/radial.py) - `torchesnufft` 2D NUFFT functions on radial data
- [`examples/radial.py`](examples/radial.py) - `torchesnufft` 1D NUFFT functions on randomly sampled data

## Performance compared to torchkbnufft
```console
----------------------------------------------------------------------- benchmark 'NUFFT (Type 1) on random data': 4 tests -----------------------------------------------------------------------
Name (time in ms)           Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torchesnufft (GPU)      18.7343 (1.0)         24.8683 (1.0)         21.3013 (1.0)       2.2158 (1.0)         20.2273 (1.0)       4.0700 (1.0)          16;0  46.9454 (1.0)          41           1
torchkbnufft (GPU)     178.5804 (9.53)       198.2222 (7.97)       186.9434 (8.78)      6.7912 (3.06)       186.0647 (9.20)      7.3885 (1.82)          2;0   5.3492 (0.11)          6           1
torchesnufft (CPU)     938.4318 (50.09)    1,054.7845 (42.41)      989.4232 (46.45)    43.5709 (19.66)      981.3666 (48.52)    55.8505 (13.72)         2;0   1.0107 (0.02)          5           1
torchkbnufft (CPU)     976.1194 (52.10)    1,065.0852 (42.83)    1,006.7708 (47.26)    35.9041 (16.20)    1,000.6046 (49.47)    47.0543 (11.56)         1;0   0.9933 (0.02)          5           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------- benchmark 'NUFFT (Type 2) on random data': 4 tests ------------------------------------------------------------------------
Name (time in ms)             Min                   Max                  Mean             StdDev                Median                 IQR            Outliers      OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
torchesnufft (GPU)        26.7485 (1.0)         34.5041 (1.0)         28.5387 (1.0)       2.6961 (1.0)         26.9654 (1.0)        4.6211 (1.0)          10;0  35.0401 (1.0)          38           1
torchkbnufft (GPU)        41.4610 (1.55)       125.7308 (3.64)        72.9889 (2.56)     36.6247 (13.58)       52.8371 (1.96)      66.2846 (14.34)         2;0  13.7007 (0.39)          7           1
torchkbnufft (CPU)       444.3747 (16.61)      473.7960 (13.73)      458.3618 (16.06)    10.5870 (3.93)       457.5047 (16.97)     11.2631 (2.44)          2;0   2.1817 (0.06)          5           1
torchesnufft (CPU)     1,452.8532 (54.32)    1,663.1492 (48.20)    1,564.7025 (54.83)    88.7630 (32.92)    1,563.0342 (57.96)    154.2930 (33.39)         2;0   0.6391 (0.02)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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