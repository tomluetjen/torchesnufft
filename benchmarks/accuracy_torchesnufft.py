# TODO: Compare to cufinufft
import finufft
import torch

from torchesnufft.functional import nufft1, nufft2, nufft3

BATCH_SIZE = 128

torch.manual_seed(0)  # for reproducibility


def generate_random_data_nufft1():
    M = 100
    x = 2 * torch.pi * torch.rand(size=(M,))
    y = 2 * torch.pi * torch.rand(size=(M,))
    z = 2 * torch.pi * torch.rand(size=(M,))
    xyz = torch.stack((x, y, z))
    c = torch.randn(size=(BATCH_SIZE, 1, M)) + 1j * torch.randn(size=(BATCH_SIZE, 1, M))
    N1, N2, N3 = 50, 75, 100
    return xyz, c, (N1, N2, N3), M


def generate_random_data_nufft2():
    M = 100
    x = 2 * torch.pi * torch.rand(size=(M,))
    y = 2 * torch.pi * torch.rand(size=(M,))
    z = 2 * torch.pi * torch.rand(size=(M,))
    xyz = torch.stack((x, y, z))
    N1, N2, N3 = 50, 75, 100
    f = torch.randn(size=(BATCH_SIZE, 1, N1, N2, N3)) + 1j * torch.randn(
        size=(BATCH_SIZE, 1, N1, N2, N3)
    )
    return xyz, f, (N1, N2, N3), M


def generate_random_data_nufft3():
    M = 100
    N = 200
    x = 2 * torch.pi * torch.rand(size=(M,))
    y = 2 * torch.pi * torch.rand(size=(M,))
    z = 2 * torch.pi * torch.rand(size=(M,))
    xyz = torch.stack((x, y, z))
    s = 2 * torch.pi * torch.rand(size=(N,))
    t = 2 * torch.pi * torch.rand(size=(N,))
    u = 2 * torch.pi * torch.rand(size=(N,))
    stu = torch.stack((s, t, u))
    c = torch.randn(
        size=(
            BATCH_SIZE,
            1,
            M,
        )
    ) + 1j * torch.randn(
        size=(
            BATCH_SIZE,
            1,
            M,
        )
    )
    return xyz, c, stu, M, N


# Parameters
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

XYZ, C, N, M = generate_random_data_nufft1()
TYPE1_cpu = nufft1(XYZ.to("cpu"), C.to("cpu"), N)
TYPE1_cuda = (
    nufft1(XYZ.to("cuda"), C.to("cuda"), N)
    if torch.cuda.is_available()
    else torch.zeros_like(TYPE1_cpu)
)
TYPE1 = torch.stack((TYPE1_cpu, TYPE1_cuda.to("cpu")))
TYPE1_FI = torch.zeros((len(DEVICES), BATCH_SIZE, 1, *N), dtype=C.dtype)
for b in range(BATCH_SIZE):
    for device in DEVICES:
        if device == "cpu":
            TYPE1_FI[0, b] = torch.from_numpy(
                finufft.nufft3d1(
                    XYZ[0].cpu().numpy(),
                    XYZ[1].cpu().numpy(),
                    XYZ[2].cpu().numpy(),
                    C[b, 0].cpu().numpy(),
                    N,
                )
            )
        elif device == "cuda":
            TYPE1_FI[1, b] = torch.from_numpy(
                finufft.nufft3d1(
                    XYZ[0].cpu().numpy(),
                    XYZ[1].cpu().numpy(),
                    XYZ[2].cpu().numpy(),
                    C[b, 0].cpu().numpy(),
                    N,
                )
            )

XYZ, F, N, M = generate_random_data_nufft2()
TYPE2_cpu = nufft2(XYZ.to("cpu"), F.to("cpu"))
TYPE2_cuda = (
    nufft2(XYZ.to("cuda"), F.to("cuda"))
    if torch.cuda.is_available()
    else torch.zeros_like(TYPE2_cpu)
)
TYPE2 = torch.stack((TYPE2_cpu, TYPE2_cuda.to("cpu")))
TYPE2_FI = torch.zeros((len(DEVICES), BATCH_SIZE, 1, M), dtype=F.dtype)
for b in range(BATCH_SIZE):
    for device in DEVICES:
        if device == "cpu":
            TYPE2_FI[0, b] = torch.from_numpy(
                finufft.nufft3d2(
                    XYZ[0].cpu().numpy(),
                    XYZ[1].cpu().numpy(),
                    XYZ[2].cpu().numpy(),
                    F[b, 0].cpu().numpy(),
                    isign=1,
                )
            )
        elif device == "cuda":
            TYPE2_FI[1, b] = torch.from_numpy(
                finufft.nufft3d2(
                    XYZ[0].cpu().numpy(),
                    XYZ[1].cpu().numpy(),
                    XYZ[2].cpu().numpy(),
                    F[b, 0].cpu().numpy(),
                    isign=1,
                )
            )

XYZ, C, STU, M, N = generate_random_data_nufft3()
TYPE3_cpu = nufft3(XYZ.to("cpu"), C.to("cpu"), STU.to("cpu"))
TYPE3_cuda = (
    nufft3(XYZ.to("cuda"), C.to("cuda"), STU.to("cuda"))
    if torch.cuda.is_available()
    else torch.zeros_like(TYPE3_cpu)
)
TYPE3 = torch.stack((TYPE3_cpu, TYPE3_cuda.to("cpu")))
TYPE3_FI = torch.zeros((len(DEVICES), BATCH_SIZE, 1, N), dtype=C.dtype)
for b in range(BATCH_SIZE):
    for device in DEVICES:
        if device == "cpu":
            TYPE3_FI[0, b] = torch.from_numpy(
                finufft.nufft3d3(
                    XYZ[0].cpu().numpy(),
                    XYZ[1].cpu().numpy(),
                    XYZ[2].cpu().numpy(),
                    C[b, 0].cpu().numpy(),
                    STU[0].cpu().numpy(),
                    STU[1].cpu().numpy(),
                    STU[2].cpu().numpy(),
                )
            )
        elif device == "cuda":
            TYPE3_FI[1, b] = torch.from_numpy(
                finufft.nufft3d3(
                    XYZ[0].cpu().numpy(),
                    XYZ[1].cpu().numpy(),
                    XYZ[2].cpu().numpy(),
                    C[b, 0].cpu().numpy(),
                    STU[0].cpu().numpy(),
                    STU[1].cpu().numpy(),
                    STU[2].cpu().numpy(),
                )
            )


# We need a custom quantile function as torch.quantile works only on tensors with limited sizes
def _quantile(tensor, q, dim=None, keepdim=False):
    """
    Computes the quantile of the input tensor along the specified dimension; by mklacho (https://github.com/pytorch/pytorch/issues/64947).

    Parameters:
    tensor (torch.Tensor): The input tensor.
    q (float): The quantile to compute, should be a float between 0 and 1.
    dim (int): The dimension to reduce. If None, the tensor is flattened.
    keepdim (bool): Whether to keep the reduced dimension in the output.
    Returns:
    torch.Tensor: The quantile value(s) along the specified dimension.
    """
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    # linear interpolation
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value


# Summaries for Absolute Error tables
def _summarize(t: torch.Tensor):
    f = t.flatten()
    minv = f.min().item()
    maxv = f.max().item()
    meanv = f.mean()
    stdv = f.std(unbiased=False)
    q1 = _quantile(f, 0.25)
    q3 = _quantile(f, 0.75)
    medianv = _quantile(f, 0.5).item()
    iqr = q3 - q1
    out_sd = ((f < meanv - stdv) | (f > meanv + stdv)).sum().item()
    out_iqr = ((f < q1 - 1.5 * iqr) | (f > q3 + 1.5 * iqr)).sum().item()
    return {
        "min": minv,
        "max": maxv,
        "mean": meanv.item(),
        "std": stdv.item(),
        "median": medianv,
        "iqr": iqr.item(),
        "out_sd": out_sd,
        "out_iqr": out_iqr,
    }


# Type 1 absolute and relative errors
_type1_abs_cpu = torch.abs(TYPE1[0].detach().cpu() - TYPE1_FI[0].detach().cpu())
_type1_abs_gpu = (
    torch.abs(TYPE1[1].detach().cpu() - TYPE1_FI[1].detach().cpu())
    if torch.cuda.is_available()
    else None
)
_type1_rel_cpu = _type1_abs_cpu / (torch.abs(TYPE1_FI[0].detach().cpu()))
_type1_rel_gpu = (
    _type1_abs_gpu / (torch.abs(TYPE1_FI[1].detach().cpu())) if torch.cuda.is_available() else None
)
# Type 2 absolute and relative errors
_type2_abs_cpu = torch.abs(TYPE2[0].detach().cpu() - TYPE2_FI[0].detach().cpu())
_type2_abs_gpu = (
    torch.abs(TYPE2[1].detach().cpu() - TYPE2_FI[1].detach().cpu())
    if torch.cuda.is_available()
    else None
)
_type2_rel_cpu = _type2_abs_cpu / (torch.abs(TYPE2_FI[0].detach().cpu()))
_type2_rel_gpu = (
    _type2_abs_gpu / (torch.abs(TYPE2_FI[1].detach().cpu())) if torch.cuda.is_available() else None
)
# Type 3 absolute and relative errors
_type3_abs_cpu = torch.abs(TYPE3[0].detach().cpu() - TYPE3_FI[0].detach().cpu())
_type3_abs_gpu = (
    torch.abs(TYPE3[1].detach().cpu() - TYPE3_FI[1].detach().cpu())
    if torch.cuda.is_available()
    else None
)
_type3_rel_cpu = _type3_abs_cpu / (torch.abs(TYPE3_FI[0].detach().cpu()))
_type3_rel_gpu = (
    _type3_abs_gpu / (torch.abs(TYPE3_FI[1].detach().cpu())) if torch.cuda.is_available() else None
)

_type1_cpu = _summarize(_type1_abs_cpu)
_type1_gpu = _summarize(_type1_abs_gpu) if _type1_abs_gpu is not None else None
_type2_cpu = _summarize(_type2_abs_cpu)
_type2_gpu = _summarize(_type2_abs_gpu) if _type2_abs_gpu is not None else None
_type3_cpu = _summarize(_type3_abs_cpu)
_type3_gpu = _summarize(_type3_abs_gpu) if _type3_abs_gpu is not None else None

_type1_rel_cpu = _summarize(_type1_rel_cpu)
_type1_rel_gpu = _summarize(_type1_rel_gpu) if _type1_rel_gpu is not None else None
_type2_rel_cpu = _summarize(_type2_rel_cpu)
_type2_rel_gpu = _summarize(_type2_rel_gpu) if _type2_rel_gpu is not None else None
_type3_rel_cpu = _summarize(_type3_rel_cpu)
_type3_rel_gpu = _summarize(_type3_rel_gpu) if _type3_rel_gpu is not None else None


# Column widths for aligned output
NAME_W, NUM_W, OUT_W = 22, 13, 13


def _fmt_row(name: str, s: dict) -> str:
    outliers = f"{s['out_sd']};{s['out_iqr']}"
    return (
        f"{name:<{NAME_W}} "
        f"{s['min']:{NUM_W}.4e} "
        f"{s['max']:{NUM_W}.4e} "
        f"{s['mean']:{NUM_W}.4e} "
        f"{s['std']:{NUM_W}.4e} "
        f"{s['median']:{NUM_W}.4e} "
        f"{s['iqr']:{NUM_W}.4e} "
        f"{outliers:>{OUT_W}}"
    )


HEADER = (
    f"{'Name':<{NAME_W}} "
    f"{'Min':>{NUM_W}} "
    f"{'Max':>{NUM_W}} "
    f"{'Mean':>{NUM_W}} "
    f"{'StdDev':>{NUM_W}} "
    f"{'Median':>{NUM_W}} "
    f"{'IQR':>{NUM_W}} "
    f"{'Outliers':>{OUT_W}}"
)
SEP = "-" * len(HEADER)

print(f"""
--------------------------- Relative Error of NUFFT Type 1 on random data: {"2" if torch.cuda.is_available() else "1"} tests ---------------------------
{HEADER}
{SEP}
{_fmt_row("torchesnufft (CPU)", _type1_rel_cpu)}
{_fmt_row("torchesnufft (GPU)", _type1_rel_gpu) if _type1_rel_gpu is not None else ""}
{SEP}

--------------------------- Relative Error of NUFFT Type 2 on random data: {"2" if torch.cuda.is_available() else "1"} tests ---------------------------
{HEADER}
{SEP}
{_fmt_row("torchesnufft (CPU)", _type2_rel_cpu)}
{_fmt_row("torchesnufft (GPU)", _type2_rel_gpu) if _type2_rel_gpu is not None else ""}
{SEP}

--------------------------- Relative Error of NUFFT Type 3 on random data: {"2" if torch.cuda.is_available() else "1"} tests ---------------------------
{HEADER}
{SEP}
{_fmt_row("torchesnufft (CPU)", _type3_rel_cpu)}
{_fmt_row("torchesnufft (GPU)", _type3_rel_gpu) if _type3_rel_gpu is not None else ""}
{SEP}
Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
""")
