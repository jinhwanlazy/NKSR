import torch 
from typing import Optional, Tuple, Union


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def create_scatter_output(src: torch.Tensor, index: torch.Tensor, dim: int,
                 dim_size: Union[int, None]):
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return torch.zeros(size, dtype=src.dtype, device=src.device)

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        out = create_scatter_output(src, index, dim, dim_size)
    return out.scatter_add_(dim, index, src)


def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    index = broadcast(index, src, dim)
    include_self = out is not None
    if out is None:
        out = create_scatter_output(src, index, dim, dim_size)
    return out.scatter_reduce_(dim, index, src, reduce="amin", include_self=include_self)

def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    index = broadcast(index, src, dim)
    include_self = out is not None
    if out is None:
        out = create_scatter_output(src, index, dim, dim_size)
    return out.scatter_reduce_(dim, index, src, reduce="amax", include_self=include_self)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out

def scatter_std(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None,
                unbiased: bool = True) -> torch.Tensor:

    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = (src - mean.gather(dim, index))
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out
