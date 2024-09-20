from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple
from enum import Enum
import numpy as np
import torch
from torch import Tensor

from evolution.core.config import Config
from evolution.cuda.cuda_utils import cuda_profile

if TYPE_CHECKING:
    from evolution.cuda.cu_algorithms import CUDAKernelManager


def eat_amt(sizes: Tensor, cfg: Config) -> Tensor:
    if cfg.eat_pct_scaling[0] == 'linear':
        size_pct = (sizes - cfg.size_range[0]) / (cfg.size_range[1] - cfg.size_range[0])
    elif cfg.eat_pct_scaling[0] == 'log':
        size_pct = torch.log(sizes / cfg.size_range[0]) / np.log(cfg.size_range[1] / cfg.size_range[0])
    else:
        raise ValueError(f"Unrecognized eat_pct_scaling: {cfg.eat_pct_scaling[0]}")

    if cfg.eat_pct_scaling[1] == 'linear':  # pylint: disable=no-else-return
        return cfg.eat_pct[0] + size_pct * (cfg.eat_pct[1] - cfg.eat_pct[0])
    elif cfg.eat_pct_scaling[1] == 'log':
        return cfg.eat_pct[0] * torch.exp(size_pct * np.log(cfg.eat_pct[1] / cfg.eat_pct[0]))
    else:
        raise ValueError(f"Unrecognized eat_pct_scaling: {cfg.eat_pct_scaling[1]}")

@cuda_profile
def vectorized_hsl_to_rgb(h: Tensor, l: Tensor) -> Tensor:
    # s is assumed to be 1
    def hue2rgb( v1, v2, vH):
        vH = torch.where(vH < 0, vH + 1, vH)
        vH = torch.where(vH > 1, vH - 1, vH)
        s1 = torch.where(vH <= 1/6, v1 + (v2-v1)*6.0*vH, 0)
        s2 = torch.where((1/6 < vH) & (vH <= 1/2), v2, 0)
        s3 = torch.where((1/2 < vH) & (vH <= 2/3), v1 + (v2-v1)*((2.0/3.0)-vH)*6.0, 0)
        s4 = torch.where(vH > 2/3, v1, 0)
        return s1 + s2 + s3 + s4

    var_2 = torch.where(l < 0.5, l * 2, 1.0)
    var_1 = 2.0 * l - var_2

    r = 255 * hue2rgb( var_1, var_2, h + ( 1.0 / 3.0 ) )
    g = 255 * hue2rgb( var_1, var_2, h )
    b = 255 * hue2rgb( var_1, var_2, h - ( 1.0 / 3.0 ) )
    return torch.stack((r, g, b), dim=-1)


def hsv_spiral(scalar_color: Tensor) -> Tensor:
    """Given a Tensor scalar_color of shape (N) with values between 1 and 255, return an RGB color.
    This parametric curve goes through HSL space to cover a wide range of distinct colors.
    0 must map to (0, 0, 0)"""
    h = scalar_color / 255
    # s = 1.0
    l = torch.where(scalar_color > 0, 0.5 + 0.15*(scalar_color*0.2).cos(), 0.0)
    return vectorized_hsl_to_rgb(h,l)

@cuda_profile
def cuda_hsv_spiral(scalar_colors: Tensor, kernels: 'CUDAKernelManager') -> Tensor:
    """Given a Tensor scalar_colors of shape (N) with values between 1 and 255, return an RGB color.
    This parametric curve goes through HSL space to cover a wide range of distinct colors.
    0 must map to (0, 0, 0)"""
    n = scalar_colors.shape[0]
    visual_colors = torch.empty((n, 3), device=scalar_colors.device, dtype=torch.float32)

    block_size = 512
    grid_size = n // block_size + 1

    kernels('hsv_spiral',
            grid_size, block_size,
            scalar_colors, visual_colors, n)

    return visual_colors



class InitializerStyle(Enum):
    """Enum for the different types of initializers that can be used for a CreatureTrait."""
    OTHER_DEPENDENT = 'other_dependent'
    FILLABLE = 'fillable'
    MUTABLE = 'mutable'
    FORCE_MUTABLE = 'force_mutable'


class Initializer:
    """Class for managing the different ways we initialize CreatureTraits."""
    def __init__(self, **kwargs):
        self.style: InitializerStyle
        self.mut_idx: int = None
        self.func: Callable
        self.name: str
        self.args: Tuple
        self.mut: float
        for k,v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def mutable(cls, name: str, *args: Any) -> 'Initializer':
        """Create an Initializer object for a mutable trait. mut_idx, which is required for this
        type of initializer, will be set later, by the CreatureArray class, as this avoids needing
        to provide a manual index for each trait.
        Args:
            name: Torch name of the intializer function when first seeding the world (non-reproductive).
            args: Arguments to pass to the initializer function."""
        return cls(style=InitializerStyle.MUTABLE, name=name, args=args)


    @classmethod
    def fillable(cls, name: str, *args: Any) -> 'Initializer':
        """Create an Initializer object for a fillable trait.
        Args:
            name: Torch name of the intializer function (e.g. ('zeros_', 'normal_')).
            args: Arguments to pass to the initializer function"""
        return cls(style=InitializerStyle.FILLABLE, name=name, args=args)

    @classmethod
    def other_dependent(cls, name: str, func: Callable, *args: Any) -> 'Initializer':
        """Create an Initializer object for a trait that depends on another trait. (e.g. energy and health
        are always initialized as functions of size).

        Args:
            name: Name of the trait that this trait depends on.
            func: Function that takes the other trait as an argument and returns the initialized trait.
            args: Extra arguments to pass to the function.
        """
        return cls(style=InitializerStyle.OTHER_DEPENDENT, name=name, func=func, args=args)

    @classmethod
    def force_mutable(cls, name: str, mut: float, *args: Any) -> 'Initializer':
        """Create an Initializer object for a trait that is treated as mutable even if it isn't (e.g. position).

        Args:
            name: Torch name of the intializer function when first seeding the world (non-reproductive).
            args: Arguments to pass to the initializer function."""
        return cls(style=InitializerStyle.FORCE_MUTABLE, name=name, mut=mut, args=args)

    def __call__(self, arg) -> Tensor:
        if self.style == InitializerStyle.OTHER_DEPENDENT:
            return self.func(arg, *self.args)

        if self.style in [InitializerStyle.FILLABLE, InitializerStyle.FORCE_MUTABLE, InitializerStyle.MUTABLE]:
            return getattr(arg, self.name)(*self.args)

        raise ValueError(f"Invalid initializer style: {self.style}")
