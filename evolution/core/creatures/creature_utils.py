import numpy as np
import torch
from torch import Tensor

from evolution.core.config import Config
from evolution.cuda.cuda_utils import cuda_profile

from .creature_trait import CreatureTrait


def normalize_rays(rays: CreatureTrait, cfg: Config):
    rays[..., :2] /= torch.norm(rays[..., :2], dim=2, keepdim=True)
    rays[..., 2] = torch.clamp(torch.abs(rays[..., 2]),
                cfg.ray_dist_range[0],
                cfg.ray_dist_range[1])

def normalize_head_dirs(head_dirs: CreatureTrait):
    head_dirs /= torch.norm(head_dirs, dim=1, keepdim=True)

def eat_amt(sizes: CreatureTrait, cfg: Config) -> Tensor:
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

def clamp(x: Tensor, min_: float, max_: float):
    x[:] = torch.clamp(x, min=min_, max=max_)

@cuda_profile
def vectorized_hsl_to_rgb(h,l):
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
    return torch.stack((r, g, b), axis=-1)


def hsv_spiral(t: Tensor) -> Tensor:
    """Given a Tensor t of shape (N) with values between 1 and 255, return an RGB color.
    This parametric curve goes through HSL space to cover a wide range of distinct colors.
    0 must map to (0, 0, 0)"""
    h = t / 255
    # s = 1.0
    l = torch.where(t > 0, 0.5 + 0.15*(t*0.2).cos(), 0.0)
    return vectorized_hsl_to_rgb(h,l)
