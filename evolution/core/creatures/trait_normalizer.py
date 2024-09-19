from typing import TYPE_CHECKING, Callable, Optional, Tuple
import torch
from torch import Tensor

from evolution.core.config import Config

if TYPE_CHECKING:
    from evolution.core.creatures.creature_trait import CreatureTrait


def normalize_rays(rays: 'CreatureTrait', cfg: Config):
    rays[..., :2] /= torch.norm(rays[..., :2], dim=2, keepdim=True)
    rays[..., 2] = torch.clamp(torch.abs(rays[..., 2]),
                cfg.ray_dist_range[0],
                cfg.ray_dist_range[1])

def normalize_head_dirs(head_dirs: 'CreatureTrait'):
    head_dirs /= torch.norm(head_dirs, dim=1, keepdim=True)

def clamp(x: 'CreatureTrait', min_: float, max_: float):
    x[:] = torch.clamp(x, min=min_, max=max_)


class Normalizer:
    def __init__(self, func: Callable, *args: Optional[Tuple], **kwargs: Optional[dict]):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, arg: 'CreatureTrait'):
        """Operate in place to normalize the trait."""
        self.func(arg, *self.args, **self.kwargs)
