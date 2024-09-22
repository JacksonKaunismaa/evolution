from operator import mul
from functools import reduce
from typing import Dict
import torch

from evolution.core.creatures.trait_initializer import InitializerStyle
from evolution.core.creatures.creature_trait import CreatureTrait
from evolution.core.benchmarking import Profile

class BatchedRandom:
    """Class to manage batched random generation of parameters for creatures.
    It generates a block of uniform normal numbers, and then it is sliced according
    to the shape of the passed-in trait name, reshaped to the appropriate shape
    and returned."""
    def __init__(self, params: Dict[str, CreatureTrait], device: torch.device):
        self.param_shapes = {}  # shape of the parameter
        self.param_elems = {}    # how many elements of the contiguous random block to take
        for k, v in params.items():
            # Only generate noise for parameters that would need it
            if not (v.init.style in [InitializerStyle.MUTABLE, InitializerStyle.FORCE_MUTABLE] or
               v.init.style == InitializerStyle.FILLABLE and v.init.name == 'normal_'):
                continue
            v_shape = v._shape
            self.param_shapes[k] = v_shape
            if len(v_shape) == 0:
                self.param_elems[k] = 1
            else:
                self.param_elems[k] = reduce(mul, v_shape, 1)

        self.gen_size = sum(self.param_elems.values())
        self.device = device
        self.buffer: torch.Tensor = None
        self.idx = 0

    @Profile.cuda_profile
    def generate(self, num):
        """Generate a new block of random numbers."""
        self.buffer = torch.randn(num, self.gen_size, device=self.device)
        self.idx = 0

    @Profile.cuda_profile
    def fetch_params(self, param):
        """Fetch the next block of random numbers for the given parameter name."""
        acc_size = self.param_elems[param]
        shape = self.param_shapes[param]

        result = self.buffer[:, self.idx:self.idx+acc_size].view(-1, *shape)
        self.idx += acc_size
        return result
