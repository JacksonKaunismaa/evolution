from typing import Tuple, TYPE_CHECKING
import torch
from torch import Size, Tensor

from evolution.core.config import Config
from evolution.core.creatures.trait_initializer import Initializer, InitializerStyle, Normalizer
from evolution.cuda.cuda_utils import cuda_profile

if TYPE_CHECKING:
    from evolution.utils.batched_random import BatchedRandom


class CreatureTrait:
    """A single trait of a CreatureArray, such as size. This class manages two Tensors, one
    which is referred to as the 'underlying' memory (_data) and the other which is the
    'current' memory (data). The current memory is a slice of the underlying memory that
    corresponds to the traits of creatures that are currently alive. We use this schema as it
    minimizes unnecessary copying of traits when creatures die or are born. The CreatureTrait
    class also manages the initialization of the underlying memory, the normalization of
    the current memory (although some normalization schemes currently depend on other
    CreatureTraits, so they can't be managed by a single instance of this class), the
    generation of child traits from parent traits, and the reindexing of traits when the
    population changes."""
    def __init__(self, shape: Tuple[int, ...], init: Initializer,
                 normalize_func: Normalizer | None, device: torch.device):
        self._shape = shape
        self.init = init
        self.normalize_func = normalize_func
        self.device = device
        self._data: Tensor = None  # the "base" data is the underlying memory
        self.data: Tensor = None   # the current view of the base data that we work with in a single generation
        self.max_size: int = None  # the maximum number of creatures that can be stored in the underlying memory

    @property
    def population(self) -> int:
        if self.data is None:
            return 0
        return self.shape[0]

    @property
    def shape(self) -> Size:
        return self.data.shape

    def init_underlying(self, max_creatures: int):
        """Allocate underlying memory. This should only be called
        for root CreatureTraits, and not for any offspring/children CreatureTraits."""
        self._data = torch.empty(max_creatures, *self._shape, device=self.device)

    def init_base(self, cfg: Config):
        """Initialize underlying memory and current memory. This should only be called
        for root CreatureTraits, and not for any offspring/children CreatureTraits."""
        self.max_size = cfg.max_creatures
        self._data = torch.empty(cfg.max_creatures, *self._shape, device=self.device)
        if self.init.style == InitializerStyle.OTHER_DEPENDENT:
            return
        self.data = self.init(self._data[:cfg.start_creatures])
        self.normalize()

    def unset_data(self):
        """Unset both underlying and current memory, so that it can be garbage collected."""
        self.data = None
        self._data = None

    def init_base_from_state_dict(self, data: Tensor, cfg: Config, start: int, new_size: int):
        """Instantiate and initialize current memory from a given state_dict. This is useful
        for loading a saved model."""
        self.init_underlying(cfg.max_creatures)
        self._data[start: start+new_size] = data.to(self.device)
        self.reindex(start, new_size)

    def init_base_from_other(self, other: 'CreatureTrait'):
        """Initialize underlying memory and current memory from a given CreatureTrait. This is useful
        for e.g. energy and health, which are initialized as a function of the size. This should
        only be called for root CreatureTraits, and not for any offspring/children CreatureTraits."""
        data = self.init(other.data)
        self._data[:data.shape[0]] = data  # copy the data in to the appropriate spot
        self.data = self._data[:data.shape[0]] # set the view to the underlying data

    def reproduce_from_other(self, other: 'CreatureTrait'):
        """Deal with initialization that depends on another CreatureTrait. This is used for e.g.
        energy and health, which are initialized as a function of the size."""
        child = CreatureTrait(self._shape, self.init, self.normalize_func, self.device)
        child.data = self.init(other.data)
        return child

    def reproduce_fillable(self, name: str, rng: 'BatchedRandom', num_reproducers: int):
        """Generate child traits from the current CreatureTrait by either sampling from random normal, or
        using some default Pytorch initialization function. The child traits are then normalized."""
        child = CreatureTrait(self._shape, self.init, self.normalize_func, self.device)
        if self.init.name == 'normal_':
            child.data = rng.fetch_params(name) * self.init.args[1] + self.init.args[0]   # * std + mean
            child.normalize()
        else:
            child.data = self.init(torch.empty(num_reproducers, *self._shape, device=self.device))
        return child

    @cuda_profile
    def reproduce_mutable(self, name: str, rng: 'BatchedRandom', reproducers: Tensor, mut: Tensor | float):
        """Generate child traits from the current CreatureTrait by adding random noise to the
        parent's traits, and then normalizing. The noise is scaled by the mut Tensor, which is
        the mutation rate of that particular trait. If force_mutable is True, then the trait
        is treated as mutable even if isn't (e.g. for position this is useful, since we can consider
        a child's position to be a 'mutation' of the parent's position).

        Args:
            rng: BatchedRandom object for generating random noise.
            reproducers: boolean Tensor (size of current memory) of the creatures that are reproducing.
            mut: Tensor of mutation rates for each trait.
            force_mutable: If True, treat the trait as mutable even if it isn't.
        """
        if isinstance(mut, Tensor):
            mut = mut[:, self.init.mut_idx].view(-1, *([1]*(self.data.dim()-1)))
        child = CreatureTrait(self._shape, self.init, self.normalize_func, self.device)
        perturb = rng.fetch_params(name) * mut
        child.data = self[reproducers] + perturb
        child.normalize()
        return child

    def reproduce(self, name: str, rng: 'BatchedRandom', reproducers: Tensor,  num_reproducers: int, mut: Tensor):
        """Select the appropriate reproduction method based on the reproduce_type attribute to
        generate the child traits.

        Args:
            rng: BatchedRandom object for generating random noise.
            reproducers: boolean Tensor (size of current memory) of the creatures that are reproducing.
            num_reproducers: Number of creatures that are reproducing.
            mut: Tensor of mutation rates for each trait.
        """
        if self.init.style == InitializerStyle.FILLABLE:
            return self.reproduce_fillable(name, rng, num_reproducers)

        if self.init.style == InitializerStyle.MUTABLE:
            return self.reproduce_mutable(name, rng, reproducers, mut)

        if self.init.style == InitializerStyle.FORCE_MUTABLE:
            return self.reproduce_mutable(name, rng, reproducers, self.init.mut)

        return None

    @cuda_profile
    def normalize(self):  # technically we should add support for list-type CreatureTraits, but we don't use that
        """Apply normalization function to the trait."""
        if self.normalize_func is not None:
            self.normalize_func(self)

    def write_new(self, idxs: Tensor, other: 'CreatureTrait'):
        """Write data from child CreatureTrait (other) to the parent CreatureTrait (self). We write
        to the underlying memory here, and then update the current memory later when we call self.reindex.

        Args:
            idxs: Tensor of indices (into underlying memory) where we want to write the new creatures.
            other: Child CreatureTrait whose data we want to write to the parent CreatureTrait.
        """
        try:
            self._data[idxs] = other.data
        except RuntimeError:
            print(self._data.shape, idxs.shape, other.data.shape)
            raise

    def rearrange_old_data(self, outer_idxs: Tensor, inner_idxs: Tensor):
        """Move creatures that are outside the best window to the inside of the best window.
        We write to the underlying memory here, and then update the current memory later
        when we call self.reindex.

        Args:
            outer_idxs: Tensor of indices (into underlying memory) of creatures that are outside the best window.
            inner_idxs: Tensor of indices (into underlying memory) where we want to move them to, inside
                        the best window
        """
        self._data[inner_idxs] = self._data[outer_idxs]

    def reindex(self, start, new_size):
        """Slice underlying memory to set current memory to the current set of active creatures.

        Args:
            start: Index of the first creature in the current set of active creatures.
            new_size: Number of active creatures.
        """
        if new_size == self.max_size:  # optimization when the population is at max size, we don't have to slice
            self.data = self._data
        else:
            sl = slice(start, start + new_size)
            self.data = self._data[sl]

    def __getitem__(self, idxs):
        return self.data[idxs]

    def __setitem__(self, idxs, val: Tensor):
        self.data[idxs] = val

    def __repr__(self):
        return f"CreatureTrait(data={self.data})"

    def __getattr__(self, name):
        """Delegate attribute access to the underlying Tensor. This is only called if the attribute
        doesn't already exist in CreatureTrait.__dict__, so no need to check that here."""
        if self.data is not None and hasattr(self.data, name):
            return getattr(self.data, name)
        raise AttributeError(f"'CreatureTrait' object has no attribute '{name}'")

    def __mul__(self, other) -> Tensor:
        return self.data * other

    def __add__(self, other) -> Tensor:
        return self.data + other

    def __sub__(self, other) -> Tensor:
        return self.data - other

    def __radd__(self, other) -> Tensor:
        return other + self.data

    def __le__(self, other) -> Tensor:
        return self.data <= other

    def __pow__(self, other) -> Tensor:
        return self.data ** other

    def __iadd__(self, other) -> 'CreatureTrait':
        self.data += other
        return self

    def __truediv__(self, other) -> Tensor:
        return self.data / other

    def __isub__(self, other) -> 'CreatureTrait':
        self.data -= other
        return self

    def __itruediv__(self, other) -> 'CreatureTrait':
        self.data /= other
        return self


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None): # pylint: disable=unused-argument
        if kwargs is None:
            kwargs = {}
        if isinstance(args[0], list):
            args = [[a.data if isinstance(a, CreatureTrait) else a for a in args[0]]]
            # print([type(t) for t in args], func)
        else:
            # print(args, func)
            args = [a.data if isinstance(a, CreatureTrait) else a for a in args]
            # print('done', args, func, [type(t) for t in args])
        # print([type(t) for t in args], func)
        ret = func(*args, **kwargs)
        return ret
