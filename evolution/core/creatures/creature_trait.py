from typing import Callable, Iterable, List, Tuple, Union, TYPE_CHECKING
import torch
torch.set_grad_enabled(False)
from torch import Tensor
from enum import Enum


if TYPE_CHECKING:
    from evolution.utils.batched_random import BatchedRandom


from evolution.core.config import Config
from evolution.cuda.cuda_utils import cuda_profile


class InitializerStyle(Enum):
    """Enum for the different types of initializers that can be used for a CreatureTrait."""
    OTHER_DEPENDENT = 'other_dependent'
    FILLABLE = 'fillable'
    MUTABLE = 'mutable'
    FORCE_MUTABLE = 'force_mutable'


class Initializer:
    def __init__(self, **kwargs):
        self.style: InitializerStyle = None
        self.mut_idx = None
        self.func = None
        self.name = None
        self.args = None
        self.is_list = False
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        if self.args is not None and len(self.args) == 1 and isinstance(self.args[0], Iterable):  # unwrap list of args
            self.args = self.args[0]
            
        if self.args is not None and isinstance(self.args, Iterable) and \
            len(self.args) > 0 and isinstance(self.args[0], Iterable):
            self.is_list = True
    
    @staticmethod
    def mutable(mut_idx: int, name: str, *args):
        """Create an Initializer object for a mutable trait. 
        Args:
            mut_idx: index into the creature's mut_rate tensor that corresponds to this trait.
            name: Torch name of the intializer function when first seeding the world (non-reproductive).
            args: Arguments to pass to the initializer function."""
        return Initializer(style=InitializerStyle.MUTABLE, mut_idx=mut_idx, name=name, args=args)
    
    
    @staticmethod
    def fillable(name: str, *args):
        """Create an Initializer object for a fillable trait.
        Args:
            name: Torch name of the intializer function (e.g. ('zeros_', 'normal_')).
            args: Arguments to pass to the initializer function"""
        return Initializer(style=InitializerStyle.FILLABLE, name=name, args=args)
    
    @staticmethod
    def other_dependent(name: str, func: Callable):
        """Create an Initializer object for a trait that depends on another trait. (e.g. energy and health
        are always initialized as functions of size).
        
        Args:
            name: Name of the trait that this trait depends on.
            func: Function that takes the other trait as an argument and returns the initialized trait."""
        return Initializer(style=InitializerStyle.OTHER_DEPENDENT, name=name, func=func)
    
    @staticmethod
    def force_mutable(name: str, *args):
        """Create an Initializer object for a trait that is treated as mutable even if it isn't (e.g. position).
        
        Args:
            name: Torch name of the intializer function when first seeding the world (non-reproductive).
            args: Arguments to pass to the initializer function."""
        return Initializer(style=InitializerStyle.FORCE_MUTABLE, name=name, args=args)
    
    def __call__(self, arg, i=None):
        if self.style == InitializerStyle.OTHER_DEPENDENT:
            return self.func(arg)
        elif self.style in [InitializerStyle.FILLABLE, InitializerStyle.FORCE_MUTABLE, InitializerStyle.MUTABLE]:
            if self.is_list:
                return getattr(arg, self.name)(*(self.args[i]))
            return getattr(arg, self.name)(*self.args)


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
    population changes. It also supports lists of Tensors, which function the exact same
    as single Tensors, but every operation is applied to each Tensor in the list. This is useful
    e.g. for the weights of a neural network, where each layer has its own weight matrix."""
    def __init__(self, shape: List[tuple] | tuple, 
                 init: Initializer | List[Initializer], normalize_func: Callable, device: str):
        self._shape = shape
        self.init = init
        self.is_list = init.is_list
        self.normalize_func = normalize_func
        self.device = device
        self._data = None  # the "base" data is the underlying memory
        self.data = None   # the current view of the base data that we work with in a single generation
        
    @property
    def population(self):
        if self.data is None:
            return 0
        return self.shape[0]
        
    @property
    def shape(self):
        if self.is_list:
            return [d.shape for d in self.data]
        return self.data.shape
    
    def init_underlying(self, max_creatures: int):
        """Allocate underlying memory. This should only be called
        for root CreatureTraits, and not for any offspring/children CreatureTraits."""
        if self.is_list:
            self._data = [torch.empty(max_creatures, *shape, device=self.device) for shape in self._shape]
        else:
            self._data = torch.empty(max_creatures, *self._shape, device=self.device)
        
    def init_base(self, cfg: Config):
        """Initialize underlying memory and current memory. This should only be called
        for root CreatureTraits, and not for any offspring/children CreatureTraits."""
        self.max_size = cfg.max_creatures
        if self.is_list:
            self._data = [torch.empty(cfg.max_creatures, *shape, device=self.device) for shape in self._shape]
            if self.init.style == InitializerStyle.OTHER_DEPENDENT:
                return
            self.data = [self.init(data[:cfg.start_creatures], i) for i, data in enumerate(self._data)]
        else:
            self._data = torch.empty(cfg.max_creatures, *self._shape, device=self.device)
            if self.init.style == InitializerStyle.OTHER_DEPENDENT:
                return
            self.data = self.init(self._data[:cfg.start_creatures])
        self.normalize()
        
    def unset_data(self):
        self.data = None
        self._data = None
        
    def init_base_from_state_dict(self, data: Union[Tensor, List[Tensor]], cfg: Config, start: int, new_size: int):
        """Instantiate and initialize current memory from a given state_dict. This is useful
        for loading a saved model."""
        self.init_underlying(cfg.max_creatures)
        if self.is_list:
            for i, d in enumerate(data):
                self._data[i][start: start+new_size] = d.to(self.device)
        else:
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
            if self.is_list:
                child.data = [self.init(torch.empty(num_reproducers, *shape, device=self.device), i)
                              for i, shape in enumerate(self._shape)]
            else:
                child.data = self.init(torch.empty(num_reproducers, *self._shape, device=self.device))
        return child
            
    #@cuda_profile
    def reproduce_mutable(self, name: str, rng: 'BatchedRandom', reproducers: Tensor, mut: Tensor):
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
        if self.init.mut_idx is not None:
            n_dims = len(self._shape[0]) if self.is_list else len(self._shape)
            mut = mut[:, self.init.mut_idx].view(-1, *([1]*n_dims))
        child = CreatureTrait(self._shape, self.init, self.normalize_func, self.device)
        if self.is_list:
            perturb = [rng.fetch_params(name, i) * mut for i in range(len(self.data))]
            child.data = [d[reproducers] + od for d, od in zip(self.data, perturb)]
        else:
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
        elif self.init.style == InitializerStyle.MUTABLE:
            return self.reproduce_mutable(name, rng, reproducers, mut)
            
    #@cuda_profile
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
        if self.is_list:
            for d, od in zip(self._data, other.data):
                try:
                    d[idxs] = od
                except RuntimeError:
                    print(d.shape, idxs.shape, od.shape)
        else:
            try:
                self._data[idxs] = other.data
            except RuntimeError:
                print(self._data.shape, idxs.shape, other.data.shape)
            
    def rearrange_old_data(self, outer_idxs: Tensor, inner_idxs: Tensor):
        """Move creatures that are outside the best window to the inside of the best window.
        We write to the underlying memory here, and then update the current memory later
        when we call self.reindex.
        
        Args:
            outer_idxs: Tensor of indices (into underlying memory) of creatures that are outside the best window.
            inner_idxs: Tensor of indices (into underlying memory) where we want to move them to, inside 
                        the best window
        """
        if self.is_list:
            for d in self._data:
                d[inner_idxs] = d[outer_idxs]
        else:
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
            if self.is_list:
                self.data = [d[sl] for d in self._data]
            else:
                self.data = self._data[sl]
                
    def __iter__(self):
        if self.is_list:
            return iter(self.data)
        else:
            raise ValueError(f"Cannot iterate over non-list CreatureTrait(shape={self.shape})")
            
    def __getitem__(self, idxs: Tensor):
        return self.data[idxs]
    
    def __setitem__(self, idxs: Tensor, val: Tensor):
        self.data[idxs] = val
        
    def __repr__(self):
        return f"CreatureTrait(data={self.data})"
                
    def long(self) -> Tensor:
        return self.data.long()
    
    def int(self) -> Tensor:
        return self.data.int()
    
    def max(self, *args, **kwargs) -> Tensor:
        return self.data.max(*args, **kwargs)
    
    def min(self, *args, **kwargs) -> Tensor:
        return self.data.min(*args, **kwargs)
    
    def argmax(self, *args, **kwargs) -> Tensor:
        return self.data.argmax(*args, **kwargs)
    
    def argmin(self, *args, **kwargs) -> Tensor:
        return self.data.argmin(*args, **kwargs)
    
    def unsqueeze(self, *args, **kwargs) -> Tensor:
        return self.data.unsqueeze(*args, **kwargs)
    
    def dim(self) -> int:
        if self.is_list:
            return [d.dim() for d in self.data]
        return self.data.dim()
    
    def data_ptr(self) -> int:
        return self.data.data_ptr()
    
    def stride(self, *args, **kwargs) -> Tuple[int]:
        return self.data.stride(*args, **kwargs)
    
    def element_size(self) -> int:
        return self.data.element_size()
    
    def size(self, *args, **kwargs) -> Tuple[int]:
        return self.data.size(*args, **kwargs)
    
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
        
    def __isub__(self, other) -> 'CreatureTrait':
        self.data -= other
        return self
    
    def __itruediv__(self, other) -> 'CreatureTrait':
        self.data /= other
        return self
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
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
        