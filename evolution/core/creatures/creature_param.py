from typing import Callable, List, Tuple, Union, TYPE_CHECKING
import torch
torch.set_grad_enabled(False)
from torch import Tensor


if TYPE_CHECKING:
    from evolution.utils.batched_random import BatchedRandom


from evolution.core.config import Config
from evolution.cuda.cuda_utils import cuda_profile



class CreatureParam:
    """A single trait of a CreatureArray, such as size. This class manages two Tensors, one
    which is referred to as the 'underlying' memory (_data) and the other which is the 
    'current' memory (data). The current memory is a slice of the underlying memory that 
    corresponds to the traits of creatures that are currently alive. We use this schema as it
    minimizes unnecessary copying of traits when creatures die or are born. The CreatureParam
    class also manages the initialization of the underlying memory, the normalization of
    the current memory (although some normalization schemes currently depend on other
    CreatureParams, so they can't be managed by a single instance of this class), the
    generation of child traits from parent traits, and the reindexing of traits when the
    population changes. It also supports lists of Tensors, which function the exact same
    as single Tensors, but every operation is applied to each Tensor in the list. This is useful
    e.g. for the weights of a neural network, where each layer has its own weight matrix."""
    def __init__(self, name: str, shape: Union[List[tuple], tuple], 
                 init: Union[List[Tuple[str,...]], Tuple[str,...], None], 
                 normalize_func: Callable, device: str,
                 mut_idx: Union[int, None]):
        self.name = name
        self._shape = shape
        self.is_list = isinstance(shape, list)
        self.init = init
        self.normalize_func = normalize_func
        self.device = device
        self._data = None  # the "base" data is the underlying memory
        self.data = None   # the current view of the base data that we work with in a single generation
        self.mut_idx = mut_idx  # also indicates whether it's a mutable parameter or not
        
        if mut_idx is not None:
            self.reproduce_type = 'mutable'
        elif init is not None and init[0] == 'zero_':
            self.reproduce_type = 'zeros'
        elif init is not None and init[0] == 'normal_':
            self.reproduce_type = 'randn'
        else:
            self.reproduce_type = 'none'
        
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
        
    def init_base(self, cfg: Config):
        """Initialize underlying memory and current memory. This should only be called
        for root CreatureParams, and not for any offspring/children CreatureParams."""
        self.max_size = cfg.max_creatures
        if self.is_list:
            self._data = [torch.empty(cfg.max_creatures, *shape, device=self.device) for shape in self._shape]
            if self.init is None:
                return
            self.data = [getattr(data[:cfg.start_creatures], init[0])(*init[1:]) 
                            for data, init in zip(self._data, self.init)]
        else:
            self._data = torch.empty(cfg.max_creatures, *self._shape, device=self.device)
            if self.init is None:
                return
            self.data = getattr(self._data[:cfg.start_creatures], self.init[0])(*self.init[1:])
        self.normalize()
            
    def init_base_from_data(self, data: Tensor):
        """Initialize underlying memory and current memory from a given Tensor. This is useful
        for e.g. energy and health, which are initialized as a function of the size. This should
        only be called for root CreatureParams, and not for any offspring/children CreatureParams."""
        self._data[:data.shape[0]] = data  # copy the data in to the appropriate spot
        self.data = self._data[:data.shape[0]]  # set the view to the underlying data
              
    def reproduce_randn(self, rng: 'BatchedRandom'):
        """Generate child traits from the current CreatureParam by sampling random noise."""
        child = CreatureParam(self.name, self._shape, self.init, self.normalize_func, self.device, self.mut_idx)
        child.data = rng.fetch_params(self.name)
        child.normalize()
        return child
    
    def reproduce_zeros(self, num_reproducers):
        """Generate child traits from the current CreatureParam by setting them to zero."""
        child = CreatureParam(self.name, self._shape, self.init, self.normalize_func, self.device, self.mut_idx)
        if self.is_list:
            child.data = [torch.zeros(num_reproducers, *shape, device=self.device) for shape in self._shape]
        else:
            child.data = torch.zeros(num_reproducers, *self._shape, device=self.device)
        return child
            
    #@cuda_profile
    def reproduce_mutable(self, rng: 'BatchedRandom', reproducers: Tensor,
                          mut: Tensor, force_mutable=False):
        """Generate child traits from the current CreatureParam by adding random noise to the
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
        if self.mut_idx is None and not force_mutable:
            return
        if self.mut_idx is not None:
            n_dims = len(self._shape[0]) if self.is_list else len(self._shape)
            mut = mut[:, self.mut_idx].view(-1, *([1]*n_dims))
        child = CreatureParam(self.name, self._shape, None, self.normalize_func, self.device, None)
        if self.is_list:
            perturb = [rng.fetch_params(self.name, i) * mut for i in range(len(self.data))]
            child.data = [d[reproducers] + od for d, od in zip(self.data, perturb)]
        else:
            perturb = rng.fetch_params(self.name) * mut
            child.data = self[reproducers] + perturb
        child.normalize()
        return child
            
    def reproduce(self, rng: 'BatchedRandom', reproducers: Tensor,  num_reproducers: int, mut: Tensor):
        """Select the appropriate reproduction method based on the reproduce_type attribute to
        generate the child traits.
        
        Args: 
            rng: BatchedRandom object for generating random noise.
            reproducers: boolean Tensor (size of current memory) of the creatures that are reproducing.
            num_reproducers: Number of creatures that are reproducing.
            mut: Tensor of mutation rates for each trait.
        """
        if self.reproduce_type == 'zeros':
            return self.reproduce_zeros(num_reproducers)
        elif self.reproduce_type == 'mutable':
            return self.reproduce_mutable(rng, reproducers, mut)
        elif self.reproduce_type == 'randn':
            return self.reproduce_randn(rng)
            
    #@cuda_profile
    def normalize(self):  # technically we should add support for list-type CreatureParams, but we don't use that
        """Apply normalization function to the trait."""
        if self.normalize_func is not None:
            self.normalize_func(self)
            
    def write_new(self, idxs: Tensor, other: 'CreatureParam'):
        """Write data from child CreatureParam (other) to the parent CreatureParam (self). We write
        to the underlying memory here, and then update the current memory later when we call self.reindex.
        
        Args:
            idxs: Tensor of indices (into underlying memory) where we want to write the new creatures.
            other: Child CreatureParam whose data we want to write to the parent CreatureParam.
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
            raise ValueError(f"Cannot iterate over non-list CreatureParam(name={self.name}, shape={self.shape})")
            
    def __getitem__(self, idxs: Tensor):
        return self.data[idxs]
    
    def __setitem__(self, idxs: Tensor, val: Tensor):
        self.data[idxs] = val
        
    def __repr__(self):
        return f"CreatureParam(name={self.name}, data={self.data})"
                
    def long(self) -> Tensor:
        return self.data.long()
    
    def int(self) -> Tensor:
        return self.data.int()
    
    def max(self, *args, **kwargs) -> Tensor:
        return self.data.max(*args, **kwargs)
    
    def argmax(self, *args, **kwargs) -> Tensor:
        return self.data.argmax(*args, **kwargs)
    
    def unsqueeze(self, *args, **kwargs) -> Tensor:
        return self.data.unsqueeze(*args, **kwargs)
    
    def dim(self) -> int:
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
    
    def __radd__(self, other) -> Tensor:
        return other + self.data
    
    def __le__(self, other) -> Tensor:
        return self.data <= other
    
    def __pow__(self, other) -> Tensor:
        return self.data ** other
    
    def __iadd__(self, other) -> 'CreatureParam':
        self.data += other
        return self
        
    def __isub__(self, other) -> 'CreatureParam':
        self.data -= other
        return self
    
    def __itruediv__(self, other) -> 'CreatureParam':
        self.data /= other
        return self
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if isinstance(args[0], list):
            args = [[a.data if isinstance(a, CreatureParam) else a for a in args[0]]]
            # print([type(t) for t in args], func)
        else:
            # print(args, func)
            args = [a.data if isinstance(a, CreatureParam) else a for a in args]
            # print('done', args, func, [type(t) for t in args])
        # print([type(t) for t in args], func)
        ret = func(*args, **kwargs)
        return ret
        