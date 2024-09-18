from typing import TYPE_CHECKING, Callable, Optional, Tuple
from enum import Enum

from torch import Tensor

if TYPE_CHECKING:
    from .creature_trait import CreatureTrait


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
    def mutable(cls, name: str, *args) -> 'Initializer':
        """Create an Initializer object for a mutable trait. mut_idx, which is required for this
        type of initializer, will be set later, by the CreatureArray class, as this avoids needing
        to provide a manual index for each trait.
        Args:
            name: Torch name of the intializer function when first seeding the world (non-reproductive).
            args: Arguments to pass to the initializer function."""
        return cls(style=InitializerStyle.MUTABLE, name=name, args=args)


    @classmethod
    def fillable(cls, name: str, *args: Optional[Tuple]) -> 'Initializer':
        """Create an Initializer object for a fillable trait.
        Args:
            name: Torch name of the intializer function (e.g. ('zeros_', 'normal_')).
            args: Arguments to pass to the initializer function"""
        return cls(style=InitializerStyle.FILLABLE, name=name, args=args)

    @classmethod
    def other_dependent(cls, name: str, func: Callable, *args: Optional[Tuple]) -> 'Initializer':
        """Create an Initializer object for a trait that depends on another trait. (e.g. energy and health
        are always initialized as functions of size).

        Args:
            name: Name of the trait that this trait depends on.
            func: Function that takes the other trait as an argument and returns the initialized trait.
            args: Extra arguments to pass to the function.
        """
        return cls(style=InitializerStyle.OTHER_DEPENDENT, name=name, func=func, args=args)

    @classmethod
    def force_mutable(cls, name: str, mut: float, *args) -> 'Initializer':
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


class Normalizer:
    def __init__(self, func: Callable, *args: Optional[Tuple], **kwargs: Optional[dict]):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, arg: 'CreatureTrait'):
        """Operate in place to normalize the trait."""
        self.func(arg, *self.args, **self.kwargs)
