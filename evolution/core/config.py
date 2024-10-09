import dataclasses
from typing import Any, Dict, Tuple
from enum import Enum
import torch

class ConfigFunction:   # for replacing functions in regular python code
    def __init__(self, name, *args):
        self.name = name
        self.func = getattr(self, name)
        self.extra_args = args

    @staticmethod
    def pow(x, m, a):
        return m * x**a

    @staticmethod
    def square_add(x, m, a):
        return m*x**2 + a

    @staticmethod
    def square(x, m):
        return m*x**2

    @staticmethod
    def linear(x, m):
        return m*x

    @staticmethod
    def bilinear(x, y, m):
        return x * y * m

    @staticmethod
    def linear_frac(x, y, m):
        return x / (1 + y) * m

    @staticmethod
    def abs_exp(x, y, m):
        return (1 - torch.exp(-torch.abs(x) * y)) * m

    def __call__(self, *args):
        return self.func(*args, *self.extra_args)

    def __repr__(self):
        return f'{self.name}({self.extra_args})'

class FunctionExpression:  # for code preprocessor in cu_algorithms
    def __init__(self, symbols, expr):
        self.symbols = symbols
        self.expr = expr

    def __repr__(self):
        return f'{self.symbols} -> {self.expr}'


class Scaling(Enum):
    LINEAR = 'linear'
    LOG = 'log'


@dataclasses.dataclass
class Config:
    """
    Configuration parameters for the simulation environment.
    """
    ### World
    size: int = 300  # width of the world in units
    start_creatures: int = 128 # number of creatures at the beginning
    max_creatures: int = 1024 # maximum number of creatures


    ### Food
    init_food_scale: float = 8.0    # food will be initialized uniformly to be in [0, init_food_scale]
    eat_pct: Tuple[float, float] = (0.001, 0.1) # (pct of food a min. size creature can in a cell, pct of food a max. size creature can eat in a cell)
    # see core.creatures.trait_initializer.eat_amt, https://www.desmos.com/calculator/kzkq5kfh0x
    eat_pct_scaling: Tuple[Scaling, Scaling] = (Scaling.LOG, Scaling.LOG) # scaling for (size, eat_pct)
    food_cover_decr: float = 0.2  # if a creature occupies a cell, the food in that cell be decreased by this amount each step
    food_cover_decr_pct: float = 10. # actual_food_decr = food_cover_decr * float_cover_decr_pct * creature_eat_pct
    food_cover_decr_incr_amt: float = 1.5e-5  # how much food_cover_decr increase per generation when increase is enabled
    neg_food_eat_mul: float = 0.1  # if food is negative, creature eating is scaled by this amount
    max_food: float = 15.     # maximum amount of food in a cell (decays past this value)
    food_decay_rate: float = 0.05 # how fast food decays (when over max_food)
    food_growth_rate: float = 10.0  # scale to apply to food growth rate
    food_recovery_rate: float = 30.0  # scale to apply to food growth rate when its negative
    food_health_recovery: float = 0.1  # multiple of food eaten that creatures gain as health
    food_step_size: float = 3e-5  # how much food grows each step


    ### Creatures

    ## brains
    brain_size: Tuple[int, ...] = (30, 40) # size of hidden layers of brains
    mem_size: int = 10 # how many output values/input values can be used for memory

    ## vision
    food_sight: int = 3 # how many squares (grid) away creatures can see food (1 => a 3x3 window centered on them)
    num_rays: int = 32 # number of rays creatures can see with
    ray_dist_range: Tuple[float, float] = (1.5, 5.0) # minimum distance of rays (as multiple of size)

    # if False, when creatures rotate, their rays don't rotate. Theoretically, the creatures can reason
    # with either. If we assume they need the delta of rays to their head_dir (since they want to know
    # where the other creature is relative to them), then this works either way. If True, then they store
    # fixed deltas from head_dir to each ray in their weights. If False, then they can store the actual
    # angles of each ray and then compute the delta instead. So probably True => easier to learn how to
    # make sense of rays, False => harder to learn, but you save ~15% of the compute per step by not
    # having to rotate the rays.
    rays_rotate: bool = False  # whether or not rays rotate with creatures


    ## vitality
    init_size_range: Tuple[float, float] = (0.5, 4.5)  # (min, max) size of creatures at the beginning
    init_energy: ConfigFunction = ConfigFunction('linear', 1.0)  # func(size)*scale to determine initial energy
    init_health: ConfigFunction = ConfigFunction('square', 1.0)  # func(size)*scale to determine initial health
    size_range: Tuple[float, float] = (0.1, 5.0)  # (minimum,maximum) size of creatures (after mutating)
    immortal: bool = False  # if True, creatures don't die

    ## mutation
    init_mut_rate_range: Tuple[float, float] = (0.01, 0.04)  # (min, max) mutation rate at the beginning



    ### Energy/Movement costs

    alive_cost: FunctionExpression = FunctionExpression(['x'], '0.01f * x') # func(size) that determines the cost of being alive at each step
    rotate_amt: FunctionExpression = FunctionExpression(['x', 'y'], '(x / (1.0f + y)) * 0.314159f') # func(out, size) that determines the amt to rotate
    rotate_cost: FunctionExpression = FunctionExpression(['x', 'y'], '0.05f * (1.0f - (expf(-abs(x) * y)))')  # func(rotate_amt, size) to determine rotation cost
    move_amt: ConfigFunction = ConfigFunction('linear_frac', 0.3)  # func(move_logit, size) to determine movement amount
    move_cost: ConfigFunction = ConfigFunction('abs_exp', 0.2)      # func(move_amt, size) to determine movement cost



    ### Attack

    # emprically measured based on creature sprite to match it as closely as possible
    attack_range: Tuple[float, float] = (1.35, 1.96)   # anywhere along size * head_dir * t + position, t \in [attack_range[0], attack_range[1]] can do damage
    attack_cost: ConfigFunction = ConfigFunction('bilinear', 0.05) # func(num_attacks, size) to determine attack cost
    attack_dmg: FunctionExpression = FunctionExpression(['x'], 'x*x*x*0.1f')  # func(size) to determine the amount of damage done in an attack
    attack_color_frac: FunctionExpression = FunctionExpression(['d'], '1. - 1. / (4.*d + 1.)')  # func(color_diff) to determine the fraction of damage done in an attack
    attack_dist_bonus: float = 0.0  # if creatures are within the dist_bonus, they can attack each other
    dead_drop_food: ConfigFunction = ConfigFunction('linear', 1.)  # func(size) to determine how much food a creature drops when it dies



    ### Reproduction

    reproduce_thresh: ConfigFunction = ConfigFunction('square_add', 11., 1.0)  # func(size) to determine the energy threshold for reproduction
    reproduce_energy_loss_frac: float = 15.  # factor to divide energy by after reproducing (and subtracting off init_energy costs)
    reproduce_dist: float = 3.  # standard deviation of distance from parent to place offspring

    ### Aging

    # one epoch is approximately 1.8 Earth months = 15(yrs, sexual maturity in humans) * / age_mature_mul(epochs) * 12 (months/yr)
    age_dmg_mul: float = 1.4e-3  # for every year past age_old, creatures take this pct more dmg (compounded)
    age_speed_size: ConfigFunction = ConfigFunction('pow', 1., 0.1323)  # (sz**0.133) is the speed at which creatures age
    age_mature_mul: float = 150.0  # age_speed * mature_age_mul is the age at which creatures can reproduce
    age_old_mul: float = 1600.0 # age_speed * age_old_mul is the age at which creatures start taking extra dmg/energy


    ### Algorithm parameters

    max_per_cell: int = 40 # maximum number of creatures in a cell when doing gridded/celled ray tracing
    cell_size: float = 16.0   # width of cells when doing gridded/celled ray tracing
    cache_size: int = 32    # number of creatures to cache when doing gridded/celled ray tracing (power of 2)
    use_cache: int = 0   # whether to use the cache when doing gridded ray tracing (1 = enabled, 0 = disabled)


    ### Game Settings

    max_game_speed: int = 1000 # maximum game speed (steps per frame)

    def update_in_place(self, other: 'Config' | Dict[str, Any]) -> 'Config':
        if isinstance(other, dict):
            for k, v in other.items():
                setattr(self, k, v)
        else:
            for field in dataclasses.fields(self):
                setattr(self, field.name, getattr(other, field.name))
        return self
