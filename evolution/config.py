import dataclasses
from typing import Tuple
import torch
torch.set_grad_enabled(False)


class ConfigFunction:   # for replacing functions in regular python code
    def __init__(self, name, mul):
        self.name = name
        self.mul = mul
        self.func = getattr(self, name)

    @staticmethod
    def square(x):
        return x**2

    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def bilinear(x, y):
        return x * y

    @staticmethod
    def linear_frac(x, y):
        return x / (1 + y)
    
    @staticmethod
    def abs_exp(x, y):
        return 1 - torch.exp(-torch.abs(x) * y)

    def __call__(self, *args):
        return self.func(*args) * self.mul
    
    def __repr__(self):
        return f'{self.name}*{self.mul}'
    
class FunctionExpression():  # for code preprocessor in cu_algorithms
    def __init__(self, symbols, expr):
        self.symbols = symbols
        self.expr = expr

    def __repr__(self):
        return f'{self.symbols} -> {self.expr}'


@dataclasses.dataclass
class Config:
    """
    Configuration parameters for the simulation environment.

    Attributes:
        size: Width of the world in units.
        start_creatures (int): Number of creatures at the beginning.
        max_creatures (int): Maximum number of creatures allowed.

        init_food_scale (float): Initial uniform food distribution range [0, init_food_scale].
        eat_pct (float): Percentage of food in a cell that creatures can eat per turn.
        max_food (float): Maximum amount of food in a cell (food decays past this value).
        food_cover_decr (float): Amount by which food in a cell decreases per step when occupied by a creature.
        food_decay_rate (float): Rate at which food decays when above max_food.
        food_growth_rate (float): Scale factor for food growth rate.

        brain_size (Tuple[int, ...]): Sizes of hidden layers in creature brains.
        mem_size (int): Number of memory slots available for input/output.
        food_sight (int): Distance in grid squares that creatures can see food (1 => a 3x3 window).
        num_rays (int): Number of vision rays creatures use.
        min_ray_dist (float): Minimum distance of vision rays.
        init_size_range (Tuple[float, float]): Initial size range (min, max) of creatures.
        init_energy (ConfigFunction): Function to determine initial energy based on size.
        init_health (ConfigFunction): Function to determine initial health based on size.
        size_min (float): Minimum size of creatures after mutation.
        init_mut_rate_range (Tuple[float, float]): Initial mutation rate range (min, max).

        alive_cost (ConfigFunction): Function to determine the cost of being alive at each step based on size.
        rotate_amt (ConfigFunction): Function to determine the amount of rotation.
        rotate_cost (ConfigFunction): Function to determine the cost of rotation based on amount and size.
        move_amt (ConfigFunction): Function to determine the movement amount.
        move_cost (ConfigFunction): Function to determine the cost of movement based on amount and size.

        attack_cost (ConfigFunction): Function to determine the cost of attacking based on number of attacks and size.
        attack_dmg (FunctionExpression): Function to determine the damage done in an attack based on size.
        attack_ignore_color_dist (float): If sum(abs(color1-color2)) < this value, creatures don't harm each other.
        attack_dist_bonus (float): Distance bonus for attacking creatures within a certain distance.

        reproduce_thresh (ConfigFunction): Function to determine the energy threshold for reproduction based on size.
        reproduce_energy_loss_frac (float): Fraction of energy lost after reproducing.
        reproduce_dist (float): Standard deviation of distance from parent to offspring location.

        max_per_cell (int): Maximum number of creatures per cell for gridded/celled ray tracing.
        cell_size (float): Width of cells for gridded/celled ray tracing.
        cache_size (int): Number of creatures to cache for gridded/celled ray tracing (must be a power of 2).
    """
    ### World
    size: int = 125  # width of the world in units
    start_creatures: int = 64 # number of creatures at the beginning
    max_creatures: int = 256 # maximum number of creatures


    ### Food
    init_food_scale: float = 8.0    # food will be initialized uniformly to be in [0, init_food_scale]
    eat_pct: float = 0.1       # what percentage of food in a given cell that creatures can eat in a turn
    max_food: float = 15.     # maximum amount of food in a cell (decays past this value)
    food_cover_decr: float = 0.2  # if a creature occupies a cell, the food in that cell be decreased by this amount each step
    food_decay_rate: float = 0.05 # how fast food decays (when over max_food)
    food_growth_rate: float = 10.0  # scale to apply to food growth rate


    ### Creatures
    # brains
    brain_size: Tuple[int, ...] = (30, 40) # size of hidden layers of brains
    mem_size: int = 10 # how many output values/input values can be used for memory

    # vision
    food_sight: int = 2 # how many squares (grid) away creatures can see food (1 => a 3x3 window centered on them)
    num_rays: int = 32 # number of rays creatures can see with
    min_ray_dist: float = 0.1 # minimum distance of rays

    # vitality
    init_size_range: Tuple[float, float] = (0.5, 4.5)  # (min, max) size of creatures at the beginning
    # func(size)*scale to determine initial energy
    init_energy: ConfigFunction = ConfigFunction('linear', 1.0) 
    # func(size)*scale to determine initial energy
    init_health: ConfigFunction = ConfigFunction('square', 1.0)   
    size_range: Tuple[float, float] = (0.1, 5.0)  # (minimum,maximum) size of creatures (after mutating)
    immortal: bool = False  # if True, creatures don't die

    # mutation
    init_mut_rate_range: Tuple[float, float] = (0.01, 0.04)  # (min, max) mutation rate at the beginning



    ### Energy/Movement costs
    # func(size) that determines the cost of being alive at each step
    alive_cost: ConfigFunction = ConfigFunction('linear', 0.03)  
    # func(out, size) that determines the amt to rotate
    rotate_amt: ConfigFunction = ConfigFunction('linear_frac', torch.pi/10)  
    # func(rotate_amt, size) to determine rotation cost
    rotate_cost: ConfigFunction = ConfigFunction('abs_exp', 0.2)  
    # func(output, size) to determine movement amount
    move_amt: ConfigFunction = ConfigFunction('linear_frac', 0.13)
    # func(move_amt, size) to determine movement cost  
    move_cost: ConfigFunction = ConfigFunction('abs_exp', 0.2)  
    


    ### Attack
    # func(num_attacks, size) to determine attack cost
    attack_cost: ConfigFunction = ConfigFunction('bilinear', 0.05)  
    # func(size) to determine the amount of damage done in an attack
    attack_dmg: FunctionExpression = FunctionExpression(['x'], 'x*x * 1.5f') 
    attack_ignore_color_dist: float = 3.0 # if sum(abs(color1-color2)) < this, they don't hurt each other
    attack_dist_bonus: float = 0.0  # if creatures are within the dist_bonus, they can attack each other
    dead_drop_food: ConfigFunction = ConfigFunction('linear', 1.)  # func(size) to determine how much food a creature drops when it dies



    ### Reproduction
    # func(size) to determine the energy threshold for reproduction
    reproduce_thresh: ConfigFunction = ConfigFunction('square', 11.)  
    reproduce_energy_loss_frac: float = 15.  # factor to divide energy by after reproducing (and subtracting off init_energy costs)
    reproduce_dist: float = 3.  # standard deviation of distance from parent to place offspring
    mature_age_mul: float = 20.0  # size * mature_age_mult is the age at which creatures can reproduce

    
    
    ## Algorithm parameters
    max_per_cell: int = 512 # maximum number of creatures in a cell when doing gridded/celled ray tracing
    cell_size: float = 4.0   # width of cells when doing gridded/celled ray tracing
    cache_size: int = 32    # number of creatures to cache when doing gridded/celled ray tracing (power of 2)


def simple_cfg(**kwargs):
    cfg = dict(size=10, start_creatures=5, max_creatures=10, mem_size=4, 
                  init_food_scale=5.0, food_cover_decr=0.0, alive_cost=ConfigFunction('linear', 0.0),
                  init_size_range=(0.1, 2.5))
    cfg.update(kwargs)
    return Config(**cfg)