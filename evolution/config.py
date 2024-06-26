import dataclasses
from typing import Tuple, List
import torch


@dataclasses.dataclass
class Config:
    ### World
    size: float = 100.  # width of the world
    start_creatures: int = 32 # number of creatures at the beginning
    max_creatures: int = 100 # maximum number of creatures


    ### Food
    init_food_scale: float = 1.0    # food will be initialized uniformly to be in [0, init_food_scale]
    eat_pct: float = 0.2       # what percentage of food in a given cell that creatures can eat in a turn
    max_food: float = 10.     # maximum amount of food in a cell (decays past this value)
    food_cover_decr: float = 0.3  # if a creature occupies a cell, the food in that cell be decreased by this amount each step
    decay_grow_ratio: float = 0.1 # how much faster food decays (when over max_food) than it grows (when under max food)


    ### Creatures
    # brains
    brain_size: List = dataclasses.field(default_factory=lambda: [30, 40]), # size of hidden layers of brains
    mem_size: int = 10 # how many output values/input values can be used for memory

    # vision
    food_sight: int = 1 # how many squares (grid) away creatures can see food (1 => a 3x3 window centered on them)
    num_rays: int = 32 # number of rays creatures can see with
    min_ray_dist: float = 0.1 # minimum distance of rays

    # vitality
    init_size_min: float = 0.5 # minimum size of creatures at the beginning
    init_size_max: float = 1.5 # maximum size of creatures at the beginning
    init_energy_scale: Tuple[str, float] = ('linear', 0.1) # func(size)*scale to determine initial energy
    init_health: Tuple[str, float] = ('square', 1.0)   # func(size)*scale to determine initial energy
    size_min: float = 0.1  # minimum size of creatures (after mutating)

    # mutation
    init_mut_rate_min: float = 0.01   # minimum mutation rate at the beginning
    init_mut_rate_max: float = 0.02   # maximum mutation rate at the beginning



    ### Energy/Movement costs
    alive_cost: Tuple[str, float] = ('square', 0.1)  # func(size) that determines the cost of being alive at each step
    rotate_amt: Tuple[str, float] = ('linear_frac', torch.pi/20)  # func(out, size) that determines the amt to rotate
    rotate_cost: Tuple[str, float] = ('abs_exp', 1.)  # func(rotate_amt, size) to determine rotation cost
    move_amt: Tuple[str, float] = ('linear_frac', 0.1)  # func(output, size) to determine movement amount
    move_cost: Tuple[str, float] = ('abs_exp', 2.)  # func(move_amt, size) to determine movement cost
    


    ### Attack
    attack_cost: Tuple[str, float] = ('bilinear', 0.1)  # func(num_attacks, size) to determine attack cost
    attack_dmg: Tuple[str, float] = ('linear', 1.0)  # func(size) to determine attack damage
    attack_ignore: float = 3.0 # max amount that abs(color1 - color2).sum() can be and still mean they can't hurt each other
    attack_dist_bonus: float = 0.5  # if creatures are within the dist_bonus, they can attack each other



    ### Reproduction
    reproduce_thresh: Tuple[str, float] = ('square', 11.)  # func(size) to determine the energy threshold for reproduction
    reproduce_cost: float = 15.  # factor to divide energy by after reproducing (and subtracting off init_energy costs)
    reproduce_dist: float = 3.  # standard deviation of distance from parent to place offspring

    

    ## Algorithm parameters
    max_per_cell: int = 512 # maximum number of creatures in a cell when doing gridded/celled ray tracing
    cell_size: float = 4.   # width of cells when doing gridded/celled ray tracing
    cache_size: int = 32  # must be exact power of 2. In gridded ray tracing, this cache stores creatures we've already checked



    def square(x):
        return x**2

    def linear(x):
        return x
    
    def bilinear(x, y):
        return x * y

    def linear_frac(x, y):
        return x / (1 + y)
    
    def abs_exp(x, y):
        return 1 - torch.exp(x * torch.abs(y))


