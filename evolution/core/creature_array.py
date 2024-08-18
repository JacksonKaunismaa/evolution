from typing import Tuple, Union
import numpy as np
import torch    
torch.set_grad_enabled(False)

from evolution.utils.batched_random import BatchedRandom

from .config import Config

class CreatureArray:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.posn_bounds = (0, cfg.size-1e-4)
        self.num_food = (cfg.food_sight*2+1)**2

    def generate_from_cfg(self):
        # objects need to stay within [0, size-1] so that indexing makes sense
        self.positions = torch.empty(self.cfg.start_creatures, 2, device='cuda').uniform_(*self.posn_bounds)
        self.sizes = torch.empty(self.cfg.start_creatures, device='cuda').uniform_(*self.cfg.init_size_range)
        self.memories = torch.zeros(self.cfg.start_creatures, self.cfg.mem_size, device='cuda')
        self.ages = torch.zeros(self.cfg.start_creatures, device='cuda')

        self.rays = torch.randn(self.cfg.start_creatures, self.cfg.num_rays, 3, device='cuda')

        self.head_dirs = torch.randn(self.cfg.start_creatures, 2, device='cuda')

        self.colors = torch.empty(self.cfg.start_creatures, 3, device='cuda').uniform_(1, 255)

        # we have mutation rate for rays, sizes, colors, weights/biases, and mutation rate itself
        self.mutation_rates = torch.empty(self.cfg.start_creatures, 6, device='cuda').uniform_(*self.cfg.init_mut_rate_range)

        # inputs: rays (x num_rays*3 colors), memory (x mem_size), food (3x3 = 9), health, energy
        # outputs: move forward/back, rotate, memory (x mem_size)
        self.layer_sizes = [self.cfg.num_rays*3 + self.cfg.mem_size + self.num_food + 2, 
                            *self.cfg.brain_size, self.cfg.mem_size + 2]


        """weights are each [N, in, out]"""
        self.weights = [(torch.rand(self.cfg.start_creatures, prev_layer, next_layer, device='cuda')-0.5) / (np.sqrt(15.0 / prev_layer))
                        for prev_layer, next_layer in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        
        self.biases = [(torch.rand(self.cfg.start_creatures, 1, next_layer, device='cuda')-0.5) / np.sqrt(next_layer)
                    for next_layer in self.layer_sizes[1:]]
        reproduce_dims = {'sizes': self.sizes, 
                        'colors': self.colors, 
                        'weights': self.weights,
                        'biases': self.biases,
                        'mutation_rates': self.mutation_rates,
                        'head_dirs': self.head_dirs,
                        'rays': self.rays,
                        'positions': self.positions}
        self.rng = BatchedRandom(reproduce_dims)
        self.normalize_and_generate_health()
            
    def generate_from_parents(self, reproducers, num_reproducers, parents: 'CreatureArray'):
        mut = parents.mutation_rates[reproducers]
        parents.rng.generate(num_reproducers)

        size_perturb = parents.rng.get('sizes') * mut[:,0]
        color_perturb = parents.rng.get('colors') * mut[:, 1, None]
        ray_perturb = parents.rng.get('rays') * mut[:, 2, None, None]        
        weight_perturbs = [parents.rng.get('weights', i) for i in range(len(parents.weights))]
        bias_perturbs = [parents.rng.get('biases', i) for i in range(len(parents.biases))]
        mut_rate_perturb = parents.rng.get('mutation_rates') * mut[:, 5, None]
        pos_perturb = parents.rng.get('positions') * parents.cfg.reproduce_dist#* new_sizes.unsqueeze(1) * parents.cfg.reproduce_dist

        self.head_dirs = parents.rng.get('head_dirs')
        self.sizes = parents.sizes[reproducers] + size_perturb
        self.colors = parents.colors[reproducers] + color_perturb
        self.weights = [w[reproducers] + wp for w, wp in zip(parents.weights, weight_perturbs)]
        self.biases = [b[reproducers] + bp for b, bp in zip(parents.biases, bias_perturbs)]
        self.mutation_rates = mut + mut_rate_perturb
        self.memories = torch.zeros_like(parents.memories[reproducers])
        self.ages = torch.zeros_like(parents.ages[reproducers])
        self.rays = parents.rays[reproducers] + ray_perturb
        self.positions = parents.positions[reproducers] + pos_perturb
        self.normalize_and_generate_health()
            
            
    def normalize_and_generate_health(self):
        self.positions = torch.clamp(self.positions, *self.posn_bounds)  # make sure we are in bounds
        self.sizes = torch.clamp(self.sizes, *self.cfg.size_range)   # clamp size
        
        self.rays[..., :2] /= torch.norm(self.rays[..., :2], dim=2, keepdim=True)  # make sure we are normalized
        self.rays[..., 2] = torch.clamp(torch.abs(self.rays[..., 2]),   # clamp ray length
                                        self.cfg.ray_dist_range[0]*self.sizes.unsqueeze(1),
                                        self.cfg.ray_dist_range[1]*self.sizes.unsqueeze(1))
        
        self.head_dirs /= torch.norm(self.head_dirs, dim=1, keepdim=True)   # make sure we are normalized
        self.colors = torch.clamp(self.colors, 1, 255)   # make sure colors are reasonable
        
        self.energies = self.cfg.init_energy(self.sizes)  # now that size has been normalized, we can generate initial
        self.healths = self.cfg.init_health(self.sizes)  # energy and health based off of size
            
    @property
    def population(self):
        return self.positions.shape[0]

    def add_with_deaths(self, alive: Union[None, torch.Tensor], num_dead: int, other: 'CreatureArray'):
        num_creatures = self.population - num_dead + other.population
        num_old = self.population - num_dead
        
        positions = torch.empty(num_creatures, 2, device='cuda')
        sizes = torch.empty(num_creatures, device='cuda')
        memories = torch.empty(num_creatures, self.cfg.mem_size, device='cuda')
        energies = torch.empty(num_creatures, device='cuda')
        healths = torch.empty(num_creatures, device='cuda')
        ages = torch.empty(num_creatures, device='cuda')
        rays = torch.empty(num_creatures, self.cfg.num_rays, 3, device='cuda')
        head_dirs = torch.empty(num_creatures, 2, device='cuda')
        colors = torch.empty(num_creatures, 3, device='cuda')
        mutation_rates = torch.empty(num_creatures, 6, device='cuda')
        weights = [torch.empty(num_creatures, *w.shape[1:], device='cuda') for w in other.weights]
        biases = [torch.empty(num_creatures, *b.shape[1:], device='cuda') for b in other.biases]

        positions[:num_old] = self.positions[alive]  # do this instead of a cat, so that we avoid copying twice
        sizes[:num_old] = self.sizes[alive]
        memories[:num_old] = self.memories[alive]
        energies[:num_old] = self.energies[alive]
        healths[:num_old] = self.healths[alive]
        ages[:num_old] = self.ages[alive]
        rays[:num_old] = self.rays[alive]
        head_dirs[:num_old] = self.head_dirs[alive]
        colors[:num_old] = self.colors[alive]
        mutation_rates[:num_old] = self.mutation_rates[alive]
        for i, w in enumerate(other.weights):
            weights[i][:num_old] = self.weights[i][alive]
        for i, b in enumerate(other.biases):
            biases[i][:num_old] = self.biases[i][alive]
        
        positions[num_old:] = other.positions
        sizes[num_old:] = other.sizes
        memories[num_old:] = other.memories
        energies[num_old:] = other.energies
        healths[num_old:] = other.healths
        ages[num_old:] = other.ages
        rays[num_old:] = other.rays
        head_dirs[num_old:] = other.head_dirs
        colors[num_old:] = other.colors
        mutation_rates[num_old:] = other.mutation_rates
        for i, w in enumerate(other.weights):
            weights[i][num_old:] = w
        for i, b in enumerate(other.biases):
            biases[i][num_old:] = b
            
        self.positions = positions
        self.sizes = sizes
        self.memories = memories
        self.energies = energies
        self.healths = healths
        self.ages = ages
        self.rays = rays
        self.head_dirs = head_dirs
        self.colors = colors
        self.mutation_rates = mutation_rates
        self.weights = weights
        self.biases = biases
        
        
    def apply_deaths(self, alive: torch.Tensor):
        self.positions = self.positions[alive]
        self.sizes = self.sizes[alive]
        self.memories = self.memories[alive]
        self.energies = self.energies[alive]
        self.healths = self.healths[alive]
        self.ages = self.ages[alive]
        self.rays = self.rays[alive]
        self.head_dirs = self.head_dirs[alive]
        self.colors = self.colors[alive]
        self.mutation_rates = self.mutation_rates[alive]
        self.weights = [w[alive] for w in self.weights]
        self.biases = [b[alive] for b in self.biases]