import numpy as np
import torch    
import torch.nn as nn
torch.set_grad_enabled(False)
from numba import cuda
import logging
# logging.basicConfig(level=logging.INFO, filename='game.log')

from . import algorithms


"""Store all relevant creature attributes in CUDA objects to ensure minimal
movement between devices. We can kill creatures by re-indexing the array. We
can add a new set of creatures by appending a new set of entries to the array."""


class CreatureArray():
    def __init__(self, num_starting, max_creatures, grid_size, mem_size=4, num_rays=32, hidden_sizes=None):
        if hidden_sizes is None:
            hidden_sizes = [30, 40]
        
        self.max_creatures = max_creatures
        self.grid_size = grid_size
        self.dev = torch.device('cuda')
        self.offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                                     [ 0, -1], [ 0, 0], [ 0, 1],
                                     [ 1, -1], [ 1, 0], [ 1, 1]], device=self.dev).unsqueeze(0)

        self.positions = torch.clamp(torch.rand(num_starting, 2, device=self.dev) * (grid_size-3) + 1, 1, grid_size-2)
        self.sizes = torch.rand(num_starting, device=self.dev) * 1 + 0.5
        self.memories = torch.zeros(num_starting, mem_size, device=self.dev)
        self.energies = self.sizes
        self.healths = self.sizes**2
        self.ages = torch.zeros(num_starting, device=self.dev)

        self.rays = torch.randn(num_starting, num_rays, 3, device=self.dev)
        self.rays[..., :2] /= torch.norm(self.rays[..., :2], dim=2, keepdim=True)
        self.rays[..., 2] = self.rays[..., 2]**2

        self.head_dirs = torch.randn(num_starting, 2, device=self.dev)
        self.head_dirs /= torch.norm(self.head_dirs, dim=1, keepdim=True)

        self.colors = torch.rand(num_starting, 3, device=self.dev) * 254 + 1

        # we have mutation rate for rays, sizes, colors, weights/biases, and mutation rate itself
        self.mutation_rates = torch.rand(num_starting, 6, device=self.dev) * 0.1 + 0.1

        # inputs: rays (x num_rays*3 colors), memory (x mem_size), food (3x3 = 9), health, energy
        # outputs: move forward/back, rotate, memory (x mem_size)
        self.layer_sizes = [num_rays*3 + mem_size + 9 + 2, *hidden_sizes, mem_size + 2]


        """weights are each [N, in, out]"""
        self.weights = [(torch.rand(num_starting, prev_layer, next_layer, device=self.dev)-0.5) / (np.sqrt(15.0 / prev_layer))
                        for prev_layer, next_layer in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        
        self.biases = [(torch.rand(num_starting, 1, next_layer, device=self.dev)-0.5) / np.sqrt(next_layer)
                       for next_layer in self.layer_sizes[1:]]
        

    def forward(self, inputs):
        """Inputs: [N, 1, num_rays + mem_size + 9 + 2]"""
        # print([w.shape for w in self.weights], [b.shape for b in self.biases])
        for w, b in zip(self.weights, self.biases):
            # print(inputs.shape, '@', w.shape, '+', b.shape)
            inputs = torch.tanh(inputs @ w + b)
        return inputs.squeeze()
    
    def kill_dead(self, food_grid):
        dead = (self.healths <= 0) | (self.energies <= 0)

        if not torch.any(dead):
            return

        # the dead drop their food on their ground
        # print(self.positions[dead].long().shape)
        # print(food_grid[self.positions[dead].long()].shape)
        # print(self.positions[dead].long())
        # print(self.sizes[dead].shape)
        dead_posns = self.positions[dead].long()
        logging.info(f"{dead_posns.shape[0]} creatures have died.")
        food_grid[dead_posns[..., 0], dead_posns[..., 1]] += self.sizes[dead]**2 + 10

        self.positions = self.positions[~dead]
        self.memories = self.memories[~dead]
        self.energies = self.energies[~dead]
        self.healths = self.healths[~dead]
        self.colors = self.colors[~dead]
        self.ages = self.ages[~dead]
        self.rays = self.rays[~dead]
        self.head_dirs = self.head_dirs[~dead]
        self.sizes = self.sizes[~dead]
        self.mutation_rates = self.mutation_rates[~dead]
        
        self.weights = [w[~dead] for w in self.weights]
        self.biases = [b[~dead] for b in self.biases]


    def eat(self, food_grid, pct):
        """Eat the food in the creatures' vicinity."""
        pos = self.positions.long()
        food = food_grid[pos[..., 0], pos[..., 1]] * pct
        self.energies += food
        food_grid[pos[..., 0], pos[..., 1]] -= food
        self.ages += 1  # if they've eaten, then they've made it to the next step


    def extract_food_windows(self, indices, food_grid):
        """Indices is [N, 2] indices into food_grid.
        Food grid is [S, S], where S is the width of the world."""
        windows = indices.unsqueeze(1) + self.offsets  # [N, 1, 2] + [1, 9, 2] = [N, 9, 2]
        return food_grid[windows[..., 0], windows[..., 1]]
        

    def collect_stimuli(self, collisions, food_grid):
        """Collisions: [N, 3], food_grid: [N, 3, 3] -> [N, F] where F is the number of features."""
        # get the food in the creature's vicinity
        food = self.extract_food_windows(self.positions.long(), food_grid)  # [N, 9]
        rays_results = collisions.view(self.positions.shape[0], -1)  # [N, 3*num_rays]

        # print(rays_results.shape, rays_results.dtype, rays_results.device,
        #       food.shape, food.dtype, food.device,
        #       self.memories.shape, self.memories.dtype, self.memories.device,
        #       self.healths.unsqueeze(1).shape, self.healths.unsqueeze(1).dtype, self.healths.unsqueeze(1).device,
        #       self.energies.unsqueeze(1).shape, self.energies.unsqueeze(1).dtype, self.energies.unsqueeze(1).device)

        return torch.cat([rays_results, 
                          self.memories, 
                          food, 
                          self.healths.unsqueeze(1), 
                          self.energies.unsqueeze(1)
                        ], 
                        dim=1)  # [N, F]
    

    def rotate_creatures(self, outputs):
        """`outputs` [N,6] are move forward/backward, rotate left/rotate, and then 4 for memory.
        Rotation is done first, and then movement. the "move" output is a scalar between -1 and 1, which
        is multiplied by the object's speed to determine how far to move. Rotation is also a scalar between
        -1 and 1, which is multiplied by the object's rotation speed to determine how far to rotate. 
        Negative rotation is clockwise, positive is counter-clockwise. """
        # rotation => need to rotate the rays and the object's direction
        # print('outputs', outputs[:,1])
        # print('sqrt sizes', torch.sqrt(self.sizes))
        rotate = outputs[:,1]*torch.pi / 20*(1 + torch.sqrt(self.sizes))
        # print('sizes', self.sizes)
        rotate_energy = 1 - torch.exp(self.sizes * torch.abs(rotate))   # energy cost of rotation
        # print('rot_nrg', rotate_energy)
        # print('rot', rotate)
        self.energies -= rotate_energy
        
        rotation_matrix = torch.empty((outputs.shape[0], 2, 2), device=self.dev)
        cos_rotate = torch.cos(rotate)
        sin_rotate = torch.sin(rotate)
        rotation_matrix[:,0,0] = cos_rotate
        rotation_matrix[:,0,1] = -sin_rotate
        rotation_matrix[:,1,0] = sin_rotate
        rotation_matrix[:,1,1] = cos_rotate

        # rotate the rays and head directions
        self.rays[..., :2] = self.rays[..., :2] @ rotation_matrix
        self.head_dirs = (self.head_dirs.unsqueeze(1) @ rotation_matrix).squeeze()


    def move_creatures(self, outputs):
        # move => need to move the object's position
        move = outputs[:,0]/10.   # maybe add some acceleration/velocity thing instead
        # print(self.head_dirs.shape, move.shape, self.positions.shape)
        self.positions += self.head_dirs * move.unsqueeze(1)   # move the object's to new position
        self.positions = torch.clamp(self.positions, 1, self.grid_size-2)  # don't let it go off the edge

    def do_attacks(self, attacks):
        """Attacks is [N, 2], where the 1st coordinate is an integer indicating how many things the 
        organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the 
        organism is taking from other things attacking it"""

        # print('nrg_before', self.energies)
        logging.info(f'Attacks:\n{attacks}')
        self.energies -= attacks[:,0] * self.sizes * 0.1  # takes 0.1 * size to do an attack
        # print('nrg_after', self.energies)
        self.healths -= attacks[:,1]


    def reproduce(self):
        """For sufficiently high energy creatures, create a new creature with mutated genes."""
        # print(self.energies)
        # print(self.sizes)
        reproducers = self.energies >= (self.sizes**2 * 10)
        # print(reproducers)
        if not torch.any(reproducers):
            return
        
        # limit reproducers so we don't exceed max size
        num_reproducers = torch.sum(reproducers)
        max_reproducers = self.max_creatures - self.positions.shape[0]
        if max_reproducers <= 0:
            return 
        
        if num_reproducers > max_reproducers:
            num_reproducers = min(num_reproducers, max_reproducers)
            non_reproducers = torch.nonzero(reproducers)[num_reproducers:]

            # if you get unlucky to not reproduce, you get to go to max health at least, but take an energy
            # hit as a result
            self.healths[non_reproducers] = self.sizes[non_reproducers]**2
            self.energies[non_reproducers] /= 3.

            reproducers[non_reproducers] = False

        self.energies[reproducers] /= 10  # they lose a bunch of energy if they reproduce
            
        mut = self.mutation_rates[reproducers]
        logging.info(f"Reproducers:\n{reproducers}")
        logging.info(f"Mut: {mut}")
        logging.info(f"self.sizes[reproducers]: {self.sizes[reproducers], self.sizes[reproducers].shape}")
        logging.info(f"self.colors[reproducers]: {self.colors[reproducers], self.colors[reproducers].shape}")
        logging.info(f"mut[:,0]: {mut[:,0], mut[:,0].shape}")
        logging.info(f"mut[:,1]: {mut[:,1], mut[:,1].shape}")
        size_perturb = torch.randn_like(self.sizes[reproducers]) * mut[:,0]
        color_perturb = torch.randn_like(self.colors[reproducers]) * mut[:, 1, None]
        ray_perturb = torch.randn_like(self.rays[reproducers]) * mut[:, 2, None, None]
        weight_perturbs = [torch.randn_like(w[reproducers]) * mut[:, 3, None, None]
                           for w in self.weights]
        bias_perturbs = [torch.randn_like(b[reproducers]) * mut[:, 4, None, None]
                         for b in self.biases]
        mut_rate_perturb = torch.randn_like(mut) * mut[:, 5, None]

        new_sizes = torch.clamp_min(self.sizes[reproducers] + size_perturb, 0.1)
        new_colors = torch.clamp(self.colors[reproducers] + color_perturb, 1, 255)
        new_weights = [w[reproducers] + wp for w, wp in zip(self.weights, weight_perturbs)]
        logging.info(f"existing bias shapes: {[b[reproducers].shape for b in self.biases]}")
        logging.info(f"perturb bias shapse: {[bp.shape for bp in bias_perturbs]}")
        logging.info(f"bias mut shape: {mut[:, 4, None].shape}")
        new_biases = [b[reproducers] + bp for b, bp in zip(self.biases, bias_perturbs)]
        logging.info(f"Bias sahpes: {[b.shape for b in new_biases]}")
        new_mutation_rates = mut + mut_rate_perturb

        new_memories = torch.zeros_like(self.memories[reproducers])
        new_energy = new_sizes
        new_health = new_sizes**2
        new_ages = torch.zeros_like(self.ages[reproducers])
        
        new_head_dirs = torch.randn_like(self.head_dirs[reproducers])
        new_head_dirs /= torch.norm(new_head_dirs, dim=1, keepdim=True)
        
        new_rays = self.rays[reproducers] + ray_perturb
        new_rays[...,:2] /= torch.norm(new_rays[...,:2], dim=2, keepdim=True)

        pos_perturb = torch.randn_like(self.positions[reproducers]) * torch.sqrt(new_sizes.unsqueeze(1))*2
        new_positions = torch.clamp(self.positions[reproducers] + pos_perturb, 1, self.grid_size-2)

        self.positions = torch.cat([self.positions, new_positions], dim=0)
        self.sizes = torch.cat([self.sizes, new_sizes], dim=0)
        self.memories = torch.cat([self.memories, new_memories], dim=0)
        self.energies = torch.cat([self.energies, new_energy], dim=0)
        self.healths = torch.cat([self.healths, new_health], dim=0)
        self.ages = torch.cat([self.ages, new_ages], dim=0)
        self.rays = torch.cat([self.rays, new_rays], dim=0)
        self.head_dirs = torch.cat([self.head_dirs, new_head_dirs], dim=0)
        self.colors = torch.cat([self.colors, new_colors], dim=0)
        self.mutation_rates = torch.cat([self.mutation_rates, new_mutation_rates], dim=0)

        self.weights = [torch.cat([w, nw], dim=0) for w, nw in zip(self.weights, new_weights)]
        self.biases = [torch.cat([b, nb], dim=0) for b, nb in zip(self.biases, new_biases)]     
