import numpy as np
import torch    
import torch.nn as nn
torch.set_grad_enabled(False)
import logging
logging.basicConfig(level=logging.INFO, filename='game.log')

from .config import Config


"""Store all relevant creature attributes in CUDA objects to ensure minimal
movement between devices. We can kill creatures by re-indexing the array. We
can add a new set of creatures by appending a new set of entries to the array."""


class CreatureArray():
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # how much indexing into the food grid needs to be adjusted to avoid OOB
        self.pad = self.cfg.food_sight 
        # self.posn_bounds = (pad, cfg.size+pad-1)
        self.posn_bounds = (0, cfg.size-1)
        self.offsets = torch.tensor([[i, j] for i in range(-self.pad, self.pad+1) 
                                     for j in range(-self.pad, self.pad+1)], device='cuda').unsqueeze(0)

        # objects need to stay within [0, size-1] so that indexing makes sense
        self.positions = torch.empty(cfg.start_creatures, 2, device='cuda').uniform_(*self.posn_bounds)
        self.sizes = torch.empty(cfg.start_creatures, device='cuda').uniform_(*cfg.init_size_range)
        logging.info(f"Initial sizes: {self.sizes}")
        self.memories = torch.zeros(cfg.start_creatures, cfg.mem_size, device='cuda')
        self.energies = cfg.init_energy(self.sizes)
        self.healths = cfg.init_health(self.sizes)
        self.ages = torch.zeros(cfg.start_creatures, device='cuda')

        self.rays = torch.randn(cfg.start_creatures, cfg.num_rays, 3, device='cuda')
        self.rays[..., :2] /= torch.norm(self.rays[..., :2], dim=2, keepdim=True)
        self.rays[..., 2] = torch.clamp_min(torch.abs(self.rays[..., 2]), cfg.min_ray_dist)

        self.head_dirs = torch.randn(cfg.start_creatures, 2, device='cuda')
        self.head_dirs /= torch.norm(self.head_dirs, dim=1, keepdim=True)

        self.colors = torch.empty(cfg.start_creatures, 3, device='cuda').uniform_(1, 255)

        # we have mutation rate for rays, sizes, colors, weights/biases, and mutation rate itself
        self.mutation_rates = torch.empty(cfg.start_creatures, 6, device='cuda').uniform_(*cfg.init_mut_rate_range)

        # inputs: rays (x num_rays*3 colors), memory (x mem_size), food (3x3 = 9), health, energy
        # outputs: move forward/back, rotate, memory (x mem_size)
        self.layer_sizes = [cfg.num_rays*3 + cfg.mem_size + self.offsets.numel()//2 + 2, *cfg.brain_size, cfg.mem_size + 2]


        """weights are each [N, in, out]"""
        self.weights = [(torch.rand(cfg.start_creatures, prev_layer, next_layer, device='cuda')-0.5) / (np.sqrt(15.0 / prev_layer))
                        for prev_layer, next_layer in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        
        self.biases = [(torch.rand(cfg.start_creatures, 1, next_layer, device='cuda')-0.5) / np.sqrt(next_layer)
                       for next_layer in self.layer_sizes[1:]]
        

    def forward(self, inputs):
        """Inputs: [N, 1, F] tensor"""
        # print([w.shape for w in self.weights], [b.shape for b in self.biases])
        for w, b in zip(self.weights, self.biases):
            # print(inputs.shape, '@', w.shape, '+', b.shape)
            inputs = torch.tanh(inputs @ w + b)
        # print('out' ,inputs.shape)
        outputs = inputs.squeeze(dim=1)  # [N, O]
        self.memories = outputs[:, 2:]  # only 2 outputs actually do something, so the rest are memory
        return outputs
    
    def kill_dead(self, food_grid):
        health_deaths = (self.healths <= 0)
        energy_deaths = (self.energies <= 0)
        dead = health_deaths | energy_deaths

        if not torch.any(dead):
            return

        logging.info(f"{health_deaths.sum()} creatures died from health.")
        logging.info(f"{energy_deaths.sum()} creatures died from energy.")

        logging.info(f"Dead from health: {self.sizes[health_deaths]}")
        logging.info(f"Dead from energy: {self.sizes[energy_deaths]}")

        # the dead drop their food on their ground
        # print(self.positions[dead].long().shape)
        # print(food_grid[self.positions[dead].long()].shape)
        # print(self.positions[dead].long())
        # print(self.sizes[dead].shape)
        dead_posns = self.food_grid_pos[dead]
        logging.info(f"{dead_posns.shape[0]} creatures have died.")
        food_grid[dead_posns[..., 1], dead_posns[..., 0]] += self.cfg.dead_drop_food(self.sizes[dead])

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

    @property
    def food_grid_pos(self):
        return self.positions.long() + self.pad
    
    @property
    def population(self):
        return self.positions.shape[0]

    def eat(self, food_grid):
        """Eat the food in the creatures' vicinity."""
        pos = self.food_grid_pos
        # print('pos', pos)
        # print('food @ pos', (food_grid[pos[..., 1], pos[..., 0]]*100).int())
        food = food_grid[pos[..., 1], pos[..., 0]] * self.cfg.eat_pct
        # gain food for eating and lose food for staying alive
        alive_cost =  self.cfg.alive_cost(self.sizes)
        logging.info(f"Food: {food}")
        logging.info(f"Alive cost: {alive_cost}")
        self.energies += food - alive_cost
        food_grid[pos[..., 1], pos[..., 0]] -= food
        self.ages += 1  # if they've eaten, then they've made it to the next step
        # print('food @ pos after', (food_grid[pos[..., 1], pos[..., 0]]*100).int())


    def extract_food_windows(self, indices, food_grid):
        """Indices is [N, 2] indices into food_grid.
        Food grid is [S, S], where S is the width of the world."""
        windows = indices.unsqueeze(1) + self.offsets  # [N, 1, 2] + [1, 9, 2] = [N, 9, 2]
        return food_grid[windows[..., 1], windows[..., 0]]
        

    def collect_stimuli(self, collisions, food_grid):
        """Collisions: [N, 3], food_grid: [N, 3, 3] -> [N, F] where F is the number of features."""
        # get the food in the creature's vicinity
        food = self.extract_food_windows(self.food_grid_pos, food_grid)  # [N, 9]
        rays_results = collisions.view(self.population, -1)  # [N, 3*num_rays]

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
        # rotate amount
        rotate = self.cfg.rotate_amt(outputs[:,1], self.sizes)
        rotate_energy = self.cfg.rotate_cost(rotate, self.sizes)
        
        logging.info(f"Rotate amt: {rotate}")
        logging.info(f"Rotate cost: {rotate_energy}")
        self.energies -= rotate_energy
        
        rotation_matrix = torch.empty((outputs.shape[0], 2, 2), device='cuda')  # [N, 2, 2]
        cos_rotate = torch.cos(rotate)
        sin_rotate = torch.sin(rotate)
        rotation_matrix[:,0,0] = cos_rotate
        rotation_matrix[:,0,1] = -sin_rotate
        rotation_matrix[:,1,0] = sin_rotate
        rotation_matrix[:,1,1] = cos_rotate

        # rotate the rays and head directions
        self.rays[..., :2] = self.rays[..., :2] @ rotation_matrix  # [N, 32, 2] @ [N, 2, 2]
        self.head_dirs = (self.head_dirs.unsqueeze(1) @ rotation_matrix).squeeze(1)  # [N, 1, 2] @ [N, 2, 2]


    def move_creatures(self, outputs):
        # move => need to move the object's position
        # maybe add some acceleration/velocity thing instead
        move = self.cfg.move_amt(outputs[:,0], self.sizes)
        move_cost = self.cfg.move_cost(move, self.sizes)
        logging.info(f"Move amt: {move}")
        logging.info(f"Move cost: {move_cost}")
        self.energies -= move_cost
        self.positions += self.head_dirs * move.unsqueeze(1)   # move the object's to new position
        self.positions = torch.clamp(self.positions, *self.posn_bounds)  # don't let it go off the edge

    def do_attacks(self, attacks):
        """Attacks is [N, 2], where the 1st coordinate is an integer indicating how many things the 
        organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the 
        organism is taking from other things attacking it"""

        # print('nrg_before', self.energies)
        # logging.info(f'Attacks:\n{attacks}')
        # self.energies -= attacks[:,0] * self.sizes * 0.1  # takes 0.1 * size to do an attack
        attack_cost = self.cfg.attack_cost(attacks[:,0], self.sizes)
        self.energies -= attack_cost
        logging.info(f"Attack energy: {attack_cost}")
        self.healths -= attacks[:,1]

    def reproduce(self):
        """For sufficiently high energy creatures, create a new creature with mutated genes."""
        # 1 + so that really small creatures don't get unfairly benefitted
        reproducers = self.energies >= 1 + self.cfg.reproduce_thresh(self.sizes)
        # print(reproducers)
        if not torch.any(reproducers):
            return
        
        # limit reproducers so we don't exceed max size
        num_reproducers = torch.sum(reproducers)
        max_reproducers = self.cfg.max_creatures - self.population
        if max_reproducers <= 0:
            return 
        
        if num_reproducers > max_reproducers:
            num_reproducers = max_reproducers
            non_reproducers = torch.nonzero(reproducers)[num_reproducers:]
            reproducers[non_reproducers] = False

        self.energies[reproducers] -= self.sizes[reproducers]  # subtract off the energy that you've put into the world
        self.energies[reproducers] /= self.cfg.reproduce_energy_loss_frac  # then lose a bit extra because this process is lossy
        logging.info(f"Energy after reproduce: {self.energies[reproducers]}")
            
        mut = self.mutation_rates[reproducers]
        logging.info(f"Num reproducers:{reproducers.sum()}")
        # logging.info(f"Mut: {mut}")
        logging.info(f"Reproducers sizes: {self.sizes[reproducers]}")
        logging.info(f"Reproducers colors: {self.colors[reproducers]}")
        # logging.info(f"mut[:,0]: {mut[:,0], mut[:,0].shape}")
        # logging.info(f"mut[:,1]: {mut[:,1], mut[:,1].shape}")
        size_perturb = torch.randn_like(self.sizes[reproducers]) * mut[:,0]
        color_perturb = torch.randn_like(self.colors[reproducers]) * mut[:, 1, None]
        ray_perturb = torch.randn_like(self.rays[reproducers]) * mut[:, 2, None, None]
        weight_perturbs = [torch.randn_like(w[reproducers]) * mut[:, 3, None, None]
                           for w in self.weights]
        bias_perturbs = [torch.randn_like(b[reproducers]) * mut[:, 4, None, None]
                         for b in self.biases]
        mut_rate_perturb = torch.randn_like(mut) * mut[:, 5, None]

        new_sizes = torch.clamp_min(self.sizes[reproducers] + size_perturb, self.cfg.size_min)
        new_colors = torch.clamp(self.colors[reproducers] + color_perturb, 1, 255)
        new_weights = [w[reproducers] + wp for w, wp in zip(self.weights, weight_perturbs)]
        # logging.info(f"existing bias shapes: {[b[reproducers].shape for b in self.biases]}")
        # logging.info(f"perturb bias shapse: {[bp.shape for bp in bias_perturbs]}")
        # logging.info(f"bias mut shape: {mut[:, 4, None].shape}")
        new_biases = [b[reproducers] + bp for b, bp in zip(self.biases, bias_perturbs)]
        # logging.info(f"Bias sahpes: {[b.shape for b in new_biases]}")
        new_mutation_rates = mut + mut_rate_perturb

        new_memories = torch.zeros_like(self.memories[reproducers])
        new_energy = self.cfg.init_energy(new_sizes)
        new_health = self.cfg.init_health(new_sizes)
        new_ages = torch.zeros_like(self.ages[reproducers])
        
        new_head_dirs = torch.randn_like(self.head_dirs[reproducers])
        new_head_dirs /= torch.norm(new_head_dirs, dim=1, keepdim=True)
        
        new_rays = self.rays[reproducers] + ray_perturb
        new_rays[..., :2] /= torch.norm(new_rays[...,:2], dim=2, keepdim=True)
        new_rays[..., 2] = torch.clamp_min(new_rays[...,2], self.cfg.min_ray_dist) 

        pos_perturb = torch.randn_like(self.positions[reproducers]) * new_sizes.unsqueeze(1) * self.cfg.reproduce_dist
        new_positions = torch.clamp(self.positions[reproducers] + pos_perturb, *self.posn_bounds)

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
