from typing import Tuple, Union
import numpy as np
import torch    
torch.set_grad_enabled(False)
# import logging
# logging.basicConfig(level=logging.ERROR, filename='game.log')

from evolution.utils.batched_random import BatchedRandom
from evolution.cuda import cuda_utils
from evolution.cuda.cu_algorithms import CUDAKernelManager

from .config import Config



class CreatureArray():
    """Class to store all relevant creature attributes in CUDA objects to ensure minimal
    movement between devices. To skip copying read-only parameters, we store 2 copies of the data.
    The first set, which is generally refered to as "discontiguous" memory is of size max_creatures 
    and stores (potentially out-of-date) data for all creatures.
    We store the set of indices into the distinguous set that correspond to creatures that are still 
    alive in self.alive.
    The second set, which is generally referred to as "contiguous" memory, is of size population, 
    and consists of all the attributes of creatures that are alive in contiguous arrays. 
    Kernels and graphics read from and write to the contiguous memory, but we want the discontiguous memory because
    it allows us to fuse the kill and reproduce operation into one operation to minimize copies.
    By forcing traits to remain on the GPU at all times, we can massively increase the speed of the simulation
    by utilizing massively parallel operations on the GPU. """
    
    def __init__(self, cfg: Config, kernels: CUDAKernelManager):
        self.cfg = cfg
        self.kernels = kernels

        # how much indexing into the food grid needs to be adjusted to avoid OOB
        self.pad = self.cfg.food_sight 
        # self.posn_bounds = (pad, cfg.size+pad-1)
        self.posn_bounds = (0, cfg.size-1e-4)
        self.offsets = torch.tensor([[i, j] for i in range(-self.pad, self.pad+1) 
                                     for j in range(-self.pad, self.pad+1)], device='cuda').unsqueeze(0)

        # objects need to stay within [0, size-1] so that indexing makes sense
        self._positions = torch.empty(cfg.max_creatures, 2, device='cuda').uniform_(*self.posn_bounds)
        self._sizes = torch.empty(cfg.max_creatures, device='cuda').uniform_(*cfg.init_size_range)
        #logging.info(f"Initial sizes: {self.sizes}")
        self._memories = torch.zeros(cfg.max_creatures, cfg.mem_size, device='cuda')
        self._energies = cfg.init_energy(self._sizes)
        self._healths = cfg.init_health(self._sizes)
        self._ages = torch.zeros(cfg.max_creatures, device='cuda')

        self._rays = torch.randn(cfg.max_creatures, cfg.num_rays, 3, device='cuda')
        self._rays[..., :2] /= torch.norm(self._rays[..., :2], dim=2, keepdim=True)
        self._rays[..., 2] = torch.clamp(torch.abs(self._rays[..., 2]), 
                                         cfg.ray_dist_range[0]*self._sizes.unsqueeze(1),
                                         cfg.ray_dist_range[1]*self._sizes.unsqueeze(1))

        self._head_dirs = torch.randn(cfg.max_creatures, 2, device='cuda')
        self._head_dirs /= torch.norm(self._head_dirs, dim=1, keepdim=True)

        self._colors = torch.empty(cfg.max_creatures, 3, device='cuda').uniform_(1, 255)

        # we have mutation rate for rays, sizes, colors, weights/biases, and mutation rate itself
        self._mutation_rates = torch.empty(cfg.max_creatures, 6, device='cuda').uniform_(*cfg.init_mut_rate_range)

        # inputs: rays (x num_rays*3 colors), memory (x mem_size), food (3x3 = 9), health, energy
        # outputs: move forward/back, rotate, memory (x mem_size)
        self.layer_sizes = [cfg.num_rays*3 + cfg.mem_size + self.offsets.numel()//2 + 2, *cfg.brain_size, cfg.mem_size + 2]


        """weights are each [N, in, out]"""
        self._weights = [(torch.rand(cfg.max_creatures, prev_layer, next_layer, device='cuda')-0.5) / (np.sqrt(15.0 / prev_layer))
                        for prev_layer, next_layer in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        
        self._biases = [(torch.rand(cfg.max_creatures, 1, next_layer, device='cuda')-0.5) / np.sqrt(next_layer)
                       for next_layer in self.layer_sizes[1:]]
        self.activations = []
        self.dead = None   # [N] contiguous boolean tensor
        self.dead_idxs = None # [N] discontiguous index tensor of creatures that died on last step
        reproduce_dims = {'sizes': self._sizes, 
                          'colors': self._colors, 
                          'weights': self._weights,
                          'biases': self._biases,
                          'mutation_rates': self._mutation_rates,
                          'head_dirs': self._head_dirs,
                          'rays': self._rays,
                          'positions': self._positions}
        self.reproduce_rand = BatchedRandom(reproduce_dims)
        
        self.alive = torch.zeros(cfg.max_creatures, dtype=torch.bool, device='cuda')
        self.alive[:cfg.start_creatures] = True
        self.population = cfg.start_creatures
        self.reindex()

    def update_old_memory(self):
        """Write back updated memory to the old creatures."""
        self._positions[self.alive] = self.positions
        self._memories[self.alive] = self.memories
        self._energies[self.alive] = self.energies
        self._healths[self.alive] = self.healths
        self._ages[self.alive] = self.ages
        self._head_dirs[self.alive] = self.head_dirs
        self._rays[self.alive] = self.rays
        
    def reindex(self):
        """Reindex the creatures so that the dead creatures are removed."""    
        # writeable params    
        self.positions = self._positions[self.alive]
        self.memories = self._memories[self.alive]
        self.energies = self._energies[self.alive]
        self.healths = self._healths[self.alive]
        self.ages = self._ages[self.alive]
        self.rays = self._rays[self.alive]
        self.head_dirs = self._head_dirs[self.alive]
        
        # readonly params
        self.sizes = self._sizes[self.alive]
        self.colors = self._colors[self.alive]
        self.mutation_rates = self._mutation_rates[self.alive]
        self.weights = [w[self.alive] for w in self._weights]
        self.biases = [b[self.alive] for b in self._biases]
        

    def fused_kill_reproduce(self, central_food_grid):
        """Kill the dead creatures and reproduce the living ones."""
        deads = self._kill_dead(central_food_grid)
        # print('deads is', deads)
        any_reproduced = self._reproduce(deads)
        if deads is not None or any_reproduced:
            self.reindex()
        
    @cuda_utils.cuda_profile
    def _kill_dead(self, central_food_grid: torch.Tensor):
        """If any creature drops below 0 health or energy, kill it. Update the food grid by depositing an amount of food
        that depends on the creature's size at death. If any creatures die, we """
        if self.cfg.immortal:
            return
        health_deaths = (self.healths <= 0)
        energy_deaths = (self.energies <= 0)
        self.dead = health_deaths | energy_deaths   # congiuous boolean tensor
        if not torch.any(self.dead):
            return
        dead_posns = self.positions[self.dead].long()
        # food_grid[dead_posns[..., 1], dead_posns[..., 0]] += self.cfg.dead_drop_food(self.sizes[self.dead])
        central_food_grid.index_put_((dead_posns[..., 1], dead_posns[..., 0]), 
                             self.cfg.dead_drop_food(self.sizes[self.dead]), 
                             accumulate=True)
        
        if self.dead is not None:
            # print('dead', self.dead)
            self.update_old_memory()
            alive_idxs = torch.nonzero(self.alive).squeeze()    # contiguous index tensor with values that are discintiguous indices
            # very important to call .item(), else population will become a tensor and mess up kernel sizes and such
            self.population -= self.dead.sum().item()
            try:
                self.dead_idxs = alive_idxs[self.dead]    # we index it with a contiguous boolean tensor -> gives discontiguous indices
            except IndexError:
                print('alive_idxs', alive_idxs)
                print('alive', self.alive)
                print('dead', self.dead.shape)
                print('population', self.population)
                raise

            self.alive[self.dead_idxs] = False
        return self.dead

    @cuda_utils.cuda_profile
    def _reproduce(self, deads) -> bool:
        """For sufficiently high energy creatures, create a new creature with mutated genes. Updates self.alive,
        contiguous memory storage (_attrs) for old creatures if this wasn't already done by _kill_dead, writes
        to contiguous memory storage the new creature attributes. Returns True if at least 1 creature reproduces,
        otherwise False."""
        max_reproducers = self.cfg.max_creatures - self.population # dont bother if none can reproduce anyway 
        if max_reproducers <= 0:
            return False
        
        # 1 + so that really small creatures don't get unfairly benefitted
        reproducers = (self.ages >= self.sizes*self.cfg.mature_age_mul) & (self.energies >= 1 + self.cfg.reproduce_thresh(self.sizes))  # contiguous boolean tensor
        if deads is not None:
            reproducers &= ~deads   # only alive creatures can reproduce
        
        # limit reproducers so we don't exceed max size
        num_reproducers = torch.sum(reproducers)
        if num_reproducers == 0:  # no one can reproduce
            return False
        
        if num_reproducers > max_reproducers:  # this benefits older creatures because they 
            num_reproducers = max_reproducers  # are more likely to be in the num_reproducers window
            reproducer_indices = torch.nonzero(reproducers)#[num_reproducers:]
            perm = torch.randperm(reproducer_indices.shape[0], device='cuda')
            non_reproducers = reproducer_indices[perm[num_reproducers:]]
            reproducers[non_reproducers] = False


        # print('reproduce', reproducers)
        # could fuse this
        self.energies[reproducers] -= self.sizes[reproducers]  # subtract off the energy that you've put into the world
        self.energies[reproducers] /= self.cfg.reproduce_energy_loss_frac  # then lose a bit extra because this process is lossy
        # #logging.info(f"Energy after reproduce: {self.energies[reproducers]}")
        
        mut = self.mutation_rates[reproducers]
        self.reproduce_rand.generate(num_reproducers)

        size_perturb = self.reproduce_rand.get('sizes') * mut[:,0]
        color_perturb = self.reproduce_rand.get('colors') * mut[:, 1, None]
        ray_perturb = self.reproduce_rand.get('rays') * mut[:, 2, None, None]        

        weight_perturbs = [self.reproduce_rand.get('weights', i) for i in range(len(self.weights))]
        bias_perturbs = [self.reproduce_rand.get('biases', i) for i in range(len(self.biases))]

        mut_rate_perturb = self.reproduce_rand.get('mutation_rates') * mut[:, 5, None]

        new_head_dirs = self.reproduce_rand.get('head_dirs')
        pos_perturb = self.reproduce_rand.get('positions') * self.cfg.reproduce_dist#* new_sizes.unsqueeze(1) * self.cfg.reproduce_dist

        new_sizes = torch.clamp(self.sizes[reproducers] + size_perturb, *self.cfg.size_range)
        new_colors = torch.clamp(self.colors[reproducers] + color_perturb, 1, 255)

        new_weights = [w[reproducers] + wp for w, wp in zip(self.weights, weight_perturbs)]
        new_biases = [b[reproducers] + bp for b, bp in zip(self.biases, bias_perturbs)]
        
        new_mutation_rates = mut + mut_rate_perturb

        new_memories = torch.zeros_like(self.memories[reproducers])
        new_energy = self.cfg.init_energy(new_sizes)
        new_health = self.cfg.init_health(new_sizes)
        new_ages = torch.zeros_like(self.ages[reproducers])
        
        
        new_rays = self.rays[reproducers] + ray_perturb
        new_rays[..., :2] /= torch.norm(new_rays[...,:2], dim=2, keepdim=True)
        new_rays[..., 2] = torch.clamp(new_rays[...,2],
                                       self.cfg.ray_dist_range[0]*new_sizes.unsqueeze(1),
                                        self.cfg.ray_dist_range[1]*new_sizes.unsqueeze(1)) 

        new_head_dirs /= torch.norm(new_head_dirs, dim=1, keepdim=True)

        new_positions = torch.clamp(self.positions[reproducers] + pos_perturb, *self.posn_bounds)
                
        new_alives = torch.nonzero(~self.alive)[:num_reproducers].squeeze(1)
        
        self._positions[new_alives] = new_positions
        self._sizes[new_alives] = new_sizes
        self._memories[new_alives] = new_memories
        self._energies[new_alives] = new_energy
        self._healths[new_alives] = new_health
        self._ages[new_alives] = new_ages
        self._rays[new_alives] = new_rays
        self._head_dirs[new_alives] = new_head_dirs
        self._colors[new_alives] = new_colors
        self._mutation_rates[new_alives] = new_mutation_rates
        
        for i, nw in enumerate(new_weights):
            self._weights[i][new_alives] = nw
            
        for i, nb in enumerate(new_biases):
            self._biases[i][new_alives] = nb
            
        if deads is None:   # if deads is not None, then we've already called update_old_memory, so we don't need to do it again
            self.update_old_memory()
        self.alive[new_alives] = True
        
        # very important to call int(), else population will become a tensor and mess up kernel sizes and such
        self.population += int(num_reproducers)
        
        return True

    def eat_grow(self, food_grid: torch.Tensor, selected_cell: Union[None, Tuple[int, int]]):
        """Compute how much food each creature eats, by taking a fixed percentage of the available
        food in the cell it is in. If there are more creatures in a cell than the reciprocal of 
        this fixed percentage (i.e. there isn't enough food to go around), then each creature 
        instead gets an equal share of the total food in the cell. 
        
        After eating, we apply food_cover costs to the cell. This represents something like the
        presence of creatures making the environment less hospitable for food growth / more toxic.
        This is a fixed cost per creature in the cell.
        
        Finally, we allow food in all cells to grow by an amount proportional to its distance from
        the 'maximum' food level. Food levels beyond the maximum food level are decayed 
        proportionally to how high above the maximum they are."""
        
        pos = self.positions.int() + self.pad
        
        # calculate how many creatures are in each position, useful for computng how much food each creature eats
        pos_counts = torch.zeros_like(food_grid, dtype=torch.int, device='cuda') 
        threads_per_block = 512
        blocks_per_grid = self.population // threads_per_block + 1
        self.kernels('setup_eat_grid', blocks_per_grid, threads_per_block,
                     pos, pos_counts, self.population, food_grid.shape[0])
        
        
        # calculate how much food each creature eats, add 1 to ages, update energies, take away living costs
        food_grid_updates = torch.zeros_like(food_grid, device='cuda')
        alive_costs = torch.zeros(self.population, device='cuda', dtype=torch.float32)
        self.kernels('eat', blocks_per_grid, threads_per_block,
                     pos, pos_counts, self.sizes, food_grid,
                     food_grid_updates, alive_costs, self.energies, self.ages,
                     self.population, food_grid.shape[0], self.cfg.food_cover_decr, 1. / self.cfg.eat_pct)
        
        # grow food, apply eating costs
        step_size = (torch.sum(alive_costs)/self.cfg.max_food/(self.cfg.size**2)).item()
        threads_per_block = (16, 16)
        blocks_per_grid = (food_grid.shape[0] // threads_per_block[0] + 1, 
                           food_grid.shape[1] // threads_per_block[1] + 1)
        
        if selected_cell is not None:
            prev_food = food_grid[selected_cell[1] + self.pad, selected_cell[0] + self.pad].item()
            
        self.kernels('grow', blocks_per_grid, threads_per_block,
                     food_grid_updates, food_grid, food_grid.shape[0], self.pad, step_size)

        if selected_cell is not None:
            num_occupants = pos_counts[selected_cell[1]+self.pad, selected_cell[0]+self.pad]
            cover_cost = num_occupants * self.cfg.food_cover_decr
            eaten_amt = food_grid_updates[selected_cell[1]+self.pad, selected_cell[0]+self.pad] - cover_cost
            post_growth_food = food_grid[selected_cell[1]+self.pad, selected_cell[0]+self.pad]
            pre_grow_food = prev_food - cover_cost - eaten_amt
            print(f"Cell at {selected_cell}:"
                  f"\n\toccupants: {num_occupants}"
                  f"\n\tcover: {cover_cost}"
                  f"\n\teaten: {eaten_amt}"
                  f"\n\tpre-growth:  {pre_grow_food}"
                  f"\n\tpost-growth: {post_growth_food}"
                  f"\n\tstep_size: {step_size}"
                  f"\n\tgrowth_amt: {post_growth_food - pre_grow_food}")


    def extract_food_windows(self, indices, food_grid):
        """Indices is [N, 2] indices into food_grid.
        Food grid is [S, S], where S is the width of the world.
        Returns a [N, W] tensor of the food in the W x W window around each index, 
        which goes as an input into creatures' brains."""
        windows = indices.unsqueeze(1) + self.offsets  # [N, 1, 2] + [1, 9, 2] = [N, 9, 2]
        return food_grid[windows[..., 1], windows[..., 0]]
        

    def collect_stimuli(self, collisions, food_grid):
        """Collisions: [N, 3], food_grid: [N, 3, 3] -> [N, F] where F is the number of features
        that get passed in to the creatures' brains."""
        # get the food in the creature's vicinity
        # we must add self.pad, because positions are stored in central_food_grid coordinates,
        # but we need to index into food_grid with food_grid coordinates (which includes padding)
        posn = self.positions.long() + self.pad
        food = self.extract_food_windows(posn, food_grid)  # [N, 9]
        rays_results = collisions.view(self.population, -1)  # [N, 3*num_rays]
        
        return torch.cat([rays_results, 
                          self.memories, 
                          food, 
                          self.healths.unsqueeze(1), 
                          self.energies.unsqueeze(1)
                        ], 
                        dim=1)  # [N, F]

    def forward(self, inputs):
        """Inputs: [N, 1, F] tensor.
        Compute the forward pass of the neural network brain for each creature, and update
        their memories by the output of the network. 
        Returns: `outputs` [N, 2+cfg.mem_size], where the first coordinate is how much 
            to move forward/backward the 2nd coordinate is how much to rotate rotate left/rotate, 
            and the last cfg.mem_size coordinates are for memory."""
        
        self.activations.clear()
        for w, b in zip(self.weights, self.biases):
            self.activations.append(inputs)
            inputs = torch.tanh(inputs @ w + b)
        outputs = inputs.squeeze(dim=1)  # [N, O]
        self.activations.append(inputs)
        self.memories = outputs[:, 2:]  # only 2 outputs actually do something, so the rest are memory
        return outputs
    
    def rotate_creatures(self, outputs):
        """Given the neural network outputs `outputs`, rotate each creature accordingly.
        Rotation is a scalar between -1 and 1, which is combined with the creatures' size
        to compute the number of radians that are rotated. Negative rotation is rightwards, 
        positive is leftwards. We also take a small energy penalty for rotating."""
        # rotation => need to rotate the rays and the object's direction
        
        block_size = 512
        grid_size = self.population // block_size + 1
        rotation_matrix = torch.empty((self.population, 2, 2), device='cuda')  # [N, 2, 2]
        self.kernels('build_rotation_matrices', grid_size, block_size,
                     outputs, self.sizes, self.energies, self.population, outputs.shape[1],
                     rotation_matrix)
        

        # rotate the rays and head directions
        self.rays[..., :2] = self.rays[..., :2] @ rotation_matrix  # [N, 32, 2] @ [N, 2, 2]
        self.head_dirs = (self.head_dirs.unsqueeze(1) @ rotation_matrix).squeeze(1)  # [N, 1, 2] @ [N, 2, 2]

    def move_creatures(self, outputs):
        """Given the neural network outputs `outputs`, move each creature accordingly.
        Moving forward is a scalar between -1 and 1, which is combined with the creatures' size
        to compute the distance that is moved. Positive scalars are movement forward, and negative
        scalars are backwards movement. We also take a small energy penalty for moving.
        We clamp the position to stay inside the grid, but penalize the creature for the full 
        movement anyway."""
        # move => need to move the object's position
        # maybe add some acceleration/velocity thing instead
        move = self.cfg.move_amt(outputs[:,0], self.sizes)
        move_cost = self.cfg.move_cost(move, self.sizes)
        #logging.info(f"Move amt: {move}")
        #logging.info(f"Move cost: {move_cost}")
        self.energies -= move_cost
        self.positions += self.head_dirs * move.unsqueeze(1)   # move the object's to new position
        self.positions = torch.clamp(self.positions, *self.posn_bounds)  # don't let it go off the edge

    def do_attacks(self, attacks):
        """Attacks is [N, 2], where the 1st coordinate is an integer indicating how many things the 
        organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the 
        organism is taking from other things attacking it. We update each 
        creature's health and energy accordingly."""

        # print('nrg_before', self.energies)
        # #logging.info(f'Attacks:\n{attacks}')
        # self.energies -= attacks[:,0] * self.sizes * 0.1  # takes 0.1 * size to do an attack
        attack_cost = self.cfg.attack_cost(attacks[:,0], self.sizes)
        # attack_cost = attacks[:, 0] * self.sizes * self.cfg.attack_cost.mul
        self.energies -= attack_cost
        #logging.info(f"Attack energy: {attack_cost}")
        self.healths -= attacks[:,1]
    
    def get_selected_creature(self, creature_id):
        """Given a discontiguous creature_id, retrieve the contiguous index. Useful
        for the interactive component, since contiguous creature ids can change their meaning
        rapidly, but discontiguous ids always mean the same thing."""
        if creature_id is None:
            return None
        
        if self.dead_idxs is not None and creature_id in self.dead_idxs:
            return None
        
        contig_creature = self.alive[:creature_id].sum().item()
        return contig_creature
            