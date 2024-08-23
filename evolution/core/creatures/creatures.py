from typing import Tuple, Union
import numpy as np
import torch    
torch.set_grad_enabled(False)
# import logging
# logging.basicConfig(level=logging.ERROR, filename='game.log')

from evolution.utils.batched_random import BatchedRandom
from evolution.cuda.cuda_utils import cuda_profile
from evolution.cuda.cu_algorithms import CUDAKernelManager
from evolution.core.config import Config

from .creature_array import CreatureArray


class Creatures(CreatureArray):
    """Class to store all relevant creature attributes in CUDA objects to ensure minimal
    movement between devices. To skip copying read-only parameters, we store 2 copies of the data.
    The first set, which is generally refered to as "underlying" memory is of size max_creatures 
    and stores (potentially out-of-date) data for all creatures.
    We store the set of indices into the underlying set that correspond to creatures that are still 
    alive in self.alive.
    The second set, which is generally referred to as "current" memory, is of size population, 
    and consists of all the attributes of creatures that are alive in Tensors. 
    Kernels and graphics read from and write to the current memory, but we want the underlying memory because
    it allows us to fuse the kill and reproduce operation into one operation to minimize copies.
    By forcing traits to remain on the GPU at all times, we can massively increase the speed of the simulation
    by utilizing massively parallel operations on the GPU. """
    
    def __init__(self, cfg: Config, kernels: CUDAKernelManager):
        super().__init__(cfg)
        self.cfg = cfg
        self.kernels = kernels

        # how much indexing into the food grid needs to be adjusted to avoid OOB
        self.pad = self.cfg.food_sight 
        self.offsets = torch.tensor([[i, j] for i in range(-self.pad, self.pad+1) 
                                     for j in range(-self.pad, self.pad+1)], device='cuda').unsqueeze(0)
        self.activations = []
        self.dead = None   # [N] current memory boolean tensor
        self._selected_creature = None
        
    def set_selected_creature(self, creature_id):
        self._selected_creature = creature_id
        
    def get_selected_creature(self):
        return self._selected_creature

    def fused_kill_reproduce(self, central_food_grid):
        """Kill the dead creatures and reproduce the living ones."""
        dead, num_dead = self._kill_dead(central_food_grid)  # determine who is dead
        alive = ~dead
        new_creatures = self._reproduce(alive, num_dead)  # determine who is reproducing
        if new_creatures or num_dead > 0:   # rearrange memory and update current data
            self._selected_creature = self.add_with_deaths(dead, alive, num_dead, new_creatures, 
                                                           self._selected_creature)        
                    
    #@cuda_profile
    def _kill_dead(self, central_food_grid: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """If any creature drops below 0 health or energy, kill it. Update the food grid by depositing an amount of food
        that depends on the creature's size at death."""
        if self.cfg.immortal:
            return None, 0
        health_deaths = (self.healths <= 0)
        energy_deaths = (self.energies <= 0)
        self.dead = health_deaths | energy_deaths   # congiuous boolean tensor
        
        num_dead = torch.sum(self.dead).item()
        if num_dead > 0:
            dead_posns = self.positions[self.dead].long()
            # food_grid[dead_posns[..., 1], dead_posns[..., 0]] += self.cfg.dead_drop_food(self.sizes[self.dead])
            central_food_grid.index_put_((dead_posns[..., 1], dead_posns[..., 0]), 
                                self.cfg.dead_drop_food(self.sizes[self.dead]), 
                                accumulate=True)
        
        return self.dead, num_dead
    
    #@cuda_profile
    def _get_reproducers(self, alive: Union[None, torch.Tensor]) -> torch.Tensor:
        # 1 + so that really small creatures don't get unfairly benefitted
        reproducers = (self.ages >= self.sizes*self.cfg.mature_age_mul) & (self.energies >= 1 + self.cfg.reproduce_thresh(self.sizes))  # current mem boolean tensor
        if alive is not None:
            reproducers &= alive
        return reproducers
            
    #@cuda_profile
    def _reduce_reproducers(self, reproducers: torch.Tensor, num_reproducers: int, 
                            max_reproducers: int) -> Tuple[torch.Tensor, int]:
        if num_reproducers > max_reproducers:  # this benefits older creatures because they 
            num_reproducers = max_reproducers  # are more likely to be in the num_reproducers window
            reproducer_indices = torch.nonzero(reproducers)#[num_reproducers:]
            perm = torch.randperm(reproducer_indices.shape[0], device='cuda')
            non_reproducers = reproducer_indices[perm[num_reproducers:]]
            reproducers[non_reproducers] = False
        return reproducers, num_reproducers

    #@cuda_profile
    def _reproduce(self, alive, num_dead) -> Union[None, 'CreatureArray']:
        """For sufficiently high energy creatures, create a new creature with mutated genes. Returns the set
        of new creatures in a CreatureArray object."""
        max_reproducers = self.cfg.max_creatures + num_dead - self.population # dont bother if none can reproduce anyway 
        if max_reproducers <= 0:
            return
        
        reproducers = self._get_reproducers(alive)  # current memory boolean tensor
        # limit reproducers so we don't exceed max size
        num_reproducers = torch.sum(reproducers)
        if num_reproducers == 0:  # no one can reproduce
            return
        
        # print(int(num_reproducers), num_dead)
        reproducers, num_reproducers = self._reduce_reproducers(reproducers, num_reproducers, max_reproducers)
        self.n_children[reproducers] += 1

        # could fuse this
        self.energies[reproducers] -= self.sizes[reproducers]  # subtract off the energy that you've put into the world
        self.energies[reproducers] /= self.cfg.reproduce_energy_loss_frac  # then lose a bit extra because this process is lossy
        
        new_creatures = self.reproduce_traits(reproducers, num_reproducers)
        return new_creatures

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
        pct_eaten = torch.zeros_like(food_grid, dtype=torch.float32, device='cuda') 
        threads_per_block = 512
        blocks_per_grid = self.population // threads_per_block + 1
        self.kernels('setup_eat_grid', blocks_per_grid, threads_per_block,
                     pos, self.eat_pcts, pct_eaten, self.population, food_grid.shape[0])
        
        if self.get_selected_creature() is not None:
            prev_energy = self.energies[self.get_selected_creature()].item()
        
        # calculate how much food each creature eats, add 1 to ages, update energies, take away living costs
        food_grid_updates = torch.zeros_like(food_grid, device='cuda')
        alive_costs = torch.zeros(self.population, device='cuda', dtype=torch.float32)
        self.kernels('eat', blocks_per_grid, threads_per_block,
                     pos, self.eat_pcts, pct_eaten, self.sizes, food_grid,
                     food_grid_updates, alive_costs, self.energies, self.healths, self.ages,
                     self.population, food_grid.shape[0], self.cfg.food_cover_decr)
        if self.get_selected_creature() is not None:
            creat_pos = pos[self.get_selected_creature()]
            # print(creat_pos, self.get_selected_creature())
            print(f"\tAge: {self.ages[self.get_selected_creature()].item()}"
                  f"\n\tChildren: {self.n_children[self.get_selected_creature()].item()}"
                  f"\n\tAlive Costs: {alive_costs[self.get_selected_creature()].item()}"
                  f"\n\tEat pct: {self.eat_pcts[self.get_selected_creature()].item()}"
                  f"\n\tFood Eaten: {self.energies[self.get_selected_creature()].item() - prev_energy}"
                  f"\n\tCell Energy: {food_grid[creat_pos[1], creat_pos[0]].item()}"
                  f"\n\tEnergy: {self.energies[self.get_selected_creature()].item()}")
        
        # grow food, apply eating costs
        # step_size = (torch.sum(alive_costs)/self.cfg.max_food/(self.cfg.size**2)).item()
        step_size = (self.cfg.max_food / (self.cfg.size**(1.9)))#.item()
        threads_per_block = (16, 16)
        blocks_per_grid = (food_grid.shape[0] // threads_per_block[0] + 1, 
                           food_grid.shape[1] // threads_per_block[1] + 1)
        
        if selected_cell is not None:
            prev_food = food_grid[selected_cell[1] + self.pad, selected_cell[0] + self.pad].item()
            
        self.kernels('grow', blocks_per_grid, threads_per_block,
                     food_grid_updates, food_grid, food_grid.shape[0], self.pad, step_size)

        # if selected_cell is not None:
        #     num_occupants = pos_counts[selected_cell[1]+self.pad, selected_cell[0]+self.pad]
        #     cover_cost = num_occupants * self.cfg.food_cover_decr
        #     eaten_amt = food_grid_updates[selected_cell[1]+self.pad, selected_cell[0]+self.pad] - cover_cost
        #     post_growth_food = food_grid[selected_cell[1]+self.pad, selected_cell[0]+self.pad]
        #     pre_grow_food = prev_food - cover_cost - eaten_amt
        #     print(f"Cell at {selected_cell}:"
        #           f"\n\toccupants: {num_occupants}"
        #           f"\n\tcover: {cover_cost}"
        #           f"\n\teaten: {eaten_amt}"
        #           f"\n\tpre-growth:  {pre_grow_food}"
        #           f"\n\tpost-growth: {post_growth_food}"
        #           f"\n\tstep_size: {step_size}"
        #           f"\n\tgrowth_amt: {post_growth_food - pre_grow_food}")


    def extract_food_windows(self, indices, food_grid):
        """Indices is [N, 2] indices into food_grid.
        Food grid is [S, S], where S is the width of the world.
        Returns a [N, W] tensor of the food in the W x W window around each index, 
        which goes as an input into creatures' brains."""
        windows = indices.unsqueeze(1) + self.offsets  # [N, 1, 2] + [1, 9, 2] = [N, 9, 2]
        if windows.min() < 0 or windows.max() >= food_grid.shape[0]:
            raise ValueError("OOB")
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
        self.memories[:] = outputs[:, 2:]  # only 2 outputs actually do something, so the rest are memory
        return outputs
    
    def rotate_creatures(self, outputs):
        """Given the neural network outputs `outputs`, rotate each creature accordingly.
        Rotation is a scalar between -1 and 1, which is combined with the creatures' size
        to compute the number of radians that are rotated. Negative rotation is rightwards, 
        positive is leftwards. We also take a small energy penalty for rotating."""
        # rotation => need to rotate the rays and the object's direction
        
        if self.get_selected_creature() is not None:
            energy_before = self.energies[self.get_selected_creature()].item()
        
        block_size = 512
        grid_size = self.population // block_size + 1
        rotation_matrix = torch.empty((self.population, 2, 2), device='cuda')  # [N, 2, 2]
        self.kernels('build_rotation_matrices', grid_size, block_size,
                     outputs, self.sizes, self.energies, self.population, outputs.shape[1],
                     rotation_matrix)
        
        if self.get_selected_creature() is not None:
            print(f"Creature {self.get_selected_creature()}:"
                  f"\n\tRotate Logit: {outputs[self.get_selected_creature(), 1].item()}"
                  f"\n\tRotate angle: {np.arccos(rotation_matrix[self.get_selected_creature(),0,0].item())}"
                  f"\n\tRotate Energy: {energy_before - self.energies[self.get_selected_creature()].item()}\n")
        

        # rotate the rays and head directions
        self.rays[..., :2] = self.rays[..., :2] @ rotation_matrix  # [N, 32, 2] @ [N, 2, 2]
        self.head_dirs[:] = (self.head_dirs.unsqueeze(1) @ rotation_matrix).squeeze(1)  # [N, 1, 2] @ [N, 2, 2]

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
        self.positions.normalize()# = torch.clamp(self.positions, *self.posn_bounds)  # don't let it go off the edge
        if self.get_selected_creature() is not None:
            print(f"\tMove Logit: {outputs[self.get_selected_creature(), 0].item()}"
                  f"\n\tMove Amt: {move[self.get_selected_creature()].item()}"
                  f"\n\tMove Energy: {move_cost[self.get_selected_creature()].item()}\n")
            

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
        
        if self.get_selected_creature() is not None:
            print(f"\tNum attacking: {attacks[self.get_selected_creature(), 0].item()}"
                  f"\n\tDamage Taken: {attacks[self.get_selected_creature(), 1].item()}"
                  f"\n\tHealth: {self.healths[self.get_selected_creature()].item()}"
                  f"\n\tAttack Energy: {attack_cost[self.get_selected_creature()].item()}\n")
            