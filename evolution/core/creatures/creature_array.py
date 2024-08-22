from functools import partial
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
torch.set_grad_enabled(False)
import torch.nn.functional as F
from torch import Tensor

from evolution.utils.batched_random import BatchedRandom

from .config import Config
from .creature_param import CreatureParam

def normalize_rays(rays: 'CreatureParam', sizes: 'CreatureParam', cfg: Config):
    rays[..., :2] /= torch.norm(rays[..., :2], dim=2, keepdim=True)
    rays[..., 2] = torch.clamp(torch.abs(rays[..., 2]),
                cfg.ray_dist_range[0]*sizes.unsqueeze(1),
                cfg.ray_dist_range[1]*sizes.unsqueeze(1))
    
def normalize_head_dirs(head_dirs: 'CreatureParam'):
    head_dirs /= torch.norm(head_dirs, dim=1, keepdim=True)
    
def clamp(x: Tensor, min_: float, max_: float):
    x[:] = torch.clamp(x, min_, max_)


class CreatureArray:
    """A class to manage the underlying memory for a Creatures class. It consists of a set of 
    CreatureParam objects that represent the different traits of the creatures. It handles
    their generation, reproduction, and death, as well as the reordering of memory needed
    when creatures die or reproduce."""
    def __init__(self, cfg: Config, device='cuda'):
        self.cfg = cfg
        self.posn_bounds = (0, cfg.size-1e-4)
        self.num_food = (cfg.food_sight*2+1)**2
        self.device = device
        
        
    def generate_from_cfg(self):
        """Generate a new set of creatures from the configuration. This should only be called for
        the root CreatureArray, that represents all the creatures in the world."""
        self.sizes: CreatureParam = CreatureParam('sizes', tuple(), ('uniform_', *self.cfg.init_size_range), 
                              partial(clamp, min_=self.cfg.size_range[0], max_=self.cfg.size_range[1]), 
                              self.device, 0)
        self.colors: CreatureParam = CreatureParam('colors', (3,), ('uniform_', 1, 255),
                              partial(clamp, min_=1, max_=255), self.device, 1)
        self.rays: CreatureParam = CreatureParam('rays', (self.cfg.num_rays, 3), ('normal_', 0, 1),
                              None, self.device, 2)
        self.mutation_rates: CreatureParam = CreatureParam('mutation_rates', (6,), 
                                                           ('uniform_', *self.cfg.init_mut_rate_range),
                              None, self.device, 5)
        self.positions: CreatureParam = CreatureParam('positions', (2,), ('uniform_', *self.posn_bounds),
                              partial(clamp, min_=self.posn_bounds[0], max_=self.posn_bounds[1]), 
                              self.device, None)
        
        self.energies: CreatureParam = CreatureParam('energies', tuple(), None, None, self.device, None)
        self.healths: CreatureParam = CreatureParam('healths', tuple(), None, None, self.device, None)
        self.memories: CreatureParam = CreatureParam('memories', (self.cfg.mem_size,), ('zero_',), None, 
                                                     self.device, None)
        self.ages: CreatureParam = CreatureParam('ages', tuple(), ('zero_',), None, self.device, None)
        self.n_children: CreatureParam = CreatureParam('n_children', tuple(), ('zero_',), None, self.device, None)
        self.head_dirs: CreatureParam = CreatureParam('head_dirs', (2,), ('normal_', 0, 1), 
                                       normalize_head_dirs, self.device, None)
        
        layer_sizes = [self.cfg.num_rays*3 + self.cfg.mem_size + self.num_food + 2, 
            *self.cfg.brain_size, self.cfg.mem_size + 2]
        weight_sizes = [(prev_layer, next_layer) for prev_layer, next_layer in zip(layer_sizes[:-1], layer_sizes[1:])]
        bias_sizes = [(1, next_layer) for next_layer in layer_sizes[1:]]
        weight_inits = []
        bias_inits = []
        for prev_layer, next_layer in weight_sizes:
            w_norm = 0.5 / np.sqrt(15.0 / prev_layer)
            b_norm = 0.5 / np.sqrt(next_layer)
            weight_inits.append(('uniform_', -w_norm, w_norm))
            bias_inits.append(('uniform_', -b_norm, b_norm))
        self.weights: CreatureParam = CreatureParam('weights', weight_sizes, weight_inits, None, self.device, 3)
        self.biases: CreatureParam = CreatureParam('biases', bias_sizes, bias_inits, None, self.device, 4)
        
        # select all CreatureParam objects
        self.variables: List[CreatureParam] = [getattr(self, p) for p in dir(self) 
                                               if isinstance(getattr(self, p), CreatureParam)]
        for v in self.variables:
            v.init_base(self.cfg)
        self.energies.init_base_from_data(self.cfg.init_energy(self.sizes.data))
        self.healths.init_base_from_data(self.cfg.init_health(self.sizes.data))
        normalize_rays(self.rays, self.sizes, self.cfg)

        self.start_idx = 0
        self.alive = torch.zeros(self.cfg.max_creatures, dtype=torch.float32, device='cuda')
        self.alive[:self.cfg.start_creatures] = 1.0
        self.rng = BatchedRandom(self.variables)

 
    def reproduce(self, reproducers: Tensor, num_reproducers: int) -> 'CreatureArray':
        """Generate descendant traits from the set of indices in reproducers and store it in 
        a new CreatureArray.
        
        Args:
            reproducers: A boolean tensor of creatures (size of current memory) that are reproducing.
            num_reproducers: The number of creatures that are reproducing.
        """
        mut = self.mutation_rates[reproducers]
        self.rng.generate(num_reproducers)
        self.n_children[reproducers] += 1
        children = CreatureArray(self.cfg, self.device)
        
        for v in self.variables:   # handle mutable parameters
            child_trait = v.reproduce(self.rng, reproducers, num_reproducers, mut)
            if child_trait is not None:
                setattr(children, v.name, child_trait)
        
        # we can pretend position is a mutable parameter since it is a random deviation from the parent
        children.positions = self.positions.reproduce_mutable(self.rng, reproducers, 
                                                              self.cfg.reproduce_dist, force_mutable=True)
        # need to call normalization weird for rays since it depends on sizes
        normalize_rays(children.rays, children.sizes, self.cfg)
        # need to initialize these weird as well because they depend on sizes
        children.energies = self.cfg.init_energy(children.sizes)
        children.healths = self.cfg.init_health(children.sizes)
        return children
            
    @property
    def population(self):
        return self.positions.population
    
    def write_new_data(self, idxs: Tensor, other: 'CreatureArray'):
        """Write the data of the creatures in `other` at the indices `idxs` into the current block
        for all CreatureParams in each.
        
        Args:
            idxs: Tensor of indices (into underlying memory) where we want to write the new creatures.
            other: The CreatureArray that contains the new creatures.
        """
        for v in self.variables:
            v.write_new(idxs, getattr(other, v.name))
            
    def reindex(self, start_idx: int, n_after: int):
        """Slice each CreatureParam according to the current block so that their `data` attribute
        is set properly for the next generation."""
        for v in self.variables:
            v.reindex(start_idx, n_after)
            
    def rearrange_old_data(self, outer_idxs: Tensor, inner_idxs: Tensor):
        """Move creatures that are outside the best window to the inside of the best window.
        We write to the underlying memory here, and then update the current memory later
        when we call self.reindex.
        
        Args:
            outer_idxs: Tensor of indices (into underlying memory) of creatures that are outside the best window.
            inner_idxs: Tensor of indices (into underlying memory) where we want to move them to, inside 
                        the best window
        """
        for v in self.variables:
            v.rearrange_old_data(outer_idxs, inner_idxs)

    def add_with_deaths(self, dead: Union[None, Tensor], alive: Union[None, Tensor], 
                        num_dead: int, other: Union[None, 'CreatureArray'], 
                        creature: Union[None, int]) -> Union[None, int]:
        """Once we have determined which creatures are reproducing and which are dying, we need
        to re-order the memory to do this as efficiently as possible. Since copying is very expensive,
        we try our best to minimize the number of copies. We have 3 separate strategies for 
        finding contiguous blocks of creatures (since we can slice a contiguous block, avoiding
        a huge copy each generation), as outlined in functions `add_with_deaths_*`. In
        addition, we also update the selected creature (in the `creature` argument), since it can
        move around in an arbitrary manner now.
        
        Args:
            dead: A boolean tensor (size of current memory) of creatures that are dead.
            alive: A boolean tensor (size of current memory) of creatures that are alive.
            num_dead: The number of creatures that are dead.
            other: The CreatureArray that contains the new creatures (if any).
            creature: The index of the selected creature (if any).
            
        Returns:
            The updated index of the selected creature (or None if it died)."""
        
        sl = slice(self.start_idx, self.start_idx + self.population)  # current block
        self.alive[sl] = alive.float()  # update which creatures are alive
        if creature is not None:  # if selected_creature is dead, then we can set it to None
            if dead[creature]:
                creature = None
        num_reproducers = other.population if other is not None else 0  # how many new creatures we are adding
        n_after = self.population - num_dead + num_reproducers  # how many creatures are alive after this step
        
        # select and apply block repair strategy
        if n_after == self.cfg.max_creatures:
            self.add_with_deaths_max_array(dead, other)
        elif num_reproducers >= num_dead:
            creature = self.add_with_deaths_fill_gaps(dead, other, num_dead, num_reproducers, sl, creature)
        else:
            creature = self.add_with_deaths_move_block(other, n_after, num_reproducers, sl, creature)
        
        # update self.alive to reflect the new block
        self.alive[self.start_idx : self.start_idx+n_after] = 1.0  # mark the new creatures as alive
        self.alive[:self.start_idx] = 0.0  # mark creatures outside the new block as dead
        self.alive[self.start_idx+n_after:] = 0.0
        # slice the underlying data arrays to give the current updated data
        self.reindex(self.start_idx, n_after)
        return creature

    def add_with_deaths_max_array(self, dead: Tensor, other: 'CreatureArray'):
        """If the number of creatures after death and reproduction will fill the entire buffer,
        we can simply put new creatures in the dead creatures spots, and set start_idx to 0.
        
        Args:
            dead: A boolean tensor (size of current memory) of creatures that are dead.
            other: The CreatureArray that contains the new creatures (if any).
        """
        self.start_idx = 0  # case where we just have to fill in the gaps
        if dead.shape[0] == self.cfg.max_creatures:
                # if prev epoch was also maxxed, then we can just use dead
            missing = dead.nonzero().squeeze(1)
        else:
                # otherwise, we need to look at the full indices of self.alive
            missing = (1-self.alive).nonzero().squeeze(1)
        self.write_new_data(missing, other)
        
    def add_with_deaths_fill_gaps(self, dead: Tensor, other: 'CreatureArray',
                                  num_dead: int, num_reproducers: int, sl: slice, creature: int) -> int:
        """If we add more creatures than we kill in this step, then we will be able to fill in all the
        gaps created in the block created by the dead creatures. Then, we put the remaining creatures on
        either side of the new block. We try our best to move the start of the block as far back to the
        left as we can, so that things stay relatively consistent throughout the generations.
        
        Args:
            dead: A boolean tensor (size of current memory) of creatures that are dead.
            other: The CreatureArray that contains the new creatures (if any).
            num_dead: The number of creatures that are dead.
            num_reproducers: The number of creatures that are reproducing.
            sl: The slice corresponding to the current block of creatures.
            creature: The index of the selected creature (if any).
        Returns:
            The updated index of the selected creature (or None if it died)."""
        missing = dead.nonzero().squeeze(1) + self.start_idx   # missing spots in current block
        num_added = num_reproducers - num_dead   # how many we need to add to the sides
        before_avail = self.start_idx   # how much we can add to the left gap
        
        if num_added >= before_avail:  # with the extra we need to add, we can totally fill in the left gap
            missing = torch.cat((torch.arange(sl.start, device=self.device), missing))  # add left gap indices
            self.start_idx = 0   # block will begin from 0 now
            num_added -= before_avail   # we've used up before_avail worth of creatures filling left gap
        else:  # otherwise, the extra creatures only partially fill in the left gap
            missing = torch.cat((torch.arange(sl.start-num_added, sl.start, device=self.device), missing))
            self.start_idx = self.start_idx - num_added   # start of block shifts back a bit
            num_added = 0  # and we are done adding creatures to the left gap
        if num_added > 0:  # add the remaining creatures to the right gap
            missing = torch.cat((missing, torch.arange(sl.stop, sl.stop+num_added, device=self.device)))
            
        # add the new creatures to the gaps in the block and to the sides
        self.write_new_data(missing, other)
        
        # block has been adjusted backwards, so we need to push the pointer forward
        if creature is not None:
            creature += (sl.start - self.start_idx)
        return creature
    
    def add_with_deaths_move_block(self, other: 'CreatureArray', n_after: int, 
                                   num_reproducers: int, sl: slice, creature: int) -> int:
        """If we have more dead creatures than new creatures, then we need to totally restart our
        block and find the best window of `n_after` creatures that we should put our block in. We
        do this by finding which window of `n_after` creatures already contains the most creatures.
        We can then complete the block by filling in the gaps with new creatures and moving old 
        creatures that are outside this block into the other gaps.
        
        Args:
            other: The CreatureArray that contains the new creatures (if any).
            n_after: The number of creatures that are alive after this step.
            num_reproducers: The number of creatures that are reproducing.
            sl: The slice corresponding to the current block of creatures.
            creature: The index of the selected creature (if any).
        Returns:
            The updated index of the selected creature (or None if it died)."""
        window = torch.ones(n_after, device='cuda', dtype=torch.float)
        # count the number of alive creatures in each window of size n_after
        windows = F.conv1d(self.alive[sl].view(1, 1, -1), window.view(1, 1, -1), padding=0).squeeze()
        best_start = windows.argmax().item() + self.start_idx
        # find the gaps in the best window identified that have to be filled by something
        missing = (1-(self.alive[best_start:best_start+n_after])).nonzero().squeeze(1) + best_start
        # print(missing)
        new_idxs = missing[:num_reproducers]   # we can fill some of them with new creatures
        old_idxs = missing[num_reproducers:]   # and fill the rest by rearranging old creatures
        
        # indices of old creatures outside the new best window that have to be moved in to it
        outer_idxs = torch.cat([
            self.alive[sl.start : best_start].nonzero().squeeze(1) + sl.start,  # left side of window
            self.alive[best_start+n_after : sl.stop].nonzero().squeeze(1) + best_start+n_after # right side of window
            ])
        
        # update the selected creature
        if creature is not None:
            # find its absolute index into the entire underlying data buffer
            discontig_creature = creature + sl.start
            # if it is outside the best window, find where it is being moved to
            move_idx = (outer_idxs == discontig_creature).nonzero().squeeze(1)
            if move_idx.shape[0] > 0:
                discontig_creature = old_idxs[move_idx]
            # turn it back into a contiguous index
            creature = discontig_creature - self.start_idx
        
        # move the old creatures that are outside the best window into the gaps in the best window
        self.rearrange_old_data(outer_idxs, old_idxs)
        # add the new creatures to the other gaps in the best window
        if other is not None:
            self.write_new_data(new_idxs, other)
        self.start_idx = best_start  # update the start of the block to the best window found
        return creature