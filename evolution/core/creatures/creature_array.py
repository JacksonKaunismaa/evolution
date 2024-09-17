from functools import partial
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from torch import Tensor

from evolution.utils.batched_random import BatchedRandom
from evolution.utils.quantize import quantize, QuantizedData
from evolution.core.config import Config
from evolution.cuda.cuda_utils import cuda_profile
from evolution.state.game_state import GameState

from .creature_trait import CreatureTrait, Initializer, InitializerStyle


def _normalize_rays(rays: 'CreatureTrait', cfg: Config):
    rays[..., :2] /= torch.norm(rays[..., :2], dim=2, keepdim=True)
    rays[..., 2] = torch.clamp(torch.abs(rays[..., 2]),
                cfg.ray_dist_range[0],
                cfg.ray_dist_range[1])

def normalize_head_dirs(head_dirs: 'CreatureTrait'):
    head_dirs /= torch.norm(head_dirs, dim=1, keepdim=True)

def _eat_amt(sizes: CreatureTrait, cfg: Config) -> Tensor:
    if cfg.eat_pct_scaling[0] == 'linear':
        size_pct = (sizes - cfg.size_range[0]) / (cfg.size_range[1] - cfg.size_range[0])
    elif cfg.eat_pct_scaling[0] == 'log':
        size_pct = torch.log(sizes / cfg.size_range[0]) / np.log(cfg.size_range[1] / cfg.size_range[0])
    else:
        raise ValueError(f"Unrecognized eat_pct_scaling: {cfg.eat_pct_scaling[0]}")

    if cfg.eat_pct_scaling[1] == 'linear':  # pylint: disable=no-else-return
        return cfg.eat_pct[0] + size_pct * (cfg.eat_pct[1] - cfg.eat_pct[0])
    elif cfg.eat_pct_scaling[1] == 'log':
        return cfg.eat_pct[0] * torch.exp(size_pct * np.log(cfg.eat_pct[1] / cfg.eat_pct[0]))
    else:
        raise ValueError(f"Unrecognized eat_pct_scaling: {cfg.eat_pct_scaling[1]}")

def clamp(x: Tensor, bounds: Tuple[float, float]):
    min_, max_ = bounds
    x[:] = torch.clamp(x, min=min_, max=max_)


class CreatureArray:
    """A class to manage the underlying memory for a Creatures class. It consists of a set of
    CreatureTrait objects that represent the different traits of the creatures. It handles
    their generation, reproduction, and death, as well as the reordering of memory needed
    when creatures die or reproduce."""
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.posn_bounds = (0, cfg.size-1e-3)
        self.num_food = (cfg.food_sight*2+1)**2
        self.device = device
        self.variables: Dict[str, CreatureTrait] = {}
        self.num_mutable = 0

    def generate_from_cfg(self):
        """Generate a new set of creatures from the configuration. This should only be called for
        the root CreatureArray, that represents all the creatures in the world."""
        self.sizes: CreatureTrait = CreatureTrait(tuple(),
                                                  Initializer.mutable('uniform_', *self.cfg.init_size_range),
                                                  partial(clamp, bounds=self.cfg.size_range),
                                                  self.device)
        color_channels = 3
        self.colors: CreatureTrait = CreatureTrait((color_channels,),
                                                   Initializer.mutable('uniform_', 1, 255),
                                                   partial(clamp, bounds=(1,255)), self.device)

        normalize_rays = partial(_normalize_rays, cfg=self.cfg)
        # the initializer here isn't fully correct, as it should be normal(0, 1) for the first 2 dimensions and
        # uniform on the last dimension, but I guess that's what you get for combining 2 traits that really should have been separate
        self.rays: CreatureTrait = CreatureTrait((self.cfg.num_rays, 3),
                                                 Initializer.mutable('normal_', 0, self.cfg.ray_dist_range[1]),
                                                 normalize_rays, self.device)


        self.positions: CreatureTrait = CreatureTrait((2,),
                                                      Initializer.force_mutable('uniform_', *self.posn_bounds),
                                                      partial(clamp, bounds=self.posn_bounds), self.device)

        self.energies: CreatureTrait = CreatureTrait(tuple(),
                                                     Initializer.other_dependent('sizes', self.cfg.init_energy),
                                                     None, self.device)

        self.reproduce_energies: CreatureTrait = CreatureTrait(tuple(),
                                                               Initializer.other_dependent('sizes', self.cfg.reproduce_thresh),
                                                               None, self.device)

        self.healths: CreatureTrait = CreatureTrait(tuple(),
                                                    Initializer.other_dependent('sizes', self.cfg.init_health),
                                                    None, self.device)
        eat_amt = partial(_eat_amt, cfg=self.cfg)
        self.eat_pcts: CreatureTrait = CreatureTrait(tuple(),
                                                     Initializer.other_dependent('sizes', eat_amt),
                                                     None, self.device)

        self.age_speeds: CreatureTrait = CreatureTrait(tuple(),
                                                       Initializer.other_dependent('sizes', self.cfg.age_speed_size),
                                                       None, self.device)
        self.age_mults: CreatureTrait = CreatureTrait(tuple(),
                                                      Initializer.fillable('fill_', 1.0),
                                                      None, self.device)

        self.memories: CreatureTrait = CreatureTrait((self.cfg.mem_size,),
                                                     Initializer.fillable('zero_'),
                                                     None, self.device)

        self.ages: CreatureTrait = CreatureTrait(tuple(),
                                                 Initializer.fillable('zero_'),
                                                 None, self.device)

        self.n_children: CreatureTrait = CreatureTrait(tuple(),
                                                    Initializer.fillable('zero_'),
                                                    None, self.device)

        self.head_dirs: CreatureTrait = CreatureTrait((2,),
                                       Initializer.fillable('normal_', 0, 1),
                                       normalize_head_dirs, self.device)

        # input: rays (num_rays*color_dims) + memory (mem_size) + num_food ((2*food_sight+1)**2) + health (1) + energy (1) +
        #         head_dir (2) + age_mult (1) + color (color_dims) + size (1)
        # output: move (1) + rotate (1) + memory (mem_size)
        layer_sizes = [
            self.cfg.num_rays*color_channels + self.cfg.mem_size + self.num_food + 1 + 1 + 2 + 1 + color_channels + 1,
            *self.cfg.brain_size,
            self.cfg.mem_size + 2
        ]
        weights = []
        biases = []
        for prev_layer, next_layer in zip(layer_sizes[:-1], layer_sizes[1:]):
            w_norm = 0.5 * np.sqrt(15. / prev_layer)
            b_norm = 0.5 / np.sqrt(next_layer)
            weights.append(CreatureTrait((prev_layer, next_layer),
                                         Initializer.mutable('uniform_', -w_norm, w_norm),
                                            None, self.device))
            biases.append(CreatureTrait((1, next_layer),
                                        Initializer.mutable('uniform_', -b_norm, b_norm),
                                        None, self.device))

        self.weights: List[CreatureTrait] = weights
        self.biases: List[CreatureTrait] = biases

        # this must be defined last, so that the right number of mutable parameters are set
        self.mutation_rates: CreatureTrait = CreatureTrait((self.num_mutable+1,),
                                                  Initializer.mutable('uniform_', *self.cfg.init_mut_rate_range),
                                                  None, self.device)


        # set position, size, color, etc. as samples from their respective init methods
        for k, v in self.variables.items():
            try:
                v.init_base(self.cfg)
            except TypeError:
                print(k)
                raise

        # set health, energy, age_speed, etc. as a function of the size
        for k, v in self.variables.items():
            if v.init.style == InitializerStyle.OTHER_DEPENDENT:
                try:
                    v.init_base_from_other(self.variables[v.init.name])
                except KeyError as exc:
                    raise KeyError(f"CreatureTrait '{k}' depends on trait '{v.init.name}'"
                                         ", but it has not been registered.") from exc
        # normalize_rays(self.rays, self.sizes, self.cfg)

        self.start_idx: int = 0
        self.alive = torch.zeros(self.cfg.max_creatures, dtype=torch.float32, device=self.device)
        self.alive[:self.cfg.start_creatures] = 1.0
        self.rng = BatchedRandom(self.variables, self.device)
        self.algos = {'max': 0, 'fill_gaps': 0, 'move_block': 0}

    def __setattr__(self, name, value):
        """Capture any CreatureTrait objects that are added to the CreatureArray, and store them
        in the `variables` dictionary, so that we can iterate over CreatureTraits and apply the
        same methods to each one. Additionally, we also can track how many mutable parametrs
        have been added so far, so that each one can be assigned a unique mut_idx when reproducing."""
        if isinstance(value, CreatureTrait):
            self.register_variable(name, value)
        elif isinstance(value, list) and all(isinstance(v, CreatureTrait) for v in value):
            for i, v in enumerate(value):
                self.register_variable(name + f"_{i}", v)
        super().__setattr__(name, value)

    def register_variable(self, name: str, v: CreatureTrait):
        """Register a new CreatureTrait with the CreatureArray. This is useful since we often like
        to iterate over all CreatureTraits in the CreatureArray and apply the same method to each one."""
        # if name in self.variables:
        #     raise ValueError(f"CreatureTrait with name '{name}' already exists in CreatureArray.")
        self.variables[name] = v
        if v.init.style == InitializerStyle.MUTABLE:
            v.init.mut_idx = self.num_mutable
            self.num_mutable += 1

    @cuda_profile
    def reproduce_most(self, reproducers: Tensor, num_reproducers: int,
                       children: 'CreatureArray', mut: Tensor):
        """Reproduce all traits that are mutable or fillable."""
        for name, v in self.variables.items():   # handle mutable parameters
            child_trait = v.reproduce(name, self.rng, reproducers, num_reproducers, mut)
            if child_trait is not None:
                setattr(children, name, child_trait)

    @cuda_profile
    def reproduce_extra(self, reproducers: Tensor, children: 'CreatureArray'):
        """Reproduce all traits that are not mutable."""
        children.positions = \
            self.positions.reproduce_mutable('positions', self.rng, reproducers,
                                             self.cfg.reproduce_dist)
        for name, v in self.variables.items():
            if v.init.style == InitializerStyle.OTHER_DEPENDENT:
                try:
                    children.variables[name] = v.reproduce_from_other(children.variables[v.init.name])
                except KeyError as exc:
                    raise KeyError(f"CreatureTrait '{name}' depends on trait '{v.init.name}' "
                                         ", but it does not been set by child.") from exc


    @cuda_profile
    def reproduce_traits(self, reproducers: Tensor, num_reproducers: int) -> 'CreatureArray':
        """Generate descendant traits from the set of indices in reproducers and store it in
        a new CreatureArray.

        Args:
            reproducers: A boolean tensor of creatures (size of current memory) that are reproducing.
            num_reproducers: The number of creatures that are reproducing.
        """
        mut = self.mutation_rates[reproducers]
        self.rng.generate(num_reproducers)
        children = CreatureArray(self.cfg, self.device)
        self.reproduce_most(reproducers, num_reproducers, children, mut)
        # we can pretend position is a mutable parameter since it is a random deviation from the parent
        self.reproduce_extra(reproducers, children)
        return children

    @property
    def population(self) -> int:
        return self.positions.population

    def write_new_data(self, idxs: Tensor, other: Union[None, 'CreatureArray']):
        """Write the data of the creatures in `other` at the indices `idxs` into the current block
        for all CreatureTraits in each.

        Args:
            idxs: Tensor of indices (into underlying memory) where we want to write the new creatures.
            other: The CreatureArray that contains the new creatures.
        """
        if other is None:  # None => no new creatures, so nothing to do
            return
        for name, v in self.variables.items():
            try:
                v.write_new(idxs, other.variables[name])
            except RuntimeError:
                print(name)
                raise

    def reindex(self, start_idx: int, n_after: int):
        """Slice each CreatureTrait according to the current block so that their `data` attribute
        is set properly for the next generation."""
        for v in self.variables.values():
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
        for v in self.variables.values():
            v.rearrange_old_data(outer_idxs, inner_idxs)

    def add_with_deaths(self, dead: Tensor, alive: Tensor,
                        num_dead: int, other: Union[None, 'CreatureArray'],
                        state: GameState) -> None:
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
        if state.selected_creature:  # if selected_creature is dead, then we can set it to None
            if dead[state.selected_creature]:
                # print("select is dead")
                state.selected_creature = None
        num_reproducers = other.population if other is not None else 0  # how many new creatures we are adding
        n_after = self.population - num_dead + num_reproducers  # how many creatures are alive after this step

        # select and apply block repair strategy
        if n_after == self.cfg.max_creatures:
            selected_creature_update = self.add_with_deaths_max_array(dead, other)
        elif num_reproducers >= num_dead:
            selected_creature_update = self.add_with_deaths_fill_gaps(dead, other, num_dead, num_reproducers, sl, state)
        else:
            selected_creature_update = self.add_with_deaths_move_block(other, n_after, num_reproducers, sl, state)

        # update self.alive to reflect the new block
        self.alive[self.start_idx : self.start_idx+n_after] = 1.0  # mark the new creatures as alive
        self.alive[:self.start_idx] = 0.0  # mark creatures outside the new block as dead
        self.alive[self.start_idx+n_after:] = 0.0
        # slice the underlying data arrays to give the current updated data
        self.reindex(self.start_idx, n_after)

        if state.selected_creature and selected_creature_update is not None:
            state.selected_creature = selected_creature_update

    def add_with_deaths_max_array(self, dead: Tensor, other: Union[None, 'CreatureArray']): # pylint: disable=useless-return
        """If the number of creatures after death and reproduction will fill the entire buffer,
        we can simply put new creatures in the dead creatures spots, and set start_idx to 0.

        Args:
            dead: A boolean tensor (size of current memory) of creatures that are dead.
            other: The CreatureArray that contains the new creatures (if any).
        Returns:
            None, since the selected creature doesn't get updated with this strategy
        """
        self.algos['max'] += 1
        self.start_idx = 0  # case where we just have to fill in the gaps
        if dead.shape[0] == self.cfg.max_creatures:
                # if prev epoch was also maxxed, then we can just use dead
            missing = dead.nonzero().squeeze(1)
        else:
                # otherwise, we need to look at the full indices of self.alive
            missing = (1-self.alive).nonzero().squeeze(1)
        self.write_new_data(missing, other)
        return None

    def add_with_deaths_fill_gaps(self, dead: Tensor, other: Union[None, 'CreatureArray'],  num_dead: int,
                                  num_reproducers: int, sl: slice, state: GameState) -> int | None:
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
            state: GameState that includes selected creature (which this updates)

        Returns:
            The updated index of the selected creature (if any). This must be deferred until after the reindex step.
        """
        self.algos['fill_gaps'] += 1
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
        if state.selected_creature:
            # print("filling gaps selected")
            return state.selected_creature + (sl.start - self.start_idx)

    def add_with_deaths_move_block(self, other: Union[None, 'CreatureArray'], n_after: int,
                                   num_reproducers: int, sl: slice, state: GameState) -> int | None:
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
            state: GameState that includes selected creature
        Returns:
            The updated index of the selected creature (if any). This must be deferred until after the reindex step.
        """
        self.algos['move_block'] += 1
        window = torch.ones(n_after, device=self.device, dtype=torch.float)
        # count the number of alive creatures in each window of size n_after
        windows = torch.conv1d(self.alive[sl].view(1, 1, -1), window.view(1, 1, -1), padding=0).squeeze()
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


        # move the old creatures that are outside the best window into the gaps in the best window
        self.rearrange_old_data(outer_idxs, old_idxs)
        # add the new creatures to the other gaps in the best window
        self.write_new_data(new_idxs, other)
        self.start_idx = best_start  # update the start of the block to the best window found

        # update the selected creature
        if state.selected_creature:
            # print("move block selected")
            # find its absolute index into the entire underlying data buffer
            discontig_creature = state.selected_creature + sl.start
            # if it is outside the best window, find where it is being moved to
            move_idx = (outer_idxs == discontig_creature).nonzero().squeeze(1)
            if move_idx.shape[0] > 0:
                discontig_creature = old_idxs[move_idx].item()
            # turn it back into a contiguous index
            return int(discontig_creature - self.start_idx)

    def state_dict(self, quantized=True):
        """Return a dictionary of the current state of the CreatureArray. This is useful for saving
        the state of the CreatureArray to disk."""
        variables = {}
        for name, v in self.variables.items():
            if quantized and name not in ['positions', 'colors', 'energies',
                                          'healths', 'ages', 'n_children']: # list of vars we shouldn't quantize
                variables[name] = quantize(v.data, map_location=torch.device('cpu'))
            else:
                variables[name] = v.data.to('cpu')

        start_idx = self.start_idx
        alive = self.alive
        return {'variables': variables, 'start_idx': start_idx, 'alive': alive, 'population': self.population}

    def load_state_dict(self, state_dict, map_location):
        """Load a state_dict into the CreatureArray. This is useful for loading the state of the CreatureArray
        from disk."""
        self.start_idx = state_dict['start_idx']
        self.alive = state_dict['alive'].to(self.device)
        population = state_dict['population']

        for name, v in self.variables.items():
            data = state_dict['variables'][name]
            if isinstance(data, QuantizedData):
                data = data.dequantize(map_location)
            v.init_base_from_state_dict(data, self.cfg, self.start_idx, population)  # sets _data parameters

    def unset_data(self):
        """Unset the underlying memory for all CreatureTraits in the CreatureArray. This allows room
        on the GPU to be made for new variables coming in (e.g. when loading a state_dict)."""
        for v in self.variables.values():
            v.unset_data()  # unsets _data parameters to make room for state_dicts to be loaded
