from typing import Tuple, Union, List
import torch
from torch import Tensor
import numpy as np
# import logging
# logging.basicConfig(level=logging.ERROR, filename='game.log')

from evolution.cuda.cuda_utils import cuda_profile
from evolution.cuda.cu_algorithms import CUDAKernelManager
from evolution.core.config import Config
from evolution.state.game_state import GameState

from .creature_array import CreatureArray
from .creature_trait import CreatureTrait
from . import trait_initializer as init
from .trait_initializer import Initializer
from . import trait_normalizer as norm
from .trait_normalizer import Normalizer


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

    def __init__(self, cfg: Config, kernels: CUDAKernelManager, device: torch.device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.kernels = kernels

        # how much indexing into the food grid needs to be adjusted to avoid OOB
        self.pad = self.cfg.food_sight
        self.offsets = torch.tensor([[i, j] for i in range(-self.pad, self.pad+1)
                                     for j in range(-self.pad, self.pad+1)], device=self.device).unsqueeze(0)
        self.activations = []
        self.dead: Tensor   # [N] current memory boolean tensor
        self.generate_from_cfg()

    def generate_from_cfg(self):
        """Definition of all creature Traits for root creature. This should only be called by the
        root Creatures object. When reproduction happens, it should create a new CreatureArray object,
        which is mostly just a container, not a Creatures object, which has specifics about how
        what particular traits exist and what they do. Each time a new CreatureTrait attribute is set,
        it is automatically added to self.variables, which allows for iterating over all traits that
        exist and applying operations to them. For lists of CreatureTraits, each element of the list
        is also added to self.variables."""
        self.sizes = CreatureTrait(tuple(),
                        Initializer.mutable('uniform_', *self.cfg.init_size_range),
                        Normalizer(norm.clamp, *self.cfg.size_range),
                        self.device)

        color_channels = 1
        self.colors = CreatureTrait(tuple(),
                          Initializer.mutable('uniform_', 1, 255),
                          Normalizer(norm.clamp, 1, 255), self.device)

        self.visual_colors = CreatureTrait((3,),
                                 Initializer.other_dependent('colors', init.cuda_hsv_spiral, self.kernels),
                                 None, self.device)

        # the initializer here isn't fully correct, as it should be normal(0, 1) for the first 2 dimensions and
        # uniform on the last dimension, but I guess that's what you get for combining 2 traits that really should have been separate
        self.rays = CreatureTrait((self.cfg.num_rays, 3),
                        Initializer.mutable('normal_', 0, self.cfg.ray_dist_range[1]),
                        Normalizer(norm.normalize_rays, self.cfg),
                        self.device)


        self.positions = CreatureTrait((2,),
                             Initializer.force_mutable('uniform_', self.cfg.reproduce_dist, *self.posn_bounds),
                             Normalizer(norm.clamp, *self.posn_bounds),
                             self.device)

        self.energies = CreatureTrait(tuple(),
                            Initializer.other_dependent('sizes', self.cfg.init_energy),
                            None, self.device)

        self.reproduce_energies = CreatureTrait(tuple(),
                                      Initializer.other_dependent('sizes', self.cfg.reproduce_thresh),
                                      None, self.device)

        self.healths = CreatureTrait(tuple(),
                           Initializer.other_dependent('sizes', self.cfg.init_health),
                           None, self.device)

        self.eat_pcts = CreatureTrait(tuple(),
                            Initializer.other_dependent('sizes', init.eat_amt, self.cfg),
                            None, self.device)

        self.age_speeds = CreatureTrait(tuple(),
                              Initializer.other_dependent('sizes', self.cfg.age_speed_size),
                              None, self.device)

        self.age_mults = CreatureTrait(tuple(),
                             Initializer.fillable('fill_', 1.0),
                             None, self.device)

        self.memories = CreatureTrait((self.cfg.mem_size,),
                            Initializer.fillable('zero_'),
                            None, self.device)

        self.ages = CreatureTrait(tuple(),
                        Initializer.fillable('zero_'),
                        None, self.device)

        self.n_children = CreatureTrait(tuple(),
                              Initializer.fillable('zero_'),
                              None, self.device)

        self.head_dirs = CreatureTrait((2,),
                             Initializer.fillable('normal_', 0, 1),
                             Normalizer(norm.normalize_head_dirs), self.device)

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
        self.mutation_rates = CreatureTrait((self.num_mutable+1,),
                                  Initializer.mutable('uniform_', *self.cfg.init_mut_rate_range),
                                  None, self.device)


        super().generate_from_cfg()

    def fused_kill_reproduce(self, central_food_grid: Tensor, state: GameState):
        """Kill the dead creatures and reproduce the living ones."""
        dead, num_dead = self._kill_dead(central_food_grid)  # determine who is dead
        alive = ~dead  # type: ignore
        new_creatures = self._reproduce(alive, num_dead)  # determine who is reproducing
        if new_creatures or num_dead > 0:   # rearrange memory and update current data
            self.add_with_deaths(dead, alive, num_dead, new_creatures, state)

    @cuda_profile
    def _kill_dead(self, central_food_grid: Tensor) -> Tuple[Tensor, int]:
        """If any creature drops below 0 health or energy, kill it. Update the food grid by depositing an amount of food
        that depends on the creature's size at death."""
        if self.cfg.immortal:
            return torch.zeros(self.population, dtype=torch.bool, device=self.device), 0
        health_deaths = self.healths <= 0
        energy_deaths = self.energies <= 0
        self.dead = health_deaths | energy_deaths   # congiuous boolean tensor

        num_dead = int(torch.sum(self.dead).item())
        if num_dead > 0:
            dead_posns = self.positions[self.dead].long()
            # food_grid[dead_posns[..., 1], dead_posns[..., 0]] += self.cfg.dead_drop_food(self.sizes[self.dead])
            central_food_grid.index_put_((dead_posns[..., 1], dead_posns[..., 0]),
                                self.cfg.dead_drop_food(self.sizes[self.dead]),
                                accumulate=True)
        return self.dead, num_dead

    @cuda_profile
    def _get_reproducers(self, alive: Union[None, Tensor]) -> Tensor:
        # 1 + so that really small creatures don't get unfairly benefitted
        reproducers = (self.ages >= self.age_speeds*self.cfg.age_mature_mul) & (self.energies >= self.reproduce_energies)  # current mem boolean tensor
        if alive is not None:
            reproducers &= alive
        return reproducers

    @cuda_profile
    def _reduce_reproducers(self, reproducers: Tensor, num_reproducers: int,
                            max_reproducers: int) -> Tuple[Tensor, int]:
        if num_reproducers > max_reproducers:  # this benefits older creatures because they
            num_reproducers = max_reproducers  # are more likely to be in the num_reproducers window
            reproducer_indices = torch.nonzero(reproducers)#[num_reproducers:]
            perm = torch.randperm(reproducer_indices.shape[0], device=self.device)
            non_reproducers = reproducer_indices[perm[num_reproducers:]]
            reproducers[non_reproducers] = False
        return reproducers, num_reproducers

    @cuda_profile
    def _reproduce(self, alive: Tensor, num_dead: int) -> Union[None, 'CreatureArray']:
        """For sufficiently high energy creatures, create a new creature with mutated genes. Returns the set
        of new creatures in a CreatureArray object."""
        max_reproducers = self.cfg.max_creatures + num_dead - self.population # dont bother if none can reproduce anyway
        if max_reproducers <= 0:
            return

        reproducers = self._get_reproducers(alive)  # current memory boolean tensor
        # limit reproducers so we don't exceed max size
        num_reproducers = int(torch.sum(reproducers))
        if num_reproducers == 0:  # no one can reproduce
            return

        # print(int(num_reproducers), num_dead)
        reproducers, num_reproducers = self._reduce_reproducers(reproducers, num_reproducers, max_reproducers)
        self.n_children[reproducers] += 1

        # could fuse this
        self.energies[reproducers] -= self.sizes[reproducers]  # subtract off the energy that you've put into the world
        self.energies[reproducers] /= self.cfg.reproduce_energy_loss_frac  # then lose a bit extra because this process is lossy

        new_creatures = self.reproduce_traits(self.mutation_rates[reproducers], reproducers, num_reproducers)
        return new_creatures

    def eat_grow(self, food_grid: Tensor, state: GameState):
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
        state.selected_creature.extract_pre_eat_state()
        state.selected_cell.extract_pre_grow_state()

        pos = self.positions.int() + self.pad

        # calculate how many creatures are in each position, useful for computng how much food each creature eats
        pct_eaten = torch.zeros_like(food_grid, dtype=torch.float32, device=self.device)
        threads_per_block = 512
        blocks_per_grid = self.population // threads_per_block + 1
        self.kernels('setup_eat_grid', blocks_per_grid, threads_per_block,
                     pos, self.eat_pcts, pct_eaten, self.population, food_grid.shape[0])

        # calculate how much food each creature eats, add 1 to ages, update energies, take away living costs
        food_grid_updates = torch.zeros_like(food_grid, device=self.device)
        alive_costs = torch.zeros(self.population, device=self.device, dtype=torch.float32)
        self.kernels('eat', blocks_per_grid, threads_per_block,
                     pos, self.eat_pcts, pct_eaten, self.sizes, self.age_speeds, food_grid,
                     food_grid_updates, alive_costs, self.energies, self.healths, self.ages, self.age_mults,
                     self.population, food_grid.shape[0], self.cfg.food_cover_decr)

        # grow food, apply eating costs
        # step_size = (torch.sum(alive_costs)/self.cfg.max_food/(self.cfg.size**2)).item()
        # step_size = (self.cfg.max_food / (self.cfg.size**(1.9)))#.item()
        step_size = self.cfg.food_step_size
        threads_per_block = (16, 16)
        blocks_per_grid = (food_grid.shape[0] // threads_per_block[0] + 1,
                           food_grid.shape[1] // threads_per_block[1] + 1)


        self.kernels('grow', blocks_per_grid, threads_per_block,
                     food_grid_updates, food_grid, food_grid.shape[0], self.pad, step_size, self.cfg.max_food)

        state.selected_cell.extract_post_grow_state(pct_eaten, food_grid_updates, step_size)
        state.selected_creature.extract_post_eat_state(pos, alive_costs)


    def extract_food_windows(self, indices, food_grid):
        """Indices is [N, 2] indices into food_grid.
        Food grid is [S, S], where S is the width of the world.
        Returns a [N, W] tensor of the food in the W x W window around each index,
        which goes as an input into creatures' brains."""
        windows = indices.unsqueeze(1) + self.offsets  # [N, 1, 2] + [1, 9, 2] = [N, 9, 2]
        # if windows.min() < 0 or windows.max() >= food_grid.shape[0]:
        #     raise ValueError("OOB")
        return food_grid[windows[..., 1], windows[..., 0]]


    def collect_stimuli(self, collisions, food_grid):
        """Collisions: [N, R], food_grid: [W, W] (W = world size) -> [N, F],
        where F is the number of features that get passed in to the creatures' brains."""
        # get the food in the creature's vicinity
        # we must add self.pad, because positions are stored in central_food_grid coordinates,
        # but we need to index into food_grid with food_grid coordinates (which includes padding)
        posn = self.positions.long() + self.pad
        food = self.extract_food_windows(posn, food_grid)  # [N, (2*sight+1)**2]
        rays_results = collisions

        return torch.cat([rays_results,  # type: ignore
                          self.memories,
                          food,
                          self.healths.unsqueeze(1),
                          self.energies.unsqueeze(1),
                          self.head_dirs,
                          self.age_mults.unsqueeze(1),
                          self.colors.unsqueeze(1),
                          self.sizes.unsqueeze(1),
                        ],
                        dim=1)

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

    def rotate_creatures(self, outputs: Tensor, state: GameState):
        """Given the neural network outputs `outputs`, rotate each creature accordingly.
        Rotation is a scalar between -1 and 1, which is combined with the creatures' size
        to compute the number of radians that are rotated. Negative rotation is leftwards,
        positive is rightwards. We also take a small energy penalty for rotating."""
        # rotation => need to rotate the rays and the object's direction

        state.selected_creature.extract_pre_rotate_state()

        block_size = 512
        grid_size = self.population // block_size + 1
        rotation_matrix = torch.empty((self.population, 2, 2), device=self.device)  # [N, 2, 2]
        self.kernels('build_rotation_matrices', grid_size, block_size,
                     outputs, self.sizes, self.energies, self.population, outputs.shape[1],
                     rotation_matrix)

        # rotate the rays and head directions
        if self.cfg.rays_rotate:
            self.rays[..., :2] = self.rays[..., :2] @ rotation_matrix  # [N, 32, 2] @ [N, 2, 2]
        self.head_dirs[:] = (self.head_dirs.unsqueeze(1) @ rotation_matrix).squeeze(1)  # [N, 1, 2] @ [N, 2, 2]

        state.selected_creature.extract_post_rotate_state(outputs[:,1], rotation_matrix)

    def move_creatures(self, outputs: Tensor, state: GameState):
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

        self.energies -= move_cost
        self.positions += self.head_dirs * move.unsqueeze(1)   # move the object's to new position
        self.positions.normalize()# = torch.clamp(self.positions, *self.posn_bounds)  # don't let it go off the edge

        state.selected_creature.extract_move_state(outputs[:,0], move, move_cost)


    def do_attacks(self, attacks: Tensor, state: GameState):
        """Attacks is [N, 2], where the 1st coordinate is an integer indicating how many things the
        organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the
        organism is taking from other things attacking it. We update each
        creature's health and energy accordingly."""

        attack_cost = self.cfg.attack_cost(attacks[:,0], self.sizes)
        self.energies -= attack_cost
        self.healths -= attacks[:,1]

        state.selected_creature.extract_attack_state(attacks, attack_cost)

    def total_energy(self):
        """Return the sum of energies of all creatures"""
        return torch.sum(self.energies).item() # type: ignore
