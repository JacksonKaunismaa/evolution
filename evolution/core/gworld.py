import os.path as osp
from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

from evolution.cuda import cu_algorithms
from evolution.core.benchmarking import Profile, BenchmarkMethods
from evolution.utils.quantize import quantize, QuantizedData
from evolution.state.game_state import GameState
from evolution.visual.interactive.camera import Camera

from .config import Config
from .creatures.creatures import Creatures

class GWorld():
    """The main class that holds the state the world, including the food grid, the creatures,
    and the game state. It also contains the methods to run the simulation, including updating the
    creatures, computing the decisions of the creatures, and running one step of the simulation."""
    def __init__(self, cfg: Config, device=torch.device('cuda'), path: Optional[str] = None):
        self.cfg: Config = cfg
        self.device = device
        self.path = path
        self.food_grid = torch.rand((cfg.size, cfg.size), device=self.device) * cfg.init_food_scale
        # pad with zeros on all sides
        self.food_grid = F.pad(self.food_grid, (cfg.food_sight,)*4, mode='constant', value=0)
        self.kernels = cu_algorithms.CUDAKernelManager(cfg)
        self.creatures: Creatures = Creatures(cfg, self.kernels, self.device)
        self.n_maxxed = 0

        self.state = GameState(cfg, self)

        # we keep track of these objects so that we can visualize them
        self.celled_world: Tuple[Tensor, Tensor] = None    # array of grid cells for fast object clicking
        self.outputs: Tensor = None   # the set of neural outputs that decide what the creatures want to do
        self.collisions: Tensor = None   # the set of ray collisions for each creature

    @Profile.cuda_profile
    def compute_grid_setup(self):
        """Returns [G, G, M] array of the grid setup for all creatures. Where G is the number of grid cells,
        (G = world_size // cell_size + 1), and M is the maximum number of objects per cell (M = max_per_cell),
        and [G, G] array of the number of objects in each cell.

        Used to speed up ray tracing by reducing the number of objects we need to check."""
        posns = self.creatures.positions
        sizes = self.creatures.sizes

        # fill in grid cells to vastly reduce the number of objects we need to check
        num_cells = int(self.cfg.size // self.cfg.cell_size + 1)
        cells = torch.full((num_cells, num_cells, self.cfg.max_per_cell), -1, dtype=torch.int32, device=self.device)
        cell_counts = torch.zeros((num_cells, num_cells), dtype=torch.int32, device=self.device)
        block_size = 512
        grid_size = self.population // block_size + 1

        # algorithms.setup_grid[grid_size, block_size](posns, sizes, cells, cell_counts, self.cfg.cell_size)
        self.kernels('setup_grid',
                     grid_size, block_size, # threads
                     posns, sizes, cells, cell_counts,
                     self.population, num_cells)
        return cells, cell_counts  # argument return order should match trace_rays_grid


    @Profile.cuda_profile
    def trace_rays_grid(self, cells, cell_counts):
        """ Returns [N, R] array of the results of ray collisions for all creatures.
        Where the coordinate is the scalar color of the ray intersection.
        """

        rays = self.creatures.rays
        colors = self.creatures.colors
        posns = self.creatures.positions
        sizes = self.creatures.sizes

        # traverse the grid and draw lines through it, checking each object that passes through the ray
        collisions = torch.zeros((rays.shape[0], rays.shape[1]), device=self.device)
        # algorithms.trace_rays_grid(self.cfg)[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors,
        #                                                                    cells, cell_counts,
        #                                                                    self.cfg.cell_size, collisions)
        # cells_hit = torch.zeros((rays.shape[0], rays.shape[1]), device=self.device, dtype=torch.int32)
        # organisms_checked = torch.zeros((rays.shape[0], rays.shape[1]), device=self.device, dtype=torch.int32)
        # cache_hits = torch.zeros((rays.shape[0], rays.shape[1]), device=self.device, dtype=torch.int32)
        self.kernels('trace_rays_grid',
                     rays.shape[0], rays.shape[1], # threads
                     rays, posns, sizes, colors,
                     cells, cell_counts, collisions,
                     self.population, cells.shape[0],
                    #  cells_hit, organisms_checked, cache_hits
                     )
        # self.all_hits['cells_hit'].append(cells_hit.cpu().numpy())
        # self.all_hits['organisms_checked'].append(organisms_checked.cpu().numpy())
        # self.all_hits['cache_hits'].append(cache_hits.cpu().numpy())
        return collisions


    def click_creature(self, mouse_pos) -> None | int:
        """Return the index of the creature clicked on, or None if no creature was clicked."""
        mouse_pos = torch.tensor(mouse_pos, device=self.device)   # [2]
        dists = torch.norm(self.creatures.positions - mouse_pos, dim=1)   # [N, 2] - [1, 2] = [N, 2]
        close_enough = dists < self.creatures.sizes
        nonzero = torch.nonzero(close_enough)
        creature = int(nonzero[0].item()) if nonzero.shape[0] != 0 else None
        return creature

    def log_total_energy(self) -> float:
        """Return the log10 of the total energy of the world, which is the sum of the energy of all
        creatures and the energy of the positive areas of the food grid."""
        return float(np.log10(F.relu(self.food_grid).sum().item() + self.creatures.total_energy()))

    def creature_histogram(self, n_bins: int = 255) -> np.ndarray:
        """Return a histogram of the energy of all creatures."""
        return self.creatures.histogram(n_bins)

    @Profile.cuda_profile
    def collect_stimuli(self, collisions):
        """Returns [N, F] array for all creature stimuli."""
        return self.creatures.collect_stimuli(collisions, self.food_grid)

    @Profile.cuda_profile
    def think(self, stimuli):
        """Return [N, O] array of outputs of the creatures' neural networks.."""
        return self.creatures.forward(stimuli.unsqueeze(1))

    @Profile.cuda_profile
    def rotate_creatures(self, outputs):
        """Rotate all creatures based on their outputs."""
        self.creatures.rotate_creatures(outputs, self.state)

    @Profile.cuda_profile
    def only_move_creatures(self, outputs):
        """Rotate and move all creatures"""
        self.creatures.move_creatures(outputs, self.state)

    def move_creatures(self, outputs):
        """Rotate and move all creatures"""
        self.rotate_creatures(outputs)
        self.only_move_creatures(outputs)

    @Profile.cuda_profile
    def compute_gridded_attacks(self, cells, cell_counts):
        """Compute which creatures are attacking which creatures are then based on that, update
        health and energy of creatures."""
        posns = self.creatures.positions
        sizes = self.creatures.sizes
        colors = self.creatures.colors
        head_dirs = self.creatures.head_dirs

        # figure out who is attacking who
        tr_results = torch.zeros((self.population, 2), device=self.device)
        block_size = 512
        grid_size = self.population // block_size + 1

        self.kernels('gridded_is_attacking',
                     grid_size, block_size,  # threads
                     posns, sizes, colors, head_dirs,
                     cells, cell_counts, tr_results,
                     self.population, cells.shape[0])
        return tr_results

    @Profile.cuda_profile
    def do_attacks(self, tr_results):
        """Update health and energy of creatures based on the results of the attacks."""
        self.creatures.do_attacks(tr_results, self.state)

    @Profile.cuda_profile
    def creatures_eat_grow(self):
        """Grow food_grid and give creatures energy for being in a square"""
        self.creatures.eat_grow(self.food_grid, self.state)

    @property
    def central_food_grid(self):
        """Return the central part of the food grid, excluding the padding."""
        pad = self.cfg.food_sight
        return self.food_grid[pad:-pad, pad:-pad]

    @property
    def population(self) -> int:
        """"Return the number of creatures in the world."""
        return self.creatures.population

    @Profile.cuda_profile
    def fused_kill_reproduce(self):
        """Kill creatures that have no energy, and reproduce creatures that have enough energy."""
        self.creatures.fused_kill_reproduce(self.central_food_grid, self.state)

    @Profile.cuda_profile
    def update_creatures(self):
        """Move creatures, update health and energy, reproduce, eat, grow food. Update all the
        states of the creatures and the world."""
        if self.outputs is None:   # skip updating if we haven't computed the outputs yet
            return

        self.move_creatures(self.outputs, )   # move creatures, update health and energy

        celled_world = self.compute_grid_setup()
        attacks2 = self.compute_gridded_attacks(*celled_world)

        self.do_attacks(attacks2)  # update health and energy of creatures
        self.fused_kill_reproduce()
        self.creatures_eat_grow()   # give energy for being in a square, and then reduce that energy

    @Profile.cuda_profile
    def compute_decisions(self):
        """Compute the decisions of all creatures. This includes running the neural networks,
        computing memories, and running vision ray tracing. Sets `outputs`, `collisions`, and
        `celled_world`."""
        self.celled_world = self.compute_grid_setup()
        self.collisions = self.trace_rays_grid(*self.celled_world)
        stimuli = self.collect_stimuli(self.collisions)
        self.outputs = self.think(stimuli)

    @Profile.cuda_profile
    def step(self) -> bool:
        """Run one step of the simulation. Returns False if there was a mass extinction."""
        # we do these steps in this (seemingly backwards) order, because we want vision and brain
        # information to be up to date when we visualize, which happens after we exit from here

        # move creatures, update health and energy, reproduce, eat, grow food
        self.update_creatures()

        if self.population == 0:
            #logging.info("Mass extinction!")
            print("Mass extinction!")
            return False

        self.compute_decisions()   # run neural networks, compute memories, do vision ray tracing
        if Profile.BENCHMARK != BenchmarkMethods.NONE:
            k = min(20, self.celled_world[1].numel())
            self.n_maxxed += (self.celled_world[1]).topk(k).values.float().mean()
        self.state.publish_all()

        # if self.state.time % 60 == 0:   # save this generation
        #     if save:
        #         self.write_checkpoint('game.ckpt')
        #     if visualize:
        #         self.visualize(None, show_rays=False)

        self.state.time += 1
        if self.state.increasing_food_decr:
            self.cfg.food_cover_decr += self.cfg.food_cover_decr_incr_amt
        return True

    def write_checkpoint(self, path=None, quantized=True, camera: Camera | None = None):
        """Write a checkpoint file to `path` that contains the state of the world, including the
        food grid, the creatures, the configuration, the time, the seed, current game settings in
        the `self.state` object, and the camera state if `camera` is not None. If `quantized` is
        True, then most of the data associated with creatures and the food grid will be quantized
        to reduce the size of the checkpoint file.
        """
        if path is None:
            path = self.path

        if path is None:
            raise ValueError("No path provided to write checkpoint to.")

        seed = self.state.time
        fgrid = quantize(self.food_grid, map_location=torch.device('cpu')) if quantized else self.food_grid
        torch.save({'food_grid': fgrid,
                    'creatures': self.creatures.state_dict(quantized),
                    'cfg': self.cfg,
                    'time': self.state.time,
                    'seed': seed,
                    'device': self.device,
                    'state': self.state.state_dict(),
                    'others': {
                        'camera': camera.state_dict() if camera is not None else None
                    }
                    }, path)
        torch.random.manual_seed(seed)

    @classmethod
    def from_checkpoint(cls, path) -> Tuple['GWorld', Dict]:
        """Instantiate a GWorld object from a checkpoint file. Returns the instance as well
        as saved data for any other objects that were saved in the checkpoint."""
        if not osp.exists(path):
            raise FileNotFoundError(f"Checkpoint file '{path}' not found.")

        print(f"Loading checkpoint '{path}'...")

        checkpoint = torch.load(path)
        # we nede to create the GWorld object so that it has the right config from the get go
        # since (eg.) Creatures objects are created based on config when Creatures is created
        instance = cls(checkpoint['cfg'], device=checkpoint['device'], path=path)

        instance.creatures.unset_data()
        del instance.food_grid

        fgrid = checkpoint['food_grid']
        if isinstance(fgrid, QuantizedData):
            fgrid = fgrid.dequantize(map_location=instance.device)
        instance.food_grid = fgrid

        instance.creatures.load_state_dict(checkpoint['creatures'], instance.device)
        instance.state.time = checkpoint.get('time', 0)
        torch.random.manual_seed(checkpoint['seed'])
        instance.state.load_state_dict(checkpoint['state'])

        return instance, checkpoint.get('others', {})
