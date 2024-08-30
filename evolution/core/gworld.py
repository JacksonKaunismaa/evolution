import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
import time
# import logging
from tqdm import trange, tqdm
import os.path as osp
from collections import defaultdict
import numpy as np

# logging.basicConfig(level=logging.info, filename='game.log')

from evolution.cuda import cu_algorithms
from evolution.cuda import cuda_utils
from evolution.cuda.cuda_utils import cuda_profile
from evolution.utils.subscribe import Publisher
from evolution.utils.quantize import quantize, QuantizedData
from evolution.state.game_state import GameState

from .config import Config, simple_cfg
from .creatures.creatures import Creatures

class GWorld():
    def __init__(self, cfg: Config, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.food_grid = torch.rand((cfg.size, cfg.size), device=self.device) * cfg.init_food_scale
        # pad with -inf on all sides
        self.food_grid = F.pad(self.food_grid, (cfg.food_sight,)*4, mode='constant', value=0)
        self.kernels = cu_algorithms.CUDAKernelManager(cfg)
        self.creatures: Creatures = Creatures(cfg, self.kernels, self.device)
        self.creatures.generate_from_cfg()
        self.n_maxxed = 0
        
        self.state = GameState(cfg, self)
        
        # we keep track of these objects so that we can visualize them
        self.celled_world = None    # array of grid cells for fast object clicking
        self.outputs = None   # the set of neural outputs that decide what the creatures want to do
        self.collisions = None   # the set of ray collisions for each creature
        self.dead_updated = False
        self.time = 0
        self.publisher = Publisher()

    @cuda_profile
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
    

    @cuda_profile
    def trace_rays_grid(self, cells, cell_counts):
        """Returns [N, R, 3] array of the results of ray collisions for all creatures."""

        rays = self.creatures.rays
        colors = self.creatures.colors
        posns = self.creatures.positions
        sizes = self.creatures.sizes

        # traverse the grid and draw lines through it, checking each object that passes through the ray
        collisions = torch.zeros((rays.shape[0], rays.shape[1], colors.shape[-1]), device=self.device)
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
    
    
    def click_creature(self, mouse_pos) -> int:
        """Return the index of the creature clicked on, or None if no creature was clicked."""
        mouse_pos = torch.tensor(mouse_pos, device=self.device)   # [2]
        dists = torch.norm(self.creatures.positions - mouse_pos, dim=1)   # [N, 2] - [1, 2] = [N, 2]
        close_enough = dists < self.creatures.sizes
        nonzero = torch.nonzero(close_enough)
        creature = nonzero[0].item() if nonzero.shape[0] != 0 else None
        return creature

    @cuda_profile
    def collect_stimuli(self, collisions):
        """Returns [N, F] array for all creature stimuli."""
        return self.creatures.collect_stimuli(collisions, self.food_grid)
    
    @cuda_profile
    def think(self, stimuli):
        """Return [N, O] array of outputs of the creatures' neural networks.."""
        return self.creatures.forward(stimuli.unsqueeze(1))
    
    @cuda_profile
    def rotate_creatures(self, outputs):
        """Rotate all creatures based on their outputs."""
        self.creatures.rotate_creatures(outputs, self.state)
    
    @cuda_profile
    def only_move_creatures(self, outputs):
        """Rotate and move all creatures"""
        self.creatures.move_creatures(outputs, self.state)

    def move_creatures(self, outputs):
        """Rotate and move all creatures"""
        self.rotate_creatures(outputs)
        self.only_move_creatures(outputs)
    
    @cuda_profile
    def compute_gridded_attacks(self, cells, cell_counts):
        """Compute which creatures are attacking which creatures are then based on that, update health 
        and energy of creatures."""
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
    
    @cuda_profile
    def only_do_attacks(self, tr_results):
        # update health and energy
        self.creatures.do_attacks(tr_results, self.state)

    def do_attacks(self, tr_results):
        # update health and energy
        self.only_do_attacks(tr_results)

    @cuda_profile
    def creatures_eat_grow(self):
        self.creatures.eat_grow(self.food_grid, self.state)        

    @property
    def central_food_grid(self):
        pad = self.cfg.food_sight
        return self.food_grid[pad:-pad, pad:-pad]
        
    @property
    def population(self):
        return self.creatures.population

    def write_checkpoint(self, path, quantized=True):
        seed = self.time
        fgrid = quantize(self.food_grid, map_location='cpu') if quantized else self.food_grid
        torch.save({'food_grid': fgrid, 
                    'creatures': self.creatures.state_dict(quantized), 
                    'cfg': self.cfg,
                    'time': self.time,
                    'seed': seed
                    }, path)
        torch.random.manual_seed(seed)
        
    def load_checkpoint(self, path):
        if not osp.exists(path):
            print(f"Warning: checkpoint file '{path}' not found, continuing anyway...")
        else:
            print(f"Loading checkpoint '{path}'...")
            
            self.creatures.unset_data()
            del self.food_grid
            
            checkpoint = torch.load(path)
            
            fgrid = checkpoint['food_grid']
            if isinstance(fgrid, QuantizedData):
                fgrid = fgrid.dequantize(map_location=self.device)
            self.food_grid = fgrid
            
            self.creatures.load_state_dict(checkpoint['creatures'], self.device)
            self.cfg = checkpoint['cfg']
            self.time = checkpoint.get('time', 0)
            torch.random.manual_seed(checkpoint['seed'])
    
    @cuda_profile
    def fused_kill_reproduce(self):
        self.creatures.fused_kill_reproduce(self.central_food_grid, self.state)
        self.dead_updated = False

    def update_creatures(self):
        """Move creatures, update health and energy, reproduce, eat, grow food. Update all the states of the creatures
        and the world."""
        if self.outputs == None:   # skip updating if we haven't computed the outputs yet
            return

        self.move_creatures(self.outputs, )   # move creatures, update health and energy
        
        celled_world = self.compute_grid_setup()
        attacks2 = self.compute_gridded_attacks(*celled_world)

        self.do_attacks(attacks2)  # update health and energy of creatures
        self.fused_kill_reproduce()
        self.creatures_eat_grow()   # give energy for being in a square, and then reduce that energy


    def compute_decisions(self):
        """Compute the decisions of all creatures. This includes running the neural networks, computing memories, and
        running vision ray tracing. Sets `outputs`, `collisions`, and `celled_world`."""
        self.celled_world = self.compute_grid_setup()
        self.n_maxxed += (self.celled_world[1] >= self.cfg.max_per_cell).sum()
        # input()
        self.collisions = self.trace_rays_grid(*self.celled_world)
        stimuli = self.collect_stimuli(self.collisions)
        self.outputs = self.think(stimuli)
        
    def toggle_increasing_food_decr(self):
        self.increasing_food_decr_enabled = not self.increasing_food_decr_enabled
        

    def step(self) -> bool:        
        # we do these steps in this (seemingly backwards) order, because we want vision and brain
        # information to be up to date when we visualize the scene, which happens after we exit from here
        self.update_creatures()   # move creatures, update health and energy, reproduce, eat, grow food
        
        if self.population == 0:
            #logging.info("Mass extinction!")
            print("Mass extinction!")
            return False

        self.compute_decisions()   # run neural networks, compute memories, do vision ray tracing
        self.publisher.publish()
        
        # if self.time % 60 == 0:   # save this generation
        #     if save:
        #         self.write_checkpoint('game.ckpt')
        #     if visualize:
        #         self.visualize(None, show_rays=False) 

        self.time += 1
        if self.state.increasing_food_decr:
            self.cfg.food_cover_decr += self.cfg.food_cover_decr_incr_amt
        return True


def _benchmark(cfg=None, max_steps=512):
    # import cProfile
    # from pstats import SortKey
    # import io, pstats
    cuda_utils.BENCHMARK = True
    cuda_utils.times.clear()

    # torch.manual_seed(1)
    if cfg is None:
        cfg = simple_cfg()
    game = GWorld(cfg)

    # pr = cProfile.Profile()
    # pr.enable()
    for i in trange(max_steps):
        if not game.step(visualize=False):   # we did mass extinction before finishing
            return game
    
    cuda_utils.times['n_maxxed'] = game.n_maxxed.item()
    cuda_utils.times['algo_max'] = game.creatures.algos['max']
    cuda_utils.times['algo_fill'] = game.creatures.algos['fill_gaps']
    cuda_utils.times['algo_move'] = game.creatures.algos['move_block']
    

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    return cuda_utils.times

def multi_benchmark(cfg, max_steps=2500, N=20, skip_first=False):
    total_times = defaultdict(list)
    for i in range(N):
        bmarks = _benchmark(cfg, max_steps=max_steps)
        if i == 0 and skip_first:  # skip first iteration for compilation weirdness
            continue
        for k, v in bmarks.items():
            total_times[k].append(v)

    for k, v in total_times.items():
        # compute mean and sample standard deviation
        mean = np.mean(v)
        std = np.std(v, ddof=1) / np.sqrt(len(v))
        print(k, mean, '+-', std)


def main(cfg=None, max_steps=99999):
    if cfg is None:
        cfg = Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0)
    cuda_utils.times.clear()
    game = GWorld(cfg)
    print("BEGINNING SIMULATION...")
    time_now = time.time()
    for i in trange(max_steps):
        if not game.step():
            break
    return cuda_utils.times, game

        

        
        