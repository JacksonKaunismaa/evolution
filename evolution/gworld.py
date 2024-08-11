from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
import time
from IPython.display import clear_output
import logging
from tqdm import trange, tqdm
from functools import partial, wraps
import functools
from typing import Callable

logging.basicConfig(level=logging.INFO, filename='game.log')

# from . import algorithms
from . import cu_algorithms
from .creature_array import CreatureArray
from .config import Config, simple_cfg

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
times = defaultdict(float)

def cuda_profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start.record()
        res = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times[func.__name__] += start.elapsed_time(end)
        return res
    return wrapper



class GWorld():
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.food_grid = torch.rand((cfg.size, cfg.size), device='cuda') * cfg.init_food_scale
        # pad with -inf on all sides
        self.food_grid = F.pad(self.food_grid, (cfg.food_sight,)*4, mode='constant', value=0)
        self.creatures: CreatureArray = CreatureArray(cfg)
        self.kernels = cu_algorithms.CUDAKernelManager(cfg)


        # we keep track of these objects so that we can visualize them
        self.celled_world = None    # array of grid cells for fast object clicking
        self.outputs = None   # the set of neural outputs that decide what the creatures want to do
        self.collisions = None   # the set of ray collisions for each creature
        self.time = 0

    @cuda_profile
    def trace_rays(self):
        """Returns [N, R, 3] array of the results of ray collisions for all creatures."""
        rays = self.creatures.rays
        posns = self.creatures.positions
        sizes = self.creatures.sizes
        colors = self.creatures.colors

        collisions = torch.zeros((rays.shape[0], rays.shape[1], colors.shape[-1]), device='cuda')  # [N, R, C]

        # algorithms.correct_ray_trace[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors, collisions)

        self.kernels('correct_ray_trace',
                     rays.shape[0], rays.shape[1],  # threads
                     rays, posns, sizes, colors, collisions, 
                     self.population)
        return collisions

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
        cells = torch.full((num_cells, num_cells, self.cfg.max_per_cell), -1, dtype=torch.int32, device='cuda')
        cell_counts = torch.zeros((num_cells, num_cells), dtype=torch.int32, device='cuda')
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
        collisions = torch.zeros((rays.shape[0], rays.shape[1], colors.shape[-1]), device='cuda')
        # algorithms.trace_rays_grid(self.cfg)[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors, 
        #                                                                    cells, cell_counts, 
        #                                                                    self.cfg.cell_size, collisions)
        self.kernels('trace_rays_grid', 
                     rays.shape[0], rays.shape[1], # threads
                     rays, posns, sizes, colors, 
                     cells, cell_counts, collisions,
                     self.population, cells.shape[0])
        return collisions#, (cell_counts >= max_per_cell).sum().item() / (num_cells**2) #, cells, cell_counts
    

    @cuda_profile
    def collect_stimuli(self, collisions):
        """Returns [N, F] array for all creature stimuli."""
        return self.creatures.collect_stimuli(collisions, self.food_grid)
    
    @cuda_profile
    def think(self, stimuli):
        """Return [N, O] array of outputs of the creatures' neural networks.."""
        return self.creatures.forward(stimuli.unsqueeze(1))
    
    @cuda_profile
    def move_creatures(self, outputs):
        """Rotate and move all creatures"""
        self.creatures.rotate_creatures(outputs)
        self.creatures.move_creatures(outputs)

    @cuda_profile
    def compute_attacks(self):
        """Compute which creatures are attacking which creatures are then based on that, update health 
        and energy of creatures."""
        posns = self.creatures.positions
        sizes = self.creatures.sizes
        colors = self.creatures.colors
        head_dirs = self.creatures.head_dirs
        tr_results = torch.zeros((self.population, 2), device='cuda')

        # figure out who is attacking who
        block_size = (32, 32)
        grid_size = (self.population//block_size[0] + 1, self.population//block_size[0] + 1)
        # blocks_per_grid (N/32, N/32), threads_per_block (32, 32)
        # print(grid_size, block_size)
        # print(posns.shape, sizes.shape, colors.shape, head_dirs.shape, nb_results.shape)
        # print(posns.dtype, sizes.dtype, colors.dtype, head_dirs.dtype, nb_results.dtype)
        # algorithms.is_attacking(self.cfg)[grid_size, block_size](posns, sizes, colors, 
        #                                                          head_dirs, tr_results)
        self.kernels('is_attacking', 
                     grid_size, block_size,  # threads
                     posns, sizes, colors, head_dirs, tr_results,
                     self.population)
        return tr_results
    
    @cuda_profile
    def compute_gridded_attacks(self, cells, cell_counts):
        """Compute which creatures are attacking which creatures are then based on that, update health 
        and energy of creatures."""
        posns = self.creatures.positions
        sizes = self.creatures.sizes
        colors = self.creatures.colors
        head_dirs = self.creatures.head_dirs

        # figure out who is attacking who
        tr_results = torch.zeros((self.population, 2), device='cuda')
        block_size = 512
        grid_size = self.population // block_size + 1
        # algorithms.gridded_is_attacking(self.cfg)[grid_size, block_size](posns, sizes, colors, head_dirs, 
        #                                                                  cells, cell_counts, 
        #                                                                  self.cfg.cell_size, tr_results)
        self.kernels('gridded_is_attacking', 
                     grid_size, block_size,  # threads
                     posns, sizes, colors, head_dirs, 
                     cells, cell_counts, tr_results,
                     self.population, cells.shape[0])
        return tr_results

    @cuda_profile
    def do_attacks(self, tr_results):
        # update health and energy
        self.creatures.do_attacks(tr_results)
        self.creatures.kill_dead(self.food_grid)

    @cuda_profile
    def creatures_eat(self):
        self.creatures.eat(self.food_grid)

    @cuda_profile
    def creatures_reproduce(self):
        self.creatures.reproduce()


    @property
    def central_food_grid(self):
        pad = self.cfg.food_sight
        return self.food_grid[pad:-pad, pad:-pad]

    @cuda_profile
    def grow_food(self):
        """The higher step_size is, the faster food grows (and the faster corpses decay)."""
        # don't grow on squares that are occupied by creatures

        # maximum growth in a single step is step_size*max_food roughly (ignoring negatives, which are rare)
        # creatures lose alive_cost energy per step at least. so the energy leaving the system is the sum of this amount
        # energy entering is step_size*max_food*(grid_size**2), so we should set step size to sum(size**2)/max_food/(grid_size**2)/10
    
        step_size = torch.sum(self.cfg.alive_cost(self.creatures.sizes))/self.cfg.max_food/(self.cfg.size**2)
        logging.info(f"Food growth step size: {step_size}")

        posns = self.creatures.positions.long()
        # this allows negative food. We can think of this as "overfeeding" -> toxicity.
        self.food_grid[posns[:, 1], posns[:, 0]] -= self.cfg.food_cover_decr

        # don't bother growing food in the padding/inaccesible area
        growing = self.central_food_grid
        # grow food and decay dead corpses slowly
        torch.where(growing < self.cfg.max_food, 
                    growing - step_size*(growing-self.cfg.max_food)*self.cfg.food_growth_rate, 
                    growing - step_size*(growing-self.cfg.max_food)*self.cfg.food_decay_rate, 
                    out=growing)
        
    
    @property
    def population(self):
        return self.creatures.population

    def write_checkpoint(self, path):
        torch.save({'food_grid': self.food_grid, 
                    'creatures': self.creatures, 
                    'cfg': self.cfg,
                    }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.food_grid = checkpoint['food_grid']
        self.creatures = checkpoint['creatures']
        self.cfg = checkpoint['cfg']


    def click_creature(self, mouse_pos) -> int:
        """Return the index of the creature clicked on, or None if no creature was clicked."""
        mouse_pos = torch.tensor(mouse_pos, device='cuda')   # [2]
        # print('mouse_pos', mouse_pos)
        dists = torch.norm(self.creatures.positions - mouse_pos, dim=1)   # [N, 2] - [1, 2] = [N, 2]
        # print('dists', dists)
        close_enough = dists < self.creatures.sizes
        # print('close_enough', close_enough)
        nonzero = torch.nonzero(close_enough)
        # print('nonzero', nonzero)
        return nonzero[0].item() if nonzero.shape[0] != 0 else None


    def update_selected_creature(self, creature_id):
        """Taking into account creatures that have died and been reborn this epoch, update the selected creature."""
        if creature_id is None:
            return None
        if self.creatures.dead is None:
            return creature_id
        
        if self.creatures.dead[creature_id]:   # if its dead, then there is no longer any creature selected
            return None  # indexing into dead is fine since it's using the old indices
        
        # otherwise, we need to compute how many creatures have died in indices before this one
        # and then adjust the index accordingly
        dead_before = torch.sum(self.creatures.dead[:creature_id])
        return creature_id - dead_before
        
    

    def update_creatures(self):
        """Move creatures, update health and energy, reproduce, eat, grow food. Update all the states of the creatures
        and the world."""
        if self.outputs == None:   # skip updating if we haven't computed the outputs yet
            return

        self.move_creatures(self.outputs)

        celled_world = self.compute_grid_setup()
        # print(celled_world)
        attacks2 = self.compute_gridded_attacks(*celled_world)
        # print(attacks2)
    
        self.do_attacks(attacks2)  # update health and energy of creatures, and kill any creatures that are dead
        if self.population == 0:
            return
        self.creatures_reproduce()    # allow high energy individuals to reproduce
        self.creatures_eat()   # give energy for being in a square, and then reduce that energy
        self.grow_food()   # give food time to grow


    def compute_decisions(self):
        """Compute the decisions of all creatures. This includes running the neural networks, computing memories, and
        running vision ray tracing. Sets `outputs`, `collisions`, and `celled_world`."""
        self.celled_world = self.compute_grid_setup()
        self.collisions = self.trace_rays_grid(*self.celled_world)
        stimuli = self.collect_stimuli(self.collisions)
        self.outputs = self.think(stimuli)
        

    def step(self, visualize=False, save=True) -> bool:
        logging.info(f"{self.population} creatures alive.")
        logging.info(f"Health:\n {self.creatures.healths}")
        logging.info(f"Energy:\n {self.creatures.energies}")
        
        self.update_creatures()   # move creatures, update health and energy, reproduce, eat, grow food

        self.compute_decisions()   # run neural networks, compute memories, do vision ray tracing
        
        if self.time % 60 == 0:   # save this generation
            if save:
                self.write_checkpoint('game.ckpt')
            if visualize:
                self.visualize(None, show_rays=False) 

        if self.population == 0:
            logging.info("Mass extinction!")
            print("Mass extinction!")
            return False

        self.time += 1
        return True

    def visualize(self, collisions, show_rays=True, legend=False):
        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Move the food grid to CPU if it's on GPU
        food_grid_cpu = self.food_grid.cpu().numpy()
        
        # Plot the food grid heatmap
        heatmap = ax.imshow(food_grid_cpu, interpolation='nearest', alpha=0.2, cmap='Greens')
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        
        # Move creature attributes to CPU if they're on GPU
        positions_cpu = self.creatures.positions.cpu().numpy() + self.creatures.pad  # add pad so it lines up with food_grid
        sizes_cpu = self.creatures.sizes.cpu().numpy()
        head_dirs_cpu = self.creatures.head_dirs.cpu().numpy()
        colors_cpu = self.creatures.colors.cpu().numpy()
        rays_cpu = self.creatures.rays.cpu().numpy()

        for i in range(len(positions_cpu)):
            pos = positions_cpu[i]
            size = sizes_cpu[i]
            color = colors_cpu[i] / 255  # Convert to 0-1 range for matplotlib
            head_dir = head_dirs_cpu[i]
            head_pos = pos + head_dir * size
            
            # Plot the creature as a circle
            circle = plt.Circle(pos, size, color=color, fill=True, alpha=0.9, label=i)
            ax.add_patch(circle)
            
            # Plot the head direction
            ax.plot([pos[0], head_pos[0]], [pos[1], head_pos[1]], color='black')
            
            if show_rays:
                # Plot the rays
                if collisions is not None:
                    collisions_cpu = collisions.cpu().numpy()
                    for j in range(len(rays_cpu[i])):
                        ray_dir = rays_cpu[i][j][:2]
                        ray_len = rays_cpu[i][j][2]
                        ray_end = pos + ray_dir * ray_len
                        collision_info = collisions_cpu[i][j]
                        if np.any(collision_info[:3] != 0):  # If any component is not zero, ray is active
                            ray_color = collision_info / 255  # Convert to 0-1 range for matplotlib
                            ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color=ray_color)
                        else:
                            ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
                else:
                    for j in range(len(rays_cpu[i])):
                        ray_dir = rays_cpu[i][j][:2]
                        ray_len = rays_cpu[i][j][2]
                        ray_end = pos + ray_dir * ray_len
                        ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
        
        ax.set_xlim([0, food_grid_cpu.shape[1]])
        ax.set_ylim([0, food_grid_cpu.shape[0]])
        ax.set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        if legend:
            plt.legend()
        plt.gca().invert_yaxis()
        plt.title(f'Step {self.time} - Population {self.population}')
        plt.show()


    def visualize_grid_setup(self, cells, cell_counts, collisions=None, show_rays=True):
        """Visualizes the grid setup for ray tracing."""
        # match the return order from self.compute_grid_setup for ease of use
        num_objects = self.population
        num_cells = cells.shape[0]
        width = num_cells * self.cfg.cell_size

        positions = self.creatures.positions.cpu().numpy()
        sizes = self.creatures.sizes.cpu().numpy()
        # print(sizes)
        colors = self.creatures.colors.cpu().numpy()
        
        for obj_idx in range(num_objects):
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Create a heatmap for the current object
            heatmap = np.ones((num_cells, num_cells, 3), dtype=int)*255
            
            for y in range(num_cells):
                for x in range(num_cells):
                    for k in range(cell_counts[y, x]):
                        if cells[y, x, k] == obj_idx:
                            heatmap[y, x] = colors[obj_idx]
            
            cax = ax.imshow(heatmap[::-1], cmap='viridis', extent=[0, width, 0, width])
            # plt.colorbar(cax, ax=ax)
            
            # Plot the circles
            for i in range(num_objects):
                x, y = positions[i]
                radius = sizes[i]
                color = colors[i] / 255  # Convert to [0, 1] range for Matplotlib
                circle = patches.Circle(
                    (x, y),
                    radius,
                    linewidth=0.4,
                    edgecolor='black',
                    facecolor=color,
                    alpha=0.5
                )
                ax.add_patch(circle)

                if show_rays:
                    rays_cpu = self.creatures.rays.cpu().numpy()
                    pos = np.array([x, y])
                    # Plot the rays
                    if collisions is not None:
                        collisions_cpu = collisions.cpu().numpy()
                        for j in range(len(rays_cpu[i])):
                            ray_dir = rays_cpu[i][j][:2]
                            ray_len = rays_cpu[i][j][2]
                            ray_end = pos + ray_dir * ray_len
                            collision_info = collisions_cpu[i][j]
                            if np.any(collision_info[:3] != 0):  # If any component is not zero, ray is active
                                ray_color = collision_info / 255  # Convert to 0-1 range for matplotlib
                                ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color=ray_color)
                            else:
                                ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
                    else:
                        for j in range(len(rays_cpu[i])):
                            ray_dir = rays_cpu[i][j][:2]
                            ray_len = rays_cpu[i][j][2]
                            ray_end = pos + ray_dir * ray_len
                            ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
            
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.title(f'Grid Setup Visualization for Object {obj_idx}')
            # show grid lines
            ax.set_xticks(np.arange(0, width, self.cfg.cell_size))#, minor=True)
            ax.set_yticks(np.arange(0, width, self.cfg.cell_size))#, minor=True)
            ax.grid(which='both', color='black', linestyle='-', linewidth=1)
            plt.gca().invert_yaxis()  # Invert y-axis to match array index representation
            plt.show()


    def plotly_visualize_grid_setup(self, cells, cell_counts):        
        """Visualizes the grid setup for ray tracing."""
        # match the return order from self.compute_grid_setup for ease of use
        num_objects = self.population
        num_cells = cells.shape[0]
        width = num_cells * self.cfg.cell_size

        positions = self.creatures.positions.cpu().numpy() + self.creatures.pad  # add pad so it lines up with food_grid
        sizes = self.creatures.sizes.cpu().numpy()
        colors = self.creatures.colors.cpu().numpy()
        
        for obj_idx in range(num_objects):
            heatmap = np.ones((num_cells, num_cells, 3), dtype=int) * 255

            for y in range(num_cells):
                for x in range(num_cells):
                    for k in range(cell_counts[y, x]):
                        if cells[y, x, k] == obj_idx:
                            heatmap[y, x] = colors[obj_idx]

            fig = go.Figure()

            # Create the heatmap for the current object
            fig.add_trace(go.Image(
                z=heatmap,
                opacity=0.5,
                x0=self.cfg.cell_size/2,
                dx=self.cfg.cell_size,
                y0=self.cfg.cell_size/2,
                dy=self.cfg.cell_size,
            ))

            # Plot the circles
            for i in range(num_objects):
                x, y = positions[i]
                radius = sizes[i]
                color = 'rgba({}, {}, {}, 0.5)'.format(*colors[i])

                min_x = x - radius
                max_x = x + radius
                min_y = y - radius
                max_y = y + radius

                fig.add_shape(
                    type='circle',
                    xref='x',
                    yref='y',
                    x0=min_x,
                    y0=min_y,
                    x1=max_x,
                    y1=max_y,
                    line=dict(
                        color='black',
                        width=1,
                    ),
                    fillcolor=color
                )

            fig.update_layout(
                title=f'Grid Setup Visualization for Object {obj_idx}',
                xaxis=dict(
                    title='Grid X',
                    tickmode='array',
                    tickvals=np.arange(0, width, self.cfg.cell_size),
                    showgrid=True,
                    gridcolor='black',
                    gridwidth=None,
                    range=[0, width]
                ),
                yaxis=dict(
                    title='Grid Y',
                    tickmode='array',
                    tickvals=np.arange(0, width, self.cfg.cell_size),
                    showgrid=True,
                    gridcolor='black',
                    gridwidth=None,
                    scaleanchor=None,
                    scaleratio=None,
                    range=[0, width]
                ),
                plot_bgcolor='rgba(255,255,255,1)',
                margin=dict(l=0, r=0, t=35, b=0),  # Reduce margins
                height=800,  # Set the height of the figure
            )

            fig.show()


def benchmark():
    import cProfile
    from pstats import SortKey
    import io, pstats
    times.clear()

    torch.manual_seed(1)
    cfg = simple_cfg()
    game = GWorld(cfg)

    pr = cProfile.Profile()
    pr.enable()
    for i in range(512):
        if not game.step(visualize=False):   # we did mass extinction before finishing
            return game

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    return times


def main(cfg=None):
    if cfg is None:
        cfg = Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0)
    times.clear()
    game = GWorld(cfg)
    print("BEGINNING SIMULATION...")
    time_now = time.time()
    for i in range(99999):
        if not game.step():
            break
        if i % 5 == 4:
            curr_time = time.time()
            print(f"FPS: {5/(curr_time - time_now)}")
            time_now = curr_time
    return times, game

        

        
        