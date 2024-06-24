from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
from numba import cuda
import time
from IPython.display import clear_output
import logging
from tqdm import trange, tqdm

logging.basicConfig(level=logging.ERROR, filename='game.log')



from . import algorithms
from .creature_array import CreatureArray

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
times = defaultdict(float)

def cuda_profile(func):
    def wrapper(*args, **kwargs):
        start.record()
        res = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        cuda.synchronize()
        times[func.__name__] += start.elapsed_time(end)
        return res
    return wrapper
    return func


class World():
    def __init__(self, start_creatures, max_creatures, size):
        self.size = size
        self.food_grid = torch.rand((size, size), device='cuda') * 5
        # pad with -inf on all sides
        self.food_grid = F.pad(self.food_grid, (1, 1, 1, 1), mode='constant', value=0)
        self.creatures = CreatureArray(start_creatures, max_creatures, size)

    @cuda_profile
    def trace_rays(self):
        """Returns [N, R, 3] array of the results of ray collisions for all creatures."""
        rays = self.creatures.rays
        posns = self.creatures.positions
        sizes = self.creatures.sizes
        colors = self.creatures.colors

        collisions = torch.zeros((rays.shape[0], rays.shape[1], 3), device='cuda')  # [N, R, 1]
        # nb_collisions = cuda.as_cuda_array(tr_collisions, sync=True)

        # blocks_per_grid (N), threads_per_block (R)
        # print out all shapes and dtypes
        # print(rays.shape, posns.shape, sizes.shape, colors.shape, nb_collisions.shape)
        # print(rays.dtype, posns.dtype, sizes.dtype, colors.dtype, nb_collisions.dtype)
        algorithms.correct_ray_trace[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors, collisions)
        return collisions

    @cuda_profile
    def compute_grid_setup(self, max_per_cell=512, cell_size=4.):
        """Returns [G, G, M] array of the grid setup for all creatures. Where G is the number of grid cells,
        (G = world_size // cell_size + 1), and M is the maximum number of objects per cell (M = max_per_cell),
        and [G, G] array of the number of objects in each cell. 
        
        Used to speed up ray tracing by reducing the number of objects we need to check."""
        rays = self.creatures.rays
        posns = self.creatures.positions
        sizes = self.creatures.sizes

        # fill in grid cells to vastly reduce the number of objects we need to check
        num_cells = int(self.size // cell_size + 1)
        cells = torch.ones((num_cells, num_cells, max_per_cell), dtype=torch.int32, device='cuda')*-1
        cell_counts = torch.zeros((num_cells, num_cells), dtype=torch.int32, device='cuda')
        block_size = 512
        grid_size = rays.shape[0] // block_size + 1

        algorithms.setup_grid[grid_size, block_size](posns, sizes, cells, cell_counts, cell_size)
        return cells, cell_counts, cell_size  # argument return order should match trace_rays_grid
    

    @cuda_profile
    def trace_rays_grid(self, cells, cell_counts, cell_size):
        """Returns [N, R, 3] array of the results of ray collisions for all creatures."""

        rays = self.creatures.rays
        colors = self.creatures.colors
        posns = self.creatures.positions
        sizes = self.creatures.sizes

        # traverse the grid and draw lines through it, checking each object that passes through the ray
        collisions = torch.zeros((rays.shape[0], rays.shape[1], colors.shape[-1]), device='cuda')
        algorithms.trace_rays_grid[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors, cells, cell_counts, cell_size, collisions)
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
        tr_results = torch.zeros((posns.shape[0], 2), device='cuda')

        # figure out who is attacking who
        block_size = (32, 32)
        grid_size = (posns.shape[0]//block_size[0] + 1, posns.shape[0]//block_size[0] + 1)
        # blocks_per_grid (N/32, N/32), threads_per_block (32, 32)
        # print(grid_size, block_size)
        # print(posns.shape, sizes.shape, colors.shape, head_dirs.shape, nb_results.shape)
        # print(posns.dtype, sizes.dtype, colors.dtype, head_dirs.dtype, nb_results.dtype)
        algorithms.is_attacking[grid_size, block_size](posns, sizes, colors, head_dirs, tr_results)
        return tr_results
    
    @cuda_profile
    def compute_gridded_attacks(self, cells, cell_counts, cell_size):
        """Compute which creatures are attacking which creatures are then based on that, update health 
        and energy of creatures."""
        posns = self.creatures.positions
        sizes = self.creatures.sizes
        colors = self.creatures.colors
        head_dirs = self.creatures.head_dirs

        # figure out who is attacking who
        tr_results = torch.zeros((posns.shape[0], 2), device='cuda')
        block_size = 512
        grid_size = posns.shape[0] // block_size + 1
        # print(grid_size, block_size)
        # print(posns.shape, sizes.shape, colors.shape, head_dirs.shape, cells.shape, cell_counts.shape, cell_size, tr_results.shape)
        algorithms.gridded_is_attacking[grid_size, block_size](posns, sizes, colors, head_dirs, cells, cell_counts, cell_size, tr_results)
        return tr_results

    @cuda_profile
    def do_attacks(self, tr_results):
        # update health and energy
        self.creatures.do_attacks(tr_results)
        self.creatures.kill_dead(self.food_grid)

    @cuda_profile
    def creatures_eat(self):
        self.creatures.eat(self.food_grid, 0.2)

    @cuda_profile
    def creatures_reproduce(self):
        self.creatures.reproduce()

    @cuda_profile
    def energy_grow(self):
        # don't grow on squares that are occupied by creatures
        posns = self.creatures.positions.long()
        self.food_grid[posns[:, 1], posns[:, 0]] -= 0.1
        self.food_grid[1:-1, 1:-1] += 0.1

    def write_checkpoint(self, path):
        torch.save({'food_grid': self.food_grid, 
                    'creatures': self.creatures, 
                    'size': self.size, 
                    }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.food_grid = checkpoint['food_grid']
        self.creatures = checkpoint['creatures']
        self.size = checkpoint['size']


    def visualize(self, collisions, show_rays=True):
        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Move the food grid to CPU if it's on GPU
        food_grid_cpu = self.food_grid.cpu().numpy()
        
        # Plot the food grid heatmap
        heatmap = ax.imshow(food_grid_cpu, interpolation='nearest', alpha=0.2, cmap='Greens')
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        
        # Move creature attributes to CPU if they're on GPU
        positions_cpu = self.creatures.positions.cpu().numpy()
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
        plt.legend()
        # plt.gca().invert_yaxis()
        plt.show()


    def visualize_grid_setup(self, cells, cell_counts, cell_size):
        """Visualizes the grid setup for ray tracing."""
        # match the return order from self.compute_grid_setup for ease of use
        num_objects = self.creatures.positions.shape[0]
        num_cells = cells.shape[0]
        width = num_cells * cell_size

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
            
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.title(f'Grid Setup Visualization for Object {obj_idx}')
            # show grid lines
            ax.set_xticks(np.arange(0, width, cell_size))#, minor=True)
            ax.set_yticks(np.arange(0, width, cell_size))#, minor=True)
            ax.grid(which='both', color='black', linestyle='-', linewidth=1)
            plt.gca().invert_yaxis()  # Invert y-axis to match array index representation
            plt.show()



    def plotly_visualize_grid_setup(self, cells, cell_counts, cell_size):
        """Visualizes the grid setup for ray tracing."""
        # match the return order from self.compute_grid_setup for ease of use
        num_objects = self.creatures.positions.shape[0]
        num_cells = cells.shape[0]
        width = num_cells * cell_size

        positions = self.creatures.positions.cpu().numpy()
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
                x0=cell_size/2,
                dx=cell_size,
                y0=cell_size/2,
                dy=cell_size,
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
                    tickvals=np.arange(0, width, cell_size),
                    showgrid=True,
                    gridcolor='black',
                    gridwidth=None,
                    range=[0, width]
                ),
                yaxis=dict(
                    title='Grid Y',
                    tickmode='array',
                    tickvals=np.arange(0, width, cell_size),
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


def main():
    # results = {}
    # n_trial = 10
    # for max_creat in tqdm([32, 64, 128, 256, 512]):
    #     for cell_size in tqdm([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0]):
    #         for j in range(n_trial):
    #             torch.random.manual_seed(j)

        times.clear()
        game = World(16, 32, 15)
        fps = 30
        pcts = []
        for i in range(99999):
            # print(i)
            logging.info(f"{game.creatures.positions.shape[0]} creatures alive.")
            logging.info(f"Health:\n {game.creatures.healths}")
            logging.info(f"Energy:\n {game.creatures.energies}")
            # if i % 100 == 0:
            #     print(f"On step{i}, {game.creatures.positions.shape[0]} creatures alive.")
            # collisions, pct_bad = game.trace_rays_grid()
            # pcts.append(pct_bad)
            celled_world = game.compute_grid_setup()
            collisions = game.trace_rays_grid(*celled_world)
            if i % 10 == 0:
                game.visualize(None, show_rays=False) 
                time.sleep(1/(2*fps))
            # input()
            stimuli = game.collect_stimuli(collisions)
            # print(stimuli.shape)
            logging.debug(f"Stimuli shape: {stimuli.shape}")
            outputs = game.think(stimuli)
            logging.debug(f"Outputs shape: {outputs.shape}")

            game.move_creatures(outputs)

            celled_world = game.compute_grid_setup()
            # if i == 23:
            #     return celled_world, game
            # attacks = game.compute_attacks()
            attacks2 = game.compute_gridded_attacks(*celled_world)
        
            game.do_attacks(attacks2)  # update health and energy of creatures, and kill any creatures that are dead

            game.creatures_eat()   # give energy for being in a square, and then reduce that energy

            game.creatures_reproduce()    # allow high energy individuals to reproduce

            game.energy_grow()   # give food time to grow
            # game.visualize(None)
            # input()
            # time.sleep(1/(2*fps))
            # # game.maybe_environmental_change()   # mass extinction time?
        return times
        # if (max_creat, cell_size) not in results:
        #     results[(max_creat, cell_size)] = []
        # results[(max_creat, cell_size)].append((sum(times.values()), np.percentile(pcts, 98)))
    # return results

        

        
        