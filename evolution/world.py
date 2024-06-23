from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
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
    def trace_rays_grid(self, max_per_cell=512, cell_size=4.):
        """Returns [N, R, 3] array of the results of ray collisions for all creatures."""
        rays = self.creatures.rays
        colors = self.creatures.colors
        posns = self.creatures.positions
        sizes = self.creatures.sizes

        # fill in grid cells to vastly reduce the number of objects we need to check
        num_cells = int(self.size // cell_size + 1)
        cells = torch.ones((num_cells, num_cells, max_per_cell), dtype=torch.int32, device='cuda')*10
        cell_counts = torch.zeros((num_cells, num_cells), dtype=torch.int32, device='cuda')
        # print(cells.shape, cell_counts.shape)
        block_size = 512
        grid_size = rays.shape[0] // block_size + 1
        # print(grid_size, block_size)
        # print(posns.shape, sizes.shape, cells.shape, cell_counts.shape, cell_size)
        # print(posns.dtype, sizes.dtype, cells.dtype, cell_counts.dtype)
        algorithms.setup_grid[grid_size, block_size](posns, sizes, cells, cell_counts, cell_size)
        # print("Pct at max", (cell_counts >= max_per_cell).sum().item() / (num_cells**2))
        # print(cell_counts)
        # print(cells[..., :5].permute(2, 0, 1))
        # return cell_counts, cells
        # then, traverse the grid and draw lines through it, checking each object that passes through the ray
        collisions = torch.zeros((rays.shape[0], rays.shape[1], colors.shape[-1]), device='cuda')
        algorithms.trace_rays_grid[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors, cells, cell_counts, cell_size, collisions)
        return collisions, (cell_counts >= max_per_cell).sum().item() / (num_cells**2) #, cells, cell_counts
    

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
    def do_attacks(self):
        """Compute which creatures are attacking which creatures are then based on that, update health 
        and energy of creatures."""
        posns = cuda.as_cuda_array(self.creatures.positions)
        sizes = cuda.as_cuda_array(self.creatures.sizes)
        colors = cuda.as_cuda_array(self.creatures.colors)
        head_dirs = cuda.as_cuda_array(self.creatures.head_dirs)
        tr_results = torch.zeros((posns.shape[0], 2), device='cuda')
        nb_results = cuda.as_cuda_array(tr_results, sync=True)

        # figure out who is attacking who
        block_size = (32, 32)
        grid_size = (posns.shape[0]//block_size[0] + 1, posns.shape[0]//block_size[0] + 1)
        # blocks_per_grid (N/32, N/32), threads_per_block (32, 32)
        # print(grid_size, block_size)
        # print(posns.shape, sizes.shape, colors.shape, head_dirs.shape, nb_results.shape)
        # print(posns.dtype, sizes.dtype, colors.dtype, head_dirs.dtype, nb_results.dtype)
        algorithms.is_attacking[grid_size, block_size](posns, sizes, colors, head_dirs, nb_results)

        # update health and energy
        self.creatures.do_attacks(tr_results)
        self.creatures.kill_dead(self.food_grid)

    @cuda_profile
    def creatures_eat(self):
        self.creatures.eat(self.food_grid, 0.1)

    @cuda_profile
    def creatures_reproduce(self):
        self.creatures.reproduce()

    @cuda_profile
    def energy_grow(self):
        self.food_grid[1:-1, 1:-1] += 0.1


    def visualize(self, collisions):
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
            size = np.sqrt(sizes_cpu[i])
            color = colors_cpu[i] / 255  # Convert to 0-1 range for matplotlib
            head_dir = head_dirs_cpu[i]
            head_pos = pos + head_dir * size
            
            # Plot the creature as a circle
            circle = plt.Circle(pos, size, color=color, fill=True, alpha=0.9, label=i)
            ax.add_patch(circle)
            
            # Plot the head direction
            ax.plot([pos[0], head_pos[0]], [pos[1], head_pos[1]], color='black')
            
            # Plot the rays
            if collisions is not None:
                collisions_cpu = collisions.cpu().numpy()
                for j in range(len(rays_cpu[i])):
                    ray_dir = rays_cpu[i][j][:2]
                    ray_len = np.sqrt(rays_cpu[i][j][2])
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
                    ray_len = np.sqrt(rays_cpu[i][j][2])
                    ray_end = pos + ray_dir * ray_len
                    ax.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], color='gray', alpha=0.3)
        
        ax.set_xlim([0, food_grid_cpu.shape[1]])
        ax.set_ylim([0, food_grid_cpu.shape[0]])
        ax.set_aspect('equal')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()


def main():
    results = {}
    n_trial = 10
    for max_creat in tqdm([32, 64, 128, 256, 512]):
        for cell_size in tqdm([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0]):
            for j in range(n_trial):
                times.clear()
                torch.random.manual_seed(j)
                game = World(512, 16384, 1000)
                fps = 30
                pcts = []
                for i in range(500):
                    logging.info(f"{game.creatures.positions.shape[0]} creatures alive.")
                    logging.info(f"Health:\n {game.creatures.healths}")
                    logging.info(f"Energy:\n {game.creatures.energies}")
                    # if i % 100 == 0:
                    #     print(f"On step{i}, {game.creatures.positions.shape[0]} creatures alive.")
                    collisions, pct_bad = game.trace_rays_grid()
                    pcts.append(pct_bad)
                    # game.visualize(collisions)
                    # input()
                    # time.sleep(1/(2*fps))
                    stimuli = game.collect_stimuli(collisions)
                    logging.debug(f"Stimuli shape: {stimuli.shape}")
                    outputs = game.think(stimuli)
                    logging.debug(f"Outputs shape: {outputs.shape}")

                    game.move_creatures(outputs)
                    game.do_attacks()  # update health and energy of creatures, and kill any creatures that are dead

                    game.creatures_eat()   # give energy for being in a square, and then reduce that energy

                    game.creatures_reproduce()    # allow high energy individuals to reproduce

                    game.energy_grow()   # give food time to grow
                    # game.visualize(None)
                    # input()
                    # time.sleep(1/(2*fps))
                    # # game.maybe_environmental_change()   # mass extinction time?
                # return times
                if (max_creat, cell_size) not in results:
                    results[(max_creat, cell_size)] = []
                results[(max_creat, cell_size)].append((sum(times.values()), np.percentile(pcts, 98)))
    return results

        

        
        