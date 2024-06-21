import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)
from numba import cuda

from . import algorithms
from .creature_array import CreatureArray


class World():
    def __init__(self, start_creatures, max_creatures, size):
        self.food_grid = torch.rand((size, size)).cuda() * 5
        self.creatures = CreatureArray(start_creatures, max_creatures, size)

    def trace_rays(self):
        """Returns [N, R, 3] array of the results of ray collisions for all creatures."""
        collisions = torch.zeros((rays.shape[0], rays.shape[1], 1), device='cuda')  # [N, R, 1]
        collisions = cuda.as_cuda_array(collisions)
        rays = cuda.as_cuda_array(self.creatures.rays)
        posns = cuda.as_cuda_array(self.creatures.positions)
        sizes = cuda.as_cuda_array(self.creatures.sizes)
        colors = cuda.as_cuda_array(self.creatures.colors)

        # blocks_per_grid (N), threads_per_block (R)
        algorithms.ray_trace[rays.shape[0], rays.shape[1]](rays, posns, sizes, colors, collisions)
        return collisions

    def collect_stimuli(self, collisions):
        """Returns [N, F] array for all creature stimuli."""
        return self.creatures.collect_stimuli(collisions, self.food_grid)
    
    def think(self, stimuli):
        """Return [N, O] array of outputs of the creatures' neural networks.."""
        return self.creatures.forward(stimuli)
    
    def move_creatures(self, outputs):
        """Rotate and move all creatures"""
        self.creatures.rotate_creatures(outputs)
        self.creatures.move_creatures(outputs)


    def do_attacks(self):
        """Compute which creatures are attacking which creatures are then based on that, update health 
        and energy of creatures."""
        posns = cuda.as_cuda_array(self.creatures.positions)
        sizes = cuda.as_cuda_array(self.creatures.sizes)
        colors = cuda.as_cuda_array(self.creatures.colors)
        head_dirs = cuda.as_cuda_array(self.creatures.head_dirs)
        results = torch.zeros((posns.shape[0], 2), device='cuda')
        results = cuda.as_cuda_array(results)

        # figure out who is attacking who
        block_size = (32, 32)
        grid_size = (posns.shape[0]//block_size[0] + 1, posns.shape[0]//block_size[0] + 1)
        # blocks_per_grid (N/32, N/32), threads_per_block (32, 32)
        algorithms.is_attacking[grid_size, block_size](posns, sizes, colors, head_dirs, results)

        # update health and energy
        self.creatures.do_attacks(results)
        self.creatures.kill_dead(self.food_grid)

    def creatures_eat(self):
        self.creatures.eat(self.food_grid)

    def creatures_reproduce(self):
        self.creatures.reproduce()

    def energy_grow(self):
        self.food_grid += 0.1


    def visualize_rays(self, rays, creatures, collisions):
        # Colormap for creatures and rays
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
        colors = {clr: colors[i] for i, clr in enumerate(set(creatures[:,-1].flatten()))}
        colors[0.0] = 'black'
        # print(colors, set(collisions.flatten()))
        # Create a plot
        fig, ax = plt.subplots()

        # Plot each object and its rays
        for i, obj in enumerate(creatures):
            x, y, radius, color_idx = obj
            color = colors[color_idx]
            
            # Plot the object (circle)
            circle = plt.Circle((x, y), radius, color=color, fill=True, alpha=0.5)
            ax.add_artist(circle)
            
            # Plot the rays
            for j, ray in enumerate(rays[i]):
                dx, dy, max_length = ray
                ray_color_idx = collisions[i][j][0]
                ray_color = colors[ray_color_idx]
                
                # Normalize the ray direction to unit vector
                norm = np.sqrt(dx**2 + dy**2)
                dx /= norm
                dy /= norm
                
                # Scale the unit vector to the ray length
                dx *= max_length
                dy *= max_length
                
                # Plot the ray
                ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=ray_color, ec=ray_color,
                         alpha=0.2 if ray_color_idx == 0.0 else 1.0)

        # Set limits and aspect
        # ax.set_xlim(-2, 12)
        # ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('creatures with Rays')


def main():
    game = World(5, 10, 1000)

    while True:
        
        collisions = game.trace_rays()
        
        stimuli = game.collect_stimuli(collisions)
        outputs = game.think(stimuli)

        game.move_creatures(outputs)
        game.do_attacks()  # update health and energy of creatures, and kill any creatures that are dead

        game.creatures_eat()   # give energy for being in a square, and then reduce that energy

        game.reproduce()    # allow high energy individuals to reproduce

        game.energy_grow()   # give food time to grow
        # # game.maybe_environmental_change()   # mass extinction time?

        

        
        