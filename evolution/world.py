import numpy as np
from matplotlib import pyplot as plt

from . import algorithms
from .creature import Creature


class World():
    def __init__(self, size):
        self.objects = [Creature() for _ in range(4)]
        self.food_grid = np.random.rand((size, size), dtype=np.float32)

    def trace_rays(self):
        rays = self.collect_rays()  # [N, R, 3]
        objects = self.collect_object_positions() # [N, 4]
        collisions = np.zeros((rays.shape[0], rays.shape[1], 1), dtype=np.float32)  # [N, R, 1]
        print(rays.dtype, rays.shape, objects.dtype, objects.shape, collisions.dtype, collisions.shape)
        algorithms.ray_trace[rays.shape[0], rays.shape[1]](rays, objects, collisions)

    def visualize_rays(self, rays, objects, collisions):
        # Colormap for objects and rays
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
        colors = {clr: colors[i] for i, clr in enumerate(set(objects[:,-1].flatten()))}
        colors[0.0] = 'black'
        # print(colors, set(collisions.flatten()))
        # Create a plot
        fig, ax = plt.subplots()

        # Plot each object and its rays
        for i, obj in enumerate(objects):
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
        plt.title('Objects with Rays')

    def collect_object_positions(self):
        return np.asarray([obj.get_pos() for obj in self.objects]).astype(np.float32)

    def collect_rays(self):
        return np.asarray([obj.get_rays() for obj in self.objects])

    def collect_stimuli(self):
        return np.asarray([obj.get_stimuli() for obj in self.objects])
    
    def collect_weights(self, layer):
        return np.asarray([obj.get_weights(layer) for obj in self.objects])
    


def main():
    game = World(100)

    while True:
        
        game.trace_rays()
        break
        
        # stimuli = game.collect_stimuli()

        # for layer in range(2):
        #     weights, biases = game.collect_weights(2)
        #     stimuli = matmul(weights, stimuli) + biases
        #     if layer < 2:
        #         stimuli = ReLU(stimuli)
        #     else:
        #         stimuli = sigmoid(stimuli)
        
        # game.move_objects(stimuli)  # deals with attacking, energy, damage

        # game.objects_eat()   # give energy for being in a square, and then reduce that energy

        # game.reproduce()    # allow high energy individuals to reproduce

        # game.energy_grow()   # give food time to grow
        # # game.maybe_environmental_change()   # mass extinction time?

        

        
        