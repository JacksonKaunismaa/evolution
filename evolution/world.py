import numpy as np

from . import algorithms
from .creature import Creature


class World():
    def __init__(self):
        self.objects = [Creature() for _ in range(5)]

    def trace_rays(self):
        rays = self.collect_rays()  # [N, R, 3]
        objects = np.asarray(self.objects) # [N, 4]
        collisions = np.zeros((rays.shape[0], rays.shape[1], 2), dtype=np.float32)  # [N, R, 2]

        algorithms.ray_trace[rays.shape[0], rays.shape[1]](rays, objects, collisions)
        
        for i, collide in enumerate(collisions):
            self.objects[i].collisions = collide

    def collect_rays(self):
        return np.asarray([obj.get_rays() for obj in self.objects])

    def collect_stimuli(self):
        return np.asarray([obj.get_stimulus() for obj in self.objects])
    
    def collect_weights(self, layer):
        return np.asarray([obj.get_weights(layer) for obj in self.objects])
    


def main():
    game = World()

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

        

        
        