import numpy as np

"""
Genome will encode the brain. The brain will be a neural network.
The neural network will have 8 outputs: move forward, move backward, turn left, turn right, and 4 for memory.
Each creature will be able to see. They will have some set of rays that defines their vision.
At each time step in the simulation, the creature will receive an input that is the result of
ray tracing their vision. There will be several inputs corresponding to these rays.
Another input will correspond to the creature's memory. Another set of inputs will correspond to 
the food levels nearby. 

Creatures can obtain energy either by eating other creatures (if they are damaged) or by staying still
long enough to collect food. Creatures that run out of energy die. Creatures that reach the maximum 
energy level reproduce.

Creatures can attack other creatures by moving into them (head needs to intersect their body).
Attacking takes a bit of energy, but deals damage to the other creature. 
Moving takes energy.


Every square of the world will have a food level (which varies over time). If a creature spends
enough time in a square, they will consume some percentage of the food and gain energy. 
If a creature kills another creature, they entire energy level of the killed creature will be deposited 
on the square where it was killed (with some added bonuses for the killer).

"""


class Creature():
    def __init__(self):
        self.position = np.random.rand(2).astype(np.float32)*7
        self.rays = np.random.normal(0, 1, (8, 2)).astype(np.float32)
        self.rays /= np.linalg.norm(self.rays, axis=1, keepdims=True)
        self.rays = self.rays.astype(np.float32)
        self.max_lengths = np.random.rand(8).astype(np.float32) * 3
        self.rays = np.concatenate([self.rays, self.max_lengths[:, None]], axis=1)
        self.size = 1.
        self.color = np.random.randint(10, 200)
        self.memory = np.random.zeros(4).astype(np.float32)

    def get_pos(self):
        return [self.position[0], self.position[1], self.size, self.color]

    def get_rays(self):
        return self.rays