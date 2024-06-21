import numpy as np

"""
Genome will encode the brain. The brain will be a neural network.
The neural network will have 8 outputs: move forward, move backward, rotate left, rotate right, and 4 for memory.
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

Size will be a very relevant parameter controlled by evolution. Smaller creatures have the benefit
of moving faster and with less energy, but have less health and do less damage. They can reproduce for 
less energy.
"""


class Creature():
    def __init__(self, genome=None, mem_size=4, num_rays=32, hidden_sizes=None):
        """Genome will be a list of floating point numbers that correspond to attributes of the creature.
        Mutations will involve applying a random perturbation to the genome. The scale of this random 
        perturbation will be based on the "mutation rate", which is also a gene. The first part of
        the genome is the weights. Next is the biases. Then rays. Then the 
        maximum size, color, and then finally the mutation rates (one for gene category)."""
        if hidden_sizes is None:
            hidden_sizes = [30, 40]

        self.layer_sizes = [num_rays + mem_size + 9 + 2, *hidden_sizes, mem_size + 2]

        if genome is None:
            pass
        else:
            pass



    def mutate_and_reproduce(self):
        mut_rates = self.genome[-1]
        self.weights, self.biases, self.rays = self.genome[:3]
        self.size, self.color = self.genome[3:5]

        weight_perturb = np.random.normal(0, mut_rates[0], self.weights.shape)
        bias_perturb = np.random.normal(0, mut_rates[1], self.biases.shape)
        ray_perturb = np.random.normal(0, mut_rates[2], self.rays.shape)
        size_perturb = np.random.normal(0, mut_rates[3])
        color_perturb = np.random.normal(0, mut_rates[4])
        mut_rate_perturb = np.random.normal(0, mut_rates[5], mut_rates.shape)

        pos_genome, neg_genome = [], []
        pos_genome.append(self.weights + weight_perturb)
        neg_genome.append(self.weights - weight_perturb)

        pos_genome.append(self.biases + bias_perturb)
        neg_genome.append(self.biases - bias_perturb)

        pos_rays = self.rays + ray_perturb
        pos_rays[:,:2] /= np.linalg.norm(pos_rays[:,:2], axis=-1)  # make sure they stay unit norm
        neg_rays = self.rays - ray_perturb
        neg_rays[:,:2] /= np.linalg.norm(neg_rays[:,:2], axis=-1)

        pos_genome.append(pos_rays)
        neg_genome.append(neg_rays)

        pos_genome.append(self.size + size_perturb)
        neg_genome.append(self.size - size_perturb)

        pos_genome.append(self.color + color_perturb)
        neg_genome.append(self.color - color_perturb)

        pos_genome.append(mut_rates + mut_rate_perturb)
        neg_genome.append(mut_rates - mut_rate_perturb)
        return Creature(pos_genome), Creature(neg_genome)



    
    def get_stimuli(self, game, idx):
        """Need inputs for each ray, memory, food level, energy level, and health"""
        rays = game.collisions[idx]
        grid_posn = np.floor(self.position).astype(np.int32)
        food = np.ravel(game.food[grid_posn[0]-1:grid_posn[0]+2, grid_posn[1]-1:grid_posn[1]+2])

        return np.concatenate([rays, self.memory, food, [self.energy, self.health]])
    

    def fight(self, attack_result):
        """First element of attack_result is the number of creatures we are attacking. 
        Second element is how much damage we are taking."""
        self.health -= attack_result[1]
        self.energy -= attack_result[0] * self.size