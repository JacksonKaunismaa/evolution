from typing import Dict, List
import moderngl as mgl
import numpy as np
import torch

from evolution.core import config
from evolution.core import gworld
from evolution.cuda import cuda_utils

from .updater import Updater

class BrainVisualizer(Updater):
    def __init__(self, cfg: config.Config, ctx: mgl.Context, world: gworld.GWorld, shaders: Dict[str, str]):
        super().__init__(world, 'brain_visualizer')
        self.cfg = cfg
        self.ctx = ctx
        self.world = world
        self.shaders = shaders

        self.image_size = (1200,1200)
        self.neuron_radius = 6
        self.line_density = 300
        self.pad = 3
        self.display_width = 0.5  # display_width / 2 is what percentage of the screen the brain takes up
        
        self.img = None
        self.neuron_indices = None

        self.visible = False
        self.prev_creature_id = None

        self.brain_tex = self.ctx.texture(self.image_size, 1, dtype='f4')
        self.brain_sampler = self.ctx.sampler(texture=self.brain_tex)
        filter_type = self.ctx.NEAREST
        self.brain_sampler.filter = (filter_type, filter_type)

        
        brain_vertices = np.asarray([   # put in top left of screen
             -1.0, 1.0,    0.0, 1.0,
             -1.0, 0.0,    0.0, 0.0,
             0.0, 0.0,    1.0, 0.0,
             0.0, 1.0,    1.0, 1.0,
        ], dtype='f4')
        brain_vertices[::4] *= self.display_width
        brain_vertices[1::4] *= self.display_width
        
        brain_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.brain_vbo = self.ctx.buffer(brain_vertices)
        self.brain_ibo = self.ctx.buffer(brain_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['brain.vs'],
            fragment_shader=self.shaders['brain.fs']
        )

        self.brain_vao = self.ctx.vertex_array(self.prog, [
            (self.brain_vbo, '2f 2f', 'pos', 'tex_coord')
        ], index_buffer=self.brain_ibo)

        self.cuda_brain = cuda_utils.register_cuda_image(self.brain_tex)

    def generate_image_and_indices(self, activations, weights):
        # Image tensor initialized to black (0, 0, 0)
        img = torch.zeros((self.image_size[0], self.image_size[1]), dtype=torch.float32, device='cuda')
        
        # Compute layer positions (x coordinates)
        n_layers = len(activations)
        layer_x = torch.linspace(self.neuron_radius, self.image_size[1] - self.neuron_radius, n_layers)

        # Compute neuron positions (y coordinates) for each layer
        neuron_positions = []
        for i, layer in enumerate(activations):
            n_neurons = layer.size(0)
            width = self.image_size[1] - 2 * self.pad
            gap = width / n_neurons
            layer_y = torch.linspace(gap/2 + self.pad, self.image_size[0] - (gap/2 + self.pad), n_neurons)
            positions = torch.stack((layer_x[i].repeat(n_neurons), layer_y), dim=1)
            neuron_positions.append(positions)
        
        # Convert to tensor for easier indexing later
        # neuron_positions = torch.stack(neuron_positions)
        
        # Draw lines for weights
        for i, weight_matrix in enumerate(weights):
            start_positions = neuron_positions[i]
            end_positions = neuron_positions[i + 1]

            # Normalize weights to be between -1 and 1 for color intensity
            weight_matrix_norm = weight_matrix / weight_matrix.abs().max()

            # Draw lines between each neuron in layer i and layer i+1
            start_x = start_positions[:, 0].view(-1, 1)
            start_y = start_positions[:, 1].view(-1, 1)
            end_x = end_positions[:, 0].view(1, -1)
            end_y = end_positions[:, 1].view(1, -1)

            # Vectorized calculation of line points and colors
            line_x = torch.linspace(0, 1, steps=self.line_density).view(-1, 1, 1) * (end_x - start_x) + start_x
            line_y = torch.linspace(0, 1, steps=self.line_density).view(-1, 1, 1) * (end_y - start_y) + start_y
            line_x = line_x.view(-1).long()
            line_y = line_y.view(-1).long()

            # Filter valid indices
            valid_indices = (line_x >= 0) & (line_x < self.image_size[1]) & (line_y >= 0) & (line_y < self.image_size[0])
            img[line_y[valid_indices], line_x[valid_indices]] = weight_matrix_norm.repeat(self.line_density, 1).view(-1)


        neuron_indices = []
        # Draw circles for neurons
        for layer_idx in range(len(activations)):
            positions = neuron_positions[layer_idx]
            
            x_positions = positions[:, 0].long()
            y_positions = positions[:, 1].long()

            layer_indices = []
            # Vectorized drawing of circles
            for idx in range(positions.size(0)):
                y, x = torch.meshgrid(
                    torch.arange(self.image_size[0]) - y_positions[idx], 
                    torch.arange(self.image_size[1]) - x_positions[idx]
                )
                mask = x**2 + y**2 <= self.neuron_radius**2
                indices = mask.nonzero()
                layer_indices.append(indices)
            neuron_indices.append(torch.cat(layer_indices))
        return img, neuron_indices

    def fast_update_image(self, creature_id):
        activations = self.fetch_activations(creature_id)
        for i, indices in enumerate(self.neuron_indices):
            num_pixels = indices.shape[0]
            num_neurons = activations[i].shape[0]
            pixels_per_neuron = num_pixels // num_neurons
            self.img[indices[:, 0], indices[:, 1]] = activations[i].repeat_interleave(pixels_per_neuron)
    
    def fetch_activations(self, creature_id):
        return [self.world.creatures.activations[i][creature_id].squeeze(0) for i in range(len(self.world.creatures.activations))]
    
    def fetch_weights(self, creature_id):
        return [self.world.creatures.weights[i][creature_id] for i in range(len(self.world.creatures.weights))]

    def _update(self, creature_id):
        if creature_id is None:
            self.visible = False
            return
        self.visible = True
        if self.prev_creature_id != creature_id:
            self.img, self.neuron_indices = self.generate_image_and_indices(self.fetch_activations(creature_id), 
                                                                            self.fetch_weights(creature_id))
        self.prev_creature_id = creature_id
        self.fast_update_image(creature_id)
        cuda_utils.copy_to_texture(self.img, self.cuda_brain)

    def render(self):
        if not self.visible:
            return
        self.brain_sampler.use()
        self.brain_vao.render()