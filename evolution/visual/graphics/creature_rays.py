import moderngl as mgl
import numpy as np

from evolution.core import config
from evolution.core import gworld
from evolution.cuda import cuda_utils

from .updater import Updater

class CreatureRays(Updater):
    def __init__(self, cfg, ctx, world, shaders):
        super().__init__(world, 'creature_rays')
        self.cfg: config.Config = cfg
        self.ctx: mgl.Context = ctx
        self.world: gworld.GWorld = world
        self.shaders = shaders
        self.visible = False

        # cos(theta), sin(theta), length of arrow * sizeof(float)
        self.rays = self.ctx.buffer(reserve=self.cfg.num_rays * 3 * 4)
        self.ray_colors = self.ctx.buffer(reserve=self.cfg.num_rays * 3 * 4)  # 3 color components * sizeof(float)

        self.cuda_rays = cuda_utils.register_cuda_buffer(self.rays)
        self.cuda_ray_colors = cuda_utils.register_cuda_buffer(self.ray_colors)

        ray_width = 0.01
        hbox_vertices = np.asarray([
             0.0,   1.0,    # 0.0, 1.0,   # top left
             0.0,  -1.0,    # 0.0, 0.0,   # bottom left
             1.0,  -1.0,    # 1.0, 0.0,   # bottom right
             1.0,   1.0,    # 1.0, 1.0,   # top right
        ], dtype='f4')
        hbox_vertices[1::2] *= ray_width
        # print(hbox_vertices)
        # hbox_vertices[::4] /= self.cfg.size   # make it relative to the screen size
        # hbox_vertices[1::4] /= self.cfg.size

        hbox_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.hbox_vbo = self.ctx.buffer(hbox_vertices)
        self.hbox_ibo = self.ctx.buffer(hbox_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['rays.vs'],
            fragment_shader=self.shaders['rays.fs']
        )

        self.vao = self.ctx.vertex_array(self.prog, [
            (self.hbox_vbo, '2f', 'hbox_coord'),
            (self.rays, '3f /i', 'rays'),   # cos, sin, length
            (self.ray_colors, '3f /i', 'ray_colors')  # r, g, b
        ],
        index_buffer=self.hbox_ibo)

    def _update(self, creature_id):
        if creature_id is None:
            self.visible = False
            return
        self.visible = True
        cuda_utils.copy_to_buffer(self.world.creatures.rays[creature_id], self.cuda_rays)
        cuda_utils.copy_to_buffer(self.world.collisions[creature_id], self.cuda_ray_colors)
        self.prog['position'] = self.world.creatures.positions[creature_id]

    def render(self):
        if not self.visible:
            return
        # print('rendering rays')
        self.vao.render(instances=self.cfg.num_rays)