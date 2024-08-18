from typing import Dict
import moderngl as mgl
import numpy as np
from cuda import cuda
from matplotlib import cm
from OpenGL.GL import *


from evolution.cuda import cuda_utils
from evolution.core import config
from evolution.core import gworld

class Heatmap:
    def __init__(self, cfg: config.Config, ctx: mgl.Context, world: gworld.GWorld, shaders: Dict[str, str]):
        self.cfg = cfg
        self.ctx = ctx
        self.world = world
        self.shaders = shaders

        # world.food_grid[50:80, 0:20] = -2
        self.heatmap_tex = self.ctx.texture((self.cfg.size, self.cfg.size), 1, dtype='f4')
        self.heatmap_sampler = self.ctx.sampler(texture=self.heatmap_tex)
        filter_type = self.ctx.NEAREST
        self.heatmap_sampler.filter = (filter_type, filter_type)


        # this thing never gets rotated or translated, so we can afford to use game coordinates
        # scr_vertices = np.asarray([
        #     -1.0,  1.0,    0.0, 1.0,
        #     -1.0, -1.0,    0.0, 0.0,
        #      1.0, -1.0,    1.0, 0.0,
        #      1.0,  1.0,    1.0, 1.0,
        # ], dtype='f4')
        scr_vertices = np.asarray([
             0.0, 1.0,    0.0, 1.0,
             0.0, 0.0,    0.0, 0.0,
             1.0, 0.0,    1.0, 0.0,
             1.0, 1.0,    1.0, 1.0,
        ], dtype='f4')
        scr_vertices[::4] *= cfg.size  # scale up to game coordinate sizes
        scr_vertices[1::4] *= cfg.size
        

        scr_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.screen_vbo = self.ctx.buffer(scr_vertices)
        self.screen_ibo = self.ctx.buffer(scr_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['heatmap.vs'],
            fragment_shader=self.shaders['heatmap.fs']
        )

        cmap = cm.RdYlGn
        cmap._init()
        self.prog['colormap'] = np.array(cmap._lut[:-3, :3], dtype='f4')
        self.prog['negGamma'] = 1. / 5
        # print(self.prog['colormap'])

        self.screen_vao = self.ctx.vertex_array(self.prog, [
            (self.screen_vbo, '2f 2f', 'pos', 'tex_coord')
        ], index_buffer=self.screen_ibo)

        self.cuda_heatmap = cuda_utils.register_cuda_image(self.heatmap_tex)

    def update(self):
        max_value, min_value = self.cfg.max_food, self.world.central_food_grid.min()
        # self.prog['minVal'] = min_value
        # self.prog['maxVal'] = max_value
        self.prog['scale'] = max(abs(min_value), abs(max_value))
        cuda_utils.copy_to_texture(self.world.central_food_grid, self.cuda_heatmap)

    def render(self):
        self.update()
        self.heatmap_sampler.use()
        self.screen_vao.render()