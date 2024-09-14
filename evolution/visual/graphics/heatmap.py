from typing import Dict
import moderngl as mgl
import numpy as np
import torch
torch.set_grad_enabled(False)
from cuda import cuda
from matplotlib import cm
from OpenGL.GL import *


from evolution.cuda import cuda_utils
from evolution.core import config
from evolution.core import gworld
from evolution.utils.subscribe import Subscriber


class Heatmap(Subscriber):
    def __init__(self, cfg: config.Config, ctx: mgl.Context, world: gworld.GWorld, shaders: Dict[str, str]):
        super().__init__(10)
        world.state.game_publisher.subscribe(self)
        
        self.cfg = cfg
        self.ctx = ctx
        self.world = world
        self.shaders = shaders

        # world.food_grid[50:80, 0:20] = -2
        self.heatmap_tex = self.ctx.texture((self.cfg.size, self.cfg.size), 1, dtype='f1')
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
        self.prog['negGamma'] = 7.0
        # print(self.prog['colormap'])

        self.screen_vao = self.ctx.vertex_array(self.prog, [
            (self.screen_vbo, '2f 2f', 'pos', 'tex_coord')
        ], index_buffer=self.screen_ibo)

        self.cuda_heatmap = cuda_utils.register_cuda_image(self.heatmap_tex)

    def _update(self):
        fgrid = self.world.central_food_grid
        max_value, min_value = max(fgrid.max(),self.cfg.max_food), min(fgrid.min(), -2.)
        # print(min_value, t.unravel_index(fgrid.argmin(), fgrid.shape))
        # self.prog['minVal'] = min_value
        # self.prog['maxVal'] = max_value
        # self.prog['scale'] = self.cfg.max_food # max(abs(min_value), abs(max_value))
        byte_grid = torch.where(fgrid < 0, 
                            (fgrid - min_value) / (0 - min_value) * 128,    # [0, 127] (technically [0, 128], with floating point error)
                            (fgrid / max_value - 1e-6) * 128 + 128).byte()  # [128, 255]. -1e-6 to avoid 1*128+128 -> 256 -> wrap to 0
        # byte_grid = ((fgrid - min_value) / (max_value - min_value) * 255).byte()
        # print(byte_grid)
        cuda_utils.copy_to_texture(byte_grid, self.cuda_heatmap)
        # print(fgrid.cpu().numpy()[::-1, :])
        # print(byte_grid.cpu().numpy()[::-1, :])

    def render(self):
        self.update()
        self.heatmap_sampler.use()
        self.screen_vao.render()