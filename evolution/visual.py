from typing import Dict
import numpy as np
from OpenGL.GL import *
import moderngl as mgl
from contextlib import contextmanager
import moderngl_window as mglw
import glob
import os.path as osp
from cuda import cuda
import torch
torch.set_grad_enabled(False)
from matplotlib import cm
import PIL.Image


from . import gworld, config
from .cu_algorithms import checkCudaErrors
from . import cuda_utils


def load_shaders(shader_path) -> Dict[str, str]:
    shaders = {}
    file_path = osp.dirname(osp.abspath(__file__))
    for path in glob.glob(osp.join(file_path, shader_path, '*')):
        fname = osp.basename(path)
        print(f'Loading shader {fname}')
        with open(path, 'r') as f:
            shaders[fname] = f.read()
    return shaders


def load_image_as_texture(ctx: mgl.Context, path: str) -> mgl.Texture:
    img = PIL.Image.open(path)
    np_img = np.array(img)
    texture = ctx.texture(np_img.shape[:2], np_img.shape[2], np_img.tobytes())
    return texture

def register_cuda_buffer(buffer: mgl.Buffer):
    return checkCudaErrors(cuda.cuGraphicsGLRegisterBuffer(
        int(buffer.glo),
        cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_NONE
    ))


class Heatmap:
    def __init__(self, cfg, ctx, world, shaders):
        self.cfg: config.Config = cfg
        self.ctx: mgl.Context = ctx
        self.world: gworld.GWorld = world
        self.shaders = shaders

        # world.food_grid[50:80, 0:20] = -2
        self.heatmap_tex = self.ctx.texture((self.cfg.size, self.cfg.size), 1, dtype='f4')
        self.heatmap_sampler = self.ctx.sampler(texture=self.heatmap_tex)
        filter_type = self.ctx.NEAREST
        self.heatmap_sampler.filter = (filter_type, filter_type)

        scr_vertices = np.asarray([
            -1.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
             1.0,  1.0,    1.0, 1.0,    
        ], dtype='f4')

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

        self.screen_vao = self.ctx.vertex_array(self.prog, [
            (self.screen_vbo, '2f 2f', 'pos', 'tex_coord')
        ], index_buffer=self.screen_ibo)

        self.cuda_heatmap = checkCudaErrors(cuda.cuGraphicsGLRegisterImage(
                                        int(self.heatmap_tex.glo), 
                                        GL_TEXTURE_2D, 
                                        cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))

    def update(self):
        max_value, min_value = self.world.central_food_grid.max(), self.world.central_food_grid.min()
        # self.prog['minVal'] = min_value
        # self.prog['maxVal'] = max_value
        self.prog['scale'] = max(abs(min_value), abs(max_value))
        cuda_utils.copy_to_texture(self.world.central_food_grid, self.cuda_heatmap)

    def render(self):
        self.heatmap_sampler.use()
        self.screen_vao.render()


class InstancedCreatures:
    def __init__(self, cfg, ctx, world, shaders):
        self.cfg: config.Config = cfg
        self.ctx: mgl.Context = ctx
        self.world: gworld.GWorld = world
        self.shaders = shaders

        self.positions = self.ctx.buffer(reserve=self.cfg.max_creatures * 2 * 4)  # 2 coordinates
        self.sizes = self.ctx.buffer(reserve=self.cfg.max_creatures * 1 * 4)  # 1 radius
        self.head_dirs = self.ctx.buffer(reserve=self.cfg.max_creatures * 2 * 4)  # 2 directions
        self.colors = self.ctx.buffer(reserve=self.cfg.max_creatures * 3 * 4)  # 3 color channels

        self.cuda_positions = register_cuda_buffer(self.positions)
        self.cuda_sizes = register_cuda_buffer(self.sizes)
        self.cuda_head_dirs = register_cuda_buffer(self.head_dirs)
        self.cuda_colors = register_cuda_buffer(self.colors)
        
        self.creature_tex = load_image_as_texture(self.ctx, './assets/grey_square_creature_sprite.png')
        self.sampler = self.ctx.sampler(texture=self.creature_tex)
        filter_type = self.ctx.NEAREST
        self.sampler.filter = (filter_type, filter_type)

        hbox_vertices = np.asarray([
            -1.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
             1.0,  1.0,    1.0, 1.0,    
        ], dtype='f4')
        hbox_vertices[::4] /= self.cfg.size   # make it relative to the screen size
        hbox_vertices[1::4] /= self.cfg.size

        hbox_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.hbox_vbo = self.ctx.buffer(hbox_vertices)
        self.hbox_ibo = self.ctx.buffer(hbox_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['creatures.vs'],
            fragment_shader=self.shaders['creatures.fs']
        )

        self.prog['width'] = self.cfg.size

        self.vao = self.ctx.vertex_array(self.prog, [
            (self.hbox_vbo, '2f 2f', 'hbox_coord', 'tex_coord'),
            (self.positions, '2f /i', 'position'),
            (self.sizes, 'f /i', 'size'),
            (self.head_dirs, '2f /i', 'head_dir'),
            (self.colors, '3f /i', 'color') 
        ],
        index_buffer=self.hbox_ibo)

    def update(self):
        cuda_utils.copy_to_buffer(self.world.creatures.positions, self.cuda_positions)
        cuda_utils.copy_to_buffer(self.world.creatures.sizes, self.cuda_sizes)
        cuda_utils.copy_to_buffer(self.world.creatures.head_dirs, self.cuda_head_dirs)
        cuda_utils.copy_to_buffer(self.world.creatures.colors, self.cuda_colors)
        
    def render(self):
        self.sampler.use()
        self.vao.render(instances=self.world.population)


class Game(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1920, 1080)

    def __init__(self, shader_path='./shaders', **kwargs):
        super().__init__(**kwargs)
        self.cfg = config.Config()
        self.shaders = load_shaders(shader_path)
        self.world: gworld.GWorld = gworld.GWorld(self.cfg)
        self.cuDevice = self.world.kernels.cuDevice
        self.heatmap = Heatmap(self.cfg, self.ctx, self.world, self.shaders)
        self.creatures = InstancedCreatures(self.cfg, self.ctx, self.world, self.shaders)

    def render(self, time, frametime):
        # This method is called every frame
        self.ctx.clear(1.0, 1.0, 1.0)
        # print(frametime)
        self.wnd.title = f'Evolution : {1./frametime: .2f} FPS'
        self.heatmap.update()
        self.heatmap.render()
        self.creatures.update()
        self.creatures.render(time)
