from typing import Dict
import numpy as np
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.gl import graphics_map_flags
from OpenGL.GL import *
import moderngl
import pycuda.autoinit
from contextlib import contextmanager
import moderngl_window as mglw
import glob
import os.path as osp

from . import world, config

@contextmanager  # helper func to copy from pytorch into a texture
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0,0)
    mapping.unmap()


class Game(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1920, 1080)

    def __init__(self, shader_path='./shaders', **kwargs):
        super().__init__(**kwargs)
        self.cfg = config.Config()
        self.shaders = self.load_shaders(shader_path)
        self.world: world.World = world.World(self.cfg)

        # we will use a green-only, very simple colormap for now
        self.heatmap_tex = self.ctx.texture((self.cfg.size, self.cfg.size), 1)
        self.cuda_heatmap = cuda_gl.RegisteredImage(int(self.heatmap_tex.glo), 
                                                    GL_TEXTURE_2D, 
                                                    graphics_map_flags.WRITE_DISCARD)
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

        self.screen_vao = self.ctx.vertex_array(self.prog, [
            (self.screen_vbo, '2f 2f', 'pos', 'tex_coord')
        ], ibo=self.screen_ibo)


    def load_shaders(self, shader_path) -> Dict[str, str]:
        shaders = {}
        for path in glob.glob(osp.join(shader_path, '*')):
            fname = osp.basename(path)
            with open(path, 'r') as f:
                shaders[fname] = f.read()
        return shaders

    def copy_texture(self, tensor, pycuda_tex: cuda_gl.RegisteredImage):
        with cuda_activate(pycuda_tex) as arr:
            cpy = cuda.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(arr)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tensor.stride(0) * tensor.element_size()
            cpy.height = tensor.size(0)
            cpy(aligned=False)
            cuda.Context.synchronize()


    def render(self, time, frametime):
        # This method is called every frame
        self.ctx.clear(1.0, 1.0, 1.0)
        self.vao.render()
