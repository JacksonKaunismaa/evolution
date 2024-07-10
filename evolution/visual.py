from typing import Dict
import numpy as np
from OpenGL.GL import *
import moderngl as mgl
from contextlib import contextmanager
import moderngl_window as mglw
import glob
import os.path as osp
from cuda import cuda, nvrtc
import torch
torch.set_grad_enabled(False)

from . import gworld, config
from .cu_algorithms import checkCudaErrors

@contextmanager  # helper func to copy from pytorch into a texture
def cuda_activate(img):
    """Context manager simplifying use cuda images"""
    checkCudaErrors(cuda.cuGraphicsMapResources(1, img, None))
    yield checkCudaErrors(cuda.cuGraphicsSubResourceGetMappedArray(img, 0, 0))
    checkCudaErrors(cuda.cuGraphicsUnmapResources(1, img, None))


def copy_texture(tensor, pycuda_tex):
    with cuda_activate(pycuda_tex) as arr:
        pitch = tensor.stride(0) * tensor.element_size()
        # print('shape', tensor.shape, 'stride', tensor.stride(0), 'size', tensor.size(0), tensor.size(1), tensor.element_size())
        # print('tensor bytes', tensor.numel()*tensor.element_size())
        cpy = cuda.CUDA_MEMCPY2D()
        cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        cpy.srcDevice = tensor.data_ptr()
        cpy.srcPitch = pitch
        cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_ARRAY
        cpy.dstArray = arr
        cpy.dstPitch = tensor.size(1) * tensor.element_size()
        cpy.WidthInBytes = tensor.size(1) * tensor.element_size()
        cpy.Height = tensor.size(0)

        checkCudaErrors(cuda.cuMemcpy2DUnaligned(cpy))


def load_shaders(shader_path) -> Dict[str, str]:
    shaders = {}
    file_path = osp.dirname(osp.abspath(__file__))
    for path in glob.glob(osp.join(file_path, shader_path, '*')):
        fname = osp.basename(path)
        print(f'Loading shader {fname}')
        with open(path, 'r') as f:
            shaders[fname] = f.read()
    return shaders

class Heatmap:
    def __init__(self, cfg, ctx, world, shaders):
        self.cfg: config.Config = cfg
        self.ctx: mgl.Context = ctx
        self.world: gworld.GWorld = world
        self.shaders = shaders
        self.heatmap_tex = self.ctx.texture((self.cfg.size, self.cfg.size), 1)
        self.heatmap_sampler = self.ctx.sampler(texture=self.heatmap_tex)
        self.heatmap_sampler.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.heatmap_sampler.use()

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
        ], index_buffer=self.screen_ibo)

        self.cuda_heatmap = checkCudaErrors(cuda.cuGraphicsGLRegisterImage(
                                        int(self.heatmap_tex.glo), 
                                        GL_TEXTURE_2D, 
                                        cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))

    def update(self):
        normalized_food_grid = (255 * (self.world.central_food_grid / self.world.central_food_grid.max())).byte()
        copy_texture(normalized_food_grid,#[2:-2, 2:-2], 
                     self.cuda_heatmap)

    def render(self):
        # self.ctx.clear(1.0, 1.0, 1.0)
        self.screen_vao.render()


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

    def render(self, time, frametime):
        # This method is called every frame
        self.ctx.clear(1.0, 1.0, 1.0)
        # print(frametime)
        self.wnd.title = f'Evolution : {1./frametime: .2f} FPS'
        self.heatmap.update()
        self.heatmap.render()
