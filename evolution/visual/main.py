from typing import Dict
from OpenGL.GL import *
from contextlib import contextmanager
import moderngl_window as mglw
from moderngl_window import settings
import glob
import os.path as osp
from cuda import cuda
import torch

from evolution.visual.controller import Controller
from evolution.visual.heatmap import Heatmap
from evolution.visual.instanced_creatures import InstancedCreatures

torch.set_grad_enabled(False)
import PIL.Image
import time
from sdl2 import timer
import sdl2.video


from .camera import Camera
from .. import gworld, config, loading_utils
from ..cu_algorithms import checkCudaErrors


class Game:
    def __init__(self, window, cfg: config.Config, shader_path='./shaders'):
        # super().__init__(**kwargs)
        self.wnd = window
        self.cfg = cfg
        self.ctx = window.ctx
        self.shaders = loading_utils.load_shaders(shader_path)
        self.world: gworld.GWorld = gworld.GWorld(self.cfg)
        self.cuDevice = self.world.kernels.cuDevice
        self.heatmap = Heatmap(self.cfg, self.ctx, self.world, self.shaders)
        self.creatures = InstancedCreatures(self.cfg, self.ctx, self.world, self.shaders)
        self.camera = Camera(self.ctx, self.cfg)
        

        self.controller = Controller(window)

        self.now = timer.SDL_GetTicks()
        self.delta_time = 0


    def step(self, n=20) -> bool:
        for _ in range(n):
            if not self.world.step(visualize=False, save=False):
                torch.cuda.synchronize()
                return False
        torch.cuda.synchronize()
        return True

    def render(self):
        # This method is called every frame
        self.ctx.clear(1.0, 1.0, 1.0)
        # print(frametime)
        # self.wnd.title = f'Evolution'#: {1./frametime: .2f} FPS'
        self.camera.ubo.write(self.camera.get_camera_matrix())
        self.camera.ubo.bind_to_uniform_block()

        self.heatmap.update()
        self.heatmap.render()
        self.creatures.update()
        self.creatures.render()

        curr_time = timer.SDL_GetTicks()
        self.delta_time = (curr_time - self.now) / 1000.0
        self.now = curr_time
        


def main():
    settings.WINDOW['class'] = 'moderngl_window.context.sdl2.Window'
    settings.WINDOW['gl_version'] = (4, 6)
    settings.WINDOW['title'] = 'Evolution'
    settings.WINDOW['size'] = (1920, 1080)
    settings.WINDOW['vsync'] = True
    window = mglw.create_window_from_settings()
    cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0)
    game = Game(window, cfg)
    populated = True

    print(sdl2.video.SDL_GL_GetSwapInterval())
    
    curr_time = timer.SDL_GetTicks()
    curr_fps = 0
    i = 0
   
    while not window.is_closing and populated:
        window.clear()
        game.render()
        # print(window.frames)
        # time.sleep(5.0)
        window.title = f'Epoch: {game.world.time} | Population: {game.world.population} | FPS: {curr_fps:.2f}'
        window.swap_buffers()
        # window.set_default_viewport()
        # window.process_events()
        # time.sleep(0.1)
        populated = game.step(1)
        i += 1
        if i % 5 == 4:
            next_time = timer.SDL_GetTicks()
            curr_fps = 5.0 / (next_time - curr_time) * 1000.0
            curr_time = next_time

    def unregister(rsrc):
        checkCudaErrors(cuda.cuGraphicsUnregisterResource(rsrc))


    # free Cuda resources
    game.heatmap.heatmap_tex.release()
    game.heatmap.heatmap_sampler.release()
    game.creatures.creature_tex.release()
    game.creatures.sampler.release()
    game.creatures.positions.release()
    game.creatures.sizes.release()
    game.creatures.head_dirs.release()
    game.creatures.colors.release()
    unregister(game.heatmap.cuda_heatmap)
    unregister(game.creatures.cuda_positions)
    unregister(game.creatures.cuda_sizes)
    unregister(game.creatures.cuda_head_dirs)
    unregister(game.creatures.cuda_colors)
    game.camera.ubo.release()
    game.world.kernels.shutdown()
    game.ctx.release()
    window.close()




            
        

