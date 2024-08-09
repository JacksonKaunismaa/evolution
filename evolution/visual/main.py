from typing import Dict
from OpenGL.GL import *
from contextlib import contextmanager
import moderngl_window as mglw
from moderngl_window import settings
import moderngl as mgl
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
from sdl2.ext import window as sdl2_window
import sdl2.video


from .camera import Camera
from .. import gworld, config, loading_utils
from ..cu_algorithms import checkCudaErrors


class Game:
    def __init__(self, window: mglw.BaseWindow, cfg: config.Config, shader_path='./shaders'):
        # super().__init__(**kwargs)
        self.wnd: mglw.BaseWindow = window
        self.cfg = cfg
        self.ctx: mgl.Context = window.ctx
        self.shaders = loading_utils.load_shaders(shader_path)
        self.world: gworld.GWorld = gworld.GWorld(self.cfg)
        self.heatmap = Heatmap(self.cfg, self.ctx, self.world, self.shaders)
        self.creatures = InstancedCreatures(self.cfg, self.ctx, self.world, self.shaders)
        self.camera = Camera(self.ctx, self.cfg, window)
        self.controller = Controller(window, self.camera, self)
        self.paused = False
        self.max_dims = self.calculate_max_window_size()


    def step(self, n=1, force=False) -> bool:
        if self.paused and not force:
            return True
        for _ in range(n):
            if not self.world.step(visualize=False, save=False):
                torch.cuda.synchronize()
                return False
        torch.cuda.synchronize()
        return True
    
    def calculate_max_window_size(self):
        # calculate non-fullscreen maximum window size, in a very hacky way
        # basically, we make the window larger than its supposed to be, do a single process_events(),
        # which captures the automatic resizing event to fit the window in the frame, coming from
        # the window manager, and then we get the size of the window
        mode = sdl2.video.SDL_DisplayMode()
        sdl2.video.SDL_GetDesktopDisplayMode(0, mode)  # these should be larger than the max possible
        max_width = mode.w * 10
        max_height = mode.h * 10
        self.wnd.resize(max_width, max_height)
        self.wnd.process_events()
        max_width, max_height = self.wnd.size
        # self.wnd.size = start_size
        return max_width, max_height
        
    
    def toggle_pause(self):
        self.paused = not self.paused

    def render(self):
        # This method is called every frame
        self.ctx.viewport = (0, self.max_dims[1] - (self.wnd.height), self.wnd.width, self.wnd.height)
        # self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        self.ctx.clear(0.0, 0.0, 0.0)
        # print(self.ctx.viewport)
        # print(frametime)
        # self.wnd.title = f'Evolution'#: {1./frametime: .2f} FPS'
        self.camera.ubo.write(self.camera.get_camera_matrix())
        self.camera.ubo.bind_to_uniform_block()

        self.heatmap.update()
        self.heatmap.render()
        self.creatures.update()
        self.creatures.render()

        self.controller.tick()
        

def main():
    settings.WINDOW['class'] = 'moderngl_window.context.sdl2.Window'
    settings.WINDOW['gl_version'] = (4, 6)
    settings.WINDOW['title'] = 'Evolution'
    settings.WINDOW['size'] = (2000, 2000)
    settings.WINDOW['aspect_ratio'] = None
    settings.WINDOW['vsync'] = True
    settings.WINDOW['resizable'] = True
    # settings.WINDOW['fullscreen'] = True
    window = mglw.create_window_from_settings()
    # print(sdl2_window.SDL_)
    # cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0)
    cfg = config.Config(start_creatures=1, max_creatures=1, size=5, food_cover_decr=0.0,
                        init_size_range=(0.1, 0.1))
    game = Game(window, cfg)
    # game.toggle_pause()
    # central_grid = torch.arange(cfg.size**2, dtype=torch.float32, device='cuda').view(cfg.size, cfg.size)
    # pad = cfg.food_sight
    # game.world.food_grid[pad:-pad, pad:-pad] = central_grid
    # print(game.world.food_grid)
    populated = True

    # sdl2_window._get_sdl_window(window._window).maximize()
    # mode = sdl2.video.SDL_DisplayMode()
    # sdl2.video.SDL_GetDesktopDisplayMode(0, mode)
    # print(mode.h)
    
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
        game.ctx.viewport = (0, 0, window.width, window.height)
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




            
        
