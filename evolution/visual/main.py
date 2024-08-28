from typing import Dict
from OpenGL.GL import *
from contextlib import contextmanager
import moderngl_window as mglw
from moderngl_window import settings
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import moderngl as mgl
import glob
import os.path as osp
from cuda import cuda
import torch

import imgui

torch.set_grad_enabled(False)
import PIL.Image
import time


from .interactive.controller import Controller
from .interactive.camera import Camera
from .graphics.heatmap import Heatmap
from .graphics.instanced_creatures import InstancedCreatures
from .graphics.creature_rays import CreatureRays
from .graphics.brain import BrainVisualizer
from .graphics.thoughts import ThoughtsVisualizer


from evolution.core import config
from evolution.core import gworld
from evolution.cuda.cu_algorithms import checkCudaErrors
from evolution.utils import loading
from evolution.utils.subscribe import Publisher


class Game:
    def __init__(self, window: mglw.BaseWindow, cfg: config.Config, shader_path='./shaders', load_path=None):
        # super().__init__(**kwargs)
        self.wnd: mglw.BaseWindow = window
        self.world: gworld.GWorld = gworld.GWorld(cfg)
        if load_path is not None:
            self.world.load_checkpoint(load_path)
            cfg = self.world.cfg

        self.cfg = cfg
        self.ctx: mgl.Context = window.ctx
        self.ctx.enable(mgl.BLEND)
        self.shaders = loading.load_shaders(shader_path)
        self.heatmap = Heatmap(self.cfg, self.ctx, self.world, self.shaders)
        self.creatures = InstancedCreatures(self.cfg, self.ctx, self.world, self.shaders)
        self.rays = CreatureRays(self.cfg, self.ctx, self.world, self.shaders)
        self.camera = Camera(self.ctx, self.cfg, window, self.world)
        self.brain_visual = BrainVisualizer(self.cfg, self.ctx, self.world, self.shaders)
        self.thoughts_visual = ThoughtsVisualizer(self.cfg, self.ctx, self, self.shaders)
        self.controller = Controller(window, self.camera, self)

        # self.max_dims = self.calculate_max_window_size()

        self.paused = False
        self.game_speed = 1
        
        self.creature_publisher = Publisher()
        self.creature_publisher.subscribe(self.rays)
        self.creature_publisher.subscribe(self.camera)
        # self.creature_publisher.subscribe(self.brain_visual)
        self.creature_publisher.subscribe(self.thoughts_visual)
        
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(window)
        
        self.setup_events()
        self.setup_imgui_extras()

    def setup_events(self):
        def create_func(evt_func):
            evt_name = evt_func[:-5]
            def func(*args, **kwargs):
                getattr(self.controller, evt_func)(*args, **kwargs)
                getattr(self.imgui, evt_name)(*args, **kwargs)
            return func
        
        for evt_func in dir(self.controller):
            if evt_func.endswith('_func'):
                setattr(self.wnd, evt_func, create_func(evt_func))
                
    def setup_imgui_extras(self):
        extra_funcs = ['resize_func', 'unicode_char_entered_func']
        def create_func(func_name):
            imgui_name = func_name[:-5]
            def func(*args, **kwargs):
                getattr(self.imgui, imgui_name)(*args, **kwargs)
            return func
        
        for func_name in extra_funcs:
            setattr(self.wnd, func_name, create_func(func_name))

    def save(self):
        self.world.write_checkpoint('game.ckpt')

    def step(self, n=None, force=False) -> bool:
        if self.paused and not force:
            return True
        if n is None:
            n = self.game_speed
        for _ in range(n):
            if not self.world.step():
                # torch.cuda.synchronize()
                return False
        # torch.cuda.synchronize()
        return True
        
    
    def toggle_pause(self):
        self.paused = not self.paused

    def selected_creature_updates(self):
        creature = self.world.get_selected_creature()
        self.creature_publisher.publish()
        for sub in self.creature_publisher.subscribers:
            sub.update(creature)
        # self.rays.update(creature)
        # self.camera.update(creature)
        # # self.brain_visual.update(creature)
        # self.thoughts_visual.update(creature)
        # if self.controller.selected_creature is not None:
        #     print('energy', self.world.creatures.energies[self.controller.selected_creature])
        

    def render(self):
        # This method is called every frame
        # self.ctx.viewport = (0, self.max_dims[1] - (self.wnd.height), self.wnd.width, self.wnd.height)
        # self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        self.ctx.clear(0.0, 0.0, 0.0)

        self.selected_creature_updates()
        
        self.camera.ubo.write(self.camera.get_camera_matrix())
        self.camera.ubo.bind_to_uniform_block()

        self.heatmap.render()
        self.creatures.render()
        self.rays.render()
        self.brain_visual.render()
        self.thoughts_visual.render()
        
        self.render_ui()

        # self.controller.tick()
        
    def render_ui(self):
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.show_test_window()

        imgui.begin("Custom window", True)
        imgui.text("Bar")
        imgui.text_colored("Eggs", 0.2, 1., 0.)
        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())
        

def main(cfg=None):
    settings.WINDOW['class'] = 'moderngl_window.context.glfw.Window'
    settings.WINDOW['gl_version'] = (4, 5)
    settings.WINDOW['title'] = 'Evolution'
    settings.WINDOW['size'] = (2000, 2000)
    settings.WINDOW['aspect_ratio'] = None
    settings.WINDOW['vsync'] = True
    settings.WINDOW['resizable'] = True
    # settings.WINDOW['fullscreen'] = True
    window = mglw.create_window_from_settings()
    if cfg is None:
        cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0, immortal=False)
        print(cfg.food_cover_decr)
    # cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
    #                     init_size_range=(0.2, 0.2), num_rays=32, immortal=True)
    game = Game(window, cfg, load_path='game.ckpt')
    game.toggle_pause()
    game.world.compute_decisions()

    populated = True

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
        populated = game.step()
        # i += 1
        # if i % 5 == 4:
        #     next_time = timer.SDL_GetTicks()
        #     curr_fps = 5.0 / (next_time - curr_time) * 1000.0
        #     curr_time = next_time

    def unregister(rsrc):
        checkCudaErrors(cuda.cuGraphicsUnregisterResource(rsrc))


    # free Cuda resources
    game.heatmap.heatmap_tex.release()
    game.heatmap.heatmap_sampler.release()
    game.creatures.creature_tex.release()
    game.creatures.creature_sampler.release()
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




            
        

