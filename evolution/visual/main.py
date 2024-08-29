from typing import Dict
from OpenGL.GL import *
from contextlib import contextmanager
import moderngl_window as mglw
from moderngl_window import settings
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import moderngl as mgl
import os.path as osp
from cuda import cuda
import torch
torch.set_grad_enabled(False)
import imgui



from .interactive.controller import Controller
from .interactive.camera import Camera
from .graphics.heatmap import Heatmap
from .graphics.instanced_creatures import InstancedCreatures
from .graphics.creature_rays import CreatureRays
from .graphics.brain import BrainVisualizer
from .graphics.thoughts import ThoughtsVisualizer
from .game_state import GameState
from .gui.ui_manager import UIManager
from .fps import FPSTracker


from evolution.core import config
from evolution.core import gworld
from evolution.cuda.cu_algorithms import checkCudaErrors
from evolution.utils import loading
from evolution.utils.subscribe import Publisher


class Game:
    def __init__(self, window: mglw.BaseWindow, cfg: config.Config, shader_path='./shaders', load_path=None):
        # super().__init__(**kwargs)
        self.wnd: mglw.BaseWindow = window
        self.state = GameState(cfg)
        self.world: gworld.GWorld = gworld.GWorld(cfg, self.state)
        
        if load_path is not None:
            self.world.load_checkpoint(load_path)
            cfg = self.world.cfg

        self.cfg = cfg
        self.ctx: mgl.Context = window.ctx
        self.ctx.enable(mgl.BLEND)
        self.shaders = loading.load_shaders(shader_path)
        self.heatmap = Heatmap(cfg, self.ctx, self.world, self.shaders)
        self.creatures = InstancedCreatures(cfg, self.ctx, self.world, self.state, self.shaders)
        self.rays = CreatureRays(cfg, self.ctx, self.world, self.shaders)
        self.camera = Camera(cfg, self.ctx, window, self.world)
        self.brain_visual = BrainVisualizer(cfg, self.ctx, self.world, self.shaders)
        self.thoughts_visual = ThoughtsVisualizer(cfg, self.ctx, self.world, window, self.state, self.shaders)
        self.controller = Controller(self.world, window, self.camera, self.state)
        self.fps_tracker = FPSTracker()
        
        self.ui_manager = UIManager(cfg, self.world, window, self.state)
        
        self.creature_publisher = Publisher()
        self.creature_publisher.subscribe(self.rays)
        self.creature_publisher.subscribe(self.camera)
        # self.creature_publisher.subscribe(self.brain_visual)
        self.creature_publisher.subscribe(self.thoughts_visual)
        
        self.setup_events()
        self.setup_imgui_extras()

    def setup_events(self):
        def create_func(evt_func):
            evt_name = evt_func[:-5]
            is_mouse_evt = 'mouse' in evt_name
            def func(*args, **kwargs):
                getattr(self.ui_manager.imgui, evt_name)(*args, **kwargs)
                if not self.ui_manager.is_hovered() or not is_mouse_evt:
                    getattr(self.controller, evt_func)(*args, **kwargs)
            return func
        
        for evt_func in dir(self.controller):
            if evt_func.endswith('_func'):
                setattr(self.wnd, evt_func, create_func(evt_func))
                
    def setup_imgui_extras(self):
        extra_funcs = ['resize_func', 'unicode_char_entered_func']
        def create_func(func_name):
            imgui_name = func_name[:-5]
            def func(*args, **kwargs):
                getattr(self.ui_manager.imgui, imgui_name)(*args, **kwargs)
            return func
        
        for func_name in extra_funcs:
            setattr(self.wnd, func_name, create_func(func_name))

    def step(self, n=None) -> bool:
        if self.state.game_paused:
            return True
        if n is None:
            n = self.state.game_speed
        for _ in range(n):
            if not self.world.step():
                # torch.cuda.synchronize()
                return False
        # torch.cuda.synchronize()
        return True

    def selected_creature_updates(self):
        creature = self.state.selected_creature
        self.creature_publisher.publish()
        for sub in self.creature_publisher.subscribers:
            sub.update(creature)
        

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0)

        self.selected_creature_updates()
        
        self.camera.ubo.write(self.camera.get_camera_matrix())
        self.camera.ubo.bind_to_uniform_block()

        self.heatmap.render()
        self.creatures.render()
        self.rays.render()
        self.brain_visual.render()
        self.thoughts_visual.render()
        
        self.ui_manager.render()
        
        self.fps_tracker.tick()

        self.wnd.title = f'CUDA-Evolution - FPS: {self.fps_tracker.fps:.2f}'
        # self.controller.tick()
        

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
    game.state.toggle_pause()
    game.world.compute_decisions()

    populated = True
   
    while not window.is_closing and populated:
        window.clear()
        game.render()

        window.swap_buffers()
        game.ctx.viewport = (0, 0, window.width, window.height)

        populated = game.step()