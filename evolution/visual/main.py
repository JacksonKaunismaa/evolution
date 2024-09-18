import os.path as osp
import moderngl_window as mglw
from moderngl_window import settings
import moderngl as mgl

from evolution.core import config
from evolution.core import gworld
from evolution.utils import loading, event_handler

from .interactive.controller import Controller
from .interactive.camera import Camera
from .graphics.heatmap import Heatmap
from .graphics.instanced_creatures import InstancedCreatures
from .graphics.creature_rays import CreatureRays
from .graphics.brain import BrainVisualizer
from .graphics.thoughts import ThoughtsVisualizer
from .gui.ui_manager import UIManager
from .fps import FPSTracker


class Game:
    """Main class for the game. Contains all the visual components and the game state."""
    def __init__(self, window: mglw.BaseWindow, cfg: config.Config, shader_path='./shaders', load_path=None):
        other_state_dicts = {}
        if load_path is not None and osp.exists(load_path):
            self.world, other_state_dicts = gworld.GWorld.from_checkpoint(load_path)
            cfg = self.world.cfg
        else:
            self.world = gworld.GWorld(cfg)

        self.wnd: mglw.BaseWindow = window
        self.state = self.world.state

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

        if 'camera' in other_state_dicts:
            self.camera.load_state_dict(other_state_dicts['camera'])

        self.ui_manager = UIManager(cfg, self.world, window, self.state)

        self.state.init_publish()
        self.setup_events()

    def setup_events(self):
        """Setup event handlers for the window by forwarding them to the controller and ui manager."""
        def create_func(evt_func):
            is_mouse_evt = 'mouse' in evt_func
            def func(*args, **kwargs):
                getattr(self.ui_manager, evt_func)(*args, **kwargs)
                if not self.ui_manager.is_hovered() or not is_mouse_evt:
                    getattr(self.controller, evt_func)(*args, **kwargs)
            return func

        for func_name in event_handler.EventHandler.__abstractmethods__:
            if hasattr(self.wnd, func_name):
                setattr(self.wnd, func_name, create_func(func_name))
            else:
                raise AttributeError(f'Event function "{func_name}" not supported by window')

    def step(self, n=None) -> bool:
        """Step the game world based on current settings and return whether the game is still populated."""
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

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0)

        self.camera.update()

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

    game = Game(window, cfg, load_path='game.ckpt')
    game.state.game_paused = True
    # we do one initial step to force compute_decisions to be called (so visualizers can be populated)
    game.world.step()

    populated = True

    while not window.is_closing and populated:
        window.clear()
        populated = game.step()
        game.render()

        window.swap_buffers()
        game.ctx.viewport = (0, 0, window.width, window.height)
