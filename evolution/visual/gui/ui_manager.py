from typing import List, TYPE_CHECKING
from imgui_bundle import imgui, implot
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import glfw
from moderngl_window import BaseWindow

from .game_status_window import GameStatusWindow
from .creature_status_window import CreatureStatusWindow
from .cell_status_window import CellStatusWindow
from .ui_element import Window
from .plots.energy_plot import EnergyPlot


from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState
from evolution.utils.event_handler import EventHandler
if TYPE_CHECKING:
    from evolution.visual.main import Game


class UIManager(EventHandler):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.ui_elements: List[Window] = []
        
        imgui.create_context()
        implot.create_context()
        self.imgui = GlfwRenderer(window._window, attach_callbacks=False)
        
        
        # things that need to be set so that the Mixin works
        self.wnd = window
        self.io = imgui.get_io()
        self.reverse_mouse_map = {v:k for k,v in self.wnd._mouse_button_map.items()}
        
        # add widgets
        self.game_status = GameStatusWindow(cfg, world, window, state)
        self.creature_info = CreatureStatusWindow(cfg, world, window, state)
        self.cell_info = CellStatusWindow(cfg, world, window, state)
        self.energy_plot = EnergyPlot(cfg, world, window, state)
        
    def __setattr__(self, name, value):
        if isinstance(value, Window):
            self.ui_elements.append(value)
        super().__setattr__(name, value)
        
    def is_hovered(self):
        return self.imgui.io.want_capture_mouse
    
    def render(self):
        imgui.new_frame()
        
        for elem in self.ui_elements:
            elem.render()
            
        # imgui.show_test_window()
        self.imgui.process_inputs()
            
        imgui.render()
        self.imgui.render(imgui.get_draw_data())
    
    
    # see moderngl_window.intergrations.imgui.ModernglWindowMixin
    def resize_func(self, width, height):
        self.imgui.resize_callback(self.wnd, width, height)
    
    def key_event_func(self, key, action, mods):
        self.imgui.keyboard_callback(self.wnd, key, None, action, mods)    
        
    def mouse_position_event_func(self, x, y, dx, dy):
        self.imgui.mouse_callback()
        
    def mouse_drag_event_func(self, x, y, dx, dy):
        self.imgui.mouse_callback()
        
    def mouse_scroll_event_func(self, x_offset, y_offset):
        self.imgui.scroll_callback(self.wnd, x_offset, y_offset)
        
    def mouse_press_event_func(self, x, y, button):
        button = self.reverse_mouse_map.get(button, button)
        self.imgui.mouse_button_callback(self.wnd, button, glfw.PRESS, None)
        
    def mouse_release_event_func(self, x, y, button):
        button = self.reverse_mouse_map.get(button, button)
        self.imgui.mouse_button_callback(self.wnd, button, glfw.RELEASE, None)
        
    # def unicode_char_entered_func(self, char):
    #     self.imgui.char_callback(self.wnd, char)
