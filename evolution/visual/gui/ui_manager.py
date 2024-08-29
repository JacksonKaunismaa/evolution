from typing import List
import imgui
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from moderngl_window import BaseWindow


from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .game_status import GameStatus
from .ui_element import UIElement


class UIManager:
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.ui_elements: List[UIElement] = []
        
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(window)
        
        self.game_status = GameStatus(cfg, world, window, state)
        
    def __setattr__(self, name, value):
        if isinstance(value, UIElement):
            self.ui_elements.append(value)
        super().__setattr__(name, value)
        
    def is_hovered(self):
        return self.imgui.io.want_capture_mouse
    
    def render(self):
        imgui.new_frame()
        
        for elem in self.ui_elements:
            elem.render()
            
        # imgui.show_test_window()
            
        imgui.render()
        self.imgui.render(imgui.get_draw_data())
         
