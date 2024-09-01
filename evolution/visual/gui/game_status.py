import imgui
from moderngl_window import BaseWindow
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .ui_element import UIElement


class GameStatus(UIElement):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.init_size = self.wnd.size
        self.collapsing_header_open = False
        self.n_lines = 3  # number of lines of text
        self.width = 250
        self.x_pos = 0  # y pos is adjustable, and height depends on n_lines * LINE_SIZE
        self.name = "Game Status"
        
    def render(self):
        # Set the position dynamically based on collapsing header state
        window_width, window_height = self.wnd.size
        
        
        if self.collapsing_header_open:
            #  set y_pos upwards so that it appears it is expanding upwards
            height = self.HEADER_SIZE + self.PADDING + self.LINE_SIZE * self.n_lines + self.SLIDER_SIZE + self.PADDING
        else:
            # hide near the bottom left corner so that it looks like it is collapsing
            height = self.HEADER_SIZE
            
        imgui.set_next_window_position(self.x_pos, 
                                        window_height - height, 
                                        condition=imgui.ALWAYS)

        imgui.set_next_window_size(self.width, 
                                   height, 
                                   condition=imgui.ALWAYS)
        # Begin a new ImGui window
        self.collapsing_header_open = imgui.begin(self.name, False, 
                    imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS | 
                    imgui.WINDOW_NO_RESIZE)[0]
        
        # Display the text when expanded
        if self.collapsing_header_open:
            imgui.text(f"food_cover_decr: {self.cfg.food_cover_decr:.6f}")
            imgui.text(f"epoch: {self.world.time}")
            imgui.text(f"population: {self.world.population}")
            changed, new_speed = imgui.slider_int('game_speed', self.state.game_speed, 1, self.cfg.max_game_speed, 
                                    flags=imgui.SLIDER_FLAGS_ALWAYS_CLAMP | imgui.SLIDER_FLAGS_LOGARITHMIC)
            if changed:
                self.state.game_speed = new_speed

        # End the ImGui window
        imgui.end()