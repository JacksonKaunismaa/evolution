import imgui
from moderngl_window import BaseWindow
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .ui_element import UIElement


class CellInfo(UIElement):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.init_size = self.wnd.size
        self.n_lines = 2
        self.width = 250
        self.y_pos = 0
        self.name = "Cell Stats"
        
    def render(self):
        # Set the position dynamically based on collapsing header state
        cell = self.state.selected_cell
        if not cell:   # if its not visible, don't show anything
            return
        window_width, window_height = self.wnd.size
        
        imgui.set_next_window_position(window_width - self.width, self.y_pos, condition=imgui.ALWAYS)

        imgui.set_next_window_size(self.width, self.HEADER_SIZE + self.LINE_SIZE * self.n_lines, condition=imgui.ALWAYS)
        # Begin a new ImGui window
        imgui.begin(self.name, False, 
                    imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS |
                    imgui.WINDOW_NO_RESIZE)
        
        # self.collapsing_header_open = imgui.collapsing_header(self.name)[0]
        # Display the text when expanded
        # if self.collapsing_header_open:
        imgui.text(f"Food {cell.food:.3f}")
        imgui.text(f"Position: {cell._selected_cell[0]}, {cell._selected_cell[1]}")


        # End the ImGui window
        imgui.end()