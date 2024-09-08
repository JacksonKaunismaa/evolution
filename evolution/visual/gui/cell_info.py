from imgui_bundle import imgui
from moderngl_window import BaseWindow

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
        self.width = 250
        self.y_pos = 0
        self.name = "Cell Stats"
        
    def render(self):
        # Set the position dynamically based on collapsing header state
        cell = self.state.selected_cell
        if not cell:   # if its not visible, don't show anything
            return
        window_width, window_height = self.wnd.size
        
        if cell.update_state_available:
            self.n_lines = 6
        else: 
            self.n_lines = 2
        
        height = self.HEADER_SIZE + self.PADDING + self.n_lines * self.LINE_SIZE + self.PADDING
        
        imgui.set_next_window_pos(imgui.ImVec2(window_width - self.width, self.y_pos), 
                                  cond=imgui.Cond_.always)

        imgui.set_next_window_size(imgui.ImVec2(self.width, height), 
                                   cond=imgui.Cond_.always)
        # Begin a new ImGui window
        imgui.begin(self.name, False, 
                    imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |
                    imgui.WindowFlags_.no_resize)
        
        imgui.text(f"Food {cell.food:.3f}")
        imgui.text(f"Position: {cell._selected_cell[0]}, {cell._selected_cell[1]}")
        if cell.update_state_available:
            imgui.text(f"Percent Eaten: {100.*cell.cell_pct_eaten:.3f}")
            imgui.text(f"Cover Cost: {cell.cover_cost:.3f}")
            imgui.text(f"Growth Amount: {cell.growth_amt:.3f}")
            imgui.text(f"Step Size: {cell.step_size:.3f}")


        # End the ImGui window
        imgui.end()