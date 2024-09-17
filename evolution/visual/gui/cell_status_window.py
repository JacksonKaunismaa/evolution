from imgui_bundle import imgui
from moderngl_window import BaseWindow

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .ui_element import Window, Lines


class CellStatusWindow(Window):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Cell Stats', 250/13)
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.y_pos = 0
        self.main_text = Lines()
        self.delta_text = Lines()

    def render(self):
        # Set the position dynamically based on collapsing header state
        cell = self.state.selected_cell
        if not cell:   # if its not visible, don't show anything
            return
        window_width, window_height = self.wnd.size   # pylint: disable=unused-variable

        height = self.height
        if not cell.update_state_available:
            height -= self.delta_text.height

        pos = (window_width - self.width, self.y_pos)
        imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)

        sz = (self.width, height)
        imgui.set_next_window_size(sz, cond=imgui.Cond_.always)

        # Begin a new ImGui window
        with self.begin(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |
                    imgui.WindowFlags_.no_resize):
            self.main_text.render([
                f"Food: {cell.food:.3f}",
                f"Position: {cell._selected_cell[0]}, {cell._selected_cell[1]}"  # pylint: disable=protected-access
            ])
            if cell.update_state_available:
                self.delta_text.render([
                    f"Percent Eaten: {100.*cell.cell_pct_eaten:.3f}",
                    f"Cover Cost: {cell.cover_cost:.3f}",
                    f"Growth Amount: {cell.growth_amt:.3f}",
                    f"Step Size: {cell.step_size:.3f}"
                ])
