import imgui
from moderngl_window import BaseWindow
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.visual.game_state import GameState

from .ui_element import UIElement


class GameStatus(UIElement):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        
    def render(self):
        window_width, window_height = self.wnd.size
        imgui.set_next_window_position(10, window_height-684, imgui.ONCE)  # Bottom-left corner

        # Begin a new ImGui window
        imgui.begin("Game Stats", False, imgui.WINDOW_NO_TITLE_BAR)

        # Create a collapsible header that expands upwards
        if imgui.collapsing_header("Game Stats", flags=imgui.TREE_NODE_DEFAULT_OPEN):
            # Display the text
            imgui.text(f"food_decr_rate: {self.cfg.food_cover_decr}")
            imgui.text(f"epoch: {self.world.time}")
            imgui.text(f"population: {self.world.population}")
            imgui.text(f"game_speed: {self.state.game_speed}")
            imgui.text(f"window_sie: {window_width}x{window_height}")
            imgui.text(f"diff: {window_height-684}")
            # imgui.text(f"cursor pos: {self.wnd.mouse_position}")

        # End the ImGui window
        imgui.end()