import imgui
from moderngl_window import BaseWindow
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .ui_element import UIElement


class CreatureInfo(UIElement):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.init_size = self.wnd.size
        self.n_lines = 7
        self.width = 250
        self.y_pos = 0
        self.name = "Creature Stats"
        
    def render(self):
        # Set the position dynamically based on collapsing header state
        creat = self.state.selected_creature
        if not creat:   # if its not visible, don't show anything
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
        imgui.text(f"Age: {int(creat.age)} ({creat.age_stage})")
        imgui.text(f"Energy: {float(creat.energy):.4f} / {float(creat.reproduce_energy):.4f}")
        imgui.text(f"Health: {float(creat.health):.4f} / {float(creat.max_health):.4f}")
        imgui.text(f"Age Multiplier: {float(creat.age_mult):.4f}")
        imgui.text(f"Num Children: {int(creat.n_children)}")
        imgui.text(f"Size: {float(creat.size):.4f}")
        imgui.text(f"Position: {float(creat.position[0]):.2f}, {float(creat.position[1]):.2f}")
        # imgui.text(f"Color: {creat.color}")


        # End the ImGui window
        imgui.end()