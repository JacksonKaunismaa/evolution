from typing import List
import imgui
from moderngl_window import BaseWindow
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .ui_element import UIElement, CollapseableHeader


class CreatureInfo(UIElement):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.init_size = self.wnd.size
        self.width = 250
        self.y_pos = 0
        self.name = "Creature Stats"
        self.base_n_lines = 9
        self.attack_panel = CollapseableHeader('Attacking', 3)
        self.eating_panel = CollapseableHeader('Eating', 4)
        self.rotate_panel = CollapseableHeader('Rotation', 3)
        self.move_panel = CollapseableHeader('Movement', 3)
        self.panels: List[CollapseableHeader] = [getattr(self, panel) for panel in dir(self) 
                                                    if isinstance(getattr(self, panel), CollapseableHeader)]
        
    def render(self):
        # Set the position dynamically based on collapsing header state
        creat = self.state.selected_creature
        if not creat:   # if its not visible, don't show anything
            return
        window_width, window_height = self.wnd.size
        
        if creat.update_state_available:
            height = self.HEADER_SIZE + self.PADDING + \
                     self.base_n_lines * self.LINE_SIZE + \
                     len(self.panels) * (self.HEADER_SIZE + 2*self.HEADER_PAD) + \
                     sum([panel.n_lines for panel in self.panels if panel.open]) * self.LINE_SIZE + \
                     self.PADDING
        else:
            height = self.HEADER_SIZE +  self.PADDING + \
                     self.base_n_lines * self.LINE_SIZE + self.PADDING
        
        imgui.set_next_window_position(window_width - self.width, self.y_pos, condition=imgui.ALWAYS)

        imgui.set_next_window_size(self.width, height, condition=imgui.ALWAYS)
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
        imgui.text(f"Color: {int(creat.color[0])}, {int(creat.color[1])}, {int(creat.color[2])}")
        imgui.text(f"Eat pct: {100.*float(creat.eat_pct):.4f}")
        
        if creat.update_state_available:
            self.eating_panel.render([
                f"Alive Cost: {float(creat.alive_cost):.6f}",
                f"Food Eaten: {float(creat.food_eaten):.6f}",
                f"Cell Eaten Energy: {float(creat.cell_energy):.2f}",
                f"Food Damage: {float(creat.food_dmg_taken):.2f}",
            ])
            
            self.rotate_panel.render([
                f"Rotate Logit: {float(creat.rotate_logit):.2f}",
                f"Rotate Angle: {float(creat.rotate_angle):.2f}",
                f"Rotate Energy: {float(creat.rotate_energy):.2f}"
            ])
            
            self.move_panel.render([
                f"Move Logit: {float(creat.move_logit):.2f}",
                f"Move Amount: {float(creat.move_amt):.2f}",
                f"Move Energy: {float(creat.move_energy):.2f}"
            ])
            
            self.attack_panel.render([
                f"Num Attacking: {int(creat.n_attacking)}",
                f"Attack Damage Taken: {float(creat.dmg_taken):.2f}",
                f"Attack Energy: {float(creat.attack_cost):.2f}"
            ])

        # End the ImGui window
        imgui.end()