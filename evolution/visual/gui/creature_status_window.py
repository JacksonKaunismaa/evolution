from typing import List
from imgui_bundle import imgui
from moderngl_window import BaseWindow

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState
from evolution.core.creatures.trait_initializer import InitializerStyle

from .ui_element import CollapseableHeader, Window, Lines, UIElement


class CreatureStatusWindow(Window):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Creature Stats', 250/13)
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.y_pos = 0
        self.main_text = Lines()
        self.mutation_element = CollapseableHeader('Mutation Rates')

        self.delta_text = Lines()
        self.rotate_element = CollapseableHeader('Rotation')
        self.move_element = CollapseableHeader('Movement')
        self.attack_element = CollapseableHeader('Attacking')
        self.eating_element = CollapseableHeader('Eating')
        self.dynamic_elements: List[UIElement] = [self.delta_text,
                                                self.rotate_element,
                                                self.move_element,
                                                self.attack_element,
                                                self.eating_element]

    def render(self):
        # Set the position dynamically based on collapsing header state
        creat = self.state.selected_creature
        if not creat:   # if its not visible, don't show anything
            return
        window_width, window_height = self.wnd.size  # pylint: disable=unused-variable

        height = self.height
        if not creat.update_state_available:
            height -= sum(p.height for p in self.dynamic_elements)

        pos = (window_width - self.width, self.y_pos)
        imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)

        sz = (self.width, height)
        imgui.set_next_window_size(sz, cond=imgui.Cond_.always)

        # Begin a new ImGui window
        with self.begin(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |
                    imgui.WindowFlags_.no_resize):

            self.main_text.render([
                f"Age: {int(creat.age)} ({creat.age_stage})",
                f"Energy: {creat.energy:.4f} / {creat.reproduce_energy:.4f}",
                f"Health: {creat.health:.4f} / {creat.max_health:.4f}",
                f"Age Multiplier: {creat.age_mult:.4f}",
                f"Num Children: {int(creat.n_children)}",
                f"Size: {creat.size:.4f}",
                f"Position: {creat.position[0]:.2f}, {creat.position[1]:.2f}",
                f"Color: {int(creat.color[0])}, {int(creat.color[1])}, {int(creat.color[2])}",
                f"Eat pct: {100.*float(creat.eat_pct):.4f}",
                f"ID: {creat._selected_creature}"  # pylint: disable=protected-access
            ])

            if creat.update_state_available:
                self.delta_text.render([f"Net Energy Delta: {creat.net_energy_delta:.6f}"])

            self.mutation_element.render([
                f"{name.replace('_', ' ').title()}: {creat.mutation_rate[param.init.mut_idx]:.6f}"
                for name,param in self.world.creatures.variables.items()
                    if param.init.style == InitializerStyle.MUTABLE  # => has a mutation rate
            ])

            if creat.update_state_available:
                self.rotate_element.render([
                    f"Rotate Logit: {creat.rotate_logit:.2f}",
                    f"Rotate Angle: {creat.rotate_angle:.2f}°",
                    f"Rotate Energy: {creat.rotate_energy:.6f}"
                ])

                self.move_element.render([
                    f"Move Logit: {creat.move_logit:.2f}",
                    f"Move Amount: {creat.move_amt:.2f}",
                    f"Move Energy: {creat.move_energy:.6f}"
                ])

                self.attack_element.render([
                    f"Num Attacking: {creat.n_attacking}",
                    f"Attack Damage Taken: {creat.dmg_taken:.5f}",
                    f"Attack Energy: {creat.attack_cost:.2f}"
                ])
                self.eating_element.render([
                    f"Alive Energy: {creat.alive_cost:.6f}",
                    f"Food Eaten: {creat.food_eaten:.6f}",
                    f"Cell Eaten Energy: {creat.cell_energy:.6f}",
                    f"Food Damage: {creat.food_dmg_taken:.6f}",
                ])
