from imgui_bundle import imgui
from moderngl_window import BaseWindow

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState

from .ui_element import UIElement, Lines, Window, Checkbox, Slider


class GameStatus(Window):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Game Status')
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.food_decr_checkbox = Checkbox()
        self.text = Lines()
        self.speed_slider = Slider.slider_int()
        self.width = 300 / 13
        self.x_pos = 0  # y pos is adjustable, and height depends on n_lines * LINE_SIZE
        
    def render(self):
        # Set the position dynamically based on collapsing header state
        window_width, window_height = self.wnd.size
        
        pos = imgui.ImVec2(imgui.get_font_size() * self.x_pos, 
                           window_height - imgui.get_font_size() * self.height)
        imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)
        
        sz = imgui.ImVec2(imgui.get_font_size() * self.width, 
                          imgui.get_font_size() * self.height)
        imgui.set_next_window_size(sz, cond=imgui.Cond_.always)
        
        # Begin a new ImGui window
        with self.begin(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings 
                             | imgui.WindowFlags_.no_resize):
        
            # Display the text when expanded
            _, self.state.increasing_food_decr = \
                self.food_decr_checkbox.render("Increasing?",
                                               self.state.increasing_food_decr,
                                               f"Food Cover Decr: {self.cfg.food_cover_decr:.6f}")

            self.text.render([f"Epoch: {self.world.time}",
                              f"Population: {self.world.population}"])                

            changed, new_speed = \
                self.speed_slider.render('Game Speed', self.state.game_speed, 
                                         1, self.cfg.max_game_speed, 
                                         flags=imgui.SliderFlags_.always_clamp | imgui.SliderFlags_.logarithmic)
            if changed:
                self.state.game_speed = new_speed