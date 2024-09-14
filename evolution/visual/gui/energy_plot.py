
from typing import List
from imgui_bundle import imgui, implot
from moderngl_window import BaseWindow
import numpy as np

from .time_plot import TimePlot
from .ui_element import CollapseableHeader, Window, Lines, UIElement

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState
from evolution.core.creatures.creature_trait import InitializerStyle


class EnergyPlot(Window):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Energy Over Time')
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.width = 450 / 13
        self.plotter = TimePlot(self.state, self.world.total_energy, 10)
        
    def render(self):
        # Set the position dynamically base on collapsing header state

        window_width, window_height = self.wnd.size
        
        height = 250 / 13

        
        pos = imgui.ImVec2(window_width - imgui.get_font_size() * self.width, 
                           window_height - imgui.get_font_size() * height)
        imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)

        sz = imgui.ImVec2(imgui.get_font_size() * self.width, 
                          imgui.get_font_size() * height)
        imgui.set_next_window_size(sz, cond=imgui.Cond_.always)
        self.plotter.update()
        # Begin a new ImGui window
        with self.begin(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |
                    imgui.WindowFlags_.no_resize):

            if len(self.plotter.values) > 0:
                if implot.begin_plot("Energy Over Time", (self.width * imgui.get_font_size(), 
                                                          200)):
                    implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
                    implot.plot_line("Energy Over Time Plot", 
                                    np.asarray(self.plotter.values),
                                    xscale=10, xstart=0)
                    implot.end_plot()