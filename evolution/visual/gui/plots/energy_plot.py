
from imgui_bundle import imgui, implot
from moderngl_window import BaseWindow

from evolution.visual.gui.ui_element import Window, PlotElement
from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState


class EnergyPlot(Window):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Energy Over Time', self.PLOT_WIDTH)
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.energy_tracker = state.energy_tracker
        self.energy_plotter = PlotElement()

    def render(self):
        # Set the position dynamically base on collapsing header state

        window_width, window_height = self.wnd.size

        pos = (window_width - self.width, window_height - self.height)
        imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)  # type: ignore
        sz = (self.width, self.height)
        imgui.set_next_window_size(sz, cond=imgui.Cond_.always)  # type: ignore

        updated = self.energy_tracker.has_updates()

        # Begin a new ImGui window
        with self.begin(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |  # type: ignore
                    imgui.WindowFlags_.no_resize):
            if implot.begin_plot("Energy Over Time", (self.width, self.energy_plotter.plot_height),  # type: ignore
                                 flags=implot.Flags_.no_legend | implot.Flags_.no_title):  # type: ignore
                # implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
                implot.setup_axes('Time', 'log10(Energy)')
                implot.setup_axis_limits(implot.ImAxis_.y1, self.energy_tracker.min_val,  # type: ignore
                                         self.energy_tracker.max_val, cond=imgui.Cond_.always)  # type: ignore

                implot.setup_axis_limits(implot.ImAxis_.x1, 0, self.energy_tracker.times[-1],  # type: ignore
                                            cond=imgui.Cond_.always if updated else imgui.Cond_.once)  # type: ignore

                implot.plot_line("Energy Over Time Plot",
                                self.energy_tracker.times, self.energy_tracker.values)#, flags=implot.LineFlags_.segments)
                                # flags=implot.Flags_.no_legend)
                implot.end_plot()
