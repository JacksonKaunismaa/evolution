
from imgui_bundle import imgui, implot
from moderngl_window import BaseWindow
import numpy as np

from evolution.visual.gui.ui_element import Window, PlotElement
from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState


class HistogramPlot(Window):
    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Population by Color', self.PLOT_WIDTH)
        self.cfg = cfg
        self.world = world
        self.state = state
        self.wnd = window
        self.histogram_tracker = state.histogram_tracker
        self.hist_plotter = PlotElement()
        self.n_bins = state.histogram_tracker.kwargs['n_bins']
        self.cmap = self.world.creatures.compute_hist_cmap(self.n_bins)
        # add alpha channel
        self.cmap = np.hstack((self.cmap, np.full((self.cmap.shape[0], 1), 255.))).astype(np.float32)
        implot.add_colormap('Histogram Colors', self.cmap/255.)

    def render(self):
        # Set the position dynamically base on collapsing header state

        window_width, window_height = self.wnd.size

        pos = (window_width - 2*self.width, window_height - self.height)
        imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)  # type: ignore
        sz = (self.width, self.height)
        imgui.set_next_window_size(sz, cond=imgui.Cond_.always)  # type: ignore

        updated = self.histogram_tracker.has_updates()

        # Begin a new ImGui window
        with self.begin(imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |  # type: ignore
                    imgui.WindowFlags_.no_resize):
            if implot.begin_plot("Population by Color", (self.width, self.hist_plotter.plot_height),  # type: ignore
                                 flags=implot.Flags_.no_legend | implot.Flags_.no_title):  # type: ignore
                # implot.setup_axis_scale(implot.ImAxis_.x1, implot.Scale_.log10)
                implot.setup_axes('Time', 'Population')
                implot.setup_axis_limits(implot.ImAxis_.y1, self.histogram_tracker.min_val,  # type: ignore
                                         self.histogram_tracker.max_val, cond=imgui.Cond_.always)  # type: ignore
                latest = self.histogram_tracker.times[-1]

                implot.setup_axis_limits(implot.ImAxis_.x1, max(0, latest-20_000), latest,  # type: ignore
                                            cond=imgui.Cond_.always if updated else imgui.Cond_.once)  # type: ignore

                implot.push_colormap('Histogram Colors')
                for b in range(self.n_bins):
                    if b == 0:
                        implot.plot_shaded(f"Color {b}", self.histogram_tracker.times, self.histogram_tracker.values[b])
                    else:
                        values = self.histogram_tracker.values[b]
                        prev_values = self.histogram_tracker.values[b-1]
                        implot.plot_shaded(f"Color {b}", self.histogram_tracker.times, values, prev_values)
                implot.pop_colormap()
                implot.end_plot()
