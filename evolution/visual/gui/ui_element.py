from typing import Dict
from contextlib import contextmanager
from abc import ABC, abstractmethod
from imgui_bundle import imgui


class UIElement(ABC):
    # all measurements are divided by 13 since that's the default font size
    LINE_SIZE = 17/13  # line size for text in pixels
    HEADER_SIZE = 19/13  # header size in pixels (not including 2*HEADER_PAD)
    HEADER_PAD = 2/13  # padding between the header and the text
    SLIDER_SIZE = 23/13 # slider size in pixels (including 2*SLIDER_PAD = 4)
    PADDING = 6/13  # padding between the top header and the start of the first line (and the last line -> bottom of window)
    PLOT_HEIGHT = 220/13  # height of the plot in pixels
    HEADER_WIDTH_PAD = 55/13  # padding between the header and the right side of the window
    PLOT_WIDTH = 350/13   # max width of plots (unless overridden)

    @abstractmethod
    def render(self): ...

    @property
    @abstractmethod
    def height(self) -> float: ...


class Window(UIElement):
    def __init__(self, title, max_width):
        self.title = title
        self.max_width = max_width
        self.open = True
        self.elements: Dict[str, UIElement] = {}

    @property
    def height(self):
        if self.open:
            return imgui.get_font_size() * (self.HEADER_SIZE + 2*self.PADDING) + \
                                            sum(el.height for el in self.elements.values())
        return imgui.get_font_size() * self.HEADER_SIZE

    @property
    def width(self):
        if self.open:
            return imgui.get_font_size() * self.max_width
        return imgui.calc_text_size(self.title)[0] + imgui.get_font_size() * self.HEADER_WIDTH_PAD

    def __setattr__(self, key, value):
        if isinstance(value, UIElement):
            if key in self.elements:
                raise ValueError(f"Key {key} already exists in elements")
            self.elements[key] = value
        super().__setattr__(key, value)

    @contextmanager
    def begin(self, flags):
        self.open = imgui.begin(self.title, False, flags=flags)[0]
        yield
        imgui.end()


class CollapseableHeader(UIElement):
    def __init__(self, title, n_lines=1):
        self.title = title
        self.open = False
        self.n_lines = n_lines

    @property
    def height(self):
        header_size = self.HEADER_SIZE + 2*self.HEADER_PAD
        if self.open:
            size = header_size + self.n_lines*self.LINE_SIZE
        else:
            size = header_size
        return imgui.get_font_size() * size


    def render(self, lines):
        self.open = imgui.collapsing_header(self.title)
        # if len(lines) != self.n_lines:
        #     raise ValueError(f"Tried to render {len(lines)} lines, but expected {self.n_lines}")
        self.n_lines = len(lines)
        if self.open:
            for line in lines:
                imgui.text(line)


class Lines(UIElement):
    def __init__(self, n_lines=1):
        self.n_lines = n_lines

    @property
    def height(self):
        return imgui.get_font_size() * self.n_lines*self.LINE_SIZE

    def render(self, lines):
        # if len(lines) != self.n_lines:
        #     raise ValueError(f"Tried to render {len(lines)} lines, but expected {self.n_lines}")
        self.n_lines = len(lines)
        for line in lines:
            imgui.text(line)

class Checkbox(UIElement):
    @property
    def height(self):
        return imgui.get_font_size() * self.SLIDER_SIZE

    def render(self, label, value, text=None):
        if text is not None:
            imgui.text(text)
            imgui.same_line()
        return imgui.checkbox(label, value)


class Slider(UIElement):
    def __init__(self, slider_type):
        self.slider_type = slider_type

    @classmethod
    def slider_int(cls) -> 'Slider':
        return cls(imgui.slider_int)

    @classmethod
    def slider_float(cls) -> 'Slider':
        return cls(imgui.slider_float)

    @property
    def height(self):
        return imgui.get_font_size() * self.SLIDER_SIZE

    def render(self, label, value, min_value, max_value, flags=0, text=None, **kwargs):
        if text is not None:
            imgui.text(text)
            imgui.same_line()

        return self.slider_type(label, value, min_value, max_value, flags=flags, **kwargs)


class PlotElement(UIElement):
    @property
    def height(self):
        return self.plot_height + imgui.get_font_size() *  2*self.HEADER_PAD

    @property
    def plot_height(self):
        # needs to return value in pixels, since it doesn't get passed through Window.width or Window.height
        return imgui.get_font_size() * self.PLOT_HEIGHT

    def render(self):
        pass
