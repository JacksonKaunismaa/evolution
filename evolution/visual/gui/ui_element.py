from typing import Dict
from imgui_bundle import imgui
from contextlib import contextmanager
from abc import ABC, abstractmethod


class UIElement:
    LINE_SIZE = 17/13  # line size for text in pixels
    HEADER_SIZE = 19/13  # header size in pixels (not including 2*HEADER_PAD)
    HEADER_PAD = 2/13  # padding between the header and the text
    SLIDER_SIZE = 23/13 # slider size in pixels (including 2*SLIDER_PAD = 4)
    PADDING = 6/13  # padding between the top header and the start of the first line (and the last line -> bottom of window)
    
    @abstractmethod
    def render(self): ...
    
    @property
    @abstractmethod
    def height(self) -> float: ...
        
        
class Window(UIElement):
    def __init__(self, title):
        self.title = title
        self.open = True
        self.elements: Dict[str, UIElement] = {}
        
    @property
    def height(self):
        if self.open:
            return self.HEADER_SIZE + 2*self.PADDING + sum([el.height for el in self.elements.values()])
        else:
            return self.HEADER_SIZE
    
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
            return header_size + self.n_lines*self.LINE_SIZE
        else:
            return header_size
        
        
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
        return self.n_lines*self.LINE_SIZE
        
    def render(self, lines):
        # if len(lines) != self.n_lines:
        #     raise ValueError(f"Tried to render {len(lines)} lines, but expected {self.n_lines}")
        self.n_lines = len(lines)
        for line in lines:
            imgui.text(line)
            
class Checkbox(UIElement):
    @property
    def height(self):
        return self.SLIDER_SIZE
    
    def render(self, label, value, text=None):
        if text is not None:
            imgui.text(text)
            imgui.same_line()
        return imgui.checkbox(label, value)
    
    
class Slider(UIElement):
    @property
    def height(self):
        return self.SLIDER_SIZE
    
    def render(self, label, value, min_value, max_value, flags=0, text=None):
        if text is not None:
            imgui.text(text)
            imgui.same_line()
        return imgui.slider_float(label, value, min_value, max_value, flags=flags)