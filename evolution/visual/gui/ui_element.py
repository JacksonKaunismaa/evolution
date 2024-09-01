import imgui


class UIElement:
    LINE_SIZE = 17  # line size for text in pixels
    HEADER_SIZE = 19  # header size in pixels (not including 2*HEADER_PAD)
    HEADER_PAD = 2  # padding between the header and the text
    SLIDER_SIZE = 23 # slider size in pixels (including 2*SLIDER_PAD = 4)
    PADDING = 6  # padding between the top header and the start of the first line (and the last line -> bottom of window)
    def render(self):
        pass
    
    
class CollapseableHeader(UIElement):
    def __init__(self, title, n_lines):
        self.title = title
        self.n_lines = n_lines
        self.open = False
        
    def render(self, lines):
        self.open = imgui.collapsing_header(self.title)[0]
        if self.open:
            for line in lines:
                imgui.text(line)