from abc import ABC, abstractmethod
    
class EventHandler(ABC):    
    @abstractmethod
    def resize_func(self, width, height): ...

    @abstractmethod
    def key_event_func(self, key, action, mods): ...
        
    @abstractmethod
    def mouse_position_event_func(self, x, y, dx, dy): ...
        
    @abstractmethod
    def mouse_drag_event_func(self, x, y, dx, dy): ...
        
    @abstractmethod
    def mouse_scroll_event_func(self, x_offset, y_offset): ...
        
    @abstractmethod
    def mouse_press_event_func(self, x, y, button): ...
        
    @abstractmethod
    def mouse_release_event_func(self, x, y, button): ...
    