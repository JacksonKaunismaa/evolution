import numpy as np


from evolution.core.config import Config

class GameState:
    """Tracks state that is relevant for control of the simulation, especially the visual aspects,
    but only if it needs to be shared between different objects.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        self._selected_cell = None
        self.selected_creature = None
        
        self.increasing_food_decr = False
        self._game_speed = 1
        self.game_paused = False
        self.creatures_visible = True
        self.hitboxes_enabled = False
    
    @property
    def selected_cell(self):
        return self._selected_cell
    
    @selected_cell.setter
    def selected_cell(self, cell):
        if cell is None:
            self._selected_cell = cell
            return
        self._selected_cell = (int(cell.x), int(cell.y))
        
    @property
    def game_speed(self):
        return self._game_speed
    
    @game_speed.setter
    def game_speed(self, speed):
        self._game_speed = np.clip(speed, 1, self.cfg.max_game_speed)
        
    def toggle_pause(self):
        self.game_paused = not self.game_paused
        
    def toggle_increasing_food_decr(self):
        self.increasing_food_decr = not self.increasing_food_decr
        
    def toggle_creatures_visible(self):
        self.creatures_visible = not self.creatures_visible
        
    def toggle_hitboxes(self):
        self.hitboxes_enabled = not self.hitboxes_enabled