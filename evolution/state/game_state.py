from typing import Tuple, Union
import numpy as np


from .creature_state import CreatureState
from .cell_state import CellState

from evolution.core.config import Config

class GameState:
    """Tracks state that is relevant for control of the simulation, especially the visual aspects,
    and if it needs to be shared between different objects (say the GUI and creatures.Creatures).
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        self._selected_cell = CellState(cfg)  
        self._selected_creature = CreatureState()
        
        self.increasing_food_decr = False
        self._game_speed = 1
        self.game_paused = False
        self.creatures_visible = True
        self.hitboxes_enabled = False
        
    @property
    def selected_creature(self) -> CreatureState:
        return self._selected_creature
    
    @selected_creature.setter
    def selected_creature(self, creature: Union[int, None]):
        self._selected_creature.set_creature(creature)
    
    @property
    def selected_cell(self) -> CellState:
        return self._selected_cell
    
    @selected_cell.setter
    def selected_cell(self, cell: Union[Tuple[int, int], None]):
        self._selected_cell.set_cell(cell)
        
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