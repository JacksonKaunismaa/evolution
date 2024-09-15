from typing import Tuple, Union, TYPE_CHECKING
import numpy as np


from .creature_state import CreatureState
from .cell_state import CellState

from evolution.utils.subscribe import Publisher
from evolution.core.config import Config

if TYPE_CHECKING:
    from evolution.core.gworld import GWorld

class GameState:
    """Tracks state that is relevant for control of the simulation, especially the visual aspects,
    and if it needs to be shared between different objects (say the GUI and creatures.Creatures).
    """
    def __init__(self, cfg: Config, world: 'GWorld'):
        self.cfg = cfg
        
        self._selected_cell = CellState(cfg, world)  
        self._selected_creature = CreatureState(world)
        
        self.creature_publisher = Publisher()   # updates when selected_creature changes
        self.game_publisher = Publisher()   # updates when game state changes (call update yourself)
        self.game_step_publisher = Publisher()   # publish and updates will be called for you every game step
        self.increasing_food_decr = False
        self._game_speed = 1
        self.game_paused = False
        self.creatures_visible = True
        self.hitboxes_enabled = False
        self.time = 0
        
    @property
    def selected_creature(self) -> CreatureState:
        return self._selected_creature
    
    @selected_creature.setter
    def selected_creature(self, creature: Union[int, None]):
        self._selected_creature.set_creature(creature)
        self.creature_publisher.publish(self.selected_creature)
    
    @property
    def selected_cell(self) -> CellState:
        return self._selected_cell
    
    @selected_cell.setter
    def selected_cell(self, cell: Union[Tuple[int, int], None]):
        self._selected_cell.set_cell(cell)
        
    def publish_all(self):
        self.game_publisher.publish()
        self.creature_publisher.publish(self.selected_creature)
        
        self.game_step_publisher.publish()
        self.game_step_publisher.update_all()
        
    def init_publish(self):
        self.game_publisher.init_publish()
        self.game_step_publisher.init_publish()
        self.game_step_publisher.update_all()
        
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