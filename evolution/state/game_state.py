from typing import Dict, Tuple, Union, TYPE_CHECKING
import numpy as np

from evolution.utils.subscribe import Publisher
from evolution.core.config import Config

from .creature_state import CreatureState
from .cell_state import CellState
from .scalar_tracker import ScalarTracker

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

        self.trackers: Dict[str, ScalarTracker] = {}
        self.energy_tracker = ScalarTracker(self, world.log_total_energy, 25)

    def __setattr__(self, name, value):
        if isinstance(value, ScalarTracker):
            self.trackers[name] = value
        super().__setattr__(name, value)

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
        """Publishes all the state changes for each publisher. For game_step_publisher, also
        updates all the subscribers.
        """
        self.game_publisher.publish()
        self.creature_publisher.publish(self.selected_creature)

        self.game_step_publisher.publish()
        self.game_step_publisher.update_all()

    def init_publish(self):
        """Send an initial publish to all subscribers so that they start out with valid data."""
        self.game_publisher.init_publish()
        self.game_step_publisher.init_publish()
        self.game_step_publisher.update_all()
        if self.selected_creature:
            self.creature_publisher.init_publish(self.selected_creature)

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

    def state_dict(self):
        return {
            'selected_creature': self.selected_creature._selected_creature,
            'selected_cell': self.selected_cell._selected_cell,
            'game_speed': self.game_speed,
            # 'game_paused': self.game_paused,  # we'd rather always start the game paused
            'creatures_visible': self.creatures_visible,
            'hitboxes_enabled': self.hitboxes_enabled,
            'time': self.time,
            'trackers': {k: tracker.state_dict() for k, tracker in self.trackers.items()}
        }

    def load_state_dict(self, state_dict):
        self.selected_creature = state_dict['selected_creature']
        self.selected_cell = state_dict['selected_cell']
        self.game_speed = state_dict['game_speed']
        self.creatures_visible = state_dict['creatures_visible']
        self.hitboxes_enabled = state_dict['hitboxes_enabled']
        self.time = state_dict['time']

        for k, tracker_state in state_dict['trackers'].items():
            self.trackers[k].load_state_dict(tracker_state)