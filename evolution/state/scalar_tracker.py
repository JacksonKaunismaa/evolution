from typing import Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from  .game_state import GameState

from evolution.utils.subscribe import Subscriber

class ScalarTracker(Subscriber):
    def __init__(self, state: 'GameState', func: Callable, poll_rate: float):
        super().__init__(poll_rate)
        state.game_step_publisher.subscribe(self)
        
        self._updated = False
        self.state = state
        self.func = func
        self._times = np.array([])
        self._values = np.array([])
        self.length = 0
        self.add_amt = 10_000
        self.min_val = None
        self.max_val = None
        
        
    @property
    def times(self):
        return self._times[:self.length]
    
    @property
    def values(self):
        return self._values[:self.length]
    
    def _update(self):
        self._updated = True
        next_val = float(self.func())
        
        if self.min_val is None or next_val < self.min_val:
            self.min_val = next_val
        if self.max_val is None or next_val > self.max_val:
            self.max_val = next_val

        if self.length >= self._times.shape[0]:
            self._times = np.append(self._times, np.empty(self.add_amt))
            self._values = np.append(self._values, np.empty(self.add_amt))

        self._times[self.length] = float(self.state.time)
        self._values[self.length] = next_val
        self.length += 1
         
    def has_updates(self) -> bool:
        """Should only be called once frame, when plotting is about to begin. Returns True if there
        were any updates since the last time this function was called, and False otherwise."""
        retval = self._updated
        self._updated = False
        return retval
    
    def state_dict(self):
        return {
            'times': self.times.copy(),  # .copy() to ensure that we don't store empty values to disk
            'values': self.values.copy(),
            'min_val': self.min_val,
            'max_val': self.max_val,
            'length': self.length
        }
        
    def load_state_dict(self, state_dict):
        self._times = state_dict['times']
        self._values = state_dict['values']
        self.min_val = state_dict['min_val']
        self.max_val = state_dict['max_val']
        self.length = state_dict['length']
        
        
       
        
