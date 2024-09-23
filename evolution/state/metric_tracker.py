from typing import Callable, TYPE_CHECKING, Tuple
import numpy as np

from evolution.utils.subscribe import Subscriber

if TYPE_CHECKING:
    from  .game_state import GameState


class MetricTracker(Subscriber):
    def __init__(self, state: 'GameState', func: Callable, poll_rate: int, **kwargs):
        super().__init__(poll_rate)
        state.game_step_publisher.subscribe(self)

        self._updated = False
        self.state = state
        self.func = func
        self.length = 0
        self.add_amt = 1000
        self.min_val = None
        self.max_val = None
        self.kwargs = kwargs

        self._times = np.array([])
        self._values = np.array([])
        self._shape: Tuple[int] = None

    @property
    def times(self):
        return self._times[:self.length]

    @property
    def values(self):
        return self._values[..., :self.length]  # make time be the last dimension so that indexing gives a contiguous array

    def _update(self):
        self._updated = True
        next_val = self.func(**self.kwargs)

        if isinstance(next_val, np.ndarray):  # infer shape from first value
            self._shape = next_val.shape
            next_min = np.min(next_val)
            next_max = np.max(next_val)
        else:
            self._shape = tuple()
            next_min = next_max = next_val

        if self.min_val is None or next_min < self.min_val:
            self.min_val = next_min
        if self.max_val is None or next_max > self.max_val:
            self.max_val = next_max

        if self.length >= self._times.shape[0]:
            self._times = np.append(self._times, np.empty(self.add_amt))
            if self._values.ndim == 1 and self._shape:
                self._values = np.empty((*self._shape, self.add_amt))
            else:
                self._values = np.append(self._values, np.empty((*self._shape, self.add_amt)), axis=-1)

        self._times[self.length] = float(self.state.time)
        self._values[..., self.length] = next_val
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
