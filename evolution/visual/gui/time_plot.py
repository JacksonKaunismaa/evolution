from typing import Callable

from evolution.utils.subscribe import Subscriber
from evolution.state.game_state import GameState

class TimePlot(Subscriber):
    def __init__(self, state: GameState, func: Callable, poll_rate: float):
        super().__init__(poll_rate)
        state.game_publisher.subscribe(self)
        
        self.state = state
        self.func = func
        self.values = []
        
    def _update(self):
        self.values.append(float(self.func()))
