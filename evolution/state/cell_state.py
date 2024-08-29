from typing import Tuple, Union
from torch import Tensor


from evolution.core.config import Config

class CellState:
    """State that tracks the state of a single cell in the food grid. We extract information
    before the growing step, as well as after, so that we can pass this information to the GUI
    to be displayed if a cell to be tracked has been selected."""
    def __init__(self, cfg: Config):
        self._selected_cell = None
        self.cfg = cfg
        # bad practice since we are using cfg.food_sight twice to define the padding
        # (both here and in creatures.Creatures), though maybe it's OK since they are constant
        self.pad = cfg.food_sight
        
    def set_cell(self, cell: Union[Tuple[int, int], None]):
        if cell is None:
            self._selected_cell = cell
            return
        self._selected_cell = (int(cell.x), int(cell.y))
            
    def extract_pre_grow_state(self, food_grid: Tensor):
        if self._selected_cell is not None:
            self.prev_food = food_grid[self._selected_cell[1] + self.pad, 
                                       self._selected_cell[0] + self.pad].item()
            
    def extract_post_grow_state(self, pct_eaten: Tensor, food_grid: Tensor, 
                                food_grid_updates: Tensor, step_size: float):            
        if self._selected_cell is not None:
            self.cell_pct_eaten = pct_eaten[self._selected_cell[1] + self.pad, 
                                       self._selected_cell[0] + self.pad].item()
            # this is also bad practice since we are calculating cover_cost twice (inside the grow.cu
            # CUDA kernel and here), but there's no way to avoid it without allocating a ton of memory
            self.cover_cost = self.cfg.food_cover_decr_pct * self.cfg.food_cover_decr * self.cell_pct_eaten
            self.eaten_amt = food_grid_updates[self._selected_cell[1] + self.pad, 
                                          self._selected_cell[0] + self.pad].item() - self.cover_cost
            self.post_growth_food = food_grid[self._selected_cell[1] + self.pad, 
                                         self._selected_cell[0] + self.pad].item()
            self.pre_grow_food = self.prev_food - self.cover_cost - self.eaten_amt
            self.growth_amt = self.post_growth_food - self.pre_grow_food
            self.step_size = step_size
