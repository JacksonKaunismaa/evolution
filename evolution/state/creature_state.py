from typing import Tuple, Union, TYPE_CHECKING
from torch import Tensor
import numpy as np


if TYPE_CHECKING:
    from evolution.core.creatures.creature_param import CreatureParam
    from evolution.core.creatures.creature_array import CreatureArray


class CreatureState:
    """State that tracks the state of a single creature over time. We extract information
    before steps and after steps that change the state of the creature. These
    values get saved to attributes here so that they can be read by the GUI to be displayed to the
    user when a creature has been selected."""
    def __init__(self):
        self._selected_creature = None
        
    def set_creature(self, creature: Union[Tuple[int, int], None]):
        self._selected_creature = creature
        
    def __bool__(self):
        """Returns True if a creature has been selected."""
        return self._selected_creature is not None
    
    def __iadd__(self, other: int) -> 'CreatureState':
        """Increments the selected creature id by the given amount."""
        self._selected_creature += other
        return self
    
    def __add__(self, other: int) -> int:
        """Returns the id of the selected creature plus the given amount."""
        return self._selected_creature + other
    
    def __index__(self) -> int:
        """Returns the id of the selected creature."""
        return self._selected_creature
    
    def extract_pre_eat_state(self, energy: 'CreatureParam'):
        if self:
            self.pre_eat_energy = energy[self._selected_creature]
        
    def extract_post_eat_state(self, pos: Tensor, alive_costs: Tensor, food_grid: Tensor, arr: 'CreatureArray'):
        if self:
            creat_pos = pos[self._selected_creature]
            
            self.post_eat_energy = arr.energies[self._selected_creature]
            self.age = arr.ages[self._selected_creature]
            self.age_mult = arr.age_mults[self._selected_creature]
            self.n_children = arr.n_children[self._selected_creature]
            self.alive_costs = alive_costs[self._selected_creature]
            self.eat_pct = arr.eat_pcts[self._selected_creature]
            self.food_eaten = self.post_eat_energy - self.pre_eat_energy
            self.cell_energy = food_grid[creat_pos[1], creat_pos[0]]
            self.energy = self.post_eat_energy
            
    def extract_pre_rotate_state(self, energy: 'CreatureParam'):
        self.pre_rot_energy = energy[self._selected_creature]
        
    def extract_post_rotate_state(self, logits: Tensor, rotation_matrix: Tensor, energy: 'CreatureParam'):
        if self:
            self.post_rot_energy = energy[self._selected_creature]
            self.rotate_logit = logits[self._selected_creature]
            self.rotate_angle = np.arccos(rotation_matrix[self._selected_creature, 0, 0].item())
            self.rotate_energy = self.post_rot_energy - self.pre_rot_energy
            self.energy = self.post_rot_energy
            
    def extract_move_state(self, logits: Tensor, move: Tensor, move_cost: Tensor, energy: 'CreatureParam'):
        if self:
            self.move_logit = logits[self._selected_creature]
            self.move_amt = move[self._selected_creature]
            self.move_energy = move_cost[self._selected_creature]
            self.energy = energy[self._selected_creature]
            
            
    def extract_attack_state(self, attacks: Tensor, attack_cost: Tensor, healths: 'CreatureParam', energy: 'CreatureParam'):
        if self._selected_creature is not None:
            self.n_attacking = attacks[self._selected_creature, 0]
            self.dmg_taken = attacks[self._selected_creature, 1]
            self.attack_cost = attack_cost[self._selected_creature]
            self.health = healths[self._selected_creature]
            self.energy = energy[self._selected_creature]
        