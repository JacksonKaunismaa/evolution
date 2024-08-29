from typing import Union, TYPE_CHECKING
from torch import Tensor
import numpy as np


if TYPE_CHECKING:
    from evolution.core.creatures.creature_array import CreatureArray


class CreatureState:
    """State that tracks the state of a single creature over time. We extract information
    before steps and after steps that change the state of the creature. These
    values get saved to attributes here so that they can be read by the GUI to be displayed to the
    user when a creature has been selected."""
    def __init__(self):
        self._selected_creature = None
        
    def set_creatures_reference(self, creatures: 'CreatureArray'):
        """Sets the reference to the creatures array so that we can extract information from it.
        This must be done after __init__ since creating Creatures requires reference to the global
        GameState object (of which this is a part), so we have a circular dependency."""
        self.creatures = creatures
        
    def set_creature(self, creature: Union[Tensor, int, None]):
        if isinstance(creature, Tensor):
            creature = creature.item()
        changed = self._selected_creature != creature
        self._selected_creature = creature
        if self and changed:
            self.extract_general_state()
        
    def __bool__(self):
        """Returns True if a creature has been selected."""
        return self._selected_creature is not None
    
    def __add__(self, other: int) -> int:
        """Returns the id of the selected creature plus the given amount."""
        return self._selected_creature + other
    
    def __index__(self) -> int:
        """Returns the id of the selected creature."""
        return self._selected_creature
    
    def extract_general_state(self):
        """Extract basic attributes for the selected creature."""
        singular_name_exceptions = {'n_children': 'n_children',
                                    'energies': 'energy'}
        for k, v in self.creatures.variables.items():
            
            if k in singular_name_exceptions:
                singular_name = singular_name_exceptions[k]
            else:
                singular_name = k[:-1]  # strip off the 's' at the end of the variable name
                
            if v.dim() == 1:
                setattr(self, singular_name, v[self._selected_creature].item())
    
    def extract_pre_eat_state(self):
        if self:
            self.pre_eat_energy = self.creatures.energies[self._selected_creature]
        
    def extract_post_eat_state(self, pos: Tensor, alive_costs: Tensor, food_grid: Tensor):
        if self:
            creat_pos = pos[self._selected_creature]
            
            self.post_eat_energy = self.creatures.energies[self._selected_creature]
            self.age = self.creatures.ages[self._selected_creature]
            self.age_mult = self.creatures.age_mults[self._selected_creature]
            self.n_children = self.creatures.n_children[self._selected_creature]
            self.alive_cost = alive_costs[self._selected_creature]
            self.eat_pct = self.creatures.eat_pcts[self._selected_creature]
            self.food_eaten = self.post_eat_energy - self.pre_eat_energy
            self.cell_energy = food_grid[creat_pos[1], creat_pos[0]]
            self.energy = self.post_eat_energy
            
    def extract_pre_rotate_state(self):
        self.pre_rot_energy = self.creatures.energies[self._selected_creature]
        
    def extract_post_rotate_state(self, logits: Tensor, rotation_matrix: Tensor):
        if self:
            self.post_rot_energy = self.creatures.energies[self._selected_creature]
            self.rotate_logit = logits[self._selected_creature]
            self.rotate_angle = np.arccos(rotation_matrix[self._selected_creature, 0, 0].item())
            self.rotate_energy = self.post_rot_energy - self.pre_rot_energy
            self.energy = self.post_rot_energy
            
    def extract_move_state(self, logits: Tensor, move: Tensor, move_cost: Tensor):
        if self:
            self.move_logit = logits[self._selected_creature]
            self.move_amt = move[self._selected_creature]
            self.move_energy = move_cost[self._selected_creature]
            self.energy = self.creatures.energies[self._selected_creature]
            
            
    def extract_attack_state(self, attacks: Tensor, attack_cost: Tensor):
        if self._selected_creature is not None:
            self.n_attacking = attacks[self._selected_creature, 0]
            self.dmg_taken = attacks[self._selected_creature, 1]
            self.attack_cost = attack_cost[self._selected_creature]
            self.health = self.creatures.healths[self._selected_creature]
            self.energy = self.creatures.energies[self._selected_creature]
        