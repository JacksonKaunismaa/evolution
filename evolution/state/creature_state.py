from typing import TYPE_CHECKING
from torch import Tensor
from torch.types import Number
import numpy as np

if TYPE_CHECKING:
    from evolution.core.gworld import GWorld


class CreatureState:
    """State that tracks the state of a single creature over time. We extract information
    before steps and after steps that change the state of the creature. These
    values get saved to attributes here so that they can be read by the GUI to be displayed to the
    user when a creature has been selected."""
    def __init__(self, world: 'GWorld'):
        self._selected_creature: int | None = None
        self.world = world
        self.update_state_available = False  # tracks whether info like amount of food eaten in last step is available

    def set_creature(self, creature: Tensor | Number | int | None):
        """Sets the selected creature to the given creature id, copying all attributes that have dimensionality
        less than or equal to 2 (so we skip copying weights, rays, and biases, for example)."""
        if isinstance(creature, Tensor):
            creature = creature.item()
        changed = self._selected_creature != creature
        self._selected_creature = creature
        if self and changed:  # technically it might be possible that someone selects a new creature on the same frame that the old creature
            self.extract_general_state()  # dies, and it happens to have the same id after the reshuffling, in which case this wouldn't trigger

    def __bool__(self):
        """Returns True if a creature has been selected."""
        return self._selected_creature is not None

    def __add__(self, other: int) -> int:
        """Returns the id of the selected creature plus the given amount."""
        if self._selected_creature is not None:
            return self._selected_creature + other
        raise ValueError("No creature selected.")

    def __index__(self) -> int:
        """Returns the id of the selected creature."""
        if self._selected_creature is not None:
            return self._selected_creature
        raise ValueError("No creature selected.")

    def extract_general_state(self):
        """Extract basic attributes for the selected creature."""
        if self._selected_creature is None:
            raise ValueError("No creature selected.")
        singular_name_exceptions = {'n_children': 'n_children',
                                    'energies': 'energy',
                                    'reproduce_energies': 'reproduce_energy'}
        for k, v in self.world.creatures.traits.variables.items():
            singular_name = singular_name_exceptions.get(k, k[:-1])
            if v.dim() <= 2:
                setattr(self, singular_name, v[self._selected_creature].cpu().numpy())

        self.max_health = self.world.cfg.init_health(self.size)
        self.set_age_stage()
        self.update_state_available = False

    def set_age_stage(self):
        """Sets the `age_stage` of the creature based on its age."""
        if not hasattr(self, 'size'):
            return
        mature_age = self.world.cfg.age_mature_mul * self.age_speed
        old_age = self.world.cfg.age_old_mul * self.age_speed

        if self.age < mature_age:
            self.age_stage = 'adolescent'
        elif self.age < old_age:
            self.age_stage = 'mature'
        else:
            self.age_stage = 'old'

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        self._age = age
        self.set_age_stage()

    def extract_pre_rotate_state(self):
        if self._selected_creature is not None:
            self.pre_rot_energy = self.world.creatures.energies[self._selected_creature].item()

    def extract_post_rotate_state(self, logits: Tensor, rotation_matrix: Tensor):
        if self._selected_creature is not None:
            self.post_rot_energy = self.world.creatures.energies[self._selected_creature].item()
            self.rotate_logit = logits[self._selected_creature].item()
            self.rotate_angle = np.arcsin(rotation_matrix[self._selected_creature, 1, 0].item()) * 180.0 / np.pi
            self.rotate_energy = self.post_rot_energy - self.pre_rot_energy
            self.energy = self.post_rot_energy

    def extract_move_state(self, logits: Tensor, move: Tensor, move_cost: Tensor):
        if self._selected_creature is not None:
            self.move_logit = logits[self._selected_creature].item()
            self.move_amt = move[self._selected_creature].item()
            self.move_energy = -move_cost[self._selected_creature].item()
            self.energy = self.world.creatures.energies[self._selected_creature].item()
            self.position = self.world.creatures.positions[self._selected_creature].cpu().numpy()

    def extract_attack_state(self, attacks: Tensor, attack_cost: Tensor):
        if self._selected_creature is not None:
            self.n_attacking = attacks[self._selected_creature, 0].item()
            self.dmg_taken = attacks[self._selected_creature, 1].item()
            self.attack_cost = -attack_cost[self._selected_creature].item()
            self.health = self.world.creatures.healths[self._selected_creature].item()
            self.energy = self.world.creatures.energies[self._selected_creature].item()

    def extract_pre_eat_state(self):
        if self._selected_creature is not None:
            self.pre_eat_energy = self.world.creatures.energies[self._selected_creature].item()
            self.pre_eat_health = self.world.creatures.healths[self._selected_creature].item()

    def extract_post_eat_state(self, pos: Tensor, alive_costs: Tensor):
        if self._selected_creature is not None:
            creat_pos = pos[self._selected_creature]

            self.post_eat_energy = self.world.creatures.energies[self._selected_creature].item()
            self.post_eat_health = self.world.creatures.healths[self._selected_creature].item()
            self.age = self.world.creatures.ages[self._selected_creature].item()
            self.age_mult = self.world.creatures.age_mults[self._selected_creature].item()
            self.n_children = self.world.creatures.n_children[self._selected_creature].item()
            self.alive_cost = -alive_costs[self._selected_creature].item()
            self.eat_pct = self.world.creatures.eat_pcts[self._selected_creature].item()
            self.food_eaten = self.post_eat_energy - self.pre_eat_energy - self.alive_cost
            self.food_dmg_taken = self.pre_eat_health - self.post_eat_health
            self.cell_energy = self.world.food_grid[creat_pos[1], creat_pos[0]].item()
            self.energy = self.post_eat_energy

            self.net_energy_delta = self.energy - self.pre_rot_energy
            self.update_state_available = True
