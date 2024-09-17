from typing import TYPE_CHECKING
import glm
# from sdl2 import timer
import moderngl_window as mglw

from evolution.state.game_state import GameState
from evolution.core.gworld import GWorld
from evolution.utils.event_handler import EventHandler

from .camera import Camera


if TYPE_CHECKING:
    from evolution.visual.main import Game


class KeyManager:
    def __init__(self):
        self.keys = {}

    def register_key(self, key, func):
        if key in self.keys:
            raise ValueError(f'Key {key} already registered')
        self.keys[key] = func

    def call(self, key, modifiers):
        if key in self.keys:
            self.keys[key](modifiers)

class Controller(EventHandler):
    def __init__(self, world: GWorld, window: mglw.BaseWindow, camera: Camera, state: GameState):
        # add all attrs of controller that end in _func to window, since these are associated with events

        self.wnd = window
        self.state = state
        self.world = world
        # self.now = timer.SDL_GetTicks()
        self.delta_time = 0
        self.camera = camera
        self.key_mgr = KeyManager()
        self.register_keys()

        self.click_pos: glm.vec2 = None
        self.mouse_pressed: bool = False
        self.mouse_pos: glm.vec2 = None

    def register_keys(self):
        """Set up key bindings for the game."""
        self.key_mgr.register_key(self.wnd.keys.H, lambda m: (  # toggle creature appearances
            self.state.toggle_creatures_visible(),
            self.set_selected_creature(None)))

        self.key_mgr.register_key(self.wnd.keys.D, lambda m:
            self.state.toggle_increasing_food_decr())  # toggle extra food decay

        self.key_mgr.register_key(self.wnd.keys.R, lambda m: (  # reset camera to starting position, deselect any creature
            self.set_selected_creature(None),
            self.camera.reset_camera()))

        self.key_mgr.register_key(self.wnd.keys.Y, lambda m:
            self.camera.toggle_follow())  # toggle whether we are locked on to the selected creature

        self.key_mgr.register_key(self.wnd.keys.U, lambda m:
            self.set_selected_creature(None))  # deselect creature

        self.key_mgr.register_key(self.wnd.keys.SPACE, lambda m:
            self.state.toggle_pause())  # pause simulation

        self.key_mgr.register_key(self.wnd.keys.RIGHT, lambda m:
            self.world.step())   # force step simulation by 1

        self.key_mgr.register_key(self.wnd.keys.C, lambda m:
            self.world.write_checkpoint('game.ckpt', camera=self.camera))  # save current state to 'game.ckpt'

        self.key_mgr.register_key(self.wnd.keys.NUMBER_0, lambda m:
            self.set_selected_creature(0))  # select creature 0

        self.key_mgr.register_key(self.wnd.keys.P, lambda m:  # increase id of followed creature by 1
            self.set_selected_creature(self.state.selected_creature + 1))

        self.key_mgr.register_key(self.wnd.keys.G, lambda m:  # jump to Genghis Khan (guy with most offspring)
            self.set_selected_creature(self.world.creatures.n_children.argmax()))

        self.key_mgr.register_key(self.wnd.keys.M, lambda m:  # jump to Methuselah (oldest creature)
            self.set_selected_creature(self.world.creatures.ages.argmax()))

        self.key_mgr.register_key(self.wnd.keys.O, lambda m:  # jump to oldest creature
            self.set_selected_creature(self.world.creatures.ages.argmax()))

        self.key_mgr.register_key(self.wnd.keys.L, lambda m:  # jump to largest creature
            self.set_selected_creature(self.world.creatures.sizes.argmax()))

        self.key_mgr.register_key(self.wnd.keys.T, lambda m:  # jump to tiniest creature
            self.set_selected_creature(self.world.creatures.sizes.argmin()))

        def speed_adjustment(modifiers, add=True):
            amt = 1
            if modifiers.shift:
                amt *= 10
            if modifiers.ctrl:
                amt *= 10
            if add:
                self.state.game_speed += amt
            else:
                self.state.game_speed -= amt

        self.key_mgr.register_key(self.wnd.keys.UP, lambda modifiers:  # speed up simulation so that we do 1 more step per frame
            speed_adjustment(modifiers, add=True))

        self.key_mgr.register_key(self.wnd.keys.DOWN, lambda modifiers:  # slow down simulation so that we do 1 less step per frame
            speed_adjustment(modifiers, add=False))

        self.key_mgr.register_key(self.wnd.keys.B, lambda m:
            self.state.toggle_hitboxes())  # turn hitboxes on/off

    def set_selected_creature(self, creature):
        """Set the selected creature to the given creature id, set the camera
        to track it, and deselect the selected cell."""
        self.camera.following = True
        self.state.selected_creature = creature
        if creature is not None:   # if setting creature, set cell to None
            self.set_selected_cell(None)

    def set_selected_cell(self, xy):
        """Set the selected cell to the given cell, and deselect the selected creature."""
        self.state.selected_cell = xy
        if xy is not None:   # if setting cell None, set creature to None
            self.set_selected_creature(None)

    def resize_func(self, width, height): ...

    def key_event_func(self, key, action, mods):
        if action == self.wnd.keys.ACTION_PRESS:
            self.key_mgr.call(key, mods)

    def mouse_press_event_func(self, x, y, button):
        # need to store everything in pixel coords so that we are closest to the actual input
        # this avoids jittering and weird camera behavior when dragging the map
        if button == self.wnd.mouse.left:
            self.click_pos = glm.vec2(x, y)
            self.camera_pos = self.camera.position.xy
            self.mouse_pressed = True
            game_click = self.camera.pixel_to_game_coords(*self.click_pos).xy
            creature_id = self.world.click_creature(game_click) if self.state.creatures_visible else None
            if creature_id is not None:
                self.set_selected_creature(creature_id)
            elif self.camera.click_in_bounds(game_click) and not self.state.selected_creature:
                self.set_selected_cell(game_click)
            else:
                self.set_selected_cell(None)

    def mouse_scroll_event_func(self, x_offset, y_offset):
        self.camera.zoom_into_point(y_offset, self.mouse_pos)

    def mouse_release_event_func(self, x, y, button):
        self.mouse_pressed = False

    def mouse_drag_event_func(self, x, y, dx, dy):
        if not self.mouse_pressed:
            return
        self.camera.drag(self.camera_pos, self.click_pos, glm.vec2(x, y))

    def mouse_position_event_func(self, x, y, dx, dy):
        self.mouse_pos = glm.vec2(x, y)

    # def tick(self):
    #     curr_time = timer.SDL_GetTicks()
    #     self.delta_time = (curr_time - self.now) / 1000.0
    #     self.now = curr_time
