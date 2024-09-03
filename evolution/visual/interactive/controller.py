import glm
# from sdl2 import timer
from typing import TYPE_CHECKING
import moderngl_window as mglw

from .camera import Camera

from evolution.state.game_state import GameState
from evolution.core.gworld import GWorld

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

class Controller:
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
        
        self.click_pos = None
        self.mouse_pressed = False
        self.mouse_pos = None
        
    def register_keys(self):
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
            self.world.write_checkpoint('game.ckpt'))  # save current state to 'game.ckpt'
        
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
        
        def speed_adjustment(modifiers):
            amt = 1
            if modifiers.shift:
                amt *= 10
            if modifiers.ctrl:
                amt *= 10
            return amt
        
        self.key_mgr.register_key(self.wnd.keys.UP, lambda modifiers:  # speed up simulation so that we do 1 more step per frame
            self.state.game_speed.__iadd__(speed_adjustment(modifiers)))
        
        self.key_mgr.register_key(self.wnd.keys.DOWN, lambda modifiers:  # slow down simulation so that we do 1 less step per frame
            self.state.game_speed.__isub__(speed_adjustment(modifiers)))
        
        self.key_mgr.register_key(self.wnd.keys.B, lambda m:
            self.state.toggle_hitboxes())  # turn hitboxes on/off
        
    def key_event_func(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            self.key_mgr.call(key, modifiers)
            
    def set_selected_creature(self, creature):
        self.camera.following = True
        self.state.selected_creature = creature
        if creature is not None:   # if setting creature, set cell to None
            self.set_selected_cell(None)
        
    def set_selected_cell(self, xy):
        self.state.selected_cell = xy
        if xy is not None:   # if setting cell None, set creature to None
            self.set_selected_creature(None)
        
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
            
    def mouse_scroll_event_func(self, xoffset, yoffset):
        self.camera.zoom_into_point(yoffset, self.mouse_pos)

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