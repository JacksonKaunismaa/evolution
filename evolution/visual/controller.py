import glm
from sdl2 import timer
from typing import TYPE_CHECKING
import moderngl_window as mglw

from .camera import Camera

if TYPE_CHECKING:
    from .main import Game

class Controller:
    def __init__(self, window: mglw.BaseWindow, camera: Camera, game: 'Game'):
        # add all attrs of controller that end in _func to window, since these are associated with events
        for attr in dir(self):
            if attr.endswith('_func'):
                setattr(window, attr, getattr(self, attr))

        self.wnd = window
        self.now = timer.SDL_GetTicks()
        self.delta_time = 0
        self.camera = camera
        self.game = game
        self.click_pos = None
        self.mouse_pressed = False
        self.mouse_pos = None

    def key_event_func(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.W:
                self.camera.process_keyboard('FORWARD', self.delta_time)

            if key == self.wnd.keys.S:
                self.camera.process_keyboard('BACKWARD', self.delta_time)

            if key == self.wnd.keys.A:
                self.camera.process_keyboard('LEFT', self.delta_time)

            if key == self.wnd.keys.D:
                self.camera.process_keyboard('RIGHT', self.delta_time)

            if key == self.wnd.keys.R:
                self.camera.reset_camera()

            if key == self.wnd.keys.SPACE:
                self.game.toggle_pause()

            if key == self.wnd.keys.RIGHT:
                self.game.step(n=1, force=True)

            if key == self.wnd.keys.B:
                self.game.creatures.toggle_hitboxes()

    def mouse_scroll_event_func(self, xoffset, yoffset):
        self.camera.zoom_into_point(yoffset, self.mouse_pos)
        
    def mouse_press_event_func(self, x, y, button):
        # need to store everything in pixel coords so that we are closest to the actual input 
        # this avoids jittering and weird camera behavior when dragging the map
        if button == self.wnd.mouse.left:
            self.click_pos = glm.vec2(x, y)
            self.camera_pos = self.camera.position.xy
            self.mouse_pressed = True
            game_click = self.camera.pixel_to_game_coords(*self.click_pos).xy
            creature = self.game.world.click_creature(game_click)
            print(game_click, self.game.world.creatures.positions, self.game.world.creatures.rays, creature)
            if creature is not None:
                self.game.select_creature(creature)

    def mouse_release_event_func(self, x, y, button):
        self.mouse_pressed = False

    def mouse_drag_event_func(self, x, y, dx, dy):
        if not self.mouse_pressed:
            return
        self.camera.drag(self.camera_pos, self.click_pos, glm.vec2(x, y))

    def mouse_position_event_func(self, x, y, dx, dy):
        self.mouse_pos = glm.vec2(x, y)

    def tick(self):
        curr_time = timer.SDL_GetTicks()
        self.delta_time = (curr_time - self.now) / 1000.0
        self.now = curr_time