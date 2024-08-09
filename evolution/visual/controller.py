import glm
from sdl2 import timer
from typing import TYPE_CHECKING
import moderngl_window as mglw

from .camera import Camera

if TYPE_CHECKING:
    from .main import Game

class Controller:
    def __init__(self, window: mglw.BaseWindow, camera: Camera, game: 'Game'):
        # add all attrs of controller that end in _func to window
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

    def mouse_scroll_event_func(self, xoffset, yoffset):
        self.camera.process_mouse_scroll(yoffset)
   
    def mouse_press_event_func(self, x, y, button):
        # need to store everything in pixel coords so that we are closest to the actual input 
        # this avoids jittering and weird camera behavior when dragging the map
        self.click_pos = glm.vec2(x, y)
        self.camera_pos = self.camera.position.xy
        # print('click pos:', self.click_pos, self.camera.pixel_to_game_coords(x, y))
        # print('position', self.game.world.creatures.positions)
        # print('sizes:', self.game.world.creatures.sizes)

    def mouse_drag_event_func(self, x, y, dx, dy):
        new_coords = glm.vec2(x, y)
        delta = self.camera.pixel_to_game_delta(new_coords - self.click_pos)  # but it should be this, so compute a delta

        # print('camera delta:', delta, game_oords - self.click_pos)

        self.camera.position.xy = self.camera_pos - delta.xy  # and move the camera by that delta
        # print('new camera pos:', self.camera.position)
        # print('camera game coord', self.camera_pos - delta)

    def mouse_move_event_func(self, x, y, dx, dy):
        self.camera.process_mouse_movement(dx, dy)

    def tick(self):
        curr_time = timer.SDL_GetTicks()
        self.delta_time = (curr_time - self.now) / 1000.0
        self.now = curr_time