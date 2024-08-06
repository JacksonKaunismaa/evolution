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

    def mouse_press_event_func(self, x, y, button):
        print(button, x, y, self.camera.game_coordinates(x, y), self.wnd.size, self.game.ctx.viewport, self.wnd.viewport, self.wnd.position)
        print('positions:', self.game.world.creatures.positions)
        # self.click_pos = (x, y)

    def mouse_scroll_event_func(self, xoffset, yoffset):
        self.camera.process_mouse_scroll(yoffset)

    def mouse_drag_event_func(self, x, y, dx, dy):
        self.camera.process_mouse_movement(dx, dy)

    def mouse_move_event_func(self, x, y, dx, dy):
        self.camera.process_mouse_movement(dx, dy)

    def tick(self):
        curr_time = timer.SDL_GetTicks()
        self.delta_time = (curr_time - self.now) / 1000.0
        self.now = curr_time