class Controller:
    def __init__(self, window) -> None:
        # add all attrs of controller that end in _func to window
        for attr in dir(self):
            if attr.endswith('_func'):
                setattr(window, attr, getattr(self, attr))

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

            if key == self.wnd.keys.Z:
                self.camera.process_mouse_movement(0, 1)

            if key == self.wnd.keys.X:
                self.camera.process_mouse_movement(0, -1)

    def mouse_scroll_event_func(self, xoffset, yoffset):
        self.camera.process_mouse_scroll(yoffset)


    def mouse_move_event_func(self, x, y, dx, dy):
        self.camera.process_mouse_movement(dx, dy)