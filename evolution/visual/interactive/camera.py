from typing import TYPE_CHECKING
import glm
from moderngl import Context
import numpy as np
from moderngl_window import BaseWindow

from evolution.core.config import Config

if TYPE_CHECKING:
    from evolution.core.gworld import GWorld

class Camera:
    """Camera class for handling camera movement and transformations."""
    def __init__(self, cfg: Config, ctx: Context, window: BaseWindow, world: 'GWorld'):
        self.ctx = ctx
        self.cfg = cfg
        self.ubo = self.ctx.buffer(reserve=2 * 4*4 * 4)  # 2 mat4s, 4x4 floats of 4 bytes
        self.window = window
        self.world = world
        self.state = self.world.state

        self.reset_camera()

    def reset_camera(self):
        """Reset the camera to its initial position, orientation, zoom, etc."""
        middle = self.cfg.size / 2
        self.position = glm.vec3(middle, middle, 5.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)   # points opposite of where camera is 'looking'
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.movement_speed = 0.5
        self.mouse_sensitivity = 0.01
        self.fov = 140.0
        self.zoom = 1.0
        self.following = False

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        return glm.scale(glm.perspective(glm.radians(self.fov),
                               self.window.size[0] / self.window.size[1],
                               0.1, 100.0),
                            glm.vec3(1./self.zoom, 1./self.zoom, 1.0))

    def rotate_to(self, dir_vec):
        """Rotate the camera so that its 'up' vector points towards `dir_vec`."""
        self.up = glm.vec3(dir_vec, 0.0)  # type: ignore
        self.right = glm.normalize(glm.cross(self.front, self.up))

    def drag(self, old_pos: glm.vec2, click_pos: glm.vec2, drag_pos: glm.vec2):
        """Compute game delta coordinates between cursor position now and when the
        click started and move the camera by that delta."""
        delta = self.pixel_to_game_delta(drag_pos - click_pos)
        if not self.following:
            self.position.xy = old_pos - delta.xy  # and move the camera by that delta

    def zoom_into_point(self, yoffset, mouse_pos):
        """Zoom in/out of the game board, and translate the camera so that the
        cursor remains in the same position in game coordinates."""
        # we'd like the game position of the cursor to be the same before and after the zoom
        # therefore, we first zoom without translating, then, we translate the camera so that
        # the cursor stays in the same place
        old_cursor_pos = self.pixel_to_game_coords(*mouse_pos)
        # self.position += self.front * yoffset / 10
        self.zoom /= (1+yoffset/5) #* self.scroll_sensitivity
        self.zoom = np.clip(self.zoom, 0.001, 200) # prevent zooming in/out too far
        # print(self.zoom)
        new_cursor_pos = self.pixel_to_game_coords(*mouse_pos)
        if not self.following:
            self.position.xy -= (new_cursor_pos - old_cursor_pos).xy

    def get_camera_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()

    def click_in_bounds(self, game_click) -> bool:
        """Args: game_click: glm.vec2, position of a given click in game coords
        Returns: bool, whether the click is within the bounds of the game"""
        return 0 <= game_click.x < self.cfg.size and 0 <= game_click.y < self.cfg.size

    def pixel_to_game_coords(self, screen_x, screen_y):
        """Extremely hacky way to go from screen coordinates to game coordinates
        to determine where a given click has hit the game board"""

        test_pt = glm.vec4(0, 0, 0.0, 1.0)
        clip_coords = self.get_camera_matrix() * test_pt
        save_w = clip_coords.w
        clip_coords /= save_w
        ndc_pos = glm.unProject(glm.vec3(screen_x, screen_y, 1),
                                 glm.mat4(1.0),
                                 glm.mat4(1.0),
                                 self.window.viewport)
        # print('ndc', game_pos)
        ndc_pos.z = clip_coords.z
        ndc_pos = glm.vec4(ndc_pos, 1.0)  # type: ignore
        # print('pre w scale', game_pos)

        ndc_pos.y *= -1   # need to invert this for some reason
        ndc_pos *= save_w
        # print('pre_inverse', game_pos)
        return glm.inverse(self.get_camera_matrix()) * ndc_pos    # in [-1, 1] (food_grid vbo coords)

    def pixel_to_game_delta(self, xy: glm.vec2):
        """Even hackier way to go from screen coordinates to game coordinates
        of a given delta. This function is necessary because pixel_to_game_coords
        assumes a specific origin, which is not the case for deltas."""
        pixel_origin = self.pixel_to_game_coords(0, 0)
        pixel_xy = self.pixel_to_game_coords(xy.x, xy.y)
        return pixel_xy - pixel_origin

    def toggle_follow(self):
        self.following = not self.following

    def update(self):
        self.ubo.write(self.get_camera_matrix())
        self.ubo.bind_to_uniform_block()

        selected_creature = self.world.state.selected_creature
        if not selected_creature:
            self.following = False
            return
        if self.following:
            self.position.xy = glm.vec2(selected_creature.position)
        # self.rotate_to(glm.vec2(self.world.creatures.head_dirs[creature_id].cpu().numpy()))

    def state_dict(self):
        return {
            'position': self.position,
            'front': self.front,
            'up': self.up,
            'right': self.right,
            'movement_speed': self.movement_speed,
            'mouse_sensitivity': self.mouse_sensitivity,
            'fov': self.fov,
            'zoom': self.zoom,
            'following': self.following
        }

    def load_state_dict(self, state):
        self.position = state['position']
        self.front = state['front']
        self.up = state['up']
        self.right = state['right']
        self.movement_speed = state['movement_speed']
        self.mouse_sensitivity = state['mouse_sensitivity']
        self.fov = state['fov']
        self.zoom = state['zoom']
        self.following = state['following']
