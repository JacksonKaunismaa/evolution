import glm
import moderngl as mgl
import numpy as np
import moderngl_window as mglw
import numpy as pn


from .. import config
from .. import gworld

class Camera:
    def __init__(self, ctx: mgl.Context, cfg: config.Config, window: mglw.BaseWindow, world: gworld.GWorld):
        self.ctx = ctx
        self.cfg = cfg
        self.ubo = self.ctx.buffer(reserve=2 * 4*4 * 4)  # 2 mat4s, 4x4 floats of 4 bytes
        self.window = window
        self.world = world

        self.reset_camera()

    def reset_camera(self):
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
    
    def process_keyboard(self, direction, delta_time):
        velocity = self.movement_speed * delta_time
        if direction == 'FORWARD':
            self.position += self.up * velocity
        if direction == 'BACKWARD':
            self.position -= self.up * velocity
        if direction == 'LEFT':
            self.position -= self.right * velocity
        if direction == 'RIGHT':
            self.position += self.right * velocity

    def rotate_to(self, dir_vec):
        self.up = glm.vec3(dir_vec, 0.0)
        self.right = glm.normalize(glm.cross(self.front, self.up))

    def drag(self, old_pos: glm.vec2, click_pos: glm.vec2, drag_pos: glm.vec2):
        # compute game delta coordinates between cursor position now and when the click started
        delta = self.pixel_to_game_delta(drag_pos - click_pos)
        self.position.xy = old_pos - delta.xy  # and move the camera by that delta

    def zoom_into_point(self, yoffset, mouse_pos):
        # we'd like the game position of the cursor to be the same before and after the zoom
        # therefore, we first zoom without translating, then, we translate the camera so that
        # the cursor stays in the same place
        old_cursor_pos = self.pixel_to_game_coords(*mouse_pos)
        # self.position += self.front * yoffset / 10
        self.zoom /= (1+yoffset/5) #* self.scroll_sensitivity
        self.zoom = np.clip(self.zoom, 0.01, 20) # prevent zooming in/out too far
        # print(self.zoom)
        new_cursor_pos = self.pixel_to_game_coords(*mouse_pos)
        self.position.xy -= (new_cursor_pos - old_cursor_pos).xy

    def get_camera_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()
    
    def pixel_to_game_coords(self, screen_x, screen_y):
        # extremely hacky way to go from screen coordinates to game coordinates to determine
        # where a given click has hit the game board

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
        ndc_pos = glm.vec4(ndc_pos, 1.0)
        # print('pre w scale', game_pos)

        ndc_pos.y *= -1   # need to invert this for some reason
        ndc_pos *= save_w
        # print('pre_inverse', game_pos)
        return glm.inverse(self.get_camera_matrix()) * ndc_pos    # in [-1, 1] (food_grid vbo coords)
    
    def pixel_to_game_delta(self, xy: glm.vec2):
        pixel_origin = self.pixel_to_game_coords(0, 0)
        pixel_xy = self.pixel_to_game_coords(xy.x, xy.y)
        return pixel_xy - pixel_origin

    def toggle_follow(self):
        self.following = not self.following
    

    def update(self, creature_id):
        if creature_id is None:
            return
        if self.following:
            self.position.xy = glm.vec2(self.world.creatures.positions[creature_id].cpu().numpy())
        # self.rotate_to(glm.vec2(self.world.creatures.head_dirs[creature_id].cpu().numpy()))