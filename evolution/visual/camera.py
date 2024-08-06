import glm
import moderngl as mgl
import numpy as np
import moderngl_window as mglw
import numpy as pn


from .. import config

class Camera:
    def __init__(self, ctx: mgl.Context, cfg: config.Config, window: mglw.BaseWindow):
        self.ctx = ctx
        self.cfg = cfg
        self.ubo = self.ctx.buffer(reserve=2 * 4*4 * 4)  # 2 mat4s, 4x4 floats of 4 bytes
        self.window = window

        self.reset_camera()

    def reset_camera(self):
        self.position = glm.vec3(0.0, 0.0, 1.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)   # points opposite of where camera is 'looking'
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.movement_speed = 0.5
        self.mouse_sensitivity = 0.01
        self.zoom = 45.0

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(self.zoom), 
                               self.window.size[0] / self.window.size[1], 
                               0.1, 100.0)
    
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

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity

        self.position += self.right * xoffset
        self.position += self.up * yoffset

    def process_mouse_scroll(self, yoffset):
        # if self.zoom < 1.0:
        #     self.zoom -= yoffset
        # if self.zoom > 45.0:
        #     self.zoom = 45.0
        self.position += self.front * yoffset / 10
        if self.position.z < 0.1:
            self.position.z = 0.1


    def get_camera_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()
    
    def game_coordinates(self, screen_x, screen_y):
        # extremely hacky way to go from screen coordinates to game coordinates to determine
        # where a given click has hit the game board

        # print(self.position)
        # z_pos = self.position.z
        # top_left = glm.vec4(-1.0, -1.0, 0.0, 1.0)
        # top_right = glm.vec4(1.0, -1.0, 0.0, 1.0)
        # bot_left = glm.vec4(-1.0, 1.0, 0.0, 1.0)
        # bot_right = glm.vec4(0, 0, 0.0, 1.0)
        # for pt in [top_left, top_right, bot_left, bot_right]:
        #     clip_coords = self.get_camera_matrix() * pt
        #     save_w = clip_coords.w
        #     clip_coords /= clip_coords.w
        #     print(clip_coords, save_w)
        # print(self.get_projection_matrix())
        # print("#"*20)
        # print(self.get_view_matrix())
        # print("#"*20)
        # print(self.get_camera_matrix())

        test_pt = glm.vec4(0, 0, 0.0, 1.0)
        clip_coords = self.get_camera_matrix() * test_pt
        save_w = clip_coords.w
        clip_coords /= save_w
        game_pos = glm.unProject(glm.vec3(screen_x, screen_y, 1),
                                 glm.mat4(1.0),
                                 glm.mat4(1.0),
                                 self.window.viewport)
        # print('ndc', game_pos)
        game_pos.z = clip_coords.z
        game_pos = glm.vec4(game_pos, 1.0)
        # print('pre w scale', game_pos)

        game_pos.y *= -1   # need to invert this for some reason
        game_pos *= save_w
        # print('pre_inverse', game_pos)
        game_pos = glm.inverse(self.get_camera_matrix()) * game_pos    # in [-1, 1] (food_grid vbo coords)
 
        game_pos = (game_pos.xy + 1) / 2   # in [0, 1]
        game_pos *= self.cfg.size
        return game_pos