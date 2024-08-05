from evolution import config


import glm
import moderngl as mgl
import numpy as np


class Camera:
    def __init__(self, ctx: mgl.Context, cfg: config.Config, window):
        self.ctx = ctx
        self.cfg = cfg
        self.ubo = self.ctx.buffer(reserve=2 * 4*4 * 4)  # 2 mat4s, 4x4 floats of 4 bytes
        self.window = window

        self.reset_camera()

    def reset_camera(self):
        self.position = glm.vec3(0.0, 0.0, 1.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.world_up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = -90.0
        self.pitch = 0.0
        self.movement_speed = 0.025
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

        self.update_camera_vectors()

    def process_mouse_scroll(self, yoffset):
        # if self.zoom < 1.0:
        #     self.zoom -= yoffset
        # if self.zoom > 45.0:
        #     self.zoom = 45.0
        self.position += self.front * yoffset / 10
        if self.position.z < 0.1:
            self.position.z = 0.1

    def update_camera_vectors(self):
        front = glm.vec3(0.0, 0.0, 0.0)
        front.x = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        front.y = np.sin(np.radians(self.pitch))
        front.z = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def get_camera_matrix(self):
        return self.get_projection_matrix() * self.get_view_matrix()
    
    def game_coordinates(self, screen_x, screen_y):
        # invert get_camera_matrix to get in game coordinates
        inv = glm.inverse(self.get_camera_matrix())
        # screen_x = screen_x / self.screen_size[0] * 2 - 1
        # screen_y = 1 - screen_y / self.screen_size[1] * 2
        # screen_z = 1
        print(screen_x, screen_y)
        screen_pos = glm.vec4(screen_x, screen_y, 0, 1)
        game_pos = inv * screen_pos

        return game_pos