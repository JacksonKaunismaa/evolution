from typing import Dict
import numpy as np
from OpenGL.GL import *
import moderngl as mgl
from contextlib import contextmanager
import moderngl_window as mglw
from moderngl_window import settings
import glob
import os.path as osp
from cuda import cuda
import torch
torch.set_grad_enabled(False)
from matplotlib import cm
import PIL.Image
import time
import glm
from sdl2 import timer


from . import gworld, config, cuda_utils, loading_utils
from .cu_algorithms import checkCudaErrors


class Heatmap:
    def __init__(self, cfg, ctx, world, shaders):
        self.cfg: config.Config = cfg
        self.ctx: mgl.Context = ctx
        self.world: gworld.GWorld = world
        self.shaders = shaders

        # world.food_grid[50:80, 0:20] = -2
        self.heatmap_tex = self.ctx.texture((self.cfg.size, self.cfg.size), 1, dtype='f4')
        self.heatmap_sampler = self.ctx.sampler(texture=self.heatmap_tex)
        filter_type = self.ctx.NEAREST
        self.heatmap_sampler.filter = (filter_type, filter_type)

        scr_vertices = np.asarray([
            -1.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
             1.0,  1.0,    1.0, 1.0,    
        ], dtype='f4')

        scr_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.screen_vbo = self.ctx.buffer(scr_vertices)
        self.screen_ibo = self.ctx.buffer(scr_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['heatmap.vs'],
            fragment_shader=self.shaders['heatmap.fs']
        )

        cmap = cm.RdYlGn
        cmap._init()
        self.prog['colormap'] = np.array(cmap._lut[:-3, :3], dtype='f4')
        self.prog['negGamma'] = 1. / 5

        self.screen_vao = self.ctx.vertex_array(self.prog, [
            (self.screen_vbo, '2f 2f', 'pos', 'tex_coord')
        ], index_buffer=self.screen_ibo)

        self.cuda_heatmap = checkCudaErrors(cuda.cuGraphicsGLRegisterImage(
                                        int(self.heatmap_tex.glo), 
                                        GL_TEXTURE_2D, 
                                        cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))

    def update(self):
        max_value, min_value = self.cfg.max_food, self.world.central_food_grid.min()
        # self.prog['minVal'] = min_value
        # self.prog['maxVal'] = max_value
        self.prog['scale'] = max(abs(min_value), abs(max_value))
        cuda_utils.copy_to_texture(self.world.central_food_grid, self.cuda_heatmap)

    def render(self):
        self.heatmap_sampler.use()
        self.screen_vao.render()


class InstancedCreatures:
    def __init__(self, cfg, ctx, world, shaders):
        self.cfg: config.Config = cfg
        self.ctx: mgl.Context = ctx
        self.world: gworld.GWorld = world
        self.shaders = shaders

        self.positions = self.ctx.buffer(reserve=self.cfg.max_creatures * 2 * 4)  # 2 coordinates
        self.sizes = self.ctx.buffer(reserve=self.cfg.max_creatures * 1 * 4)  # 1 radius
        self.head_dirs = self.ctx.buffer(reserve=self.cfg.max_creatures * 2 * 4)  # 2 directions
        self.colors = self.ctx.buffer(reserve=self.cfg.max_creatures * 3 * 4)  # 3 color channels

        self.cuda_positions = cuda_utils.register_cuda_buffer(self.positions)
        self.cuda_sizes = cuda_utils.register_cuda_buffer(self.sizes)
        self.cuda_head_dirs = cuda_utils.register_cuda_buffer(self.head_dirs)
        self.cuda_colors = cuda_utils.register_cuda_buffer(self.colors)
        
        self.creature_tex = loading_utils.load_image_as_texture(self.ctx, 
                                                                './assets/grey_square_creature_sprite.png')
        self.sampler = self.ctx.sampler(texture=self.creature_tex)
        filter_type = self.ctx.NEAREST
        self.sampler.filter = (filter_type, filter_type)

        hbox_vertices = np.asarray([
            -1.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
             1.0,  1.0,    1.0, 1.0,    
        ], dtype='f4')
        hbox_vertices[::4] /= self.cfg.size   # make it relative to the screen size
        hbox_vertices[1::4] /= self.cfg.size

        hbox_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.hbox_vbo = self.ctx.buffer(hbox_vertices)
        self.hbox_ibo = self.ctx.buffer(hbox_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['creatures.vs'],
            fragment_shader=self.shaders['creatures.fs']
        )

        self.prog['width'] = self.cfg.size

        self.vao = self.ctx.vertex_array(self.prog, [
            (self.hbox_vbo, '2f 2f', 'hbox_coord', 'tex_coord'),
            (self.positions, '2f /i', 'position'),
            (self.sizes, 'f /i', 'size'),
            (self.head_dirs, '2f /i', 'head_dir'),
            (self.colors, '3f /i', 'color') 
        ],
        index_buffer=self.hbox_ibo)

    def update(self):
        cuda_utils.copy_to_buffer(self.world.creatures.positions, self.cuda_positions)
        cuda_utils.copy_to_buffer(self.world.creatures.sizes, self.cuda_sizes)
        cuda_utils.copy_to_buffer(self.world.creatures.head_dirs, self.cuda_head_dirs)
        cuda_utils.copy_to_buffer(self.world.creatures.colors, self.cuda_colors)
        
    def render(self):
        self.sampler.use()
        self.vao.render(instances=self.world.population)


class Camera:
    def __init__(self, ctx: mgl.Context, cfg: config.Config, screen_size=(1920, 1080)):
        self.ctx = ctx
        self.cfg = cfg
        self.ubo = self.ctx.buffer(reserve=2 * 4*4 * 4)  # 2 mat4s, 4x4 floats of 4 bytes
        self.screen_size = screen_size

        self.position = glm.vec3(0.0, 0.0, 1.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)
        self.world_up = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = -90.0
        self.pitch = 0.0
        self.movement_speed = 0.025
        self.mouse_sensitivity = 0.1
        self.zoom = 45.0

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)
    
    def get_projection_matrix(self):
        return glm.perspective(glm.radians(self.zoom), self.screen_size[0] / self.screen_size[1], 0.1, 100.0)
    
    
    
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

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self.update_camera_vectors()

    def process_mouse_scroll(self, yoffset):
        self.zoom -= yoffset
        if self.zoom < 1.0:
            self.zoom = 1.0
        if self.zoom > 45.0:
            self.zoom = 45.0

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


class Game:
    def __init__(self, window, cfg: config.Config, shader_path='./shaders'):
        # super().__init__(**kwargs)
        self.wnd = window
        self.cfg = cfg
        self.ctx = window.ctx
        self.shaders = loading_utils.load_shaders(shader_path)
        self.world: gworld.GWorld = gworld.GWorld(self.cfg)
        self.cuDevice = self.world.kernels.cuDevice
        self.heatmap = Heatmap(self.cfg, self.ctx, self.world, self.shaders)
        self.creatures = InstancedCreatures(self.cfg, self.ctx, self.world, self.shaders)
        self.camera = Camera(self.ctx, self.cfg)
        self.now = timer.SDL_GetTicks()
        
        # print(self.heatmap.prog['Matrices'].binding, self.creatures.prog['Matrices'].binding)
        # self.camera.ubo.bind_to_uniform_block(self.heatmap.prog['Matrices'].binding)
        # self.camera.ubo.bind_to_uniform_block(self.creatures.prog['Matrices'].binding)

    def key_event(self, key, action, modifiers):
        now = timer.SDL_GetTicks()
        delta_time = (now - self.now) / 1000.0
        self.now = now
        print(now, delta_time)
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.W:
                print('W')
                self.camera.process_keyboard('FORWARD', delta_time)

            if key == self.wnd.keys.S:
                print('S')
                self.camera.process_keyboard('BACKWARD', delta_time)

            if key == self.wnd.keys.A:
                self.camera.process_keyboard('LEFT', delta_time)

            if key == self.wnd.keys.D:
                self.camera.process_keyboard('RIGHT', delta_time)

            if key == self.wnd.keys.Z:
                self.camera.process_mouse_movement(0, 1)

            if key == self.wnd.keys.X:
                self.camera.process_mouse_movement(0, -1)


    def step(self, n=20) -> bool:
        for _ in range(n):
            if not self.world.step(visualize=False, save=False):
                torch.cuda.synchronize()
                return False
        torch.cuda.synchronize()
        return True

    def render(self):
        # This method is called every frame
        self.ctx.clear(1.0, 1.0, 1.0)
        # print(frametime)
        # self.wnd.title = f'Evolution'#: {1./frametime: .2f} FPS'
        self.camera.ubo.write(self.camera.get_camera_matrix())
        self.camera.ubo.bind_to_uniform_block()

        self.heatmap.update()
        self.heatmap.render()
        self.creatures.update()
        self.creatures.render()

def mouse_scroll_callback(window, game, xoffset, yoffset):
    game.camera.process_mouse_scroll(yoffset)


def main():
    settings.WINDOW['class'] = 'moderngl_window.context.sdl2.Window'
    settings.WINDOW['gl_version'] = (4, 6)
    settings.WINDOW['title'] = 'Evolution'
    settings.WINDOW['size'] = (1920, 1080)
    settings.WINDOW['vsync'] = False  # avoid fps cap
    window = mglw.create_window_from_settings()
    cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0)
    game = Game(window, cfg)
    populated = True
    window.mouse_scroll_event_func = lambda x, y: mouse_scroll_callback(window, game, x, y)
    window.key_event_func = game.key_event
   
    while not window.is_closing and populated:
        window.clear()
        game.render()
        # print(window.frames)
        # time.sleep(5.0)
        window.title = f'Epoch: {game.world.time} | Population: {game.world.population}'
        window.swap_buffers()
        # time.sleep(0.1)
        populated = game.step(1)
        

