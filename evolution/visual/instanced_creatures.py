import moderngl as mgl
import numpy as np

from .. import config
from .. import gworld
from .. import cuda_utils
from .. import loading_utils


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
                                                                './assets/circle_creature_spiky.png')
        self.circle_tex = loading_utils.load_image_as_texture(self.ctx,
                                                              './assets/circle_outline.png')
        self.creature_sampler = self.ctx.sampler(texture=self.creature_tex)
        self.circle_sampler = self.ctx.sampler(texture=self.circle_tex)
        filter_type = self.ctx.NEAREST
        self.creature_sampler.filter = (filter_type, filter_type)
        self.circle_sampler.filter = (filter_type, filter_type)
        self.hitboxes_on = False

        creature_vertices = np.asarray([
            -1.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
             1.0,  1.0,    1.0, 1.0,
        ], dtype='f4')
        # we scale up the creature sprites by this factor so that they match their circular hitbox better
        # this was calculated empirically by measuring pixels
        hbox_vertices = np.copy(creature_vertices)  # we make a copy of the creature sprite vbo to render the hitbox
        scale_fudge_factor = 2#.05859534504
        creature_vertices[::4] *= scale_fudge_factor   # make it relative to the screen size
        creature_vertices[1::4] *= scale_fudge_factor

        hbox_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.creature_vbo = self.ctx.buffer(creature_vertices)
        self.hbox_vbo = self.ctx.buffer(hbox_vertices)
        self.creature_ibo = self.ctx.buffer(hbox_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['creatures.vs'],
            fragment_shader=self.shaders['creatures.fs']
        )

        # self.prog['spriteTexture'].value = 0
        # self.prog['circleTexture'].value = 1

        self.creature_vao = self.ctx.vertex_array(self.prog, [
            (self.creature_vbo, '2f 2f', 'hbox_coord', 'tex_coord'),
            (self.positions, '2f /i', 'position'),
            (self.sizes, 'f /i', 'size'),
            (self.head_dirs, '2f /i', 'head_dir'),
            (self.colors, '3f /i', 'color')],
        index_buffer=self.creature_ibo)

        self.hbox_vao = self.ctx.vertex_array(self.prog, [
            (self.hbox_vbo, '2f 2f', 'hbox_coord', 'tex_coord'),
            (self.positions, '2f /i', 'position'),
            (self.sizes, 'f /i', 'size'),
            (self.head_dirs, '2f /i', 'head_dir'),
            (self.colors, '3f /i', 'color')],
        index_buffer=self.creature_ibo)

    def toggle_hitboxes(self):
        self.hitboxes_on = not self.hitboxes_on

    def update(self):
        cuda_utils.copy_to_buffer(self.world.creatures.positions, self.cuda_positions)
        cuda_utils.copy_to_buffer(self.world.creatures.sizes, self.cuda_sizes)
        cuda_utils.copy_to_buffer(self.world.creatures.head_dirs, self.cuda_head_dirs)
        cuda_utils.copy_to_buffer(self.world.creatures.colors, self.cuda_colors)

    def render(self):
        self.update()
        # self.creature_sampler.use(self.creature_tex_loc)
        # self.circle_sampler.use(self.circle_tex_loc)
        self.creature_sampler.use(0)
        # self.circle_sampler.use(1)
        self.creature_vao.render(instances=self.world.population)

        if self.hitboxes_on:
            self.circle_sampler.use(0)
            self.hbox_vao.render(instances=self.world.population)