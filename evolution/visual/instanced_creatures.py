import moderngl as mgl
import numpy as np


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