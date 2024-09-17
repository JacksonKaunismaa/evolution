import moderngl as mgl
import numpy as np

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.cuda import cuda_utils
from evolution.utils.subscribe import Subscriber


class CreatureRays(Subscriber):
    def __init__(self, cfg: Config, ctx: mgl.Context, world: GWorld, shaders):
        super().__init__()
        world.state.creature_publisher.subscribe(self)

        self.cfg = cfg
        self.ctx = ctx
        self.world = world
        self.shaders = shaders
        self.visible = False

        # cos(theta), sin(theta), length of arrow * sizeof(float)
        self.rays = self.ctx.buffer(reserve=self.cfg.num_rays * 3 * 4)
        # 3 color components * sizeof(float)
        self.ray_colors = self.ctx.buffer(reserve=self.cfg.num_rays * 3 * 4)
        # cos(theta), sin(theta) of head direction
        self.head_dir = self.ctx.buffer(reserve=2* 4)

        self.cuda_rays = cuda_utils.register_cuda_buffer(self.rays)
        self.cuda_ray_colors = cuda_utils.register_cuda_buffer(self.ray_colors)
        self.cuda_head_dir = cuda_utils.register_cuda_buffer(self.head_dir)

        ray_width = 0.01
        head_dir_width = 0.02
        hbox_vertices = np.asarray([
             0.0,   1.0,    # 0.0, 1.0,   # top left
             0.0,  -1.0,    # 0.0, 0.0,   # bottom left
             1.0,  -1.0,    # 1.0, 0.0,   # bottom right
             1.0,   1.0,    # 1.0, 1.0,   # top right
        ], dtype='f4')
        ray_hbox_vertices = np.copy(hbox_vertices)
        ray_hbox_vertices[1::2] *= ray_width

        head_hbox_vertices = np.copy(hbox_vertices)
        head_hbox_vertices[1::2] *= head_dir_width


        hbox_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.ray_hbox_vbo = self.ctx.buffer(ray_hbox_vertices)
        self.head_hbox_vbo = self.ctx.buffer(head_hbox_vertices)
        self.hbox_ibo = self.ctx.buffer(hbox_indices)

        self.ray_prog = self.ctx.program(
            vertex_shader=self.shaders['rays.vs'],
            fragment_shader=self.shaders['rays.fs']
        )

        self.head_prog = self.ctx.program(
            vertex_shader=self.shaders['head_dir.vs'],
            fragment_shader=self.shaders['head_dir.fs']
        )

        self.ray_vao = self.ctx.vertex_array(self.ray_prog, [
            (self.ray_hbox_vbo, '2f', 'hbox_coord'),
            (self.rays, '3f /i', 'rays'),   # cos, sin, length
            (self.ray_colors, '3f /i', 'ray_colors')  # r, g, b
        ],
        index_buffer=self.hbox_ibo)

        self.head_vao = self.ctx.vertex_array(self.head_prog, [
            (self.head_hbox_vbo, '2f', 'hbox_coord'),
            (self.head_dir, '2f /r', 'head_dir')
        ],
        index_buffer=self.hbox_ibo)
        self.head_prog['head_start'] = self.cfg.attack_range[0]
        self.head_prog['head_len'] = self.cfg.attack_range[1] - self.cfg.attack_range[0]
        self.head_prog['color'] = (0.5, 0.5, 0.5)


    def _update(self, creature_id):
        if not creature_id:
            self.visible = False
            return
        self.visible = True
        cuda_utils.copy_to_buffer(self.world.creatures.rays[creature_id], self.cuda_rays)
        cuda_utils.copy_to_buffer(self.world.collisions[creature_id], self.cuda_ray_colors)
        cuda_utils.copy_to_buffer(self.world.creatures.head_dirs[creature_id], self.cuda_head_dir)

        self.ray_prog['position'] = self.world.creatures.positions[creature_id]
        self.head_prog['position'] = self.world.creatures.positions[creature_id]

        self.ray_prog['size'] = self.world.creatures.sizes[creature_id]
        self.head_prog['size'] = self.world.creatures.sizes[creature_id]

    def render(self):
        self.update()
        if not self.visible:
            return
        # print('rendering rays')
        self.ray_vao.render(instances=self.cfg.num_rays)
        self.head_vao.render()