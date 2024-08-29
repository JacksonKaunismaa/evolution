from typing import Dict, List, TYPE_CHECKING
from moderngl import Context
from moderngl_window import BaseWindow
import numpy as np

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.cuda import cuda_utils
from evolution.utils import loading
from evolution.utils.subscribe import Subscriber
from evolution.visual.game_state import GameState


class ThoughtsVisualizer(Subscriber):
    """Display memories and current outputs of a creature's thoughts."""
    def __init__(self, cfg: Config, ctx: Context, world: GWorld, wnd: BaseWindow,
                 state: GameState, shaders: Dict[str, str]):
        super().__init__()
        world.publisher.subscribe(self)
        
        self.cfg = cfg
        self.ctx = ctx
        self.shaders = shaders
        self.world = world
        self.state = state
        self.wnd = wnd
        
        self.visible = False
        self.pad = 0.05
        self.width = 0.5
        self.height = 0.1
        self.neuron_size = 0.02

        self.thoughts_tex = loading.load_image_as_texture(self.ctx, 
                                                          'assets/circle_outline_faded.png')
        self.thoughts_sampler = self.ctx.sampler(texture=self.thoughts_tex)
        filter_type = self.ctx.LINEAR
        self.thoughts_sampler.filter = (filter_type, filter_type)

        thoughts_vertices = np.asarray([
            -1.0,  1.0,    0.0, 1.0,
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
             1.0,  1.0,    1.0, 1.0,
        ], dtype='f4')

        # we will let the first row be the memories
        # second row will be the ouptuts

        n_mems = self.cfg.mem_size
        n_outs = 2 
        left_edge = -1.0 + self.pad
        right_edge = left_edge + self.width
        top_edge = 1.0 - self.pad
        bottom_edge = top_edge - self.height
        gap_mems = (right_edge - left_edge) / n_mems
        gap_outs = (right_edge - left_edge) / n_outs

        mem_position_x = np.linspace(left_edge + gap_mems/2, right_edge - gap_mems/2, n_mems)
        out_position_x = np.linspace(left_edge + gap_outs/2, right_edge - gap_outs/2, n_outs)
        position_y = np.linspace(top_edge, bottom_edge, 2)


        mem_position = np.stack([mem_position_x, position_y[0] * np.ones(n_mems)], dtype='f4', axis=1)
        out_position = np.stack([out_position_x, position_y[1] * np.ones(n_outs)], dtype='f4', axis=1)
        self.np_positions = np.concatenate([mem_position, out_position], axis=0)
        self.positions = self.ctx.buffer(self.np_positions)

        self.activations = self.ctx.buffer(reserve=(n_mems + n_outs)*4)
        self.cuda_activations = cuda_utils.register_cuda_buffer(self.activations)

        thoughts_indices = np.array([0, 1, 2, 0, 2, 3], dtype='u4')
        self.thoughts_vbo = self.ctx.buffer(thoughts_vertices)
        self.thoughts_ibo = self.ctx.buffer(thoughts_indices)

        self.prog = self.ctx.program(
            vertex_shader=self.shaders['thoughts.vs'],
            fragment_shader=self.shaders['thoughts.fs']
        )

        self.prog['neuron_size'] = self.neuron_size

        self.thoughts_vao = self.ctx.vertex_array(self.prog, [
            (self.thoughts_vbo, '2f 2f', 'hbox_coord', 'tex_coord'),
            (self.positions, '2f /i', 'position'),
            (self.activations, '1f /i', 'activation')
        ], index_buffer=self.thoughts_ibo)

    def _update(self, creature_id):
        if creature_id is None:
            self.visible = False
            return
        self.visible = True
        cuda_utils.copy_to_buffer(self.world.outputs[creature_id], self.cuda_activations)

    def render(self):
        if not self.visible:
            return
        self.prog["aspect_ratio"] = self.wnd.aspect_ratio
        self.thoughts_sampler.use()
        self.thoughts_vao.render(instances=self.np_positions.shape[0])