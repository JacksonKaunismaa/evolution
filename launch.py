#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
torch.set_grad_enabled(False)

import evolution.core.gworld
import evolution.visual.main
from evolution.core import config


# torch.random.manual_seed(0)
cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.2,
                    cell_size=2.0, cache_size=128, max_per_cell=128, use_cache=0)
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=False, init_food_scale=15.)

evolution.visual.main.main(cfg)

# evolution.core.gworld.multi_benchmark(cfg, max_steps=2500, N=10)

    
"""
compute_grid_setup 482.79530933611096 +- 1.3718758401742894
trace_rays_grid 597.2399807721376 +- 1.9672694575245235
collect_stimuli 275.7592362921685 +- 0.533262782880255
think 1101.3345553658903 +- 2.3915887145288726
rotate_creatures 920.5014999762177 +- 1.9111978756777044
only_move_creatures 455.2341208655387 +- 0.6508250423657339
compute_gridded_attacks 266.3877790927887 +- 1.8072232657506504
only_do_attacks 194.22384267919696 +- 0.12167123525153227
_kill_dead 796.6284264333547 +- 8.081302892973936
_reproduce 3994.457520944811 +- 45.91538070955325
fused_kill_reproduce 4609.480364069063 +- 51.6521775443228
creatures_eat_grow 472.77455417402086 +- 0.3659262050781333
n_maxxed 0.0 +- 0.0
"""
