#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
torch.set_grad_enabled(False)

import evolution.core.gworld
import evolution.visual.main
from evolution.core import config


torch.random.manual_seed(0)
cfg = config.Config(start_creatures=8192, max_creatures=262144, size=5000, food_cover_decr=0.2,
                    cell_size=16.0, cache_size=128, max_per_cell=20, use_cache=0,
                    food_sight=4, ray_dist_range=(1.5, 5.0), size_range=(0.1, 5.0),
                    brain_size=(50, 40))
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=False, init_food_scale=15.)

# evolution.visual.main.main(cfg)

evolution.core.gworld.multi_benchmark(cfg, max_steps=8000, N=8)
 
# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0], 
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]}, 
#                                             max_steps=8000, N=2, skip_errors=True)  

    
"""
compute_grid_setup 1385.9103773261886 +- 1.645064903446794
trace_rays_grid 2738.372640791349 +- 30.085589286100994
collect_stimuli 5840.324085130356 +- 65.49329436702084
think 49852.59275526367 +- 644.0776853145261
rotate_creatures 17442.324219211005 +- 221.85531172089634
only_move_creatures 1762.3459830768406 +- 1.8541982019993335
compute_gridded_attacks 974.1734792934731 +- 5.650673273169528
only_do_attacks 718.6291065961123 +- 0.3769048239065077
fused_kill_reproduce 25032.3290771991 +- 53.58736735226269
creatures_eat_grow 6678.975730642676 +- 13.315276436929718
n_maxxed 9.974519653320314 +- 0.12266358685980219
algo_max 2402.875 +- 66.22969702374564
algo_fill 4593.25 +- 62.34258748468589
algo_move 905.125 +- 11.04930104447465
"""
