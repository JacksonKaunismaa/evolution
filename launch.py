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
                    cell_size=2.0, cache_size=128, max_per_cell=128, use_cache=0,
                    food_sight=4, ray_dist_range=(1.5, 5.0), size_range=(0.1, 5.0),
                    brain_size=(50, 40))
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=False, init_food_scale=15.)

# evolution.visual.main.main(cfg)

# evolution.core.gworld.multi_benchmark(cfg, max_steps=8000, N=8)
 
evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0], 
                                                  'max_per_cell': [80, 90, 100]}, 
                                            max_steps=8000, N=2, skip_errors=True)  

    
"""
compute_grid_setup 53692.75039097667 +- 20.19830530762004
trace_rays_grid 2757.452476482489 +- 29.737342925962313
collect_stimuli 5786.189814642072 +- 74.42757561644584
think 51215.3072708603 +- 681.996083486991
rotate_creatures 17917.481818640605 +- 229.8669914487292
only_move_creatures 1802.0272597642615 +- 1.6330036558314653
compute_gridded_attacks 724.0041976823122 +- 1.902282685647933
only_do_attacks 731.9873527185991 +- 0.3991855130358173
fused_kill_reproduce 25216.593125466257 +- 41.99008727946535
creatures_eat_grow 6653.306500054896 +- 11.298172678878707
n_maxxed 0.0 +- 0.0
algo_max 2537.0 +- 71.33973046526839
algo_fill 4493.875 +- 69.6071726343116
algo_move 870.375 +- 8.914149066031404

"""
