#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.set_grad_enabled(False)

import evolution.core.gworld
import evolution.visual.main
from evolution.core import config


torch.random.manual_seed(0)
cfg = config.Config(start_creatures=8192, max_creatures=262144, size=5000, food_cover_decr=0.2,
                    cell_size=16.0, cache_size=128, max_per_cell=20, use_cache=0,
                    food_sight=4, size_range=(0.1, 5.0),
                    brain_size=(50, 40))
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.5,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=True, init_food_scale=15.)

evolution.visual.main.main(cfg)

# evolution.core.gworld.multi_benchmark(cfg, max_steps=16000, N=4)
 
# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0], 
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]}, 
#                                             max_steps=8000, N=2, skip_errors=True)  

"""
normalize 6342.463975309569 +- 11.613968982797424
compute_grid_setup 2785.297907390166 +- 2.9983754324222884
trace_rays_grid 8952.37781687826 +- 39.172202027772975
collect_stimuli 20582.18938582018 +- 103.9450446167116
think 175386.53027764335 +- 866.6664600204674
rotate_creatures 58467.88456666842 +- 294.5147117779566
only_move_creatures 4300.792872432619 +- 4.515987942017493
compute_gridded_attacks 2101.9149130750448 +- 3.2614510751357675
only_do_attacks 1468.9931306187063 +- 1.110977224122373
_kill_dead 11011.559420531616 +- 43.25165651732099
_get_reproducers 1229.3790456331335 +- 1.835783905821063
_reproduce 52435.71083160117 +- 77.9043048773482
fused_kill_reproduce 72037.67144123465 +- 69.06274493548524
creatures_eat_grow 11082.034298986197 +- 9.28781165479784
_reduce_reproducers 2404.2431920841336 +- 10.455050862612437
generate 569.5371361221187 +- 0.6486364518706548
fetch_params 4001.6438271479856 +- 9.73750577547072
reproduce_mutable 29474.4281854257 +- 51.515800750457224
reproduce_most 32309.61750483513 +- 53.33730051510179
reproduce_extra 6659.425844609737 +- 12.360480851768811
reproduce_traits 41490.108481526375 +- 69.7656652072182
n_maxxed 9.902875900268555 +- 0.04806991368929596
algo_max 11176.0 +- 66.97636399009231
algo_fill 3970.0 +- 58.26233774918408
algo_move 830.5 +- 14.121496615680176
"""
