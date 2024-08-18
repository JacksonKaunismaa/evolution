#!/usr/bin/env python3

import evolution
import moderngl as mgl
import moderngl_window as mglw

import evolution.core.gworld
import evolution.visual.main
import torch
import numpy as np
torch.set_grad_enabled(False)
# torch.random.manual_seed(0)

# Blocking call entering rendering/event loop
# mglw.run_window_config(evolution.visual.Game)

evolution.visual.main.main()

from evolution.core import config
import pickle
import os.path as osp

# full_results = {}
# if osp.exists('benchmark.pkl'):
#     with open("benchmark.pkl", "rb") as f:
#         full_results = pickle.load(f)


# for cell_size in [2.0, 3.0, 4.0]:
#     for cache_size in [64, 128, 256]:
#         for max_per_cell in [128, 256, 512]:
#             if (cell_size, cache_size, max_per_cell) in full_results:
#                 continue
#             cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0,
#                                 cell_size=cell_size, cache_size=cache_size, max_per_cell=max_per_cell)
#             print(cell_size, cache_size, max_per_cell)
#             benchmarks = evolution.gworld.benchmark(cfg, steps=2500)
#             full_results[(cell_size, cache_size, max_per_cell)] = benchmarks.copy()

#             with open("benchmark.pkl", "wb") as f:
#                 pickle.dump(full_results, f)

# cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0,
#                     cell_size=2.0, cache_size=128, max_per_cell=128)
# torch.random.manual_seed(0)
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=32, immortal=True, init_food_scale=15.)
# print(evolution.gworld.benchmark(cfg, max_steps=2500))

# from collections import defaultdict
# total_times = defaultdict(list)
# N = 20
# for i in range(N):
#     cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0,
#                         cell_size=2.0, cache_size=128, max_per_cell=128, use_cache=0)
#     bmarks = evolution.gworld.benchmark(cfg, max_steps=2500)
#     # if i == 0:  # skip first iteration for compilation weirdness
#     #     continue
#     for k, v in bmarks.items():
#         total_times[k].append(v)

# for k, v in total_times.items():
#     # compute mean and sample standard deviation
#     mean = np.mean(v)
#     std = np.std(v, ddof=1) / np.sqrt(len(v))
#     print(k, mean, '+-', std)
    
"""
compute_grid_setup 764.1916149377823 +- 3.2383586987344666
trace_rays_grid 611.8303262654692 +- 3.673799606921044
collect_stimuli 336.2895743921399 +- 0.3240023653903362
think 1100.118756171316 +- 2.6005534543299813
rotate_creatures 1115.2860229305923 +- 1.6630199949508853
only_move_creatures 379.9413187041879 +- 0.5163189904907349
compute_gridded_attacks 479.75453137233853 +- 3.666007786101306
only_do_attacks 179.75919150598347 +- 0.10734763266403956
_kill_dead 1703.6854702923447 +- 3.4499792726779934
_reproduce 4156.490024921298 +- 6.89442924056734
fused_kill_reproduce 7490.299363550544 +- 12.059096895927329
creatures_eat_grow 884.3601694256067 +- 0.5502520483596781
n_maxxed 0.0 +- 0.0
"""
