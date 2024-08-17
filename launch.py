#!/usr/bin/env python3

import evolution
import moderngl as mgl
import moderngl_window as mglw

import evolution.gworld
import evolution.visual.main
import torch
import numpy as np
torch.set_grad_enabled(False)
# torch.random.manual_seed(0)

# Blocking call entering rendering/event loop
# mglw.run_window_config(evolution.visual.Game)

# evolution.visual.main.main()

from evolution import config
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
# print(evolution.gworld.benchmark(cfg, max_steps=2500))

from collections import defaultdict
total_times = defaultdict(list)
N = 20
for i in range(N):
    cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0,
                        cell_size=2.0, cache_size=128, max_per_cell=128, use_cache=0)
    bmarks = evolution.gworld.benchmark(cfg, max_steps=2500)
    # if i == 0:  # skip first iteration for compilation weirdness
    #     continue
    for k, v in bmarks.items():
        total_times[k].append(v)

for k, v in total_times.items():
    # compute mean and sample standard deviation
    mean = np.mean(v)
    std = np.std(v, ddof=1) / np.sqrt(len(v))
    print(k, mean, '+-', std)
    

"""
compute_grid_setup 735.4816209129989 +- 2.9475672066372063
trace_rays_grid 382.1047769848257 +- 0.6659491256059795
collect_stimuli 359.2628526508808 +- 1.2570186312897107
think 1122.200986341387 +- 2.8981567554656746
rotate_creatures 1142.5258647695184 +- 1.359790614698702
only_move_creatures 418.1067409504205 +- 2.607459955029635
compute_gridded_attacks 407.9113952383399 +- 0.4744203299946343
only_do_attacks 188.8393968731165 +- 0.2797034819583504
_kill_dead 1842.5515991400928 +- 5.656057938812128
_reproduce 4479.234886916727 +- 7.777394625255884
fused_kill_reproduce 7917.413489667326 +- 14.325337994053912
creatures_eat 1611.95025011003 +- 8.368854853301821
grow_food 1052.3607590228319 +- 1.6071112410528354
n_maxxed 0.0 +- 0.0
"""
