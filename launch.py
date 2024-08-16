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
N = 10
for i in range(N):
    cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0,
                        cell_size=2.0, cache_size=128, max_per_cell=128)
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
