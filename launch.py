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
normalize 2171.5985902252287 +- 6.047653950473858
compute_grid_setup 1337.1783066773787 +- 1.6913972594134443
trace_rays_grid 2718.289716772735 +- 16.58390186889657
collect_stimuli 5816.539173133671 +- 35.00808472084344
think 49745.10871951468 +- 340.83290740923195
rotate_creatures 17363.32311447244 +- 121.22250659783492
only_move_creatures 2044.4653550442308 +- 1.8985783486897791
compute_gridded_attacks 949.1751352641732 +- 2.9720313886477845
only_do_attacks 689.0664435774088 +- 0.43995093444835626
_kill_dead 5234.961017228197 +- 18.996688077221695
_get_reproducers 602.710151217645 +- 0.48094812535447834
_reproduce 23677.25457563065 +- 28.83341008662993
fused_kill_reproduce 34124.42126004025 +- 55.61118389436492
creatures_eat_grow 6688.007939182222 +- 10.474161452501981
_reduce_reproducers 649.506620165892 +- 6.787129255094251
generate 293.7848396310583 +- 0.8714537761899028
fetch_params 1777.704912107729 +- 1.1884018466227304
reproduce_mutable 12720.848611157387 +- 15.382372386000283
reproduce_most 14093.852790042758 +- 16.526126014336857
reproduce_extra 3826.6938991323113 +- 3.8964876029743976
reproduce_traits 19071.871578335762 +- 21.19733159956723
n_maxxed 9.822968994140624 +- 0.09150010337119292
algo_max 2427.375 +- 41.803937570862516
algo_fill 4581.875 +- 36.36837200292112
algo_move 888.5 +- 9.623260510717916
"""
