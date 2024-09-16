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

# evolution.visual.main.main(cfg)

evolution.core.gworld.multi_benchmark(cfg, max_steps=16000, N=4)
 
# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0], 
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]}, 
#                                             max_steps=8000, N=2, skip_errors=True)  

"""
normalize 6435.839140797645 +- 11.176450513031263
compute_grid_setup 2803.143079571193 +- 8.622425101706698
trace_rays_grid 8980.954525617883 +- 90.24564541944181
collect_stimuli 20492.39701579511 +- 50.01958347094158
think 174597.17025128007 +- 498.23840699738076
rotate_creatures 58146.47748408094 +- 147.51245059609326
only_move_creatures 4283.670355234295 +- 4.609049927839974
compute_gridded_attacks 2104.6793379280716 +- 2.9684179403648034
only_do_attacks 1446.312854967313 +- 1.2442810845916317
_kill_dead 11019.144858980551 +- 38.30215018293473
_get_reproducers 1254.7081609561574 +- 0.39652281424372726
_reproduce 52692.21511463076 +- 41.06561002284245
fused_kill_reproduce 72244.11771890521 +- 53.309906874434766
creatures_eat_grow 11081.880957573652 +- 10.255986808761394
_reduce_reproducers 2413.3186161192134 +- 6.496070805583529
generate 574.3352563781664 +- 0.3196823345208669
fetch_params 3998.0609010837798 +- 3.711200136968038
reproduce_mutable 29824.831106003374 +- 27.628031371159768
reproduce_most 32604.85484686494 +- 34.216525904256486
reproduce_extra 6624.9781293570995 +- 5.920529566929506
reproduce_traits 41719.00386226177 +- 40.12024602136226
n_maxxed 10.000925540924072 +- 0.06863785581377661
algo_max 11120.75 +- 49.90052604933138
algo_fill 4020.25 +- 38.73064032519989
algo_move 835.75 +- 10.857984773121268
"""
