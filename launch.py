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

# evolution.core.gworld.multi_benchmark(cfg, max_steps=8000, N=8)
 
# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0], 
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]}, 
#                                             max_steps=8000, N=2, skip_errors=True)  

"""    
normalize 2162.5563343781396 +- 6.173245886719688
compute_grid_setup 1330.5866098391125 +- 1.5546554176955392
trace_rays_grid 2618.0051402552053 +- 21.674202579258704
collect_stimuli 5588.36129109934 +- 54.72900321130269
think 47557.14813023433 +- 493.59935342960847
rotate_creatures 16648.658287806436 +- 162.9884407888983
only_move_creatures 2037.837090222165 +- 2.4299288767116214
compute_gridded_attacks 987.6036717072129 +- 3.9423271166353753
only_do_attacks 684.6994428858161 +- 0.239281600315717
_kill_dead 5225.3890763493255 +- 20.02754600244486
_get_reproducers 599.0550951317418 +- 0.6589707453501008
_reproduce 23667.58671713993 +- 20.09709712355362
fused_kill_reproduce 34177.99520755187 +- 52.97963351628498
creatures_eat_grow 6657.855213366449 +- 11.049065646048739
_reduce_reproducers 610.9984243144281 +- 10.011429628992436
generate 298.529643710237 +- 1.070200541110568
fetch_params 1789.0008459775418 +- 1.6411960667856138
reproduce_mutable 12763.813628057018 +- 14.141859387009607
reproduce_most 14138.412400841713 +- 15.052983910589369
reproduce_extra 3816.061680331826 +- 2.6372874072975616
reproduce_traits 19110.25380796194 +- 18.89890736478484
n_maxxed 9.496658325195312 +- 0.07378639497830274
algo_max 2196.875 +- 59.001948363713254
algo_fill 4773.875 +- 53.77215075137474
algo_move 929.375 +- 9.01177305370195
"""
