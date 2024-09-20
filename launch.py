#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.set_grad_enabled(False)

from evolution.visual.main import main       # pylint: disable=wrong-import-position
from evolution.core import config  # pylint: disable=wrong-import-position
from evolution.core import benchmarking  # pylint: disable=wrong-import-position


torch.random.manual_seed(0)
cfg = config.Config(start_creatures=8192, max_creatures=262144, size=5000, food_cover_decr=0.2,
                    cell_size=16.0, cache_size=128, max_per_cell=20, use_cache=0,
                    food_sight=4, size_range=(0.1, 5.0),
                    brain_size=(50, 40))
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=True, init_food_scale=15.)

# main(cfg)

benchmarking.multi_benchmark(cfg, max_steps=16000, num_simulations=4)

# benchmarking.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0],
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]},
#                                             max_steps=8000, num_simulations=2, skip_errors=True)

"""
normalize 6765.99859212825 +- 39.18557204645398
cuda_hsv_spiral 868.1116719720885 +- 7.757733050075682
compute_grid_setup 2724.597315106541 +- 9.247042284599328
trace_rays_grid 8130.490883322433 +- 73.1691175268831
collect_stimuli 19116.4511959292 +- 95.89382700570457
think 131257.87553381547 +- 672.0775234943148
rotate_creatures 3305.2452416196465 +- 17.256443634684533
only_move_creatures 4316.8157075122 +- 22.127146476933973
compute_gridded_attacks 2057.5644617211074 +- 5.11376956850841
do_attacks 1513.3882777858526 +- 3.3768693199437876
_kill_dead 10005.728835785761 +- 57.72501129600949
_get_reproducers 1918.9523014556617 +- 6.274836911168553
_reproduce 46956.00446071476 +- 89.07789617048365
fused_kill_reproduce 65248.20877311379 +- 337.39013840820644
creatures_eat_grow 11111.536156356335 +- 11.74532746688042
_reduce_reproducers 1250.193449859973 +- 8.59978473928233
generate 474.2881532830652 +- 2.511514186467809
fetch_params 3777.3352960085904 +- 8.384115546207155
reproduce_mutable 26963.27366984263 +- 41.622704235673176
reproduce_most 34395.93253540993 +- 48.809546433323106
reproduce_extra 4496.495620943606 +- 19.717387364360707
reproduce_traits 40515.90368127823 +- 71.97634331770308
n_maxxed 9.751562118530273 +- 0.05886050160706993
algo_max 11141.0 +- 97.07471349429778
algo_fill 4036.25 +- 85.87527292533049
algo_move 794.25 +- 11.419099497479358
"""