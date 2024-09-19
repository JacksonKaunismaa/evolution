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
normalize 6876.1397655412 +- 32.8051113308554
cuda_hsv_spiral 874.7715357653797 +- 7.267613182445763
compute_grid_setup 2708.521784548182 +- 8.875133427742632
trace_rays_grid 8227.432561207563 +- 111.14213164681622
collect_stimuli 18373.20987765491 +- 87.47334499078941
think 131598.41739911214 +- 616.2409344298842
rotate_creatures 3338.3048396166414 +- 16.993108837263552
only_move_creatures 4291.666531283408 +- 21.4393421349234
compute_gridded_attacks 2060.6858774144202 +- 8.132076185079852
do_attacks 1508.6878923401237 +- 2.0618651266199963
_kill_dead 11036.904395200312 +- 52.6793779109046
_get_reproducers 1237.4012364982627 +- 3.814372928957472
_reproduce 57686.048066508025 +- 48.283655411859726
fused_kill_reproduce 77693.3338727653 +- 245.84487515299028
creatures_eat_grow 11123.144544944167 +- 14.442986038863936
_reduce_reproducers 2024.1249439914245 +- 11.665223334580512
generate 485.67145600169897 +- 3.078963728884732
fetch_params 3765.3090252972615 +- 5.244226073706686
reproduce_mutable 33142.28563321754 +- 26.480473313852237
reproduce_most 40568.92363125086 +- 27.63275751452469
reproduce_extra 4418.111201055348 +- 18.188060335289762
reproduce_traits 46616.81597959995 +- 49.172057404984244
n_maxxed 9.72036600112915 +- 0.07294644699913756
algo_max 11171.25 +- 83.0936971135597
algo_fill 4005.75 +- 74.66076055510463
algo_move 795.5 +- 8.69386757049665
"""
