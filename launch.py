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

main(cfg)

# benchmarking.multi_benchmark(cfg, max_steps=16000, num_simulations=4)

# benchmarking.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0],
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]},
#                                             max_steps=8000, num_simulations=2, skip_errors=True)

"""
normalize 6918.596209836716 +- 19.14371515458316
cuda_hsv_spiral 868.9743925724179 +- 1.1865525966554429
compute_grid_setup 2726.1442276472226 +- 1.5347112161877148
trace_rays_grid 8097.445769045502 +- 40.34513170535536
collect_stimuli 18184.718607071787 +- 75.6946159511703
think 130094.8362557292 +- 556.9521650943856
rotate_creatures 58012.760024793446 +- 258.34171854319646
only_move_creatures 4358.648212559521 +- 1.8021869028289017
compute_gridded_attacks 2079.8093599043787 +- 4.642041164417196
do_attacks 1525.6510779801756 +- 1.3316833870323073
_kill_dead 11058.428297648206 +- 38.32162042456917
_get_reproducers 1251.6394834476523 +- 1.2105644977930246
_reproduce 58327.32830679789 +- 17.519911813633417
fused_kill_reproduce 78241.11775494367 +- 130.49516260993568
creatures_eat_grow 11088.026119068265 +- 10.49008126245908
_reduce_reproducers 2027.6772243755404 +- 13.761123632499315
generate 487.90588654228486 +- 0.9050059884491293
fetch_params 3820.999415740487 +- 4.548441777357465
reproduce_mutable 33566.392489712685 +- 16.793186455415157
reproduce_most 41053.49539428949 +- 13.455051354110054
reproduce_extra 4454.761796168983 +- 2.399134640052681
reproduce_traits 47159.992691636086 +- 14.334501887100194
n_maxxed 9.861493587493896 +- 0.040515756269523036
algo_max 10999.75 +- 75.46674212304826
algo_fill 4146.0 +- 68.8754915287966
algo_move 823.25 +- 9.437293044088436
"""