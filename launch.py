#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.set_grad_enabled(False)

from evolution.visual import main       # pylint: disable=wrong-import-position
from evolution.core import config  # pylint: disable=wrong-import-position
from evolution.core import benchmarking  # pylint: disable=wrong-import-position


# torch.random.manual_seed(0)
cfg = config.Config(start_creatures=8192, max_creatures=262144, size=5000, food_cover_decr=0.2,
                    cell_size=16.0, cache_size=128, max_per_cell=20, use_cache=0,
                    food_sight=4, size_range=(0.1, 5.0),
                    brain_size=(50, 40))
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=True, init_food_scale=15.)

# main.main(cfg)

benchmarking.multi_benchmark(cfg, max_steps=16000, num_simulations=4)

# benchmarking.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0],
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]},
#                                             max_steps=8000, num_simulations=2, skip_errors=True)

"""
normalize 6906.073668628262 +- 21.803742416539503
vectorized_hsl_to_rgb 16088.44529247284 +- 17.478475755351653
compute_grid_setup 2716.260695851175 +- 18.063553246742
trace_rays_grid 8112.125282675028 +- 31.854554861868074
collect_stimuli 18174.207554467022 +- 85.67233412691083
think 130124.40049314871 +- 613.8531312585607
rotate_creatures 58058.29030571878 +- 269.253755019194
only_move_creatures 4403.6108279787 +- 44.400012415085044
compute_gridded_attacks 2053.7578427735716 +- 5.444696990688869
do_attacks 1514.3728369604796 +- 2.8565419394540634
_kill_dead 11042.67342479527 +- 35.519566169674384
_get_reproducers 1251.9230408631265 +- 5.228250150369933
_reproduce 74570.67208642513 +- 81.33448563925471
fused_kill_reproduce 94326.18511641026 +- 129.06898492323805
creatures_eat_grow 11088.206848859787 +- 14.1106520831044
_reduce_reproducers 2001.576976204873 +- 9.762492482159935
generate 493.41177504183725 +- 2.910497669319865
fetch_params 3838.763204727904 +- 7.427777906619437
reproduce_mutable 33148.31925938465 +- 16.025749387754654
reproduce_most 40579.83311796188 +- 25.507525030247738
reproduce_extra 21206.42800372839 +- 42.85235270127523
reproduce_traits 64244.58814710379 +- 70.66264060059034
n_maxxed 9.848551034927368 +- 0.03018771151992263
algo_max 11048.0 +- 74.34715865451753
algo_fill 4101.25 +- 64.78988475577547
algo_move 820.25 +- 12.331362455138525
"""