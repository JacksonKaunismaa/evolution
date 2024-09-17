#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.set_grad_enabled(False)

import evolution.core.gworld       # pylint: disable=wrong-import-position
import evolution.visual.main       # pylint: disable=wrong-import-position
from evolution.core import config  # pylint: disable=wrong-import-position


torch.random.manual_seed(0)
cfg = config.Config(start_creatures=8192, max_creatures=262144, size=5000, food_cover_decr=0.2,
                    cell_size=16.0, cache_size=128, max_per_cell=20, use_cache=0,
                    food_sight=4, size_range=(0.1, 5.0),
                    brain_size=(50, 40))
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.5,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=True, init_food_scale=15.)

evolution.visual.main.main(cfg)

# evolution.core.gworld.multi_benchmark(cfg, max_steps=16000, num_simulations=4)

# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0],
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]},
#                                             max_steps=8000, num_simulations=2, skip_errors=True)
"""
normalize 6889.796549459803 +- 22.84581029182901
compute_grid_setup 2715.954143888317 +- 25.782408118814224
trace_rays_grid 9221.880169427022 +- 212.35933316777542
collect_stimuli 20143.41929356754 +- 56.15784718358779
think 172611.3547473736 +- 409.7754202438147
rotate_creatures 57404.96380696446 +- 172.33294378422403
only_move_creatures 4420.617051463574 +- 64.26845731164667
compute_gridded_attacks 2105.2003707755357 +- 10.042365116419738
do_attacks 1511.3158377883956 +- 4.865597481959948
_kill_dead 11094.22226296924 +- 43.58927972536114
_get_reproducers 1242.3811398902908 +- 6.340512153415847
_reproduce 56597.9601280503 +- 112.21962108626775
fused_kill_reproduce 76311.67822471261 +- 130.92428581960812
creatures_eat_grow 11033.726673588157 +- 17.616835683013512
_reduce_reproducers 1994.34645590547 +- 18.879055172854464
generate 502.5745270838961 +- 2.987122373167675
fetch_params 3782.805432217705 +- 2.8642755099333863
reproduce_mutable 33377.95160616562 +- 35.734018632250816
reproduce_most 37048.96750199795 +- 49.27440971746215
reproduce_extra 6669.331218831241 +- 23.134880583253278
reproduce_traits 46224.55834072828 +- 78.31361479544807
n_maxxed 10.178117990493774 +- 0.06137840792946284
algo_max 10887.25 +- 37.699193183232275
algo_fill 4239.0 +- 28.527764254026874
algo_move 849.0 +- 8.962886439832502
"""
