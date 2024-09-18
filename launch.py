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

# evolution.visual.main.main(cfg)

evolution.core.gworld.multi_benchmark(cfg, max_steps=16000, num_simulations=4)

# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0],
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]},
#                                             max_steps=8000, num_simulations=2, skip_errors=True)

"""
normalize 6987.639738655314 +- 29.568942873026028
compute_grid_setup 2754.24838769692 +- 18.78971991964697
trace_rays_grid 8831.100685136393 +- 70.08593504529023
collect_stimuli 20319.41288983822 +- 75.54670950935443
think 173584.72338827327 +- 635.5261744466998
rotate_creatures 57749.895659685135 +- 252.53184591428288
only_move_creatures 4447.625068482012 +- 50.161215874808875
compute_gridded_attacks 2104.582506345585 +- 2.7381943884218853
do_attacks 1519.5561096277088 +- 4.033989443056919
_kill_dead 11112.284233991057 +- 37.381610914332974
_get_reproducers 1244.619283619104 +- 7.397728035510526
_reproduce 57297.48744433746 +- 179.7101373867731
fused_kill_reproduce 77094.47611740977 +- 270.58758920461594
creatures_eat_grow 11085.96262255311 +- 8.556584939334286
_reduce_reproducers 2024.7143121719128 +- 21.35579962045535
generate 507.4347831810592 +- 4.5480550927666945
fetch_params 3859.276140678703 +- 7.45065770806851
reproduce_mutable 33726.00303603336 +- 81.62274352555535
reproduce_most 37692.988218307495 +- 99.39704444457826
reproduce_extra 6692.473036400974 +- 27.06980823055139
reproduce_traits 46856.791553616524 +- 135.82228842439866
n_maxxed 10.132127285003662 +- 0.05964018548261637
algo_max 10980.25 +- 56.733844983983474
algo_fill 4146.75 +- 55.65425260780467
algo_move 848.25 +- 12.638400478963572
"""