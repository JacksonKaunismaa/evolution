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

evolution.core.gworld.multi_benchmark(cfg, max_steps=8000, N=8)
 
# evolution.core.gworld.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0], 
#                                                   'max_per_cell': [20, 40, 60, 80, 90, 100]}, 
#                                             max_steps=8000, N=2, skip_errors=True)  

"""
normalize 2949.5328905979113 +- 6.269549099279109
compute_grid_setup 1325.3970098119462 +- 1.5798338491667496
trace_rays_grid 3229.180037780665 +- 24.526331771766902
collect_stimuli 6934.798803652637 +- 50.18553007932075
think 59001.547338152304 +- 448.06771373598883
rotate_creatures 19888.614659111947 +- 149.55517010188046
only_move_creatures 2002.7982082664967 +- 1.6668261964311442
compute_gridded_attacks 1016.537638769485 +- 2.8774901699425266
only_do_attacks 678.660398190259 +- 0.361853856857679
_kill_dead 5347.968305308372 +- 19.83953160231117
_get_reproducers 592.7126552612754 +- 0.27261786369189656
_reproduce 24092.385690730065 +- 29.359527273367025
fused_kill_reproduce 34854.85674738884 +- 37.83430668749208
creatures_eat_grow 6790.355463832617 +- 9.180696878557466
_reduce_reproducers 761.3031921572983 +- 5.8906981380927155
generate 288.63428325764835 +- 0.3750240548077724
fetch_params 1845.7307366057503 +- 1.8225709094679126
reproduce_mutable 13771.55714399647 +- 16.879383948354057
reproduce_most 15153.19855980575 +- 17.958755286982363
reproduce_extra 3013.0522519536316 +- 2.75925899132422
reproduce_traits 19332.983874619007 +- 21.854800915706512
n_maxxed 9.931983276367188 +- 0.05031916835378128
algo_max 3142.875 +- 35.63577531277883
algo_fill 3998.25 +- 32.8126445575047
algo_move 833.25 +- 6.040784007773447
"""
