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
cfg = config.Config(start_creatures=256, max_creatures=16384, size=500, food_cover_decr=0.0,
                    cell_size=2.0, cache_size=128, max_per_cell=128, use_cache=0)
# cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
#                 init_size_range=(0.2, 0.2), num_rays=3, immortal=False, init_food_scale=15.)

evolution.visual.main.main(cfg)

# evolution.core.gworld.multi_benchmark(cfg, max_steps=2500, N=20)

    
"""
compute_grid_setup 764.1916149377823 +- 3.2383586987344666
trace_rays_grid 611.8303262654692 +- 3.673799606921044
collect_stimuli 336.2895743921399 +- 0.3240023653903362
think 1100.118756171316 +- 2.6005534543299813
rotate_creatures 1115.2860229305923 +- 1.6630199949508853
only_move_creatures 379.9413187041879 +- 0.5163189904907349
compute_gridded_attacks 479.75453137233853 +- 3.666007786101306
only_do_attacks 179.75919150598347 +- 0.10734763266403956
_kill_dead 1703.6854702923447 +- 3.4499792726779934
_reproduce 4156.490024921298 +- 6.89442924056734
fused_kill_reproduce 7490.299363550544 +- 12.059096895927329
creatures_eat_grow 884.3601694256067 +- 0.5502520483596781
n_maxxed 0.0 +- 0.0
"""


""" refactored
compute_grid_setup 470.1475288461894 +- 4.142590736994931
trace_rays_grid 532.3615250043571 +- 1.906800252167498
collect_stimuli 264.5651738656685 +- 0.5916395434025206
think 1102.3577881030737 +- 5.052401685092371
rotate_creatures 896.995627734065 +- 3.5783934050631583
only_move_creatures 392.21322349179536 +- 1.104663640927039
compute_gridded_attacks 252.21232261601835 +- 2.4036518553956294
only_do_attacks 181.73948060385882 +- 0.174596075457209
_kill_dead 1856.5456350415946 +- 6.0402589089193475
_reproduce 4246.70085051544 +- 14.025561641498205
fused_kill_reproduce 7640.3123897664245 +- 27.857600815756605
creatures_eat_grow 626.0984256379306 +- 1.8549017252323483
n_maxxed 0.0 +- 0.0
"""
"""
compute_grid_setup 478.77917404286563 +- 2.586394785536291
trace_rays_grid 574.3233335442841 +- 2.3889125017084964
collect_stimuli 263.6030537366867 +- 1.8987749907826341
think 1093.045412451774 +- 1.9604768590536599
rotate_creatures 887.1171906992793 +- 1.8125863217793163
only_move_creatures 385.56314977239816 +- 2.978466355979785
compute_gridded_attacks 256.4626652345061 +- 1.3815362557752466
only_do_attacks 179.1576996996999 +- 0.9463831132690869
_kill_dead 867.8524613339454 +- 8.14396789911397
_reproduce 9012.97521788641 +- 70.10486876560964
fused_kill_reproduce 9077.576701232883 +- 70.36024678188393
creatures_eat_grow 476.11893786713483 +- 3.5500289757295858
n_maxxed 0.0 +- 0.0
"""
