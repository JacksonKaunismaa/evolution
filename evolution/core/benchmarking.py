from collections import defaultdict
from typing import Dict, List
import itertools
import os.path as osp
import pickle
from tqdm import trange
import numpy as np

from evolution.core.config import Config, simple_cfg
from evolution.core.gworld import GWorld
from evolution.cuda import cuda_utils

def _benchmark(cfg=None, max_steps=512):
    # import cProfile
    # from pstats import SortKey
    # import io, pstats
    cuda_utils.BENCHMARK = True
    cuda_utils.times.clear()
    cuda_utils.n_times.clear()

    # torch.manual_seed(1)
    if cfg is None:
        cfg = simple_cfg()
    game = GWorld(cfg)

    # pr = cProfile.Profile()
    # pr.enable()
    for _ in trange(max_steps):
        if not game.step():   # we did mass extinction before finishing
            break

    cuda_utils.times['n_maxxed'] = game.n_maxxed / game.state.time
    cuda_utils.times['algo_max'] = game.creatures.algos['max']
    cuda_utils.times['algo_fill'] = game.creatures.algos['fill_gaps']
    cuda_utils.times['algo_move'] = game.creatures.algos['move_block']

    del game
    cuda_utils.clear_mem()

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    return cuda_utils.times, cuda_utils.n_times

def multi_benchmark(cfg, max_steps=2500, num_simulations=20, skip_first=False):
    """Run the simulation with `cfg` `num_simulations` times for `max_steps steps` each and then
    compute mean and standard deviation of benchmark times for each `#@cuda_profile`'d function."""
    total_times = defaultdict(list)
    for i in range(num_simulations):
        times, n_times = _benchmark(cfg, max_steps=max_steps)
        if i == 0 and skip_first:  # skip first iteration for compilation weirdness
            continue
        for k, v in times.items():
            if k in n_times:
                total_times[k].append(v)
            else:
                total_times[k].append(float(v))
    results = {}
    for k, v in total_times.items():
        # compute mean and sample standard deviation
        mean = np.mean(v)
        if len(v) == 1:
            std = 0
        else:
            std = np.std(v, ddof=1) / np.sqrt(len(v))
        results[k] = (mean, std)
        print(k, mean, '+-', std)
    return results

def hyperparameter_search(cfg: Config, hyperparameters: Dict[str, List], max_steps, num_simulations,
                          force_restart=False, path='hyper.pkl', skip_errors=False):
    """Run the simulation with all combinations of hyperparameters in `hyperparameters`
    and save the results to `path`. If `skip_errors` is True, then when the simulation crashes, we
    store the 'result' of that run as the exception that was raised. This function can be run
    multiple times between crashes, and it will pick up where it left off, unless `force_restart` is
    True, in which case it will start from scratch."""
    hyp_keys = list(hyperparameters.keys())
    hyp_vals = list(hyperparameters.values())
    choices = itertools.product(*hyp_vals)

    if osp.exists(path) and not force_restart:
        with open(path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    for choice in choices:
        if choice in results:
            continue
        cfg.update_in_place(dict(zip(hyp_keys, choice)))
        print(f"Testing {dict(zip(hyp_keys, choice))}")
        try:
            results[choice] = multi_benchmark(cfg, max_steps=max_steps,
                                              num_simulations=num_simulations)
        except Exception as e:
            if skip_errors:
                results[choice] = e
                raise e
        finally:
            with open(path, 'wb') as f:
                pickle.dump(results, f)
    return results
