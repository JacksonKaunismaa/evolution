from collections import defaultdict
from enum import Enum
from functools import wraps
from typing import Dict, List
import itertools
import os.path as osp
import os
import pickle
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from tqdm import trange
import numpy as np

from evolution.core.config import Config
from evolution.cuda import cuda_utils


class BenchmarkMethods(Enum):
    """
    Enum for the different benchmark methods
    """
    CUDA_EVENTS = 'cuda_events'
    NONE = 'none'
    NSYS = 'nsys'


class Profile:
    """Static class for managing how/whether the application is being profiled."""
    BENCHMARK = BenchmarkMethods.NONE  # method of benchmarking to use
    times: Dict[str, float] = defaultdict(float)
    n_times: Dict[str, float] = defaultdict(int)

    @staticmethod
    def reset():
        Profile.times.clear()
        Profile.n_times.clear()

    @staticmethod
    def cuda_profile(func):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        @wraps(func)
        def wrapper(*args, nsys_extra=None, **kwargs):
            match (Profile.BENCHMARK):
                case BenchmarkMethods.NONE:
                    return func(*args, **kwargs)

                case BenchmarkMethods.NSYS:
                    # nsys_extra: optional argument to allow annotating nsys timeline even further
                    # is ignored if a benchmark method other than NSYS is used
                    name = func.__name__
                    if nsys_extra is not None:
                        name = f"{func.__name__}_{nsys_extra}"

                    torch.cuda.nvtx.range_push(name)
                    retval = func(*args, **kwargs)
                    torch.cuda.nvtx.range_pop()
                    return retval

                case BenchmarkMethods.CUDA_EVENTS:
                    start.record()  # type: ignore
                    res = func(*args, **kwargs)
                    end.record()   # type: ignore
                    torch.cuda.synchronize()
                    Profile.times[func.__name__] += start.elapsed_time(end)
                    Profile.n_times[func.__name__] += 1
                    return res

        return wrapper



def _benchmark(cfg, init_steps=8000, steps=500, method=BenchmarkMethods.CUDA_EVENTS):
    # need to define the import here to avoid circular imports
    from evolution.core.gworld import GWorld  # pylint: disable=import-outside-toplevel

    # import cProfile
    # from pstats import SortKey
    # import io, pstats
    Profile.BENCHMARK = BenchmarkMethods.NONE
    Profile.times.clear()
    Profile.n_times.clear()

    game = GWorld(cfg)

    # pr = cProfile.Profile()
    for _ in trange(init_steps):
        game.step()

    # pr.enable()
    Profile.BENCHMARK = method
    if method == BenchmarkMethods.NSYS:
        torch.cuda.cudart().cudaProfilerStart() # type: ignore

    for _ in trange(steps):
        game.step()

    if method == BenchmarkMethods.NSYS:
        torch.cuda.cudart().cudaProfilerStop()  # type: ignore

    Profile.times['n_maxxed'] = game.n_maxxed / steps
    Profile.times['algo_max'] = game.creatures.algos['max']
    Profile.times['algo_fill'] = game.creatures.algos['fill_gaps']
    Profile.times['algo_move'] = game.creatures.algos['move_block']

    del game
    cuda_utils.clear_mem()

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    return Profile.times, Profile.n_times

def multi_benchmark(cfg, init_steps=8000, steps=500, num_simulations=20, skip_first=False, method=BenchmarkMethods.CUDA_EVENTS):
    """Run the simulation with `cfg` `num_simulations` times for `max_steps steps` each and then
    compute mean and standard deviation of benchmark times for each `@Profile.cuda_profile`'d function."""

    if method == BenchmarkMethods.NSYS and num_simulations > 1:
        raise ValueError("Cannot run multiple simulations with NSYS method")

    total_times = defaultdict(list)
    for i in range(num_simulations):
        times, n_times = _benchmark(cfg, init_steps=init_steps, steps=steps, method=method)
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

def hyperparameter_search(cfg: Config, hyperparameters: Dict[str, List], init_steps, steps, num_simulations,
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
            results[choice] = multi_benchmark(cfg, init_steps=init_steps, steps=steps,
                                              num_simulations=num_simulations)
        except Exception as e:
            if skip_errors:
                results[choice] = e
                raise e
        finally:
            with open(path, 'wb') as f:
                pickle.dump(results, f)
    return results


def torch_profile(cfg: Config, init_steps=8000, steps=500, log_dir='./log'):
    """Generate a torch profiler trace for the simulation with `cfg`."""
    # need to define the import here to avoid circular imports
    from evolution.core.gworld import GWorld # pylint: disable=import-outside-toplevel

    os.makedirs(log_dir, exist_ok=True)
    world = GWorld(cfg)

    my_schedule = schedule(skip_first=init_steps, wait=1, warmup=1, active=23, repeat=0)

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, schedule=my_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                with_stack=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)  # pylint: disable=protected-access
                ) as prof:
        for _ in range(init_steps + steps):
            prof.step()
            world.step()
