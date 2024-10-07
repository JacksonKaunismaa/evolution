#!/usr/bin/env python3
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import torch
torch.set_grad_enabled(False)

from evolution.visual.main import main       # pylint: disable=wrong-import-position
from evolution.core import config  # pylint: disable=wrong-import-position
from evolution.core import benchmarking  # pylint: disable=wrong-import-position



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolution simulator')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--config', choices=['small', 'medium', 'large'], default='large',
                        help='Choose a predefined configuration')
    parser.add_argument('--load_path', type=str, default='game.ckpt',
                        help='Load path for simulation state (ignored if --style is set)')

    benchmark_opts = parser.add_argument_group('Benchmarking options')
    benchmark_opts.add_argument('--style', choices=['nsys', 'torch-profile', 'cuda-events', 'tune-hyperparams'],
                                default=None, help='Benchmarking style')
    benchmark_opts.add_argument('--num_simulations', type=int, default=1, help='Number of simulations to use in benchmarking')
    benchmark_opts.add_argument('--log_dir', type=str, default='./log', help='Log directory for torch-profile')
    benchmark_opts.add_argument('--init_steps', type=int, default=8000, help='Steps to run simulation for before benchmarking')
    benchmark_opts.add_argument('--steps', type=int, default=100, help='Steps to run simulation for during benchmarking')
    args = parser.parse_args()

    match args.config:
        case 'small':
            cfg = config.Config(start_creatures=3, max_creatures=100, size=5, food_cover_decr=0.0,
                    init_size_range=(0.2, 0.2), num_rays=3, immortal=True, init_food_scale=15.)

        case 'medium':
            cfg = config.Config()

        case 'large':
            cfg = config.Config(start_creatures=8192, max_creatures=262144, size=5000, food_cover_decr=0.2,
                                cell_size=16.0, max_per_cell=40,
                                food_sight=4, size_range=(0.1, 5.0),
                                brain_size=(50, 40))

    if args.seed >= 0:
        torch.random.manual_seed(args.seed)

    match args.style:
        case None:
            main(cfg, args.load_path)  # Run simulation with visualization window

        # Run simulation for benchmarking/debugging/testing/hyperparameter tuning
        case 'nsys':
            benchmarking.multi_benchmark(cfg, init_steps=args.init_steps, steps=args.steps,
                                            num_simulations=args.num_simulations,
                                            method=benchmarking.BenchmarkMethods.NSYS)
        case 'torch-profile':
            benchmarking.torch_profile(cfg, init_steps=args.init_steps, steps=args.steps, log_dir=args.log_dir)

        case 'cuda-events':
            benchmarking.multi_benchmark(cfg, init_steps=args.init_steps, steps=args.steps,
                                            num_simulations=args.num_simulations,
                                            method=benchmarking.BenchmarkMethods.CUDA_EVENTS)

        case 'tune-hyperparams':
            benchmarking.hyperparameter_search(cfg, {'cell_size': [15.0, 16.0, 17.0],
                                                     'max_per_cell': [20, 40, 60, 80, 90, 100]},
                                                    init_steps=args.init_steps, steps=args.steps,
                                                    num_simulations=args.num_simulations, skip_errors=True)
