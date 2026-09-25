"""Headless runner for the two companion notebooks; results use the same caches.

From the repository root:
  .venv/Scripts/python.exe chronos/bayesian/support_scripts/run_comparison.py --task all
  .venv/Scripts/python.exe chronos/bayesian/support_scripts/run_comparison.py --task all --smoke
"""
import argparse
from dataclasses import replace
from pathlib import Path

import torch

import chronos.support_scripts.comparison_lib as cl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['all', 'solar', 'recovery', 'benchmark'], default='all')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip-benchmark', action='store_true', help='Resume after an already recorded benchmark')
    parser.add_argument('--output', type=Path, default=Path(__file__).parents[1]/'_run/comparison')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    csv = Path(__file__).parents[2]/'data/dataset/Dataset-SolarTechLab.csv'
    recovery, solar = cl.RecoveryConfig(), cl.SolarConfig()
    models = cl.MODELS
    if args.smoke:
        recovery = replace(recovery, mode='smoke', n_backgrounds=4)
        solar = replace(solar, mode='smoke', n_origins=64, block_size=32)
        models = (cl.MODELS[0], cl.MODELS[4])
    print(f"Mode={'smoke' if args.smoke else 'full'}; torch threads={torch.get_num_threads()}; "
          f"checkpoint count={len(models)}", flush=True)
    if args.task in ('all', 'solar', 'benchmark') and not args.skip_benchmark:
        cl.benchmark_solar(csv, args.output/'benchmarks'/solar.mode, solar, models, args.device)
    if args.task in ('all', 'recovery'):
        for family, (table, path) in cl.run_recovery(args.output, recovery, models, args.device).items():
            print(family, path, '\n', table.to_string(index=False), flush=True)
    if args.task in ('all', 'solar'):
        table, path = cl.run_solar(csv, args.output, solar, models, args.device)
        print(path, '\n', table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
