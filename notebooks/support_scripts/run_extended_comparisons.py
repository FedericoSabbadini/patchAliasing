"""Local entry point; multiprocessing-safe on Windows and Linux."""
import argparse
import os
from pathlib import Path
from dataclasses import replace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['solar', 'sweep', 'chronos2', 'benchmark'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(8)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if __package__:
        from . import comparison_lib as c
    else:
        import comparison_lib as c
    if __package__:
        from . import solar_benchmark_lib as sb
    else:
        import solar_benchmark_lib as sb
    if __package__:
        from . import complex_sweep_lib as sw
    else:
        import complex_sweep_lib as sw
    if __package__:
        from . import solar_chronos2_lib as s2
    else:
        import solar_chronos2_lib as s2
    bayes = Path(__file__).resolve().parents[1]
    root = bayes/'_run/comparison'
    csv = bayes/'data/Dataset-SolarTechLab.csv'
    if args.task in ('solar','benchmark'):
        cfg, models = sb.SolarBenchmarkConfig(), c.MODELS
        if args.smoke:
            cfg = replace(cfg, mode='smoke', n_draws=8, batch_size=4, workers=2)
            models = (c.MODELS[0], c.MODELS[4])
        if args.task == 'benchmark':
            print(sb.benchmark_cpu(csv, root/'solar_benchmark_timing', cfg, models).to_string(index=False))
        else:
            frames, out = sb.run_benchmark(csv, root, cfg, models)
            sb.plot_results(frames, out)
            print(frames['summary'].to_string(index=False))
            print(out)
    elif args.task == 'sweep':
        cfg = sw.SweepConfig()
        if args.smoke:
            cfg = replace(cfg, mode='smoke', n_backgrounds=2, frequency_step=32,
                          n_phase_probe=2, n_phase_rollout=1, rollout_length=64)
        for family in ('tsmixup','kernelsynth'):
            outputs, out = sw.run_family(family, root, cfg)
            sw.plot_family(family, outputs, out)
            print(out)
    else:
        cfg = s2.Chronos2Config()
        if args.smoke:
            cfg = replace(cfg, mode='smoke', n_draws=4, batch_size=2)
        table, pred, truth, windows, out = s2.run_comparison(csv, root, cfg)
        s2.plot_comparison(table, pred, truth, windows, out)
        print(table.to_string(index=False))
        print(out)
    plt.close('all')


if __name__ == '__main__':
    main()
