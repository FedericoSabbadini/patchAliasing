"""Compare compiled evaluators for the identical reduced log density and gradient."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits
from validate_A3_reduced import namespace, BAYES, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'output/a3_reduced_backend_benchmark_v2')
    output = parser.parse_args().output.resolve()
    with threadpool_limits(limits=1):
        g = namespace(output)
        base = g['pd'].read_parquet(BAYES/'_run/full/model_A2_localisation_bernoulli_v2/data/A_counts.parquet')
        frame, _ = g['sparse_recovery_data'](base)
        model = g['model_A3'](frame)
        point = model.initial_point()
        rng = np.random.default_rng(7042)
        points = [point] + [{k: v+rng.normal(0, .025, np.shape(v)) for k, v in point.items()} for _ in range(2)]
        results, times = {}, {}
        for mode in (None, 'NUMBA'):
            print('Compiling evaluator:', mode, flush=True)
            logp = model.compile_logp(mode=mode)
            grad = model.compile_dlogp(mode=mode)
            results[str(mode)] = [(float(logp(p)), grad(p)) for p in points]
            timings = []
            for _ in range(12):
                start = time.perf_counter()
                logp(point)
                grad(point)
                timings.append(time.perf_counter()-start)
            times[str(mode)] = float(np.median(timings))
            print('Median logp+gradient seconds:', times[str(mode)], flush=True)
        for reference, candidate in zip(results['None'], results['NUMBA']):
            np.testing.assert_allclose(reference[0], candidate[0], rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(reference[1], candidate[1], rtol=1e-8, atol=1e-8)
        result = dict(status='EVALUATOR_PARITY_PASS_NOT_MCMC', points=3, seconds=times,
                      speedup=times['None']/times['NUMBA'])
        (output/'benchmark.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
