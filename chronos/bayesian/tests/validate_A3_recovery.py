"""Run A3's fixed synthetic recovery at the complete PILOT design dimension.

This is synthetic inference validation, never an empirical result. It reads only
the design and trial counts from an existing A_counts.parquet, discards its hits,
and uses the notebook's unchanged recovery generator and per-chain checkpoints.
Run from the repository root with --scenario moderate, then --scenario sparse.
"""
import argparse
import ast
import hashlib
import json
import runpy
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['moderate', 'sparse'], required=True)
    parser.add_argument('--counts', type=Path, default=ROOT / 'chronos/bayesian/_run/full/model_A2_localisation_bernoulli_v2/data/A_counts.parquet')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-minutes', type=float, default=30.)
    args = parser.parse_args()
    helpers = runpy.run_path(str(Path(__file__).with_name('test_model_A3_notebook.py')))
    nb, g = helpers['namespace']()
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    from threadpoolctl import threadpool_limits

    run = args.output or ROOT / f'output/a3_validation/{args.scenario}_diag'
    run.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.counts)
    frame = frame.loc[frame.bg_id < 3].copy().reset_index(drop=True).assign(hits=0)
    generator = g['recovery_data'] if args.scenario == 'moderate' else g['sparse_recovery_data']
    frame, truth = generator(frame)
    design = g['site_design'](frame)
    recovery_function = next(n for n in ast.parse(nb.cells[14].source).body
                             if isinstance(n, ast.FunctionDef) and n.name == '_a3_recovery')
    specification = dict(kind='SYNTHETIC_ONLY', scenario=args.scenario,
                         model=g['MODEL_VERSION'], init=g['NUTS_INIT'], seed=g['SEED'],
                         tune=1500, draws=2000, chains=4, target_accept=g['TARGET_ACCEPT'],
                         groups=len(frame), trials=int(frame.n_trials.sum()),
                         sites=len(design['sites']), contrasts=len(design['contrasts']),
                         source_counts_sha256=hashlib.sha256(args.counts.read_bytes()).hexdigest(),
                         model_code_sha256=hashlib.sha256(nb.cells[12].source.encode()).hexdigest(),
                         recovery_code_sha256=hashlib.sha256(ast.dump(recovery_function).encode()).hexdigest(),
                         synthetic_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest(),
                         pymc=pm.__version__, arviz=az.__version__)
    g.update(OUTPUT_ROOT=run, CHECKPOINT_ROOT=run/'checkpoints',
             MANIFEST_PATH=run/'analysis_manifest.json', CHAINS=4, DRAWS=2000, TUNE=1500,
             PROGRESS_EVERY=200, ANALYSIS_FINGERPRINT=g['cp'].fingerprint(specification))
    g['CHECKPOINT_ROOT'].mkdir(exist_ok=True)
    if g['MANIFEST_PATH'].exists():
        previous = json.loads(g['MANIFEST_PATH'].read_text(encoding='utf-8'))
        if previous['analysis_fingerprint'] != g['ANALYSIS_FINGERPRINT']:
            raise ValueError('Validation settings changed; use a new output directory.')
    else:
        g['cp'].atomic_json(g['MANIFEST_PATH'], dict(analysis_fingerprint=g['ANALYSIS_FINGERPRINT'], artifacts={}))
    frame.to_parquet(run/'synthetic_counts.parquet', index=False)
    truth.to_csv(run/'truth.csv', index=False)
    with threadpool_limits(limits=1):
        model = g['model_A3'](frame)
        specification['sampled_dimension'] = sum(np.size(v) for v in model.initial_point().values())
    assert (len(frame), int(frame.n_trials.sum()), specification['sampled_dimension']) == (3690, 33390, 663)
    g['cp'].atomic_json(run/'preflight.json', specification)
    print(json.dumps(specification), flush=True)
    print('Synthetic hit rate:', frame.hits.sum()/frame.n_trials.sum(), flush=True)
    started = time.monotonic()
    original_sample = pm.sample

    def bounded_sample(**kwargs):
        callback = kwargs['callback']
        def bounded_callback(trace, draw):
            callback(trace, draw)
            if time.monotonic() - started > args.max_minutes*60:
                raise RuntimeError('Validation time limit reached; completed chains remain saved.')
        return original_sample(**{**kwargs, 'callback': bounded_callback})

    with threadpool_limits(limits=1), patch.object(pm, 'sample', bounded_sample):
        fitted = g['fit_or_load'](f'synthetic_{args.scenario}_recovery_A3', frame)
    diagnostics, table = g['diagnostics_for'](fitted, 400)
    summary, families = g['recovery_summary'](fitted, truth)
    table.to_csv(run/'parameter_diagnostics.csv')
    summary.to_csv(run/'recovery_summary.csv', index=False)
    families.to_csv(run/'recovery_families.csv', index=False)
    stats = g['_dataset'](fitted, 'sample_stats')
    energy = np.asarray(stats.energy.transpose('chain', 'draw'), float)
    bfmi = np.mean(np.diff(energy, axis=1)**2, axis=1)/np.var(energy, axis=1, ddof=1)
    chain_rows = []
    for c in range(4):
        row = dict(chain=c, bfmi=float(bfmi[c]), divergences=int(np.asarray(stats.diverging)[c].sum()))
        for name in ('acceptance_rate', 'step_size_bar', 'tree_depth', 'n_steps'):
            if name in stats:
                values = np.asarray(stats[name].isel(chain=c), float)
                row[name+'_median'] = float(np.median(values))
                row[name+'_max'] = float(np.max(values))
        chain_rows.append(row)
    pd.DataFrame(chain_rows).to_csv(run/'sampler_health.csv', index=False)
    headline_covered = bool(summary.set_index('parameter').loc['gamma_design', 'covered'])
    result = dict(**specification, diagnostics=diagnostics,
                  coverage=float(summary.covered.mean()), headline_covered=headline_covered,
                  passed=bool(diagnostics['passed'] and headline_covered and summary.covered.mean() >= .8),
                  elapsed_seconds=time.monotonic()-started, sampler_health=chain_rows)
    g['cp'].atomic_json(run/'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print(table.sort_values('ess_bulk').head(12).to_string(), flush=True)
    print(summary[summary.variable.isin(['gamma_design', 'sigma_local_baseline', 'sigma_local_lock', 'sigma_local_side'])].to_string(index=False), flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
