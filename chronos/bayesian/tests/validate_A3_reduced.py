"""Run the standalone reduced model's recovery with its actual per-chain checkpoints.

All outputs are synthetic. --smoke tests software and resume only; it cannot pass a gate.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BAYES = ROOT / 'chronos/bayesian'
NOTEBOOK = BAYES / 'model_A3_reduced_localisation_standalone_colab.ipynb'
os.environ.setdefault('PYTENSOR_FLAGS', 'base_compiledir=' + str(ROOT/'tmp/pytensor_a3_reduced').replace('\\', '/'))
sys.path.insert(0, str(BAYES))


def namespace(output, draws=2000, tune=1500, chains=4, compile_mode='NUMBA'):
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    import xarray as xr
    import checkpointing as cp
    from scipy.special import expit, ndtr, logit, gammaln
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    g = dict(__name__='reduced_validation', np=np, pd=pd, pm=pm, az=az, xr=xr, cp=cp,
             hashlib=hashlib, json=json, time=time, Path=Path, gc=gc,
             expit=expit, ndtr=ndtr, logit=logit, gammaln=gammaln,
             PRIOR_SCALE=.5, BASELINE_SCALE=1.5, NU=4, LOCAL_SCALE=1., SEED=42,
             REFERENCE_HZ=64., DRAWS=draws, TUNE=tune, CHAINS=chains,
             TARGET_ACCEPT=.95, NUTS_INIT='jitter+adapt_diag',
             COMPILE_MODE=compile_mode,
             MODEL_VERSION='A3-reduced-lock-side-v1', PROGRESS_EVERY=100,
             PPC_DRAWS=800, EFFECT_DRAWS=2000, RHAT_MAX=1.01,
             SUPPORT_LOG_OR=float(np.log(.8)), ROPE_LOG_OR=float(np.log(1.1)), PROB_CUTOFF=.95,
             OUTPUT_ROOT=output, CHECKPOINT_ROOT=output/'checkpoints', MANIFEST_PATH=output/'manifest.json')
    for i in (8, 12, 14):
        exec(compile(''.join(nb['cells'][i]['source']), f'reduced_cell_{i}', 'exec'), g)
    # Hash all executed helper code and runtime versions, independently from notebook outputs.
    spec = dict(source=hashlib.sha256(''.join(''.join(nb['cells'][i]['source']) for i in (8,12,14)).encode()).hexdigest(),
                pymc=pm.__version__, arviz=az.__version__, draws=draws, tune=tune, chains=chains, compile_mode=compile_mode,
                status='SYNTHETIC_ONLY')
    g['ANALYSIS_FINGERPRINT'] = cp.fingerprint(spec)
    output.mkdir(parents=True, exist_ok=True)
    g['CHECKPOINT_ROOT'].mkdir(exist_ok=True)
    if g['MANIFEST_PATH'].exists():
        g['read_manifest']()
    else:
        cp.atomic_json(g['MANIFEST_PATH'], dict(analysis_fingerprint=g['ANALYSIS_FINGERPRINT'],
                                              spec=spec, artifacts={}))
    return g


def validate(g, idata, frame, truth, ess_min):
    diagnostic, diagnostic_table = g['diagnostics_for'](idata, ess_min)
    table, families = g['recovery_summary'](idata, truth)
    ppc = g['predictive_check'](idata, frame)
    fine = g['fine_predictive_check'](idata, frame)
    coverage = ppc.groupby('check_set').passed.mean().to_dict()
    coverage['geometry_frequency_role'] = float(fine.passed.mean())
    headline = bool(table.set_index('parameter').loc['gamma_design', 'covered'])
    for filename, value in (('diagnostics', diagnostic_table.reset_index(names='parameter')),
                            ('recovery_summary', table), ('recovery_families', families),
                            ('ppc', ppc), ('fine_ppc', fine)):
        g['save_table'](g['OUTPUT_ROOT']/f'{filename}.parquet', value)
    result = dict(status='SYNTHETIC_ONLY', diagnostic=diagnostic,
                  groups=len(frame), trials=int(frame.n_trials.sum()),
                  background_levels=len(g['_codes'](frame.generator+'#'+frame.bg_id.astype(str))[1]),
                  ess_threshold=ess_min, recovery_coverage=float(table.covered.mean()),
                  headline_covered=headline, ppc_coverage=coverage,
                  passed=bool(diagnostic['passed'] and table.covered.mean() >= .8
                              and headline and all(v >= .9 for v in coverage.values())))
    finite = bool(g['np'].isfinite(diagnostic_table.to_numpy(float)).all())
    result['pilot_convergence_passed'] = bool(finite and diagnostic['max_rhat'] < 1.01
                                              and diagnostic['min_ess_bulk'] > 400
                                              and diagnostic['min_ess_tail'] > 400
                                              and diagnostic['divergences'] == 0)
    result['pilot_recovery_passed'] = bool(result['pilot_convergence_passed']
                                          and result['recovery_coverage'] >= .8
                                          and headline and all(v >= .9 for v in coverage.values()))
    result['posterior_sizes'] = dict(idata.posterior.sizes)
    g['cp'].atomic_json(g['OUTPUT_ROOT']/'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print(diagnostic_table.sort_values('ess_bulk').head(10).to_string(), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenario', choices=['moderate', 'sparse'], default='sparse')
    p.add_argument('--draws', type=int, default=2000)
    p.add_argument('--tune', type=int, default=1500)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--ess-min', type=int, default=1000)
    p.add_argument('--posterior', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--compile-mode', choices=['NUMBA', 'native'], default='NUMBA')
    p.add_argument('--early-reject', action='store_true',
                   help='Stop a synthetic screen after 200 retained draws if >=5 divergences occur; never accept early.')
    args = p.parse_args()
    # Load numerical libraries before threadpoolctl discovers their thread pools.
    import numpy
    import scipy.linalg
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        g = namespace(args.output.resolve(), 8 if args.smoke else args.draws,
                      15 if args.smoke else args.tune, 2 if args.smoke else args.chains,
                      None if args.compile_mode == 'native' else args.compile_mode)
        base = g['pd'].read_parquet(BAYES/'_run/full/model_A2_localisation_bernoulli_v2/data/A_counts.parquet')
        frame, truth = g['sparse_recovery_data' if args.scenario == 'sparse' else 'recovery_data'](base)
        if args.smoke:
            frame = frame[frame.model.isin(frame.model.unique()[:2]) & (frame.bg_id < 2)].copy()
            frame, truth = g['recovery_data'](frame)
        model = g['model_A3'](frame)
        point = model.initial_point()
        np = g['np']
        assert np.isfinite(model.compile_logp()(point))
        assert np.isfinite(model.compile_dlogp()(point)).all()
        assert 'r_site' not in model.named_vars and 'sigma_local_baseline' not in model.named_vars
        print('Preflight: finite logp/gradient; sampled dimension=', sum(np.size(v) for v in point.values()), flush=True)
        if args.smoke:
            # Independent expansion verifies exact Binomial aggregation at several parameters.
            parts = []
            for _, row in frame.iterrows():
                part = g['pd'].DataFrame([row.to_dict()] * int(row.n_trials))
                part['h'] = [1] * int(row.hits) + [0] * int(row.n_trials-row.hits)
                parts.append(part)
            trials = g['pd'].concat(parts, ignore_index=True)
            bern = g['model_A3'](trials, encoding='bernoulli')
            count_logp, bern_logp = model.compile_logp(), bern.compile_logp()
            constant = np.sum(g['gammaln'](frame.n_trials+1)-g['gammaln'](frame.hits+1)
                              -g['gammaln'](frame.n_trials-frame.hits+1))
            for shift in (-.6, 0., .4):
                test_point = {k: np.array(v, copy=True) for k, v in point.items()}
                test_point['gamma_config'] += shift
                np.testing.assert_allclose(count_logp(test_point)-bern_logp(test_point), constant, atol=1e-7)
            print('Exact Bernoulli/Binomial likelihood: PASS', flush=True)
            del bern, count_logp, bern_logp
        del model
        if args.posterior:
            idata = g['_load_idata'](args.posterior)
        else:
            sample = g['pm'].sample
            class RejectedScreen(RuntimeError):
                pass
            def screened_sample(*pos, **kwargs):
                callback = kwargs.get('callback')
                retained, divergences = 0, 0
                def progress(trace, draw):
                    nonlocal retained, divergences
                    if callback:
                        callback(trace, draw)
                    if not draw.tuning:
                        retained += 1
                        divergences += int(draw.stats[0].get('diverging', False))
                        if retained >= 200 and divergences >= 5:
                            failure = dict(status='SYNTHETIC_SCREEN_EARLY_REJECTION', passed=False,
                                           reason='Zero-divergence criterion already violated repeatedly',
                                           retained_in_interrupted_chain=retained,
                                           divergences_in_interrupted_chain=divergences,
                                           posterior_complete=False, rhat=None, ess=None)
                            g['cp'].atomic_json(g['OUTPUT_ROOT']/'early_rejection.json', failure)
                            print(json.dumps(failure, indent=2), flush=True)
                            raise RejectedScreen(failure['reason'])
                kwargs['callback'] = progress
                return sample(*pos, **kwargs)
            from unittest.mock import patch
            try:
                with patch.object(g['pm'], 'sample', screened_sample if args.early_reject else sample):
                    idata = g['fit_or_load']('synthetic_'+args.scenario+'_reduced', frame)
            except RejectedScreen:
                return 2
        if args.smoke:
            from unittest.mock import patch
            with patch.object(g['pm'], 'sample', side_effect=AssertionError('Cache invoked sampler')):
                cached = g['fit_or_load']('synthetic_'+args.scenario+'_reduced', frame)
            np.testing.assert_equal(idata.posterior.beta.values, cached.posterior.beta.values)
            # Delete only the combined test artifact to exercise chain-by-chain resume.
            combined = g['CHECKPOINT_ROOT']/('synthetic_'+args.scenario+'_reduced.nc')
            manifest = g['read_manifest']()
            manifest['artifacts'].pop(combined.relative_to(g['OUTPUT_ROOT']).as_posix())
            combined.unlink()
            g['cp'].atomic_json(g['MANIFEST_PATH'], manifest)
            with patch.object(g['pm'], 'sample', side_effect=AssertionError('Chain resume invoked sampler')):
                resumed = g['fit_or_load']('synthetic_'+args.scenario+'_reduced', frame)
            np.testing.assert_equal(idata.posterior.beta.values, resumed.posterior.beta.values)
            datasets = {name: g['_dataset'](idata, name) for name in g['_group_names'](idata)}
            datasets['posterior'] = datasets['posterior'].isel(draw=slice(0, 0))
            empty = g['_make_idata'](datasets)
            try:
                g['validate_posterior'](empty)
            except ValueError:
                pass
            else:
                raise AssertionError('Empty posterior was accepted')
            original_draws = g['DRAWS']
            g['DRAWS'] += 1
            try:
                g['fit_or_load']('synthetic_'+args.scenario+'_reduced', frame)
            except ValueError:
                pass
            else:
                raise AssertionError('Incompatible checkpoint settings were accepted')
            finally:
                g['DRAWS'] = original_draws
            validate(g, idata, frame, truth, args.ess_min)
            g['cp'].atomic_json(g['OUTPUT_ROOT']/'software_smoke.json', dict(
                status='SOFTWARE_SMOKE_PASS', reportable=False,
                checks=['finite_logp_gradient', 'bernoulli_binomial_equivalence',
                        'complete_posterior_reload', 'per_chain_resume_without_sampling',
                        'empty_posterior_rejected', 'incompatible_settings_rejected',
                        'recovery_and_all_ppc_sets_execute']))
            print('SOFTWARE_SMOKE_PASS; convergence is deliberately not expected.', flush=True)
            return 0
        result = validate(g, idata, frame, truth, args.ess_min)
        return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
