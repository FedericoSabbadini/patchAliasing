"""Controlled synthetic comparison: change only three hierarchical coordinates.

One chain per variant diagnoses geometry; it cannot establish convergence or reportability.
"""
import ast
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from validate_A3_reduced import namespace, BAYES

OUT = BAYES.parents[1] / 'output/a3_sparse_coordinate_diagnosis_v1'


def ncp_source(source):
    replacements = {
        "beta = pm.Normal('beta', mu=beta_mu, sigma=tau, dims='config')":
        "z_beta = pm.Normal('z_beta', 0, 1, dims='config')\n        beta = pm.Deterministic('beta', beta_mu + tau*z_beta, dims='config')",
        "gamma = pm.Normal('gamma_config', mu=gamma_bar, sigma=sigma_gamma, dims='config')":
        "z_gamma = pm.Normal('z_gamma', 0, 1, dims='config')\n        gamma = pm.Deterministic('gamma_config', gamma_bar + sigma_gamma*z_gamma, dims='config')",
        "kappa = pm.Normal('kappa_config', mu=kappa_bar, sigma=sigma_kappa, dims='config')":
        "z_kappa = pm.Normal('z_kappa', 0, 1, dims='config')\n        kappa = pm.Deterministic('kappa_config', kappa_bar + sigma_kappa*z_kappa, dims='config')",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1
        source = source.replace(old, new)
    ast.parse(source)
    return source


def main():
    g = namespace(OUT, draws=500, tune=1500, chains=1)
    import pymc as pm
    import pandas as pd
    import scipy.linalg
    nb = json.loads((BAYES/'model_A3_reduced_localisation_standalone_colab.ipynb').read_text(encoding='utf8'))
    src = ''.join(nb['cells'][12]['source'])
    ncp = dict(g)
    exec(compile(ncp_source(src), 'ncp_model', 'exec'), ncp)
    base = pd.read_parquet(BAYES/'_run/full/model_A2_localisation_bernoulli_v2/data/A_counts.parquet')
    frame, truth = g['sparse_recovery_data'](base[base.bg_id < 3])
    frame.to_parquet(OUT/'synthetic_counts.parquet', index=False)
    truth.to_csv(OUT/'truth.csv', index=False)
    models = {name: env['model_A3'](frame) for name, env in [('centered',g), ('ncp_config',ncp)]}
    # Exact density equality including the coordinate-change Jacobian at three points.
    old, new = models['centered'], models['ncp_config']
    oldlogp, newlogp = old.compile_logp(), new.compile_logp()
    _, configs = g['_codes'](frame.model)
    for shift in [-.5, 0., .7]:
        p = old.initial_point()
        for key in ['tau_log__','sigma_gamma_log__','sigma_kappa_log__']:
            p[key] = p[key] + shift
        q = {k: v for k,v in p.items() if k not in ['beta','gamma_config','kappa_config']}
        mu = p['beta_bar'] + p['delta_O']*g['_overlap_scaled'](frame,configs) + p['delta_P']*g['_logP_centred'](frame,configs)
        q['z_beta'] = (p['beta']-mu)/np.exp(p['tau_log__'])
        q['z_gamma'] = (p['gamma_config']-p['gamma_bar'])/np.exp(p['sigma_gamma_log__'])
        q['z_kappa'] = (p['kappa_config']-p['kappa_bar'])/np.exp(p['sigma_kappa_log__'])
        jac = len(configs)*sum(p[k] for k in ['tau_log__','sigma_gamma_log__','sigma_kappa_log__'])
        np.testing.assert_allclose(newlogp(q), oldlogp(p)+jac, atol=1e-8)
    print('Exact joint density + Jacobian parity: PASS', flush=True)
    results = []
    for name in ['ncp_config','centered']:
        start = time.monotonic()
        def progress(trace, draw):
            if (draw.draw_idx+1) % 200 == 0:
                print(name, draw.draw_idx+1, 'warmup' if draw.tuning else 'posterior',
                      round((time.monotonic()-start)/60,1), 'min', flush=True)
        with models[name]:
            idata = pm.sample(draws=500, tune=1500, chains=1, cores=1,
                              random_seed=3351273248, target_accept=.95,
                              init='jitter+adapt_diag', nuts_sampler='pymc',
                              compile_kwargs={'mode':'NUMBA'}, callback=progress,
                              progressbar=False, compute_convergence_checks=False)
        g['cp'].atomic_netcdf(OUT/(name+'.nc'),idata)
        health = g['sampler_health_for'](idata)
        health.to_csv(OUT/(name+'_health.csv'),index=False)
        div = np.asarray(idata.sample_stats.diverging).ravel().astype(bool)
        scale_rows = []
        for var in ['tau','sigma_gamma','sigma_kappa','sigma_harm','sigma_bg']:
            values = np.asarray(idata.posterior[var]).ravel()
            for flag in [False,True]:
                x = values[div==flag]
                scale_rows.append(dict(variable=var,divergent=flag,n=len(x),
                                       median=float(np.median(x)) if len(x) else None))
        pd.DataFrame(scale_rows).to_csv(OUT/(name+'_divergence_scales.csv'),index=False)
        row = dict(variant=name,seconds=time.monotonic()-start,divergences=int(div.sum()),
                   retained=500,chains=1,reportable=False,health=health.to_dict('records'))
        results.append(row)
        (OUT/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    # Imports inside namespace initialise BLAS pools, so keep the limit active afterwards too.
    import scipy.linalg
    with threadpool_limits(limits=1):
        main()
