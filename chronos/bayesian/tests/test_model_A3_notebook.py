"""A3 notebook contract and bounded numerical checks; no empirical inference.

Run from the repository root:
  .venv/Scripts/python.exe chronos/bayesian/tests/test_model_A3_notebook.py -v
Use --smoke with this file directly for tiny actual NUTS fits and reporting cells.
"""
import ast
import gc
import hashlib
import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('PYTENSOR_FLAGS', 'base_compiledir='+str(ROOT/'tmp/pytensor_a3_validation').replace('\\', '/'))
sys.path.insert(0, str(ROOT/'chronos/bayesian'))
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy.special import expit, ndtr, logit, gammaln
import checkpointing as cp

NOTEBOOK = ROOT/'chronos/bayesian/model_A3_localisation_standalone_colab.ipynb'


def namespace():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    g = dict(__name__='a3_validation', np=np, pd=pd, pm=pm, az=az, xr=xr, cp=cp,
             hashlib=hashlib, time=time, Path=Path, json=json, gc=gc, plt=plt,
             PRIOR_SCALE=.5, BASELINE_SCALE=1.5, NU=4, LOCAL_SCALE=1., SEED=42,
             REFERENCE_HZ=64., DRAWS=8, TUNE=15, CHAINS=4, TARGET_ACCEPT=.95,
             NUTS_INIT='jitter+adapt_diag', MODEL_VERSION='A3-local-baseline-lock-side-v1',
             ANALYSIS_FINGERPRINT='software-smoke-nonreportable', PROGRESS_EVERY=200,
             PPC_DRAWS=20, EFFECT_DRAWS=20, expit=expit, ndtr=ndtr, logit=logit, gammaln=gammaln,
             RHAT_MAX=1.01, SUPPORT_LOG_OR=float(np.log(.8)), ROPE_LOG_OR=float(np.log(1.1)), PROB_CUTOFF=.95)
    for i in (8, 12, 14):
        exec(compile(nb.cells[i].source, f'A3_cell_{i}', 'exec'), g)
    # Exercise the notebook's actual initialization choice, not a test-only default.
    for statement in ast.parse(nb.cells[6].source).body:
        if (isinstance(statement, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'NUTS_INIT' for t in statement.targets)):
            g['NUTS_INIT'] = ast.literal_eval(statement.value)
    return nb, g


def counts_fixture():
    rows = []
    for P, S in ((16, 16), (32, 16)):
        for f, n in ((32., 3), (64., 5), (96., 7)):
            for generator in ('tsmixup', 'kernelsynth'):
                for bg_id in (0, 1):
                    for role, side in (('lo', -1), ('lock', 0), ('hi', 1)):
                        rows.append(dict(model=f'p{P}-s{S}', P=P, S=S, overlap=(P-S)/P,
                                         f_lock=f, generator=generator, bg_id=bg_id, role=role,
                                         side=side, is_lock=int(role == 'lock'), n_trials=n,
                                         hits=(bg_id+side+int(f/32)) % (n+1)))
    return pd.DataFrame(rows)


def posterior_fixture(g, frame, draws=8):
    rng = np.random.default_rng(289)
    sd = g['site_design'](frame)
    cl, hl, bl = (g['_codes'](value)[1] for value in
                  (frame.model, frame.f_lock.round(3).astype(str), frame.generator+'#'+frame.bg_id.astype(str)))
    coords = dict(chain=[0], draw=np.arange(draws), config=cl, harmonic=hl, background=bl,
                  site=sd['sites'], site_contrast=sd['contrasts'], local_family=['baseline', 'lock', 'side'])
    values = {}
    for name in ('beta_bar','delta_O','delta_P','tau','gamma_bar','sigma_gamma','kappa_bar',
                 'sigma_kappa','sigma_harm','sigma_bg','sigma_local_baseline','sigma_local_lock','sigma_local_side'):
        values[name] = (('chain', 'draw'), rng.uniform(.2, .8, (1, draws)))
    for name, dim in (('beta','config'), ('gamma_config','config'), ('kappa_config','config'),
                      ('u_harm','harmonic'), ('u_bg','background')):
        values[name] = (('chain','draw',dim), rng.normal(size=(1, draws, len(coords[dim]))))
    gamma, kappa = values['gamma_config'][1], values['kappa_config'][1]
    z = rng.normal(size=(1, draws, 3, len(sd['contrasts'])))
    values['z_local'] = (('chain','draw','local_family','site_contrast'), z)
    for k, name in enumerate(('r_site','ell_site','s_site')):
        scale = values[('sigma_local_baseline','sigma_local_lock','sigma_local_side')[k]][1]
        values[name] = (('chain','draw','site'), (z[:,:,k] @ sd['basis'].T)*scale[...,None])
    values['gamma_site'] = (('chain','draw','site'), gamma[:,:,sd['site_ci']]+values['ell_site'][1])
    values['kappa_site'] = (('chain','draw','site'), kappa[:,:,sd['site_ci']]+values['s_site'][1])
    values['gamma_design'] = (('chain','draw'), gamma.mean(axis=-1))
    values['odds_ratio_design'] = (('chain','draw'), np.exp(gamma.mean(axis=-1)))
    posterior = xr.Dataset(values, coords=coords)
    stats = xr.Dataset(dict(diverging=(('chain','draw'),np.zeros((1,draws),bool)),
                            lp=(('chain','draw'),np.zeros((1,draws))),
                            energy=(('chain','draw'),np.ones((1,draws)))),
                       coords={k:coords[k] for k in ('chain','draw')})
    return g['_make_idata'](dict(posterior=posterior,sample_stats=stats))


class A3NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb, cls.g = namespace()
        cls.frame = counts_fixture()

    def test_notebook_contract(self):
        nbformat.validate(self.nb)
        for cell in self.nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        self.assertIn('RUN_MODE = "PILOT"', self.nb.cells[6].source)
        self.assertIn('ALLOW_NEW_FORECASTS = False', self.nb.cells[6].source)
        self.assertIn('"A2-varying-lock-and-side-v1"', self.nb.cells[10].source)
        self.assertIn('FULL requires a matching A3 PILOT PASS', self.nb.cells[16].source)
        self.assertIn('FINE_PPC_OK', self.nb.cells[32].source)
        self.assertIn('STRESS_RECOVERY_OK', self.nb.cells[32].source)
        self.assertEqual(self.g['NUTS_INIT'], 'jitter+adapt_diag')

    def test_diagonal_adaptation_stays_valid_at_full_pilot_dimension(self):
        from pymc.step_methods.hmc.quadpotential import QuadPotentialDiagAdapt
        with pm.Model() as model:
            pm.Normal('theta', shape=663)
            _, step = pm.init_nuts(init=self.g['NUTS_INIT'], chains=1, random_seed=981,
                                   progressbar=False)
        self.assertIsInstance(step.potential, QuadPotentialDiagAdapt)
        rng = np.random.default_rng(981)
        for _ in range(500):
            step.potential.update(rng.normal(size=663), np.zeros(663), tune=True)
            self.assertTrue(np.isfinite(step.potential._var).all())
            self.assertTrue((step.potential._var > 0).all())
            np.testing.assert_allclose(step.potential._stds**2, step.potential._var, rtol=1e-14)

    def test_sampler_health_matches_energy_definition(self):
        fixture = posterior_fixture(self.g, self.frame)
        datasets = {name:self.g['_dataset'](fixture, name).copy(deep=True)
                    for name in self.g['_group_names'](fixture)}
        energy = np.array([1., 3., 2., 4., 5., 2., 1., 3.])
        datasets['sample_stats']['energy'] = (('chain', 'draw'), energy[None,:])
        datasets['sample_stats']['acceptance_rate'] = (('chain', 'draw'), np.full((1,8), .97))
        datasets['sample_stats']['tree_depth'] = (('chain', 'draw'), np.arange(1,9)[None,:])
        health = self.g['sampler_health_for'](self.g['_make_idata'](datasets))
        self.assertAlmostEqual(health.bfmi.iloc[0], np.mean(np.diff(energy)**2)/np.var(energy,ddof=1))
        self.assertAlmostEqual(health.acceptance_rate_mean.iloc[0], .97)
        self.assertEqual(health.tree_depth_max.iloc[0], 8)
        self.assertEqual(health.retained_draws.iloc[0], 8)

    def test_failed_recovery_stops_before_later_fits(self):
        nb, g = namespace()
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            run = Path(folder)
            g.update(OUTPUT_ROOT=run, MANIFEST_PATH=run/'manifest.json', RUN_MODE='PILOT', IS_FULL=False,
                     CORE_FINGERPRINT='recovery-gate-test', groups=self.frame, PILOT_ESS_MIN=400,
                     RECOVERY_DRAWS=8, RECOVERY_TUNE=15, RECOVERY_MIN_COVERAGE=.8,
                     display=lambda x:None)
            cp.atomic_json(g['MANIFEST_PATH'], dict(analysis_fingerprint=g['ANALYSIS_FINGERPRINT'],artifacts={}))
            fixture = posterior_fixture(g, self.frame)
            diagnostic = dict(max_rhat=2., min_ess_bulk=4., min_ess_tail=5., divergences=0, passed=False)
            from unittest.mock import Mock
            fitter = Mock(return_value=fixture)
            g.update(fit_or_load=fitter, diagnostics_for=lambda *args:(diagnostic,pd.DataFrame({'ess_bulk':[4.]})))
            with self.assertRaisesRegex(RuntimeError, 'moderate recovery failed'):
                exec(compile(nb.cells[22].source, 'A3_recovery_gate', 'exec'), g)
            self.assertEqual(fitter.call_count, 1)
            verdict = json.loads((run/'final_verdict.json').read_text())
            self.assertFalse(verdict['pilot_route_ok'])
            self.assertEqual(verdict['stopped_at'], 'synthetic_moderate_recovery')
            with self.assertRaisesRegex(RuntimeError, 'recovery cell successfully'):
                exec(compile(nb.cells[24].source, 'A3_primary_gate', 'exec'), g)
            self.assertEqual(fitter.call_count, 1)
            # True flags left in memory from another run must not unlock this one.
            g.update(RECOVERY_OK=True, STRESS_RECOVERY_OK=True, RECOVERY_VALIDATED_FOR='another-run')
            with self.assertRaisesRegex(RuntimeError, 'recovery cell successfully'):
                exec(compile(nb.cells[24].source, 'A3_primary_gate', 'exec'), g)
            self.assertEqual(fitter.call_count, 1)

    def test_projection_has_exact_weighted_gaussian_prior(self):
        sd = self.g['site_design'](self.frame)
        for c in range(len(sd['configs'])):
            pos = np.flatnonzero(sd['site_ci'] == c)
            w, B = sd['weights'][pos], sd['basis'][pos]
            projector = np.eye(len(pos))-np.ones((len(pos),1))*w[None,:]
            np.testing.assert_allclose(w @ B, 0, atol=1e-14)
            np.testing.assert_allclose(B @ B.T, projector @ projector.T, atol=1e-14)
        changed = self.frame.assign(hits=0)
        np.testing.assert_array_equal(sd['basis'], self.g['site_design'](changed)['basis'])

    def test_bernoulli_counts_and_reference_joint_density(self):
        from pymc.model.transform.conditioning import remove_value_transforms
        frame, g = self.frame, self.g
        trials = frame.loc[frame.index.repeat(frame.n_trials)].copy().reset_index(drop=True)
        trials['h'] = np.concatenate([np.r_[np.ones(r.hits),np.zeros(r.n_trials-r.hits)] for r in frame.itertuples()])
        np.testing.assert_allclose(g['site_design'](frame)['basis'], g['site_design'](trials, 'bernoulli')['basis'])
        cm, bm = g['model_A3'](frame), g['model_A3'](trials, encoding='bernoulli')
        clp, blp = cm.compile_logp(), bm.compile_logp()
        constant = np.sum(gammaln(frame.n_trials+1)-gammaln(frame.hits+1)-gammaln(frame.n_trials-frame.hits+1))
        for shift in (-.6, 0., .4):
            point = cm.initial_point()
            point['gamma_config'] += shift
            point['z_local'] += np.arange(point['z_local'].size).reshape(point['z_local'].shape)*.03
            np.testing.assert_allclose(clp(point)-blp(point), constant, atol=1e-7, rtol=1e-9)
        # Compare the implemented likelihood with the declared linear predictor.
        named = [cm[name] for name in ('beta','u_harm','u_bg','r_site','gamma_site','kappa_site')]
        evaluate = cm.compile_fn(cm.replace_rvs_by_values(named), inputs=cm.value_vars,
                                 on_unused_input='ignore', point_fn=True)
        beta, harm, bg, r, gamma, kappa = evaluate(point)
        ci = g['_codes'](frame.model)[0]
        hi = g['_codes'](frame.f_lock.round(3).astype(str))[0]
        bi = g['_codes'](frame.generator+'#'+frame.bg_id.astype(str))[0]
        si = g['site_design'](frame)['si']
        eta = beta[ci]+harm[hi]+bg[bi]+r[si]+gamma[si]*frame.is_lock.to_numpy()+kappa[si]*frame.side.to_numpy()
        expected = np.sum(gammaln(frame.n_trials+1)-gammaln(frame.hits+1)-gammaln(frame.n_trials-frame.hits+1)
                          -frame.hits*np.logaddexp(0,-eta)-(frame.n_trials-frame.hits)*np.logaddexp(0,eta))
        np.testing.assert_allclose(cm.compile_logp(vars=cm.observed_RVs)(point), expected, atol=1e-8)
        self.assertTrue(np.isfinite(cm.compile_dlogp()(cm.initial_point())).all())
        generative = remove_value_transforms(g['model_A3_generative'](frame))
        reference = g['model_A3'](frame)
        reference.rvs_to_initial_values = {rv:None for rv in reference.rvs_to_initial_values}
        reference = remove_value_transforms(reference)
        old_lp, new_lp = generative.compile_logp(), reference.compile_logp()
        rng = np.random.default_rng(311)
        for _ in range(3):
            p = generative.initial_point()
            for name in ('beta','gamma_config','kappa_config','z_local'):
                p[name] = rng.normal(size=p[name].shape)
            for name in ('u_harm','u_bg'):
                p[name] = rng.normal(size=p[name].shape); p[name] -= p[name].mean()
            q = {k:np.array(p[k], copy=True) for k in reference.initial_point() if k in p}
            ref = list(generative.coords['harmonic']).index('64.0')
            q['frequency_difference'] = np.delete(p['u_harm']-p['u_harm'][ref], ref)
            q['baseline_at_reference'] = p['beta']+p['u_harm'][ref]
            np.testing.assert_allclose(new_lp(q)-old_lp(p)+.5*np.log(len(generative.coords['harmonic'])), 0, atol=1e-7)
        probit = g['model_A3'](frame, link='probit', scale=.5/1.6, baseline_scale=1.5/1.6, local_scale=1./1.6)
        self.assertTrue(np.isfinite(probit.compile_logp()(probit.initial_point())))

    def test_predictors_and_ppc_match_independent_groupby(self):
        g = self.g
        frame = self.frame.sample(frac=1., random_state=190).reset_index(drop=True)
        idata = posterior_fixture(g, frame)
        # Reference calculations independently select labeled posterior coordinates.
        post = idata.posterior
        predictions = []
        for draw in range(8):
            eta = []
            for row in frame.itertuples():
                site = f'{row.model}@{round(row.f_lock,3)}'
                at = dict(chain=0, draw=draw)
                value = (post.beta.sel(**at, config=row.model)+post.u_harm.sel(**at, harmonic=str(round(row.f_lock,3)))
                         +post.u_bg.sel(**at, background=f'{row.generator}#{row.bg_id}')
                         +post.r_site.sel(**at, site=site)+post.gamma_site.sel(**at, site=site)*row.is_lock
                         +post.kappa_site.sel(**at, site=site)*row.side)
                eta.append(float(value))
            predictions.append(expit(eta))
        rng = np.random.default_rng(1642)
        keys = ['model','f_lock','role']
        expected = []
        for p in predictions:
            simulated = frame.assign(hits=rng.binomial(frame.n_trials.to_numpy(),p))
            agg = simulated.groupby(keys, sort=True)[['hits','n_trials']].sum()
            expected.append((agg.hits/agg.n_trials).to_numpy())
        expected_intervals = np.quantile(expected, [.025,.975], axis=0)
        actual = g['fine_predictive_check'](idata, frame)
        np.testing.assert_allclose(actual[['rep_low','rep_high']].to_numpy().T, expected_intervals, atol=1e-14)
        observed = frame.groupby(keys, sort=True)[['hits','n_trials']].sum()
        np.testing.assert_allclose(actual.observed_rate, observed.hits/observed.n_trials)
        self.assertEqual(actual.n_trials.sum(), frame.n_trials.sum())
        site_table = g['site_effect_table'](idata, frame)
        self.assertEqual(len(site_table), 6)
        effects = g['comparable_effects'](idata, frame, link='probit')
        self.assertEqual(effects.shape, (8,2))
        self.assertTrue(np.isfinite(effects).all())
        sd = g['site_design'](frame)
        for c in range(2):
            pos = np.flatnonzero(sd['site_ci'] == c)
            local = np.asarray(post.gamma_site)[0,:,pos].T
            np.testing.assert_allclose(local @ sd['weights'][pos], np.asarray(post.gamma_config)[0,:,c], atol=1e-13)

    def test_recovery_includes_all_new_scientific_effects(self):
        for generator in (self.g['recovery_data'], self.g['sparse_recovery_data']):
            frame, truth = generator(self.frame)
            repeat, repeat_truth = generator(self.frame.assign(hits=0))
            pd.testing.assert_frame_equal(frame[['hits']], repeat[['hits']])
            pd.testing.assert_frame_equal(truth, repeat_truth)
            self.assertTrue({'r_site','ell_site','s_site','gamma_site','kappa_site','sigma_local_lock'}.issubset(set(truth.variable)))
            self.assertTrue((frame.hits.between(0,frame.n_trials)).all())
            self.assertGreater(truth[truth.variable.eq('ell_site')].truth.abs().max(), .1)

    def test_full_registered_design_shape_when_local_fixture_exists(self):
        path = ROOT/'chronos/bayesian/_run/full/model_A2_localisation_bernoulli_v2/data/A_counts.parquet'
        if not path.is_file():
            self.skipTest('Optional local design fixture is unavailable')
        frame = pd.read_parquet(path)
        frame = frame[frame.bg_id < 3].copy().reset_index(drop=True)
        sd = self.g['site_design'](frame)
        self.assertEqual(len(sd['sites']),205)
        self.assertEqual(3*len(sd['contrasts']),570)
        fixture = posterior_fixture(self.g,frame)
        fine = self.g['fine_predictive_check'](fixture,frame)
        self.assertEqual(len(fine),615)
        self.assertEqual(fine.n_trials.sum(),33390)
        # Outcomes in this historical table are used only to verify partitions.

    def test_checkpoint_interruption_datatree_and_empty_rejection(self):
        _, g = namespace()
        fixture = posterior_fixture(g, self.frame)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            root = Path(folder)
            g.update(OUTPUT_ROOT=root, CHECKPOINT_ROOT=root/'checkpoints', MANIFEST_PATH=root/'analysis_manifest.json')
            g['CHECKPOINT_ROOT'].mkdir()
            cp.atomic_json(g['MANIFEST_PATH'], dict(analysis_fingerprint=g['ANALYSIS_FINGERPRINT'], artifacts={}))
            # Exercise the ArviZ 1 constructor path without relying on hasattr's alias.
            class Arviz1Shim:
                def __getattr__(self, name):
                    if name == 'InferenceData':
                        return xr.DataTree
                    raise AttributeError(name)
            original_az = g['az']; g['az'] = Arviz1Shim()
            tree = g['_make_idata'](dict(posterior=fixture.posterior, sample_stats=fixture.sample_stats), {'a3_test':'yes'})
            self.assertIsInstance(tree, xr.DataTree)
            cp.atomic_netcdf(root/'tree.nc', tree)
            loaded = g['_load_idata'](root/'tree.nc')
            self.assertEqual(loaded.attrs['a3_test'], 'yes')
            g['validate_posterior'](loaded,8,1)
            g['az'] = original_az
            calls = []
            def sample(**kwargs):
                calls.append(kwargs['random_seed'])
                if len(calls) == 3:
                    raise KeyboardInterrupt('deliberate checkpoint recovery test')
                return g['_make_idata'](dict(posterior=fixture.posterior.copy(deep=True), sample_stats=fixture.sample_stats.copy(deep=True)))
            with patch.object(pm, 'sample', sample):
                with self.assertRaises(KeyboardInterrupt):
                    g['fit_or_load']('resume', self.frame)
                self.assertEqual(len(list(g['CHECKPOINT_ROOT'].glob('resume.chain_*.nc'))),2)
                combined = g['fit_or_load']('resume', self.frame)
                g['validate_posterior'](combined,8,4)
                g['fit_or_load']('resume', self.frame)
                self.assertEqual(len(calls),5)
                self.assertEqual(calls[2],calls[3])
                with self.assertRaises(ValueError):
                    g['fit_or_load']('resume', self.frame, draws=9)
                with self.assertRaises(ValueError):
                    g['fit_or_load']('resume', self.frame, local_scale=2.)
                g['NUTS_INIT'] = 'adapt_diag'
                with self.assertRaisesRegex(ValueError, 'settings/completeness differ'):
                    g['fit_or_load']('resume', self.frame)
                g['NUTS_INIT'] = 'jitter+adapt_full'
                with self.assertRaisesRegex(ValueError, 'diagonal adaptation'):
                    g['fit_or_load']('resume', self.frame)
                g['NUTS_INIT'] = 'jitter+adapt_diag'
            empty = g['_make_idata'](dict(posterior=fixture.posterior.isel(draw=slice(0,0)), sample_stats=fixture.sample_stats.isel(draw=slice(0,0))))
            with patch.object(pm, 'sample', return_value=empty):
                with self.assertRaises(RuntimeError):
                    g['fit_or_load']('empty', self.frame)
            self.assertFalse((g['CHECKPOINT_ROOT']/'empty.nc').exists())
            self.assertEqual(len(list((g['CHECKPOINT_ROOT']/'partial').glob('empty*.nc'))),1)
            for bad in (fixture.posterior.isel(draw=slice(0,4)),
                        fixture.posterior.assign_coords(draw=[0]*8),
                        fixture.posterior.assign(r_site=fixture.posterior.r_site*np.nan)):
                with self.assertRaises(ValueError):
                    g['validate_posterior'](g['_make_idata'](dict(posterior=bad,sample_stats=fixture.sample_stats)),8,1)


def actual_smoke():
    nb, g = namespace()
    root = ROOT/'output/a3/smoke'
    root.mkdir(parents=True, exist_ok=True)
    # A fresh subdirectory prevents any existing empirical cache from being used.
    run = Path(tempfile.mkdtemp(prefix='a3_software_', dir=root))
    g.update(OUTPUT_ROOT=run, CHECKPOINT_ROOT=run/'checkpoints', FIGURE_ROOT=run/'figures',
             DATA_ROOT=run/'data', MANIFEST_PATH=run/'analysis_manifest.json', CHAINS=2,
             DRAWS=8, TUNE=15, RECOVERY_DRAWS=8, RECOVERY_TUNE=15,
             FULL_ESS_MIN=1000, PILOT_ESS_MIN=400, IS_FULL=False, RUN_MODE='SOFTWARE_SMOKE',
             RECOVERY_MIN_COVERAGE=.8, PPC_MIN_COVERAGE=.90, FINE_PPC_MIN_COVERAGE=.90,
             CORE_FINGERPRINT='software-smoke-nonreportable', MEASUREMENT_OK=False,
             CALIBRATION_COMPLETE=False, display=lambda x: None, groups=counts_fixture())
    for path in (g['CHECKPOINT_ROOT'], g['FIGURE_ROOT'], g['DATA_ROOT']):
        path.mkdir()
    cp.atomic_json(g['MANIFEST_PATH'], dict(analysis_fingerprint=g['ANALYSIS_FINGERPRINT'],artifacts={}))
    started = time.monotonic()
    # The production recovery gate must stop a deliberately undersized smoke.
    try:
        exec(compile(nb.cells[22].source, 'A3_smoke_recovery', 'exec'), g)
    except RuntimeError as exc:
        assert 'moderate recovery failed' in str(exc), str(exc)
    else:
        raise AssertionError('Eight draws cannot satisfy the recovery gate')
    sparse, _ = g['sparse_recovery_data'](g['groups'])
    g['fit_or_load']('synthetic_sparse_recovery_A3', sparse)
    # Test the downstream components directly, leaving recovery flags false.
    # Only this harness removes the already-tested guard; the notebook does not.
    primary_body = ast.parse(nb.cells[24].source)
    assert isinstance(primary_body.body[0], ast.If)
    exec(compile(ast.Module(body=primary_body.body[1:],type_ignores=[]), 'A3_smoke_primary_components', 'exec'), g)
    for i in (26,28,30,32):
        print(f'ACTUAL SOFTWARE SMOKE: cell {i}', flush=True)
        exec(compile(nb.cells[i].source, f'A3_cell_{i}', 'exec'),g)
    original = pm.sample
    with patch.object(pm, 'sample', side_effect=AssertionError('A completed fit must reload')):
        g['fit_or_load']('primary_A3_logit',g['groups'])
    # Exercise the actual probit sampling branch and its odds conversion as well.
    probit = g['fit_or_load']('probit_smoke',g['groups'],link='probit')
    assert np.isfinite(g['comparable_effects'](probit,g['groups'],link='probit')).all()
    result = json.loads((run/'final_verdict.json').read_text())
    assert result['gate_ok'] is False and result['verdict']=='NOT REPORTABLE'
    cp.atomic_json(ROOT/'output/a3/smoke_result.json', dict(path=str(run), seconds=time.monotonic()-started,
                   pymc=pm.__version__, arviz=az.__version__, chain_files=len(list(g['CHECKPOINT_ROOT'].glob('*.chain_*.nc'))),
                   note='Tiny synthetic software test only. Not an empirical A3 PILOT or convergence result.'))
    print('ACTUAL SOFTWARE SMOKE COMPLETED',run,flush=True)


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        actual_smoke()
    else:
        unittest.main()
