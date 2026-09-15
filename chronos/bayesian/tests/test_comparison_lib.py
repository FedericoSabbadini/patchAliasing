"""Analytical examples and invariants for the checkpoint comparison."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comparison_lib as cl


class ComparisonTests(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(len(cl.MODELS), 16)
        self.assertEqual(sum(s.official for s in cl.MODELS), 1)
        self.assertEqual(sum((s.P, s.S) == (16, 16) for s in cl.MODELS), 2)

    def test_peak_selection(self):
        t = np.arange(64)/512
        self.assertEqual(cl.select_peaks(np.ones(64))["frequencies"], [])
        peaks = cl.select_peaks(np.sin(2*np.pi*64*t))["frequencies"]
        self.assertTrue(cl.recovered(peaks, 64))
        self.assertTrue(cl.recovered([65], 64))
        self.assertFalse(cl.recovered([65.00001], 64))
        self.assertFalse(cl.recovered([], 64))
        with self.assertRaises(ValueError):
            cl.select_peaks(np.full(64, np.nan))
        peaks = cl.select_peaks(np.sin(2*np.pi*64*t)+.1*np.sin(2*np.pi*160*t))["frequencies"]
        self.assertFalse(cl.recovered(peaks, 160))

    def test_pairs_all_geometries(self):
        cfg = cl.RecoveryConfig()
        bg = np.random.default_rng(1).normal(size=(100, 544)).astype(np.float32)
        bg /= bg.std(axis=1)[:, None]
        seeds = np.arange(100)
        for p, s in cl.FROZEN15:
            x, rows = cl.injection_trials(bg, seeds, p, s, cfg)
            self.assertEqual(x.shape, (200, 544))
            lock = rows[rows.arm == "lock"]
            control = rows[rows.arm == "control"]
            self.assertEqual(len(lock), 100)
            self.assertEqual(len(control), 100)
            counts = lock.frequency_hz.value_counts()
            self.assertLessEqual(counts.max()-counts.min(), 1)
            np.testing.assert_array_equal(lock.phase, control.phase)
            np.testing.assert_allclose(lock.amplitude, 1.5, atol=1e-6)
            self.assertGreater(np.min(abs(control.frequency_hz.to_numpy()[:, None]-cl.lock_frequencies(p, s))), 2)
            self.assertEqual(int((control.frequency_hz.to_numpy() > lock.frequency_hz.to_numpy()).sum()), 50)
        a = cl.injection_trials(bg, seeds, 16, 16, cfg)[0]
        b = cl.injection_trials(bg, seeds, 16, 16, cfg)[0]
        np.testing.assert_array_equal(a, b)

    def test_known_moments_and_partition_invariance(self):
        target = np.array([[0., 0., np.nan], [0., 0., 0.]])
        pred = np.array([[1., -1., 999.], [2., -2., 0.]])
        a = cl.ErrorMoments()
        a.update(pred, target)
        r = a.result()
        self.assertEqual(r['n_errors'], 5)
        self.assertAlmostEqual(r['MAE_W'], 1.2)
        self.assertAlmostEqual(r['bias_W'], 0.)
        self.assertAlmostEqual(r['variance_W2'], 2.)
        b = cl.ErrorMoments()
        for p, t in zip(pred, target):
            b.update(p, t)
        self.assertEqual(a.result(), b.result())
        with self.assertRaises(ValueError):
            a.update(np.full_like(pred, np.nan), target)

    def test_context_no_future_leakage(self):
        cfg = cl.SolarConfig(context_length=5)
        values = np.array([5., np.nan, 9., np.nan, np.nan, 100., 10.])
        night = np.zeros(len(values), bool)
        x = cl.context_batch(values, night, [5], cfg)
        np.testing.assert_allclose(x[0], [5., 7., 9., 0., 0.])
        values[5:] = -999.
        np.testing.assert_array_equal(x, cl.context_batch(values, night, [5], cfg))
        # An absent left endpoint cannot be borrowed from before the context either.
        np.testing.assert_array_equal(cl.fill_context([np.nan, 4.], [False, False]), [0., 4.])

    def test_origin_boundaries_gaps_and_mask(self):
        cfg = cl.SolarConfig(mode="smoke", context_length=4, horizon=2, min_observed=2, n_origins=3)
        ts = pd.date_range("2020-01-01", periods=12, freq="min")
        values = np.ones(12)
        targets = values.copy()
        eligible = cl.eligible_origins(ts, values, targets, cfg)
        np.testing.assert_array_equal(eligible, np.arange(4, 11))
        cut = cl.sample_origins(eligible, cfg)
        self.assertEqual(len(np.unique(cut)), 3)
        np.testing.assert_array_equal(cut, cl.sample_origins(eligible, cfg))
        targets[4:6] = np.nan
        self.assertNotIn(4, cl.eligible_origins(ts, values, targets, cfg))
        ts = ts.to_numpy(copy=True)
        ts[6:] += np.timedelta64(1, 'm')
        eligible = cl.eligible_origins(ts, values, targets, cfg)
        self.assertNotIn(5, eligible)
        self.assertNotIn(9, eligible)
        self.assertIn(10, eligible)

    def test_solar_target_policy(self):
        cfg = cl.SolarConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"solar.csv"
            pd.DataFrame({'Time': ['01-Jan-2017 00:00:00', '01-Jan-2017 12:00:00',
                                  '01-Jan-2017 12:01:00', '01-Jan-2017 12:02:00'],
                          'PV_Power': [-999999., -999999., 100., 5000.]}).to_csv(path, sep=';', index=False)
            _, raw, night, target = cl.load_solar(path, cfg)
            self.assertEqual(target[0], 0.)
            self.assertTrue(np.isnan(target[1]))
            self.assertEqual(target[2], 100.)
            self.assertTrue(np.isnan(target[3]))

    def test_cache_resume_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"prediction.npz"
            x = np.ones((2, 480), dtype=np.float32)
            calls = []
            def forecast(x):
                calls.append(1)
                return np.ones((len(x), 64), dtype=np.float32)
            a = cl.cached_forecast(path, x, forecast, 'a')
            b = cl.cached_forecast(path, x, forecast, 'a')
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(a, b)
            with self.assertRaises(ValueError):
                cl.cached_forecast(path, x, forecast, 'b')
            with self.assertRaises(ValueError):
                cl.cached_forecast(path, x*2, forecast, 'a')
            bad = Path(tmp)/"bad.npz"
            with self.assertRaises(ValueError):
                cl.cached_forecast(bad, x, lambda _: np.full((2, 64), np.nan), 'a')
            self.assertFalse(bad.exists())


if __name__ == '__main__':
    unittest.main()
