import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support_scripts"))
import solar_benchmark_lib as sb
import solar_chronos2_lib as s2
import complex_sweep_lib as sw


class ExtendedTests(unittest.TestCase):
    def test_disjoint_and_overlap(self):
        cfg = replace(sb.SolarBenchmarkConfig(), context_length=20, horizon=4, n_draws=15)
        eligible = np.arange(20, 500)
        cuts = sb.sample_windows(eligible, cfg)
        self.assertEqual(len(cuts), 15)
        self.assertTrue((np.diff(cuts) >= 24).all())
        np.testing.assert_array_equal(cuts, sb.sample_windows(eligible, cfg))
        overlap = sb.sample_windows(eligible, replace(cfg, n_draws=400, allow_overlap=True))
        self.assertEqual(len(np.unique(overlap)), 400)
        with self.assertRaises(ValueError):
            sb.sample_windows(eligible, replace(cfg, n_draws=30))

    def test_frequency_and_units(self):
        x = np.sin(2*np.pi*np.arange(2048)*8/2048)
        frequencies = sb.dominant_context_frequency(np.array([x, np.ones(2048), np.full(2048,np.nan)]))
        self.assertEqual(frequencies[0]*16, 1/16)
        self.assertTrue(np.isnan(frequencies[1]))
        self.assertTrue(np.isnan(frequencies[2]))

    def test_metrics_and_missing(self):
        m = sb.metrics(np.array([[2., 5., 8.]]), np.array([[1., 2., np.nan]]))
        self.assertEqual(m["MAE"], 2)
        self.assertEqual(m["bias"], 2)
        self.assertEqual(m["variance"], 1)
        self.assertEqual(m["n_errors"], 2)
        with self.assertRaises(ValueError):
            sb.metrics(np.array([[np.nan]]), np.array([[1.]]))

    def test_covariates_no_future(self):
        raw = pd.DataFrame(dict(Time=np.arange(20), PV_Power=np.arange(20), G_h=np.arange(20, dtype=float)))
        before = s2.past_covariates(raw, 12, 8)
        raw.loc[12:, "G_h"] = 99999
        after = s2.past_covariates(raw, 12, 8)
        np.testing.assert_array_equal(before["G_h"], after["G_h"])
        self.assertEqual(set(before), {"G_h"})

    def test_exact_components_and_amplitude(self):
        cfg = replace(sw.SweepConfig(), n_backgrounds=1, frequency_step=80, n_phase_probe=2)
        bg, components = sw.backgrounds("tsmixup", cfg)
        self.assertEqual(len(components), 10)
        t = np.arange(bg.shape[1])/512
        reconstructed = sum(r.amplitude*np.cos(2*np.pi*r.frequency_hz*t+r.cosine_phase)
                            for r in components.itertuples())
        np.testing.assert_allclose(bg[0], reconstructed, atol=2e-7)
        x, trials = sw.make_inputs(bg, components, cfg, 2)
        self.assertTrue(np.allclose(trials.amplitude, 1.5*components.amplitude.max()))
        r = trials.iloc[0]
        np.testing.assert_allclose(x[0], bg[0,:512]+r.amplitude*np.sin(2*np.pi*r.injected_hz*t[:512]+r.phase), atol=3e-7)

    def test_dominant_flat_and_invalid(self):
        t = np.arange(512)/512
        y = np.array([np.sin(2*np.pi*77*t), np.zeros(512)])
        result = sw.dominant_output(y)
        self.assertEqual(result[0], 77)
        self.assertTrue(np.isnan(result[1]))
        with self.assertRaises(ValueError):
            sw.dominant_output(np.full((1,64), np.nan))

    def test_ks_ten_term_reconstruction(self):
        cfg = replace(sw.SweepConfig(), n_backgrounds=1)
        bg, components = sw.backgrounds("kernelsynth", cfg)
        t = np.arange(bg.shape[1])/512
        reconstructed = sum(r.amplitude*np.cos(2*np.pi*r.frequency_hz*t+r.cosine_phase)
                            for r in components.itertuples())
        self.assertEqual(len(components), 10)
        self.assertEqual(components.frequency_hz.nunique(), 10)
        np.testing.assert_allclose(bg[0], reconstructed, atol=2e-7)

    def test_quantile_resume_and_invalid_block(self):
        cfg = replace(sb.SolarBenchmarkConfig(), mode="smoke", batch_size=2)
        spec = sb.c.MODELS[0]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            x = np.ones((3, 2048), dtype=np.float32)
            sb.cp.atomic_npy(out/"contexts.npy", x)
            for start in (0, 2):
                block = x[start:start+2]
                sb.c.save_npz(out/spec.tag/f"quantiles_{start:07d}.npz",
                    key="run", input_hash=sb.c.array_hash(block), prediction=np.ones((len(block),9,64)))
            with patch.object(sb.c, "Forecaster", side_effect=AssertionError("Unexpected inference")):
                sb._worker(spec, cfg, str(out), "run", "cpu")
            path = out/spec.tag/"quantiles_0000000.npz"
            sb.c.save_npz(path, key="wrong", input_hash=sb.c.array_hash(x[:2]), prediction=np.ones((2,9,64)))
            with self.assertRaises(ValueError):
                sb._worker(spec, cfg, str(out), "run", "cpu")

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows transient sharing lock')
    def test_atomic_transient_windows_lock(self):
        original = sb.cp.os.replace
        attempts = []
        def locked_once(source, destination):
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError('temporary reader')
            return original(source, destination)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'progress.json'
            with patch.object(sb.cp.os, 'replace', side_effect=locked_once):
                sb.cp.atomic_json(path, {'done': 12})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(__import__('json').loads(path.read_text()), {'done': 12})


if __name__ == "__main__":
    unittest.main()
