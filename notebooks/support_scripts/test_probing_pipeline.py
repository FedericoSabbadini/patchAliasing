"""Resume, cache-identity, and hook-lifecycle checks without model downloads."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

if __package__:
    from . import probing_lib as pipeline
else:
    import probing_lib as pipeline


class FakeExtractor:
    def __init__(self):
        self.capture_calls = 0
        self.close_calls = 0

    def capture(self, contexts):
        self.capture_calls += 1
        features = np.column_stack((contexts.mean(axis=1), contexts.std(axis=1),
                                    np.arange(len(contexts), dtype=float)))
        return {stage: features.copy() for stage in pipeline.STAGES}

    def randomised(self, seed=42):
        return FakeExtractor()

    def close(self):
        self.close_calls += 1


def valid_trials(features, design, trial_features, trials, cfg, metadata, network="trained"):
    """A cheap valid lock/control result; the real scorer is tested with the protocol."""
    tones = trials[["pair", "arm", "frequency", "lock_hz", "branch", "boundary_distance"]].drop_duplicates()
    rows = [dict(metadata, network=network, stage=stage, **tone, n_contexts=cfg.n_phases,
                 located=1, level1=2, level2=1, level3=1, bits_per_context=1.5, n_withheld=2)
            for stage in pipeline.STAGES for tone in tones.to_dict("records")]
    return pd.DataFrame(rows)


def valid_scores(features, design, cfg, metadata, control_features=None):
    """A cheap valid scoring result: these tests target persistence, not MDL fitting."""
    if set(features) != set(pipeline.STAGES):
        raise ValueError("Missing stage features")
    if any(not np.isfinite(values).all() or len(values) != len(design)
           for values in features.values()):
        raise ValueError("Invalid fake features")
    rows = []
    for stage in pipeline.STAGES:
        for task, low, high, _boundary in pipeline.protocol.TASKS:
            n = int(design.frequency.between(low, high).sum())
            rows.append(dict(metadata, stage=stage, task=task, n=n, L_bits=float(n),
                             L_uniform=float(n), SV=0., L_shuffled_bits=n * 1.05,
                             SV_shuffled=-.05, first_chunk_n=max(1, n // 5),
                             L_random_init_bits=float(n), SV_random_init=0.))
    return pd.DataFrame(rows)


class PipelineCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.cfg = pipeline.ProbeConfig.smoke()
        self.design = pipeline.protocol.make_design(self.cfg)
        self.spec = next(model for model in pipeline.MODELS if model.tag == "retrained-p16-s16")
        self.fake = FakeExtractor()
        self.identity = patch.object(pipeline.comparison, "checkpoint_identity", return_value={"revision": "test-only"})
        self.identity.start()
        self.addCleanup(self.identity.stop)
        self.trials = patch.object(pipeline.protocol, "score_trials", side_effect=valid_trials)
        self.trials.start()
        self.addCleanup(self.trials.stop)

    @property
    def shard_path(self):
        return self.out / "shards" / self.spec.tag / "pure" / "background_-01.csv"

    @property
    def features_path(self):
        return self.shard_path.with_name("background_-01_features.npz")

    def dataset(self, extractor=None):
        return pipeline._dataset(self.out, "test-key", self.spec, "pure", -1, -1,
                                 None, self.cfg, self.design, extractor, "cpu")

    def test_resume_committed_shard_does_not_capture_or_score_again(self):
        with patch.object(pipeline, "StageExtractor", return_value=self.fake) as factory:
            with patch.object(pipeline.protocol, "score_features", side_effect=valid_scores) as scorer:
                first, first_trials, extractor, cached = self.dataset()
                self.assertIs(extractor, self.fake)
                self.assertFalse(cached)
                self.assertEqual(self.fake.capture_calls, 1)
                self.assertFalse(self.features_path.exists())
                second, second_trials, resumed_extractor, cached = self.dataset()
                self.assertIsNone(resumed_extractor)
                self.assertTrue(cached)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(scorer.call_count, 1)
                assert_frame_equal(first, second, check_dtype=False, check_exact=False)
                assert_frame_equal(first_trials, second_trials, check_dtype=False, check_exact=False)
                self.assertEqual(set(first_trials.network), {"trained", "random_init"})

    def test_interrupted_scoring_keeps_features_and_resumes_without_capture(self):
        with patch.object(pipeline, "StageExtractor", return_value=self.fake):
            with patch.object(pipeline.protocol, "score_features", side_effect=RuntimeError("interrupted scoring")):
                with self.assertRaisesRegex(RuntimeError, "interrupted scoring"):
                    self.dataset()
        self.assertTrue(self.features_path.exists())
        self.assertFalse(self.shard_path.exists())
        with patch.object(pipeline, "StageExtractor", side_effect=AssertionError("must reuse features")):
            with patch.object(pipeline.protocol, "score_features", side_effect=valid_scores):
                frame, _trials, extractor, cached = self.dataset()
        self.assertIsNone(extractor)
        self.assertFalse(cached)
        self.assertEqual(len(frame), 70)
        self.assertEqual(self.fake.capture_calls, 1)
        self.assertFalse(self.features_path.exists())
        self.assertTrue(self.shard_path.exists())

    def test_new_extractor_is_closed_when_scoring_fails(self):
        with patch.object(pipeline, "StageExtractor", return_value=self.fake):
            with patch.object(pipeline.protocol, "score_features", side_effect=RuntimeError("scoring failed")):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    self.dataset()
        self.assertEqual(self.fake.close_calls, 1)

    def test_incomplete_committed_shard_fails_loudly(self):
        with patch.object(pipeline, "StageExtractor", return_value=self.fake):
            with patch.object(pipeline.protocol, "score_features", side_effect=valid_scores):
                self.dataset()
        frame = pd.read_csv(self.shard_path)
        frame.iloc[:-1].to_csv(self.shard_path, index=False)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.dataset()
        self.assertEqual(self.fake.capture_calls, 1)

    def test_altered_result_identity_is_not_silently_reused(self):
        with patch.object(pipeline, "StageExtractor", return_value=self.fake):
            with patch.object(pipeline.protocol, "score_features", side_effect=valid_scores):
                self.dataset()
        frame = pd.read_csv(self.shard_path)
        frame["run_key"] = "different-run"
        frame.to_csv(self.shard_path, index=False)
        with self.assertRaisesRegex(ValueError, "Shard identity mismatch: run_key"):
            self.dataset()

    def test_altered_feature_identity_is_not_silently_reused(self):
        with patch.object(pipeline, "StageExtractor", return_value=self.fake):
            with patch.object(pipeline.protocol, "score_features", side_effect=RuntimeError("interrupt")):
                with self.assertRaises(RuntimeError):
                    self.dataset()
        with np.load(self.features_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["input_hash"] = np.array("different-input")
        pipeline._npz(self.features_path, **arrays)
        with self.assertRaisesRegex(ValueError, "Feature cache mismatch"):
            self.dataset()
        self.assertFalse(self.shard_path.exists())


class TrialDesignTests(unittest.TestCase):
    def test_sites_and_controls_match_the_recovery_count(self):
        protocol, comparison = pipeline.protocol, pipeline.comparison
        cfg = pipeline.ProbeConfig()
        for spec in pipeline.MODELS:
            locks = comparison.lock_frequencies(spec.P, spec.S)
            np.testing.assert_allclose(protocol.lock_frequencies(spec.P, spec.S), locks)
            for lock in locks:
                self.assertEqual(protocol.control_delta(spec.P, spec.S, lock),
                                 comparison.control_delta(spec.P, spec.S, lock))
            trials = protocol.background_trials(spec.P, spec.S, 100, cfg)
            rng = np.random.default_rng(np.random.SeedSequence([42001, spec.P, spec.S, 131]))
            schedule = np.resize(rng.permutation(locks), 100)
            rng.shuffle(schedule)
            lock_rows = trials.loc[trials.arm.eq("lock") & trials.phase_index.eq(0)]
            np.testing.assert_allclose(lock_rows.frequency, schedule)
            controls = trials.loc[trials.arm.eq("control")]
            distance = np.abs(controls.frequency.to_numpy()[:, None] - locks[None, :]).min(axis=1)
            self.assertTrue((distance >= 2).all())

    def test_one_pair_per_background_and_all_sites_for_pure_tones(self):
        cfg = pipeline.ProbeConfig()
        spec = next(model for model in pipeline.MODELS if model.tag == "retrained-p24-s20")
        pure = pipeline._trial_design(spec, "pure", -1, cfg)
        sites = pipeline.protocol.lock_frequencies(24, 20)
        self.assertEqual(len(pure), len(sites) * 3 * cfg.n_phases)
        self.assertEqual(set(pure.branch), {"patch", "stride", "both"})
        one = pipeline._trial_design(spec, "tsmixup", 7, cfg)
        self.assertEqual(len(one), 2 * cfg.n_phases)
        self.assertEqual(set(one.pair), {7})


class HookLifecycleTests(unittest.TestCase):
    def test_forward_failure_removes_all_registered_hooks(self):
        import torch

        modules = [torch.nn.Identity() for _ in range(9)]
        model = SimpleNamespace(encoder=SimpleNamespace(block=modules[:4]),
                                decoder=SimpleNamespace(block=modules[4:8]),
                                output_patch_embedding=modules[8])

        class FailingPipeline:
            def __init__(self):
                self.model = model

            def predict(self, inputs, prediction_length):
                values = inputs[:, None, :3]
                for block in self.model.decoder.block[:2]:
                    values = block(values)
                raise RuntimeError("injected forward failure")

        extractor = object.__new__(pipeline.StageExtractor)
        extractor.pipe = FailingPipeline()
        extractor.batch_size = 4
        extractor.device = "cpu"
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            extractor.capture(np.ones((6, 512), dtype=np.float32))
        self.assertTrue(all(len(module._forward_hooks) == 0 for module in modules))


if __name__ == "__main__":
    unittest.main()
