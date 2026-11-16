"""Protocol checks: run with python -m unittest discover -s notebooks/support_scripts -p test_probing_protocol.py."""
from __future__ import annotations

import unittest
import unittest.mock
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from probing_protocol import (
    ProbeConfig, STAGES, TASKS, context_length, grouped_chunks, make_design,
    prequential_mdl, score_features,
)


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.frequency = np.repeat(np.arange(2, 102), 4)
        self.y = (self.frequency > 51).astype(int)
        self.chunks = grouped_chunks(self.frequency, self.y)
        rng = np.random.default_rng(42)
        self.features = np.column_stack((self.y * 10, rng.normal(size=(len(self.y), 7))))

    def test_full_design_has_ten_distinct_phases(self):
        cfg = ProbeConfig()
        design = make_design(cfg)
        self.assertEqual(len(design), 2490)
        self.assertEqual(design.frequency.nunique(), 249)
        for frequency in (5, 10, 25, 125, 250):
            phases = design.loc[design.frequency == frequency, "phase"].to_numpy()
            self.assertEqual(len(np.unique(phases)), 10)
            signals = np.sin(2 * np.pi * frequency * np.arange(512) / 512 + phases[:, None])
            self.assertEqual(len(np.unique(signals.round(7), axis=0)), 10)
        self.assertTrue(design.phase.between(0, 2 * np.pi, inclusive="left").all())

    def test_smoke_does_not_look_like_full(self):
        cfg = ProbeConfig.smoke()
        self.assertEqual((cfg.mode, cfg.n_backgrounds, cfg.n_phases, cfg.frequency_step),
                         ("smoke", 2, 2, 4))
        with self.assertRaises(ValueError):
            ProbeConfig(n_backgrounds=2)
        with self.assertRaises(ValueError):
            ProbeConfig(n_phases=2)

    def test_context_covers_every_sample_for_all_geometries(self):
        geometries = ((8, 8), (16, 8), (16, 12), (16, 16), (24, 8), (24, 12),
                      (24, 16), (24, 20), (24, 24), (32, 8), (32, 12), (32, 16),
                      (32, 20), (32, 24), (32, 32))
        for P, S in geometries:
            length = context_length(P, S)
            coverage = np.zeros(length, dtype=int)
            for start in range(0, length - P + 1, S):
                coverage[start:start + P] += 1
            self.assertTrue((coverage > 0).all())
            self.assertEqual(length % P, 0)
            self.assertEqual((length - P) % S, 0)
            self.assertFalse(any(candidate % P == 0 and (candidate-P) % S == 0
                                 for candidate in range(length+1, 513)))
        self.assertEqual(context_length(16, 12), 496)
        self.assertEqual(context_length(24, 20), 504)
        for geometry in ((0, 0), (8, 16), (1024, 8)):
            with self.assertRaises(ValueError):
                context_length(*geometry)

    def test_chunks_hold_all_phases_and_both_classes(self):
        assert_array_equal(np.sort(np.concatenate(self.chunks)), np.arange(len(self.y)))
        seen = set()
        for chunk in self.chunks:
            groups = set(self.frequency[chunk])
            self.assertFalse(groups & seen)
            self.assertEqual(set(self.y[chunk]), {0, 1})
            seen |= groups
        repeated = grouped_chunks(self.frequency, self.y)
        for one, two in zip(self.chunks, repeated):
            assert_array_equal(one, two)

    def test_chunk_builder_rejects_inconsistent_or_insufficient_groups(self):
        bad_labels = self.y.copy()
        bad_labels[0] = 1
        with self.assertRaises(ValueError):
            grouped_chunks(self.frequency, bad_labels)
        with self.assertRaises(ValueError):
            grouped_chunks(np.arange(8), np.tile([0, 1], 4))

    def test_scaler_never_sees_current_or_future_chunks(self):
        from sklearn.preprocessing import StandardScaler
        fits = []

        class AuditedScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                fits.append(np.array(X, copy=True))
                return super().fit(X, y=y, sample_weight=sample_weight)

        # Unique row identifiers make the transform's fit membership inspectable.
        features = self.features.copy()
        features[:, 1] = np.arange(len(self.y))
        with patch("sklearn.preprocessing.StandardScaler", AuditedScaler):
            prequential_mdl(features, self.y, self.chunks)
        self.assertEqual(len(fits), 4)
        for k, seen in enumerate(fits, start=1):
            train = np.concatenate(self.chunks[:k])
            assert_array_equal(seen, features[train])

    def test_separable_task_compresses_and_shuffled_control_does_not(self):
        result = prequential_mdl(self.features, self.y, self.chunks)
        self.assertGreater(result["SV"], .55)
        self.assertLess(result["SV_shuffled"], .05)
        self.assertLessEqual(result["SV"], 1-result["first_chunk_n"]/result["n"])
        self.assertEqual(result["L_uniform"], len(self.y))
        self.assertAlmostEqual(result["SV"], 1-result["L_bits"]/result["n"])
        self.assertAlmostEqual(result["SV_shuffled"], 1-result["L_shuffled_bits"]/result["n"])

    def test_uniform_predictor_has_exactly_zero_saving(self):
        class UniformClassifier:
            classes_ = np.array([0, 1])
            def __init__(self, **kwargs):
                pass
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                return np.full((len(X), 2), .5)

        with patch("sklearn.linear_model.LogisticRegression", UniformClassifier):
            result = prequential_mdl(self.features, self.y, self.chunks)
        self.assertEqual(result["L_bits"], len(self.y))
        self.assertEqual(result["SV"], 0)
        self.assertEqual(result["SV_shuffled"], 0)

    def test_codelength_expansion_stays_negative(self):
        class OverconfidentClassifier:
            classes_ = np.array([0, 1])
            def __init__(self, **kwargs):
                pass
            def fit(self, X, y):
                return self
            def predict_proba(self, X):
                return np.tile([1e-6, 1-1e-6], (len(X), 1))

        with patch("sklearn.linear_model.LogisticRegression", OverconfidentClassifier):
            result = prequential_mdl(self.features, self.y, self.chunks)
        self.assertLess(result["SV"], 0)
        self.assertGreater(result["L_bits"], result["L_uniform"])

    def test_reproducible_control_and_projection(self):
        one = prequential_mdl(self.features, self.y, self.chunks)
        two = prequential_mdl(self.features, self.y, self.chunks)
        self.assertEqual(one, two)
        same_labels = prequential_mdl(self.features, self.y, self.chunks, control_labels=self.y)
        self.assertEqual(same_labels["L_bits"], same_labels["L_shuffled_bits"])

    def test_invalid_features_or_partition_raise(self):
        with self.assertRaises(ValueError):
            prequential_mdl(self.features[:-1], self.y, self.chunks)
        bad = self.features.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            prequential_mdl(bad, self.y, self.chunks)
        with self.assertRaises(ValueError):
            prequential_mdl(self.features, self.y, [self.chunks[0]] * 5)

    def test_score_contract_has_70_results_and_metadata(self):
        cfg = ProbeConfig.smoke()
        design = make_design(cfg)
        frequency = design.frequency.to_numpy()
        features = {stage: np.column_stack([frequency, frequency ** 2,
                    np.sin(design.phase), np.cos(design.phase)]) for stage in STAGES}
        metadata = {"model": "retrained-p16-s16", "P": 16, "S": 16,
                    "family": "pure", "background_id": 0}
        scores = score_features(features, design, cfg, metadata)
        self.assertEqual(len(scores), 70)
        self.assertEqual(set(scores.task), {task[0] for task in TASKS})
        self.assertEqual(set(scores.stage), set(STAGES))
        self.assertEqual(set(scores.family), {"pure"})
        self.assertFalse(scores.duplicated(["stage", "task"]).any())
        assert_allclose(scores.SV, 1-scores.L_bits/scores.n)
        self.assertNotIn("SV_random_init", scores)
        with self.assertRaises(ValueError):
            score_features({"output_head": features["output_head"]}, design, cfg, metadata)

    def test_random_init_control_is_scored_with_the_same_blocks(self):
        cfg = ProbeConfig.smoke()
        design = make_design(cfg)
        frequency = design.frequency.to_numpy()
        informative = np.column_stack([frequency, frequency ** 2])
        noise = np.random.default_rng(0).normal(size=(len(design), 6))
        metadata = {"model": "m", "family": "pure", "background_id": -1}
        scores = score_features({stage: informative for stage in STAGES}, design, cfg, metadata,
                                {stage: noise for stage in STAGES})
        assert_allclose(scores.SV_random_init, 1 - scores.L_random_init_bits / scores.n)
        self.assertTrue((scores.SV > scores.SV_random_init).all())
        self.assertLess(scores.SV_random_init.median(), .05)
        with self.assertRaises(ValueError):
            score_features({stage: informative for stage in STAGES}, design, cfg, metadata,
                           {"output_head": noise})

    def test_task_path_places_a_tone_in_an_eighth_of_the_band(self):
        import probing_protocol as protocol
        self.assertEqual(protocol.task_path(20), [("Mid", 0), ("L", 0), ("LL", 0)])
        self.assertEqual(protocol.task_path(64), [("Mid", 0), ("L", 0), ("LL", 1)])
        self.assertEqual(protocol.task_path(64.5), [("Mid", 0), ("L", 1), ("LH", 0)])
        self.assertEqual(protocol.task_path(128), [("Mid", 1), ("H", 0), ("HL", 0)])
        self.assertEqual(protocol.task_path(230.4), [("Mid", 1), ("H", 1), ("HH", 1)])
        self.assertEqual(protocol.boundary_distance(128), 2.0)
        with self.assertRaises(ValueError):
            protocol.task_path(251)

    def test_trials_are_located_from_informative_states_only(self):
        import probing_protocol as protocol
        cfg = ProbeConfig()
        design = make_design(cfg)
        trials = protocol.pure_trials(16, 16, cfg)
        self.assertEqual(set(trials.arm), {"lock", "control"})
        self.assertEqual(trials.groupby("pair").frequency.nunique().unique().tolist(), [3])

        def state(frame):
            f = frame.frequency.to_numpy(float)
            return np.column_stack([f, np.sin(frame.phase), np.cos(frame.phase)])
        rng = np.random.default_rng(1)
        stages = STAGES[:1]
        with unittest.mock.patch.object(protocol, "STAGES", stages):
            good = protocol.score_trials({s: state(design) for s in stages}, design,
                                         {s: state(trials) for s in stages}, trials, cfg, {"model": "m"})
            blind = protocol.score_trials(
                {s: rng.normal(size=(len(design), 3)) for s in stages}, design,
                {s: rng.normal(size=(len(trials), 3)) for s in stages}, trials, cfg, {"model": "m"},
                network="random_init")
        self.assertEqual(len(good), trials[["pair", "arm", "frequency"]].drop_duplicates().shape[0])
        self.assertTrue((good.n_contexts == cfg.n_phases).all())
        self.assertTrue((good.located <= good[["level1", "level2", "level3"]].min(axis=1)).all())
        far = good.loc[good.boundary_distance >= 8]
        self.assertGreater(far.located.sum() / far.n_contexts.sum(), .95)
        self.assertLess(blind.located.sum() / blind.n_contexts.sum(), .4)
        # six frequencies at most are withheld around the three tones of a pair
        self.assertTrue(good.n_withheld.between(cfg.n_phases, 9 * cfg.n_phases).all())


if __name__ == "__main__":
    unittest.main()
