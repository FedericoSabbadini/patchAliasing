from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

BAYESIAN_DIR = Path(__file__).resolve().parents[1]
if str(BAYESIAN_DIR) not in sys.path:
    sys.path.insert(0, str(BAYESIAN_DIR))

import chronos.bayesian.tests.h1_offset_lib as h1
import probe_lib as pl


class FakeProbe:
    instances: list["FakeProbe"] = []

    def __init__(self, P: int, S: int):
        self.P = P
        self.S = S
        self.label = f"fake-p{P}-s{S}-seed42-v1"
        self.frequency_batches: list[np.ndarray] = []
        self.closed = False
        self.__class__.instances.append(self)

    def recovery(self, contexts, futures, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        self.frequency_batches.append(frequencies.copy())
        recovery = 0.5 + frequencies / 1_000.0
        phase_error = np.zeros_like(recovery)
        return recovery, phase_error

    def close(self):
        self.closed = True


def fake_background_pool(_generator, n_bg, length, _seed0):
    return [np.zeros(length, dtype=np.float32) for _ in range(n_bg)]


def fake_build_context(_background, frequency, phase, n):
    return np.full(n, float(frequency) + float(phase) * 0.0, dtype=np.float32)


class H1OffsetLibraryTests(unittest.TestCase):
    def setUp(self):
        FakeProbe.instances.clear()

    def test_frozen_manifest_is_the_21_model_bayesian_population(self):
        self.assertEqual(len(h1.FROZEN_BAYES_MODELS), 21)
        self.assertEqual(len(set(h1.FROZEN_BAYES_MODELS)), 21)
        self.assertIn((8, 5), h1.FROZEN_BAYES_MODELS)
        self.assertNotIn((32, 28), h1.FROZEN_BAYES_MODELS)
        self.assertEqual(tuple(pl.BAYES_MODELS), h1.FROZEN_BAYES_MODELS)

    def test_common_sites_apply_both_offset_rules(self):
        cfg = h1.H1OffsetConfig()
        sites = h1.select_common_sites(32, 15, cfg)
        self.assertEqual(len(sites), 20)
        self.assertNotIn(238.933333, sites["f_lock"].tolist())
        self.assertNotIn(240.0, sites["f_lock"].tolist())
        self.assertTrue(np.allclose(sites["delta_fixed_hz"], 1.0))
        for row in sites.itertuples(index=False):
            self.assertTrue(
                pl.controls_are_clean(row.f_lock, row.delta_fixed_hz, 32, 15)
            )
            self.assertTrue(
                pl.controls_are_clean(row.f_lock, row.delta_adaptive_hz, 32, 15)
            )

        total = sum(
            len(h1.select_common_sites(P, S, cfg)) for P, S in h1.FROZEN_BAYES_MODELS
        )
        self.assertEqual(total, 271)

    def test_synthetic_data_are_deterministic_valid_and_non_reportable(self):
        cfg = h1.H1OffsetConfig(generators=("tsmixup",), seed=42)
        first = h1.synthetic_paired_data(cfg, models=((8, 5),), n_bg=1, n_phase=2)
        second = h1.synthetic_paired_data(cfg, models=((8, 5),), n_bg=1, n_phase=2)
        self.assertTrue(first.equals(second))
        self.assertEqual(set(first["data_origin"]), {"synthetic_smoke"})
        self.assertFalse(first["reportable"].any())
        self.assertTrue(first["live_fixed"].any())
        self.assertTrue((~first["live_fixed"].astype(bool)).any())
        summary = h1.validate_paired_data(first, cfg)
        self.assertEqual(summary["rows"], len(first))

        long = h1.to_long_contrasts(first)
        self.assertEqual(len(long), 2 * len(first))
        self.assertEqual(set(long["offset_kind"]), {"fixed", "adaptive"})
        self.assertFalse(long["reportable"].any())
        lock_counts = long.groupby("pair_key")["R_lock"].nunique()
        self.assertTrue((lock_counts == 1).all())

    def test_validation_rejects_incomplete_or_inconsistent_pairs(self):
        cfg = h1.H1OffsetConfig(generators=("tsmixup",))
        good = h1.synthetic_paired_data(cfg, models=((8, 8),), n_bg=1, n_phase=1)

        missing = good.drop(columns=["R_adaptive_hi"])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            h1.validate_paired_data(missing, cfg)

        duplicate = good.iloc[[0, 0]].copy()
        with self.assertRaisesRegex(ValueError, "pair_key"):
            h1.validate_paired_data(duplicate, cfg)

        asymmetric = good.copy()
        asymmetric.loc[0, "f_fixed_hi"] += 0.25
        with self.assertRaisesRegex(ValueError, "symmetric"):
            h1.validate_paired_data(asymmetric, cfg)

        stale = good.copy()
        stale.loc[0, "d_fixed"] += 0.1
        with self.assertRaisesRegex(ValueError, "d_fixed"):
            h1.validate_paired_data(stale, cfg)

        incomplete = good.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            h1.validate_paired_data(incomplete, cfg)

    def test_joint_collector_shards_manifests_and_resumes_without_probe(self):
        cfg = h1.H1OffsetConfig(
            n_phase=1,
            n_bg=1,
            generators=("fake",),
            seed=42,
        )

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            patches = (
                mock.patch.object(
                    h1.pl, "background_pool", side_effect=fake_background_pool
                ),
                mock.patch.object(
                    h1.pl, "build_context", side_effect=fake_build_context
                ),
            )
            with patches[0], patches[1]:
                first = h1.collect_paired_offsets(
                    out,
                    cfg,
                    models=((8, 8),),
                    probe_factory=lambda P, S, _cfg: FakeProbe(P, S),
                )

            self.assertEqual(len(FakeProbe.instances), 1)
            probe = FakeProbe.instances[0]
            self.assertTrue(probe.closed)
            measured = np.concatenate(probe.frequency_batches)
            self.assertEqual(len(measured), 5 * len(first))
            for f_lock in first["f_lock"]:
                self.assertEqual(int(np.isclose(measured, f_lock).sum()), 1)

            shard = out / "raw" / "h1_paired_offsets__p8-s8.parquet"
            manifest_path = out / h1.MANIFEST_NAME
            self.assertTrue(shard.exists())
            self.assertTrue((out / h1.MERGED_NAME).exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(len(manifest["frozen_bayes_models"]), 21)
            self.assertIn("h1_offset_lib", manifest["source_provenance"])
            self.assertEqual(
                len(manifest["source_provenance"]["h1_offset_lib"]["sha256"]),
                64,
            )
            self.assertEqual(manifest["experimental_constants"]["fs_hz"], 512)
            self.assertEqual(
                {
                    (item["model"], round(item["f_lock_hz"], 6))
                    for item in manifest["excluded_sites"]
                },
                set(),
            )
            self.assertEqual(manifest["shards"]["p8-s8"]["status"], "written")
            self.assertIn("label", manifest["shards"]["p8-s8"]["checkpoint"])

            resumed = h1.collect_paired_offsets(
                out,
                cfg,
                models=((8, 8),),
                probe_factory=lambda P, S, _cfg: FakeProbe(P, S),
            )
            self.assertTrue(first.equals(resumed))
            self.assertEqual(len(FakeProbe.instances), 2)
            self.assertEqual(FakeProbe.instances[1].frequency_batches, [])
            self.assertTrue(FakeProbe.instances[1].closed)
            resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed_manifest["shards"]["p8-s8"]["status"], "reused")
            self.assertEqual(
                resumed_manifest["shards"]["p8-s8"]["checkpoint"],
                manifest["shards"]["p8-s8"]["checkpoint"],
            )

            def changed_checkpoint(P, S, _cfg):
                probe = FakeProbe(P, S)
                probe.label = f"fake-p{P}-s{S}-seed42-v2"
                return probe

            with self.assertRaisesRegex(ValueError, "resolved checkpoint changed"):
                h1.collect_paired_offsets(
                    out,
                    cfg,
                    models=((8, 8),),
                    probe_factory=changed_checkpoint,
                )

    def test_resume_refuses_a_different_design(self):
        cfg = h1.H1OffsetConfig(n_phase=1, n_bg=1, generators=("fake",))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with (
                mock.patch.object(
                    h1.pl, "background_pool", side_effect=fake_background_pool
                ),
                mock.patch.object(
                    h1.pl, "build_context", side_effect=fake_build_context
                ),
            ):
                h1.collect_paired_offsets(
                    out,
                    cfg,
                    models=((8, 8),),
                    probe_factory=lambda P, S, _cfg: FakeProbe(P, S),
                )
            changed = replace_config(cfg, fixed_offset_hz=2.0)
            with self.assertRaisesRegex(ValueError, "different config"):
                h1.collect_paired_offsets(out, changed, models=((8, 8),))
            with self.assertRaisesRegex(
                ValueError, "different config/model subset/schema"
            ):
                h1.collect_paired_offsets(out, cfg, models=((8, 8), (8, 5)))


def replace_config(config: h1.H1OffsetConfig, **changes) -> h1.H1OffsetConfig:
    values = {name: getattr(config, name) for name in config.__dataclass_fields__}
    values.update(changes)
    return h1.H1OffsetConfig(**values)


if __name__ == "__main__":
    unittest.main()
