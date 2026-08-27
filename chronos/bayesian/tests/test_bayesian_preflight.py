from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


BAYES_DIR = Path(__file__).resolve().parents[1]
GENERATOR_DIR = BAYES_DIR.parent / "data" / "synthetic" / "generators"
for path in (BAYES_DIR, GENERATOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bayesian_checks as bc
import checkpointing as cp
import collect
import model_loader as ml
import probe_lib as pl
from kernelsynth_generator import AWS_PERIODS, KernelSynthGenerator, _k_periodic


class KernelSynthContractTests(unittest.TestCase):
    def test_periodic_bank_matches_deliverable3_scaling(self):
        length = 64
        generator = KernelSynthGenerator(l_syn=length, seed=7)
        t = np.linspace(0.0, 1.0, length)
        observed = generator.kernel_bank[0][1](t, t)
        expected = _k_periodic(t, t, periodicity=AWS_PERIODS[0] / length)
        np.testing.assert_allclose(observed, expected)
        self.assertGreater(float(np.ptp(observed[0])), 0.1)

    def test_pool_quality_rejects_duplicate_draws(self):
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        distinct = [np.sin(t), np.cos(t), np.sin(2 * t)]
        distinct = [(x - x.mean()) / x.std() for x in distinct]
        self.assertTrue(pl.background_pool_quality(distinct)["quality_ok"])
        duplicated = [distinct[0], distinct[0].copy(), distinct[2]]
        self.assertFalse(pl.background_pool_quality(duplicated)["quality_ok"])


class CheckpointContractTests(unittest.TestCase):
    def test_partial_local_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "p16-s16-seed42"
            (local / "checkpoint-50000").mkdir(parents=True)
            with mock.patch.object(ml, "_WEIGHTS_DIR", root):
                with self.assertRaisesRegex(ValueError, "incomplete checkpoint"):
                    ml.resolve_local_checkpoint(16, 16)

    def test_complete_local_checkpoint_geometry_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "p16-s16-seed42"
            local.mkdir(parents=True)
            (local / "model.safetensors").write_bytes(b"test weights")
            (local / "config.json").write_text(json.dumps({
                "chronos_config": {
                    "input_patch_size": 16,
                    "input_patch_stride": 16,
                    "prediction_length": 64,
                }
            }), encoding="utf-8")
            (local / "run_config.json").write_text(json.dumps({
                "status": "done", "steps_completed": 100_000, "seed": 42
            }), encoding="utf-8")
            with mock.patch.object(ml, "_WEIGHTS_DIR", root):
                self.assertEqual(ml.resolve_local_checkpoint(16, 16), local)
                with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                    ml.validate_checkpoint_dir(local, 8, 8)

    def test_huggingface_load_is_revision_pinned(self):
        calls = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls.append((args, kwargs))
                return object()

        with mock.patch.object(ml, "resolve_local_checkpoint", return_value=None):
            pl.load_checkpoint(16, 16, pipeline_cls=FakePipeline)
        self.assertEqual(calls[0][1]["revision"], ml.SWEEP_REVISION)
        self.assertEqual(calls[0][1]["subfolder"], "p16-s16-seed42")


class D2ContractTests(unittest.TestCase):
    def test_resolution_aware_assignment_and_one_site_spacing(self):
        family, residual = pl.assign_site_family(42.8, 16, 12, tol_hz=1.5)
        self.assertEqual(family, "stride")
        self.assertLess(residual, 0.2)
        self.assertEqual(pl.assign_site_family(32.0, 16, 16)[0], "both")
        summary = pl.site_summary([42.8])
        self.assertEqual(summary["n_sites"], 1)
        self.assertTrue(np.isnan(summary["delta_hat"]))

    def test_derive_sites_retains_branch_and_unassigned_counts(self):
        frame = pd.DataFrame({
            "model": ["p16-s12"] * 3,
            "P": [16] * 3,
            "S": [12] * 3,
            "mode": ["pure"] * 3,
            "rep": [0] * 3,
            "f": [40.0, 42.8, 45.0],
            "z": [1.0, 0.0, 1.0],
        })
        sites = collect.derive_sites(frame, collect.Config.smoke_cfg())
        stride = sites[(sites["branch"] == "stride") & (sites["rep"] == 0)].iloc[0]
        self.assertEqual(int(stride["n_sites"]), 1)
        self.assertIn("n_unassigned", sites.columns)


class CollectionContractTests(unittest.TestCase):
    @staticmethod
    def _frames(tag: str) -> dict[str, pd.DataFrame]:
        contrasts = pd.DataFrame({
            "model": [tag], "P": [16], "S": [16], "generator": ["tsmixup"],
            "R_lock": [0.3], "R_lo": [0.8], "R_hi": [0.9], "d": [-1.0],
            "live": [True], "f_lock": [32.0], "phase_idx": [0],
        })
        mdl = pd.DataFrame({
            "model": [tag, tag], "P": [16, 16], "S": [16, 16],
            "stage": ["enc_0", "enc_0"], "is_locked": [0, 1], "L_bits": [20.0, 25.0],
        })
        collapse_rows = []
        for mode in ("pure", "tsmixup"):
            for f, z in ((30.0, 1.0), (32.0, 0.0), (34.0, 1.0)):
                collapse_rows.append({
                    "model": tag, "P": 16, "S": 16, "mode": mode, "rep": 0,
                    "f": f, "z": z, "z_norm": z,
                })
        return {
            "contrasts": contrasts,
            "mdl_cells": mdl,
            "collapse": pd.DataFrame(collapse_rows),
        }

    def test_merge_uses_manifest_exactly_and_ignores_stale_glob(self):
        cfg = collect.Config.smoke_cfg(band_tasks=False)
        planned = [(16, 16)]
        tag = "p16-s16"
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            manifest = collect._load_or_create_manifest(out, cfg, planned)
            for table, frame in self._frames(tag).items():
                path = collect._shard(out, table, tag)
                cp.atomic_parquet(path, frame)
                manifest["shards"][f"{table}__{tag}"] = {
                    "file": str(path.relative_to(out)),
                    "sha256": cp.sha256_file(path),
                    "rows": len(frame),
                    "columns": list(frame.columns),
                    "checkpoint_identity_sha256": manifest["design"]["checkpoints"][tag][
                        "identity_sha256"],
                }
            cp.atomic_json(out / collect.MANIFEST_NAME, manifest)
            stale = out / "raw" / "contrasts__p99-s99.parquet"
            cp.atomic_parquet(stale, self._frames(tag)["contrasts"])

            merged = collect.merge(out, cfg, planned, manifest=manifest)
            self.assertEqual(set(merged["contrasts"]["model"]), {tag})
            self.assertTrue(stale.exists())

    def test_design_validation_raises_instead_of_printing_only(self):
        with self.assertRaisesRegex(ValueError, "design validation failed"):
            collect.check_design({}, [(16, 16)], collect.Config.smoke_cfg(band_tasks=False))


class BayesianGateTests(unittest.TestCase):
    def test_fail_closed_and_three_way_rules(self):
        self.assertEqual(bc.three_way_verdict(0.99, 0.0, False), "NOT REPORTABLE")
        self.assertEqual(bc.three_way_verdict(0.96, 0.01, True), "supported")
        self.assertEqual(bc.three_way_verdict(0.01, 0.96, True), "refuted")
        self.assertEqual(bc.three_way_verdict(0.5, 0.5, True), "inconclusive")
        self.assertEqual(bc.joint_support_verdict(0.99, False, True), "inconclusive")

    def test_d2_identification_bar(self):
        sites = pd.DataFrame({
            "branch": ["stride", "patch"],
            "n_sites": [9, 10],
            "rep": [-1, -1],
            "mode": ["tsmixup", "kernelsynth"],
        })
        self.assertEqual(
            bc.identified_branches(sites, minimum_sites=10),
            {"stride": False, "patch": True},
        )


if __name__ == "__main__":
    unittest.main()
