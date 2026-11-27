"""Meaningful population and denominator checks for SV report exports."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

if __package__:
    from . import probing_exports as exports
else:
    import probing_exports as exports


def results(models=exports.MODELS, n_backgrounds=100):
    rows = []
    for model in models:
        for family in exports.FAMILIES:
            for background_id in range(n_backgrounds):
                sv = (-.02, 0., .04, .10)[background_id % 4]
                for stage in exports.STAGES:
                    for task in exports.TASKS:
                        lo, hi = exports.TASK_BANDS[task]
                        n = 10 * (hi - lo + 1)
                        rows.append(dict(
                            model=model.tag, P=model.P, S=model.S, official=model.official,
                            checkpoint="immutable-test-" + model.tag, family=family,
                            background_id=background_id, stage=stage, task=task,
                            n=n, L_bits=n * (1 - sv), L_uniform=n, SV=sv,
                            L_shuffled_bits=n * 1.05, SV_shuffled=-.05,
                            first_chunk_n=int(n / 5),
                            L_random_init_bits=n * 1.01, SV_random_init=-.01))
    return pd.DataFrame(rows)


def trial_results(models=exports.MODELS, families=exports.FAMILIES, n_pairs=100):
    rows = []
    for model in models:
        for family in families:
            for pair in range(n_pairs):
                near = pair % 4 == 0
                for network in exports.NETWORKS:
                    for stage in exports.STAGES:
                        for arm, located in (("lock", 4 if near else 8), ("control", 10)):
                            rows.append(dict(
                                model=model.tag, P=model.P, S=model.S, official=model.official,
                                family=family, background_id=pair if family != "pure" else -1,
                                network=network, stage=stage, pair=pair, arm=arm,
                                frequency=64. if arm == "lock" else 60.,
                                lock_hz=64., branch="both",
                                boundary_distance=0. if near and arm == "lock" else 9.,
                                n_contexts=10, located=located, level1=10, level2=10,
                                level3=located, bits_per_context=.4))
    return pd.DataFrame(rows)


class CompressionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = results()
        cls.cfg = SimpleNamespace(mode="full", n_backgrounds=100)

    def test_strict_positive_threshold_and_negative_values_preserved(self):
        counts, distribution = exports.compression_summary(self.frame, exports.MODELS, self.cfg)
        self.assertEqual(len(counts), 16 * 2 * 7)
        self.assertTrue((counts.compressed == 50).all())
        self.assertTrue((counts.compresses == "50/100").all())
        self.assertTrue((counts.shuffled_compressed == 0).all())
        self.assertTrue(np.allclose(distribution.SV_min, -.02))
        self.assertEqual(len(distribution), 16 * 2 * 10 * 7)

    def test_missing_background_stops_full_publication(self):
        incomplete = self.frame.loc[self.frame.background_id != 99]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            exports.compression_summary(incomplete, exports.MODELS, self.cfg)

    def test_duplicate_background_is_rejected(self):
        duplicate = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            exports.compression_summary(duplicate, exports.MODELS, self.cfg)

    def test_nonfinite_results_are_rejected(self):
        invalid = self.frame.copy()
        invalid.loc[0, "SV"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            exports.compression_summary(invalid, exports.MODELS, self.cfg)

    def test_full_mode_requires_exact_100_and_all_models(self):
        with self.assertRaisesRegex(ValueError, "100 backgrounds"):
            exports.compression_summary(self.frame, exports.MODELS, SimpleNamespace(mode="full", n_backgrounds=99))
        with self.assertRaisesRegex(ValueError, "all 16"):
            exports.compression_summary(self.frame, exports.MODELS[:1], self.cfg)

    def test_background_ids_are_verified_not_only_row_count(self):
        invalid = self.frame.copy()
        invalid.loc[invalid.background_id == 99, "background_id"] = 100
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            exports.compression_summary(invalid, exports.MODELS, self.cfg)

    def test_codelength_formula_is_verified(self):
        invalid = self.frame.copy()
        invalid.loc[0, "SV"] = .6
        with self.assertRaisesRegex(ValueError, "does not match"):
            exports.compression_summary(invalid, exports.MODELS, self.cfg)

    def test_smoke_exports_are_labelled_and_never_report_latex(self):
        models = exports.MODELS[:1]
        frame = results(models=models, n_backgrounds=2)
        with tempfile.TemporaryDirectory() as directory:
            paths = exports.export_complex(frame, Path(directory), models,
                                           SimpleNamespace(mode="smoke", n_backgrounds=2),
                                           trials=trial_results(models, n_pairs=2))
            self.assertNotIn("table_tex", paths)
            self.assertTrue(all(path.name.startswith("smoke_") for path in paths.values()))
            self.assertEqual(list(Path(directory).glob("*.tex")), [])
            self.assertIn("SMOKE TEST ONLY", paths["caption"].read_text(encoding="utf-8"))
            table = pd.read_csv(paths["counts_csv"])
            self.assertTrue((table.n_backgrounds == 2).all())


class LockControlExportTests(unittest.TestCase):
    def test_rates_net_and_off_boundary_reading(self):
        summary = exports.lock_control_summary(trial_results(exports.MODELS[:2], ("pure",), 8))
        self.assertEqual(len(summary), 2 * 2 * 10)
        self.assertTrue(np.allclose(summary.lock_pct, 70) and np.allclose(summary.control_pct, 100))
        self.assertTrue(np.allclose(summary.net_pct, -30))
        # pairs with a tone at a task boundary leave the off-boundary reading whole
        self.assertTrue(np.allclose(summary.lock_pct_off, 80) and (summary.pairs_off == 6).all())
        self.assertTrue((summary.pairs == 8).all() and (summary.lock_n == 80).all())

    def test_full_exports_write_the_report_tables(self):
        frame = results()
        with tempfile.TemporaryDirectory() as directory:
            paths = exports.export_complex(frame, Path(directory), exports.MODELS,
                                           SimpleNamespace(mode="full", n_backgrounds=100),
                                           trials=trial_results())
            table = paths["lock_control_tex"].read_text(encoding="utf-8")
            self.assertIn("tab:probingLockControlBg", table)
            self.assertEqual(table.count("$70/100$"), 16 * 2 * 10)
            self.assertIn("_{", paths["table_tex"].read_text(encoding="utf-8"))
        pure = frame.loc[(frame.family == "tsmixup") & (frame.background_id == 0)].assign(
            family="pure", background_id=-1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "lock/control"):
                exports.export_pure(pure, Path(directory))
            with self.assertRaisesRegex(ValueError, "random-initialisation"):
                exports.export_pure(pure.drop(columns="SV_random_init"), Path(directory))
            paths = exports.export_pure(pure, Path(directory),
                                        trials=trial_results(families=("pure",), n_pairs=8))
            self.assertIn("tab:probingLockControl}", paths["lock_control_tex"].read_text(encoding="utf-8"))
            self.assertIn("Untrained network", paths["stage_tex"].read_text(encoding="utf-8"))
            self.assertTrue(paths["reference_pdf"].is_file())
            self.assertNotIn("Final output", paths["caption"].read_text(encoding="utf-8"))



class LockSummaryExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pure = exports.lock_control_summary(trial_results(families=("pure",), n_pairs=8))
        cls.complex = exports.lock_control_summary(trial_results(n_pairs=8))

    def test_means_cover_the_retrained_geometries_only(self):
        table = exports.lock_summary(self.pure, self.complex)
        self.assertEqual(len(table), 3 * 3 * 10)
        self.assertTrue((table.geometries == 15).all())
        self.assertTrue(np.allclose(table.loc[~table.off_boundary, "mean_net_pct"], -30))
        self.assertTrue(np.allclose(table.loc[table.off_boundary, "mean_net_pct"], -20))
        self.assertTrue(np.allclose(table.loc[~table.off_boundary, "mean_lock_pct"], 70))
        self.assertTrue(np.allclose(table.loc[~table.off_boundary, "mean_control_pct"], 100))
        # the published checkpoint may differ without moving the retrained means
        moved = self.pure.copy()
        moved.loc[moved.official, "net_pct"] = 50.
        self.assertTrue(np.allclose(exports.lock_summary(moved, self.complex).mean_net_pct,
                                    table.mean_net_pct))

    def test_a_missing_geometry_or_stage_stops_the_table(self):
        partial = self.pure.loc[self.pure.model != "retrained-p8-s8"]
        with self.assertRaisesRegex(ValueError, "each retrained geometry"):
            exports.lock_summary(partial, self.complex)
        untrained_missing = self.complex.loc[self.complex.network != "random_init"]
        with self.assertRaisesRegex(ValueError, "each retrained geometry"):
            exports.lock_summary(self.pure, untrained_missing)

    def test_export_reads_the_published_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with self.assertRaises(FileNotFoundError):
                exports.export_lock_summary(directory)
            self.pure.to_csv(directory / "pure_lock_control.csv", index=False)
            self.complex.to_csv(directory / "complex_lock_control.csv", index=False)
            paths = exports.export_lock_summary(directory)
            table = paths["summary_tex"].read_text(encoding="utf-8")
            self.assertIn("tab:probingLockSummary", table)
            # per signal and stage, Lock, Control, Diff, Untr. and Net, in the all-pairs and off-boundary blocks
            for cell, count in (("$70$", 30), ("$80$", 30), ("$100$", 60), ("$-30$", 90), ("$-20$", 30),
                                ("$10$", 30), ("$0$", 30)):
                self.assertEqual(table.count(cell), count, cell)
            self.assertNotIn("$-0$", table)
            self.assertIn("Off boundary", table)
            self.assertEqual(len(pd.read_csv(paths["summary_csv"])), 90)

if __name__ == "__main__":
    unittest.main()
