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
                            first_chunk_n=int(n / 5)))
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
        self.assertEqual(len(distribution), 16 * 2 * 5 * 7)

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
            paths = exports.export_complex(frame, Path(directory), models, SimpleNamespace(mode="smoke", n_backgrounds=2))
            self.assertNotIn("table_tex", paths)
            self.assertTrue(all(path.name.startswith("smoke_") for path in paths.values()))
            self.assertEqual(list(Path(directory).glob("*.tex")), [])
            self.assertIn("SMOKE TEST ONLY", paths["caption"].read_text(encoding="utf-8"))
            table = pd.read_csv(paths["counts_csv"])
            self.assertTrue((table.n_backgrounds == 2).all())


if __name__ == "__main__":
    unittest.main()
