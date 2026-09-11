"""Numerical regression checks using invented, non-participant measurements."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from plax_measurement import cmr


def synthetic_reader_rows(n=8):
    """Entirely artificial cases; identifiers have no source-data counterpart."""
    rows = []
    for i in range(n):
        for r, reader in enumerate(cmr.READERS):
            for m, method in enumerate(cmr.METHODS):
                rows.append(dict(
                    case_id=f"synthetic_case_{i:02d}", reader=reader, method=method,
                    lvidd_display_px=26 + i * .9 + r * .4 + m * .2,
                    lvids_display_px=19 + i * .3 + r * .15 + m * .6 + (i % 3) * .4,
                    resize_ratio_y=1 + r * .05, reference_ef=48 + i * 2 + (i % 2) * 3,
                ))
    return pd.DataFrame(rows)


class CMRTests(unittest.TestCase):
    def test_teichholz_and_consensus_order(self):
        d = np.array([3., 5., 7.])
        s = np.array([2., 3., 4.])
        expected = 100 * (1 - (s ** 3 / (2.4 + s)) / (d ** 3 / (2.4 + d)))
        np.testing.assert_allclose(cmr.teichholz_ef(d, s), expected)
        rows = synthetic_reader_rows()
        cases = cmr.prepare_reader_measurements(rows)
        pred = cmr.build_harmonized_predictions(cases, set(cases.case_id))
        matrix = cases[cmr.metric_columns("dynamic", "lvef")].to_numpy()
        np.testing.assert_allclose(pred.consensus_dynamic_raw_ef, matrix.mean(axis=1))
        ef_from_mean_diameters = cmr.teichholz_ef(
            cases[cmr.metric_columns("dynamic", "lvidd")].mean(axis=1),
            cases[cmr.metric_columns("dynamic", "lvids")].mean(axis=1))
        self.assertGreater(float(np.max(abs(pred.consensus_dynamic_raw_ef - ef_from_mean_diameters))), 1e-4)

    def test_dynamic_shares_static_ed_and_scale(self):
        rows = synthetic_reader_rows()
        rows.loc[rows.method == "dynamic", "lvidd_display_px"] = 999
        cases = cmr.prepare_reader_measurements(rows)
        for reader in cmr.READERS:
            np.testing.assert_array_equal(cases[f"{reader}_dynamic_lvidd_harmonized"], cases[f"{reader}_static_lvidd_harmonized"])
        self.assertAlmostEqual(cases.reader01_static_lvidd_harmonized.iloc[0], 26 * .15)

    def test_no_clipping_and_physiologic_subset(self):
        rows = synthetic_reader_rows()
        selected = (rows.case_id == "synthetic_case_00") & (rows.reader == "reader01") & (rows.method == "dynamic")
        rows.loc[selected, "lvids_display_px"] = 45
        cases = cmr.prepare_reader_measurements(rows)
        self.assertLess(cases.reader01_dynamic_lvef_harmonized.iloc[0], 0)
        masks = cmr.sensitivity_masks(cases)
        self.assertTrue(masks["primary"].all())
        self.assertEqual(int(masks["physiologic"].sum()), 7)

    def test_loocv_held_out_target_has_no_effect(self):
        x = np.array([0., 1., 2., 4., 8., 9.])
        y = np.array([2., 5., 6., 13., 19., 25.])
        initial = cmr.loocv_linear_predictions(x, y)
        y[2] = 999
        altered = cmr.loocv_linear_predictions(x, y)
        self.assertEqual(initial[2], altered[2])
        np.testing.assert_allclose(cmr.loocv_linear_predictions(x, 3*x+2), 3*x+2)

    def test_icc_absolute_agreement_and_average_relation(self):
        x = np.arange(1., 7.)
        perfect = np.column_stack([x, x, x])
        np.testing.assert_allclose(cmr.icc_absolute(perfect), [1., 1.])
        biased = perfect + np.array([0., 1., 2.])
        single, average = cmr.icc_absolute(biased)
        self.assertLess(single, 1)
        self.assertAlmostEqual(average, 3*single/(1+2*single))
        np.testing.assert_allclose(cmr.icc_absolute(biased * 2 + 20), [single, average])

    def test_agreement_and_bootstrap_direction(self):
        y = np.array([41., 49., 53., 65., 71., 73.])
        row = cmr.agreement_row("synthetic", "raw", y, y+5, 0, seed=13, reps=50)
        for metric in ("bias", "mae", "rmse", "loa_low", "loa_high"):
            self.assertAlmostEqual(row[metric], 5.)
        self.assertEqual(row["within10_n"], len(y))
        self.assertAlmostEqual(row["pearson_r"], 1.)
        self.assertEqual(cmr.two_sided_bootstrap_p(np.array([-3., -2., -1.])), 0.)
        self.assertEqual(cmr.two_sided_bootstrap_p(np.array([-1., 1.])), 1.)
        low, high = cmr.fisher_ci(.6, 30)
        self.assertAlmostEqual(np.arctanh(high) - np.arctanh(.6), np.arctanh(.6) - np.arctanh(low))

    def test_rejects_incomplete_duplicate_and_nonfinite_data(self):
        rows = synthetic_reader_rows()
        for invalid in (rows.iloc[1:], pd.concat([rows, rows.iloc[:1]]), rows.assign(resize_ratio_y=0), rows.assign(reference_ef=np.nan)):
            with self.assertRaises(ValueError):
                cmr.prepare_reader_measurements(invalid)

    def test_subset_refit_and_shared_ed_interval(self):
        rows = synthetic_reader_rows()
        rows["reannotated"] = rows.case_id == "synthetic_case_00"
        results = cmr.analyze_reader_table(rows, seed=23, reps=25)
        subset = results["subsets"]["exclude_reannotated"]
        p = subset["predictions"]
        np.testing.assert_allclose(p.consensus_dynamic_loocv_ef,
                                  cmr.loocv_linear_predictions(p.consensus_dynamic_raw_ef.to_numpy(), p.reference_ef.to_numpy()))
        reliability = subset["reliability"]
        columns = [c for c in reliability if c.startswith("icc_")]
        pd.testing.assert_frame_equal(
            reliability.loc[(reliability.method == "static") & (reliability.metric == "lvidd"), columns].reset_index(drop=True),
            reliability.loc[(reliability.method == "dynamic") & (reliability.metric == "lvidd"), columns].reset_index(drop=True))
        mean_row = subset["agreement"].query("method == 'mean_only'").iloc[0]
        y = p.reference_ef.to_numpy()
        self.assertAlmostEqual(mean_row.mae, np.abs((y.sum()-y)/(len(y)-1)-y).mean())

    def test_paired_delta_mae_bootstrap_uses_frozen_predictions(self):
        cases = cmr.prepare_reader_measurements(synthetic_reader_rows())
        pred = cmr.build_harmonized_predictions(cases, set(cases.case_id))
        _, contrasts = cmr.manuscript_statistics(pred, seed=41, reps=30)
        y = pred.reference_ef.to_numpy()
        index = np.random.default_rng(41 + 500_000).integers(0, len(y), (30, len(y)))
        diff = np.abs(pred.consensus_dynamic_loocv_ef.to_numpy()-y) - np.abs(pred.consensus_static_loocv_ef.to_numpy()-y)
        expected_ci = np.quantile(diff[index].mean(axis=1), [.025, .975])
        row = contrasts.query("metric == 'delta_mae' and contrast == 'dynamic_minus_static'").iloc[0]
        np.testing.assert_allclose([row.ci_low, row.ci_high], expected_ci)
        self.assertAlmostEqual(row.observed, diff.mean())

    def test_cli_outputs_aggregate_tables_by_default(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_cmr.py"
        with tempfile.TemporaryDirectory(prefix="plax_synthetic_cmr_") as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            synthetic_reader_rows().to_csv(source, index=False)
            output = root / "analysis"
            result = subprocess.run([sys.executable, str(script), "--input", str(source),
                                     "--output-dir", str(output), "--reps", "20", "--seed", "7"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "primary" / "agreement_summary.csv").is_file())
            self.assertFalse((output / "primary" / "case_predictions.csv").exists())
            self.assertFalse((output / "reader_measurements_harmonized.csv").exists())
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["bootstrap_reps"], 20)
            self.assertFalse(manifest["case_level_outputs_written"])
            self.assertEqual(len(manifest["input_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
