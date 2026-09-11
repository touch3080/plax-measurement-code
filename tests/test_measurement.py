"""Synthetic numerical and failure-mode tests; no clinical data or weights."""
import unittest

import numpy as np

from plax_measurement.measurement import (
    analyze_trajectory, build_segmented_plax_curve_payload, geometry,
    pair_volume_extrema, same_phase_comparison, smooth_t01, teichholz_volume_cm,
    validate_endpoint_rows,
)


def rows_from_diameters(diameters):
    return [dict(frame_id=i, x1=10., y1=20., x2=10. + float(d), y2=20., confidence=1.)
            for i, d in enumerate(diameters)]


class MeasurementTests(unittest.TestCase):
    def test_teichholz_units_and_geometry(self):
        self.assertAlmostEqual(float(teichholz_volume_cm(5.)), 875 / 7.4)
        result = same_phase_comparison([[0, 0], [100, 0]], [[20, 0], [80, 0]],
                                       [[30, 10], [70, 10]], spacing_cm=.05)
        self.assertAlmostEqual(result["dynamic"]["midpoint_offset_over_ed"], .1)
        self.assertAlmostEqual(result["dynamic"]["acute_angle_deg"], 0.)
        self.assertAlmostEqual(result["delta_lvids_px"], -20.)
        self.assertGreater(result["delta_ef_percentage_points"], 0.)

    def test_axis_offset_is_perpendicular_and_swap_invariant(self):
        ed = np.array([[0., 0.], [10., 0.]])
        es = np.array([[100., 3.], [110., 3.]])
        expected = geometry(ed, es)
        self.assertAlmostEqual(expected["midpoint_offset_px"], 3.)
        self.assertEqual(expected, geometry(ed[::-1], es[::-1]))
        vertical = geometry(ed, [[5., -5.], [5., 5.]])
        self.assertAlmostEqual(vertical["acute_angle_deg"], 90.)

    def test_t01_removes_isolated_outlier_and_endpoint_swaps(self):
        rows = rows_from_diameters([20.] * 9)
        rows[4].update(x1=200., x2=220., confidence=.1)
        rows[6].update(x1=30., x2=10.)
        output, metrics = smooth_t01(rows)
        self.assertEqual(metrics["outlier_count"], 1)
        self.assertEqual(output[4]["temporal_outlier"], 1)
        self.assertEqual(output[6]["temporal_outlier"], 0)
        np.testing.assert_allclose([[r[k] for k in ("x1_temporal", "y1_temporal", "x2_temporal", "y2_temporal")]
                                    for r in output], np.tile([10., 20., 30., 20.], (9, 1)))
        self.assertTrue(metrics["uses_future_frames"])

    def test_chronological_pairing_uses_lowest_es_before_next_ed(self):
        pairs = pair_volume_extrema([10, 9, 3, 4, 12, 8, 6], [0, 4], [1, 2, 5, 6])
        self.assertEqual(pairs, [(0, 2, 70.), (4, 6, 50.)])

    def test_automatic_cycles_and_constant_signal(self):
        diameters = 70 + 15 * np.cos(2 * np.pi * np.arange(121) / 30)
        result = analyze_trajectory(rows_from_diameters(diameters), spacing_cm=.05, fps=30)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["payload"]["ed_index"], [30, 60, 90])
        self.assertEqual(result["payload"]["es_index"], [15, 45, 75, 105])
        self.assertEqual(result["cycle_count"], 3)
        self.assertGreater(result["lvef_percent"], 0)
        flat = analyze_trajectory(rows_from_diameters([70] * 40), spacing_cm=.05)
        self.assertEqual(flat["status"], "no_valid_cycle")
        self.assertIsNone(flat["lvef_percent"])

    def test_invalid_gap_never_forms_cycle(self):
        points = np.array([[[0., 0.], [20., 0.]]] * 20)
        points[14, 1, 0] = 10.
        valid = [True] * 20
        valid[10] = False
        payload = build_segmented_plax_curve_payload(points, valid, .05,
                                                     marker_indices=([5], [14]))
        self.assertEqual(payload["ef"], [])
        self.assertEqual(payload["excluded_frames"], [11])
        self.assertEqual(len(payload["valid_segments"]), 2)

    def test_input_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            teichholz_volume_cm(0)
        with self.assertRaises(ValueError):
            geometry([[0, 0], [0, 0]], [[0, 0], [1, 0]])
        bad = rows_from_diameters([20, 20])
        bad[1]["frame_id"] = 3
        with self.assertRaises(ValueError):
            validate_endpoint_rows(bad)
        with self.assertRaises(ValueError):
            analyze_trajectory(rows_from_diameters([20] * 20), spacing_cm=-.1)


if __name__ == "__main__":
    unittest.main()
