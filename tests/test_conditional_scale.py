"""Synthetic checks for units, pairing and the diagnostic completeness boundary."""
from pathlib import Path
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from plax_measurement.cmr import METHODS, READERS, teichholz_ef
from plax_measurement.conditional_scale import (
    conditional_diameter_cm, diagnose_conditional_scale, effective_spacing_mm,
)


def synthetic_rows(n=5):
    rows = []
    for i in range(n):
        for ri, reader in enumerate(READERS):
            for mi, method in enumerate(METHODS):
                for phase in ('ED', 'ES'):
                    rows.append(dict(case_id=f'synthetic_{i:02d}', reader=reader, method=method, phase=phase,
                        nominal_diameter_cm=(3.5 + .8 * ri + .15 * i + .1 * mi) if phase == 'ED' else (2.5 + .35 * ri + .04 * i + .15 * mi),
                        line_dx_px=30 + 3 * ri - 2 * mi, line_dy_px=40 - 2 * ri + 3 * mi,
                        reference_ef=50 + 3 * i + i % 2, fov_x_mm=300 + 6 * i, fov_y_mm=280 + 4 * i,
                        recon_matrix_x=200, recon_matrix_y=160, stored_width_px=400))
    return pd.DataFrame(rows)


def explicit_axes(rows):
    rows = rows.copy()
    rows['a_sx_mm_per_px'] = rows.fov_x_mm / rows.recon_matrix_x
    rows['a_sy_mm_per_px'] = rows.fov_y_mm / rows.recon_matrix_y
    rows['b_sx_mm_per_px'] = rows.fov_x_mm / rows.stored_width_px
    rows['b_sy_mm_per_px'] = rows.fov_y_mm / rows.recon_matrix_y
    return rows


def test_axis_units_direction_sign_and_length_invariance():
    assert effective_spacing_mm(10, 0, 2, 3) == 2
    assert effective_spacing_mm(0, 10, 2, 3) == 3
    expected = math.sqrt((3 / 5 * 2) ** 2 + (4 / 5 * 3) ** 2)
    assert effective_spacing_mm(3, 4, 2, 3) == pytest.approx(expected, abs=1e-14)
    assert effective_spacing_mm(-3, -4, 2, 3) == effective_spacing_mm(3, 4, 2, 3)
    assert conditional_diameter_cm(3, 10, 0, 2, 3) == 4  # 30 mm * (2 / 1.5) = 40 mm
    assert conditional_diameter_cm(3, 100, 0, 2, 3) == 4  # line length is not the measured LVID
    assert conditional_diameter_cm(3, 10, 0, 3, 2) == 6  # x/y cannot be interchanged


@pytest.mark.parametrize('dx,dy', [(0, 0), (np.nan, 1), (1, np.inf)])
def test_unavailable_direction_never_gets_isotropic_fallback(dx, dy):
    assert np.isnan(effective_spacing_mm(dx, dy, 1.5, 1.5))


def test_exact_isotropic_parity_and_dynamic_shared_static_ed():
    rows = explicit_axes(synthetic_rows())
    rows[['a_sx_mm_per_px', 'a_sy_mm_per_px', 'b_sx_mm_per_px', 'b_sy_mm_per_px']] = 1.5
    unused = (rows.method == 'dynamic') & (rows.phase == 'ED')
    rows.loc[unused, ['nominal_diameter_cm', 'line_dx_px', 'line_dy_px']] = [999, 0, 0]
    result = diagnose_conditional_scale(rows, 'axes')
    assert result.aggregate['selection']['common_diagnostic_cases'] == 5
    assert result.aggregate['validation']['isotropic_1_5_vs_nominal_EF_max_difference_pp'] == 0
    m = result.measurements
    assert np.array_equal(m.nominal_EF, m.conditional_EF)
    paired = m.pivot(index=['case_id', 'scenario', 'reader'], columns='method', values='conditional_ED_cm')
    assert np.array_equal(paired.static, paired.dynamic)
    assert not np.array_equal(paired.static, paired.peak_to_peak)


def test_missing_one_used_direction_excludes_whole_case_for_both_assumptions():
    rows = synthetic_rows()
    missing = (rows.case_id == 'synthetic_01') & (rows.reader == 'reader02') & (rows.method == 'dynamic') & (rows.phase == 'ES')
    rows.loc[missing, 'line_dx_px'] = np.nan
    result = diagnose_conditional_scale(rows)
    assert result.aggregate['selection']['common_diagnostic_cases'] == 4
    assert result.aggregate['selection']['excluded_missing_direction_cases'] == 1
    assert 'synthetic_01' not in set(result.measurements.case_id)
    assert set(result.measurements.groupby(['scenario', 'method']).size()) == {12}
    for scenario in ('A', 'B'):
        assert {v['n'] for v in result.aggregate['results'][scenario]['consensus'].values()} == {4}


def test_no_available_reader_average_and_no_incomplete_phase_rows():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match='18'):
        diagnose_conditional_scale(rows.iloc[1:])
    with pytest.raises(ValueError, match='Duplicate'):
        diagnose_conditional_scale(pd.concat([rows, rows.iloc[:1]]))


def test_mean_of_three_efs_precedes_comparator_statistics():
    result = diagnose_conditional_scale(synthetic_rows())
    rows = result.measurements.query("scenario == 'B' and method == 'dynamic'")
    means = rows.groupby('case_id').mean(numeric_only=True)
    ef_of_mean_diameter = teichholz_ef(means.conditional_ED_cm.to_numpy(), means.conditional_ES_cm.to_numpy())
    assert np.max(abs(ef_of_mean_diameter - means.conditional_EF)) > .1
    expected_mae = np.abs(means.conditional_EF - means.reference_EF).mean()
    actual = result.aggregate['results']['B']['consensus']['dynamic']['conditional_agreement_same_subset']['mae_pp']
    assert actual == pytest.approx(expected_mae, abs=1e-13)


def test_explicit_axes_and_fov_formulas_have_identical_outputs():
    rows = synthetic_rows()
    a = diagnose_conditional_scale(rows)
    b = diagnose_conditional_scale(explicit_axes(rows), 'axes')
    assert a.aggregate['results'] == b.aggregate['results']
    pd.testing.assert_frame_equal(a.measurements, b.measurements)


@pytest.mark.parametrize('column,value,message', [
    ('fov_x_mm', 0, 'positive'), ('recon_matrix_x', 200.5, 'integers'),
    ('reference_ef', np.nan, 'finite'), ('nominal_diameter_cm', -1, 'diameters'),
])
def test_invalid_units_and_measurements_fail(column, value, message):
    rows = synthetic_rows()
    rows[column] = value
    with pytest.raises(ValueError, match=message):
        diagnose_conditional_scale(rows)


def test_case_metadata_must_be_consistent_and_reader_names_explicit():
    rows = synthetic_rows()
    rows.loc[0, 'fov_x_mm'] += 1
    with pytest.raises(ValueError, match='identical'):
        diagnose_conditional_scale(rows)
    rows = synthetic_rows()
    rows.loc[0, 'reader'] = 'fourth_reader'
    with pytest.raises(ValueError, match='Unexpected reader'):
        diagnose_conditional_scale(rows)


def test_no_ef_clipping_and_undefined_correlation_is_null():
    rows = synthetic_rows()
    rows.loc[rows.phase == 'ES', 'nominal_diameter_cm'] = 12
    rows.reference_ef = 50
    result = diagnose_conditional_scale(rows)
    assert (result.measurements.conditional_EF < 0).all()
    for scenario in ('A', 'B'):
        assert result.aggregate['results'][scenario]['consensus']['static']['conditional_agreement_same_subset']['pearson_r'] is None
    json.dumps(result.aggregate, allow_nan=False)


def test_optional_frozen_reader_values_are_a_check_not_a_target():
    rows = synthetic_rows()
    base = diagnose_conditional_scale(rows).measurements.query("scenario == 'A'")
    rows = rows.merge(base[['case_id', 'reader', 'method', 'nominal_EF']], on=['case_id', 'reader', 'method'])
    rows = rows.rename(columns={'nominal_EF': 'frozen_reader_ef'})
    assert diagnose_conditional_scale(rows).aggregate['validation']['nominal_vs_supplied_frozen_max_EF_residual_pp'] == 0
    rows.frozen_reader_ef += 1
    with pytest.raises(ValueError, match='disagrees'):
        diagnose_conditional_scale(rows)


def test_cli_writes_aggregate_only_by_default_and_preserves_existing_output(tmp_path):
    source = tmp_path / 'synthetic.csv'
    synthetic_rows().to_csv(source, index=False)
    output = tmp_path / 'result'
    command = [sys.executable, str(Path(__file__).parents[1] / 'scripts/analyze_conditional_scale.py'),
               '--input', str(source), '--output-dir', str(output)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert [p.name for p in output.iterdir()] == ['conditional_scale_aggregate.json']
    text = (output / 'conditional_scale_aggregate.json').read_text()
    assert 'synthetic_00' not in text
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and 'new or empty' in again.stderr
