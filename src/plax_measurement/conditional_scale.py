"""Conditional A/B CMR scale diagnostics on explicitly supplied local records.

This module does not select source candidates, establish identity, measure scanner
spacing, reconstruct images, fit a calibration or change frozen study estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy import stats

from .cmr import METHODS, READERS, teichholz_ef

PHASES = ('ED', 'ES')
NOMINAL_MM_PER_PIXEL = 1.5
FOV_COLUMNS = ('fov_x_mm', 'fov_y_mm', 'recon_matrix_x', 'recon_matrix_y', 'stored_width_px')
AXIS_COLUMNS = ('a_sx_mm_per_px', 'a_sy_mm_per_px', 'b_sx_mm_per_px', 'b_sy_mm_per_px')


@dataclass
class ConditionalScaleResult:
    aggregate: dict
    measurements: pd.DataFrame
    selection: pd.DataFrame


def effective_spacing_mm(line_dx, line_dy, sx, sy):
    """mm/source pixel along a line; missing/zero directions stay unavailable.

    x denotes stored-frame columns and y denotes rows. sx/sy must use these axes.
    Sign reversal has no effect. Equal axis scales use the exact isotropic value.
    """
    dx, dy, sx, sy = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (line_dx, line_dy, sx, sy)])
    if not (np.isfinite(sx).all() and np.isfinite(sy).all() and (sx > 0).all() and (sy > 0).all()):
        raise ValueError('Conditional axis scales must be finite positive mm/pixel values')
    length = np.fromiter((math.hypot(x, y) for x, y in zip(dx.flat, dy.flat)), dtype=float).reshape(dx.shape)
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(length) & (length > 0)
    ux = np.divide(dx, length, out=np.full(dx.shape, np.nan), where=valid)
    uy = np.divide(dy, length, out=np.full(dy.shape, np.nan), where=valid)
    spacing = np.where(sx == sy, sx, np.hypot(ux * sx, uy * sy))
    return np.where(valid, spacing, np.nan)


def conditional_diameter_cm(nominal_cm, line_dx, line_dy, sx, sy):
    """Rescale a frozen nominal diameter; line length supplies direction only."""
    return np.asarray(nominal_cm, dtype=float) * (
        effective_spacing_mm(line_dx, line_dy, sx, sy) / NOMINAL_MM_PER_PIXEL)


def _summary(values):
    a = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(a).all():
        raise ValueError('Summary requires the common complete subset')
    return dict(n=int(a.size), mean=float(a.mean()), min=float(a.min()),
                q1=float(np.quantile(a, .25)), median=float(np.median(a)),
                q3=float(np.quantile(a, .75)), max=float(a.max()),
                mean_absolute=float(np.abs(a).mean()), max_absolute=float(np.abs(a).max()))


def _agreement(pred, ref):
    delta = pred - ref
    constant = np.ptp(pred) == 0 or np.ptp(ref) == 0
    return dict(n=int(len(delta)), bias_pp=float(delta.mean()), mae_pp=float(np.abs(delta).mean()),
                rmse_pp=float(np.sqrt(np.mean(delta ** 2))),
                pearson_r=None if constant else float(stats.pearsonr(pred, ref).statistic),
                spearman_rho=None if constant else float(stats.spearmanr(pred, ref).statistic))


def _difference(a, b):
    return None if a is None or b is None else a - b


def _prepare(rows, scale_mode):
    if scale_mode not in ('fov', 'axes'):
        raise ValueError('scale_mode must be fov or axes')
    scale_columns = FOV_COLUMNS if scale_mode == 'fov' else AXIS_COLUMNS
    required = {'case_id', 'reader', 'method', 'phase', 'nominal_diameter_cm',
                'line_dx_px', 'line_dy_px', 'reference_ef', *scale_columns}
    if required - set(rows.columns):
        raise ValueError('Missing columns: ' + ', '.join(sorted(required - set(rows.columns))))
    data = rows.copy()
    keys = ['case_id', 'reader', 'method', 'phase']
    for key in keys:
        if data[key].isna().any() or data[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'{key} must contain nonempty values')
        data[key] = data[key].astype(str)
    for name, allowed in [('reader', READERS), ('method', METHODS), ('phase', PHASES)]:
        if not set(data[name]).issubset(allowed):
            raise ValueError(f'Unexpected {name}; expected {allowed}')
    if data.duplicated(keys).any():
        raise ValueError('Duplicate case/reader/method/phase row')
    if data.empty or not data.groupby('case_id').size().eq(18).all():
        raise ValueError('Every case must have all 18 three-reader/three-method/ED-ES rows')
    numeric = ['nominal_diameter_cm', 'line_dx_px', 'line_dy_px', 'reference_ef', *scale_columns]
    if 'frozen_reader_ef' in data:
        numeric.append('frozen_reader_ef')
    for key in numeric:
        data[key] = pd.to_numeric(data[key], errors='raise')
    for key in ['reference_ef', *scale_columns]:
        if not np.isfinite(data[key]).all() or not data.groupby('case_id')[key].nunique().eq(1).all():
            raise ValueError(f'{key} must be finite and identical across each case')
    if (data[list(scale_columns)] <= 0).any().any():
        raise ValueError('FOV, grid dimensions and conditional mm/pixel scales must be positive')
    if scale_mode == 'fov':
        grid = data[list(FOV_COLUMNS[2:])].to_numpy(float)
        if not np.equal(grid, np.floor(grid)).all():
            raise ValueError('Grid dimensions must be positive integers')
    uids = sorted(data.case_id.unique())
    indexed = data.set_index(keys).sort_index()
    nominal = np.full((len(uids), 3, 3, 2), np.nan)
    delta = np.full(nominal.shape + (2,), np.nan)
    frozen = np.full(nominal.shape[:-1], np.nan)
    for i, uid in enumerate(uids):
        for ri, reader in enumerate(READERS):
            for mi, method in enumerate(METHODS):
                for pi, phase in enumerate(PHASES):
                    source_method = 'static' if method == 'dynamic' and phase == 'ED' else method
                    row = indexed.loc[(uid, reader, source_method, phase)]
                    nominal[i, ri, mi, pi] = row.nominal_diameter_cm
                    delta[i, ri, mi, pi] = row[['line_dx_px', 'line_dy_px']].to_numpy(float)
                if 'frozen_reader_ef' in data:
                    values = indexed.loc[(uid, reader, method), 'frozen_reader_ef'].to_numpy(float)
                    if not np.isfinite(values).all() or not np.equal(values, values[0]).all():
                        raise ValueError('frozen_reader_ef must agree between both phases')
                    frozen[i, ri, mi] = values[0]
    if not np.isfinite(nominal).all() or (nominal[..., 0] <= 0).any() or (nominal[..., 1] < 0).any():
        raise ValueError('Used nominal ED diameters must be positive and ES nonnegative, all finite, in cm')
    first = data.groupby('case_id', sort=True).first().loc[uids]
    if scale_mode == 'fov':
        axes = {'A': np.column_stack((first.fov_x_mm / first.recon_matrix_x, first.fov_y_mm / first.recon_matrix_y)),
                'B': np.column_stack((first.fov_x_mm / first.stored_width_px, first.fov_y_mm / first.recon_matrix_y))}
    else:
        axes = {s: first[[f'{s.lower()}_sx_mm_per_px', f'{s.lower()}_sy_mm_per_px']].to_numpy(float) for s in ('A', 'B')}
    return uids, nominal, delta, first.reference_ef.to_numpy(float), axes, frozen if 'frozen_reader_ef' in data else None


def diagnose_conditional_scale(rows: pd.DataFrame, scale_mode='fov') -> ConditionalScaleResult:
    """Compare both declared assumptions on one complete set of supplied cases.

    Caller performs source-candidate selection before supplying rows. Missing or
    degenerate used directions exclude the entire case from A/B and all methods;
    incomplete row sets and invalid diameter/scale/reference fields raise errors.
    """
    uids, nominal, delta, gold, axes, frozen = _prepare(rows, scale_mode)
    dx, dy = delta[..., 0], delta[..., 1]
    iso = effective_spacing_mm(dx, dy, NOMINAL_MM_PER_PIXEL, NOMINAL_MM_PER_PIXEL)
    complete = np.isfinite(iso).all(axis=(1, 2, 3))
    if complete.sum() < 3:
        raise ValueError('At least three common complete cases are required')
    nominal_ef = teichholz_ef(nominal[..., 0], nominal[..., 1])
    if not np.isfinite(nominal_ef).all():
        raise ValueError('Nominal EF must be finite')
    nominal_cons = nominal_ef.mean(axis=1)
    isotropic = nominal * (iso / NOMINAL_MM_PER_PIXEL)
    isotropic_ef = teichholz_ef(isotropic[..., 0], isotropic[..., 1])
    assert np.array_equal(isotropic_ef[complete], nominal_ef[complete])
    residual = None if frozen is None else float(np.max(np.abs(nominal_ef - frozen)))
    if residual is not None and residual > 1e-10:
        raise ValueError('Nominal EF disagrees with supplied frozen_reader_ef by more than 1e-10 percentage points')
    aggregate = {'scope': 'Conditional anisotropic scale point diagnostics; neither A nor B is verified physical spacing.',
                 'scale_mode': scale_mode, 'selection': {'supplied_preselected_cases': len(uids),
                    'common_diagnostic_cases': int(complete.sum()), 'excluded_missing_direction_cases': int((~complete).sum()),
                    'same_cases_all_scenarios_readers_methods': True},
                 'validation': {'isotropic_1_5_vs_nominal_EF_max_difference_pp': 0.,
                    'nominal_vs_supplied_frozen_max_EF_residual_pp': residual, 'dynamic_ED_shared_static_exact': True},
                 'results': {}, 'limitations': [
                    'Source identity, grid/axis correspondence and FOV semantics must be evaluated upstream; this CLI does not verify them.',
                    'A/B are explicit assumptions, not fitted to the reference, and do not exhaust possible spatial transforms.',
                    'No missing direction is filled and no available-reader averaging, EF clipping, calibration, refitting or bootstrap is performed.',
                    'The supplied frozen reference is a comparator, not an independently validated clinical ground truth.']}
    measurements = []
    for scenario, axis in axes.items():
        sx, sy = axis[:, 0, None, None, None], axis[:, 1, None, None, None]
        effective = effective_spacing_mm(dx, dy, sx, sy)
        diameter = nominal * (effective / NOMINAL_MM_PER_PIXEL)
        ef = teichholz_ef(diameter[..., 0], diameter[..., 1])
        consensus = ef.mean(axis=1)
        if not np.isfinite(ef[complete]).all():
            raise ValueError('Conditional EF is nonfinite for the common subset')
        current = {'case_axis_spacing_mm_per_pixel': {'x': _summary(axis[complete, 0]), 'y': _summary(axis[complete, 1])},
                   'effective_used_phase_spacing_mm_per_pixel': _summary(effective[complete]),
                   'consensus': {}, 'reader': {}, 'same_subset_method_point_comparisons': {}}
        for mi, method in enumerate(METHODS):
            current['consensus'][method] = {'n': int(complete.sum()),
                'EF_change_pp': _summary(consensus[complete, mi] - nominal_cons[complete, mi]),
                'nominal_agreement_same_subset': _agreement(nominal_cons[complete, mi], gold[complete]),
                'conditional_agreement_same_subset': _agreement(consensus[complete, mi], gold[complete])}
            for ri, reader in enumerate(READERS):
                current['reader'][reader + '_' + method] = {
                    'EF_change_pp': _summary(ef[complete, ri, mi] - nominal_ef[complete, ri, mi]),
                    'ED_diameter_change_mm': _summary(10 * (diameter[complete, ri, mi, 0] - nominal[complete, ri, mi, 0])),
                    'ES_diameter_change_mm': _summary(10 * (diameter[complete, ri, mi, 1] - nominal[complete, ri, mi, 1]))}
                for i in np.flatnonzero(complete):
                    measurements.append(dict(case_id=uids[i], scenario=scenario, reader=reader, method=method,
                        sx_mm=axis[i, 0], sy_mm=axis[i, 1], effective_ED_mm=effective[i, ri, mi, 0],
                        effective_ES_mm=effective[i, ri, mi, 1], nominal_ED_cm=nominal[i, ri, mi, 0],
                        nominal_ES_cm=nominal[i, ri, mi, 1], conditional_ED_cm=diameter[i, ri, mi, 0],
                        conditional_ES_cm=diameter[i, ri, mi, 1], nominal_EF=nominal_ef[i, ri, mi],
                        conditional_EF=ef[i, ri, mi], EF_change_pp=ef[i, ri, mi] - nominal_ef[i, ri, mi], reference_EF=gold[i]))
        for mi, method in enumerate(METHODS[1:], 1):
            points = {}
            for label, values in [('nominal', nominal_cons), ('conditional', consensus)]:
                points[label + '_paired_MAE_difference_pp'] = float(np.mean(
                    abs(values[complete, mi] - gold[complete]) - abs(values[complete, 0] - gold[complete])))
                key = label + '_agreement_same_subset'
                points[label + '_r_difference'] = _difference(current['consensus'][method][key]['pearson_r'],
                                                             current['consensus']['static'][key]['pearson_r'])
            current['same_subset_method_point_comparisons'][method + '_minus_static'] = points
        aggregate['results'][scenario] = current
    selection = pd.DataFrame({'case_id': uids, 'all_used_directions_complete': complete,
                              'included_common_diagnostic_subset': complete})
    return ConditionalScaleResult(aggregate, pd.DataFrame(measurements), selection)
