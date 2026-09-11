"""Synthetic checks of the estimand, paired uncertainty, and patient isolation."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from plax_measurement.clinical import (
    adjusted_batch, analyze, analyze_cohorts, cluster_weights, group_cv, interval,
    rank_kernel, validate_frame, weighted_fit, weighted_rho_from_kernels,
)


def synthetic_frame():
    rng = np.random.default_rng(418)
    counts = np.array([3, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1])
    codes = np.repeat(np.arange(len(counts)), counts)
    n = len(codes)
    age = rng.uniform(30, 85, len(counts))[codes] + rng.normal(0, .1, n)
    renal = rng.normal(0, .5, n)
    ef = rng.uniform(22, 72, n)
    y = 2.7 + .012 * age + .3 * renal - .018 * ef + rng.normal(0, .25, n)
    return pd.DataFrame({"patient_id": [f"synthetic-{g:02}" for g in codes],
        "study_id": [f"synthetic-study-{i:03}" for i in range(n)], "age_proxy": age,
        "log2_creatinine": renal, "ntprobnp": 10 ** y, "f04": ef,
        "echonet_lvh": np.clip(ef + rng.normal(0, 6, n), 0, 100),
        "report_lvef": np.clip(ef + rng.normal(0, 8, n), 0, 100)})


@pytest.mark.parametrize("patient_equal", [False, True])
def test_partial_correlation_matches_independent_weighted_residualization(patient_equal):
    frame = synthetic_frame()
    codes, _ = pd.factorize(frame.patient_id)
    weights = np.ones(len(frame))
    if patient_equal:
        weights /= np.bincount(codes)[codes]
    base = np.column_stack([np.ones(len(frame)), frame.age_proxy / 10, frame.log2_creatinine])
    ef = frame.f04.to_numpy() / 10
    y = np.log10(frame.ntprobnp.to_numpy())
    sqrtw = np.sqrt(weights)
    residuals = []
    for value in (ef, y):
        beta = np.linalg.lstsq(base * sqrtw[:, None], value * sqrtw, rcond=None)[0]
        residuals.append(value - base @ beta)
    rx, ry = residuals
    expected = np.sum(weights * rx * ry) / np.sqrt(np.sum(weights * rx ** 2) * np.sum(weights * ry ** 2))
    fit = adjusted_batch({"base": base, "f04": np.column_stack([base, ef])}, y, weights)
    np.testing.assert_allclose(fit["f04"]["partial_r"][0], expected, atol=1e-10)


@pytest.mark.parametrize("patient_equal", [False, True])
def test_bootstrap_cv_matches_explicit_refits_removing_all_patient_copies(patient_equal):
    frame = synthetic_frame()
    codes, names = pd.factorize(frame.patient_id)
    x = np.column_stack([np.ones(len(frame)), frame.age_proxy / 10, frame.log2_creatinine, frame.f04 / 10])
    y = np.log10(frame.ntprobnp.to_numpy())
    # The first patient has three visits and is drawn three times. Removing one
    # visit or one resampled copy would leak the held patient's other observations.
    mult = np.array([3, 0, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    count = np.bincount(codes)
    weights = mult[codes].astype(float)
    if patient_equal:
        weights /= count[codes]
    expected = np.full(len(y), np.nan)
    for held in range(len(names)):
        train = (codes != held) & (weights > 0)
        test = codes == held
        assert not set(codes[train]) & set(codes[test])
        beta = np.linalg.lstsq(x[train] * np.sqrt(weights[train, None]), y[train] * np.sqrt(weights[train]), rcond=None)[0]
        expected[test] = x[test] @ beta
    actual = group_cv(x, y, codes, mult, patient_equal)
    np.testing.assert_allclose(actual["prediction"][0], expected, atol=1e-10)
    expected_rmse = np.sqrt(np.average((expected - y) ** 2, weights=weights))
    np.testing.assert_allclose(actual["rmse"][0], expected_rmse, atol=1e-10)
    changed = y.copy()
    changed[codes == 0] += 9
    changed_fit = group_cv(x, changed, codes, mult, patient_equal)
    np.testing.assert_allclose(changed_fit["prediction"][0, codes == 0], expected[codes == 0], atol=1e-9)
    fixed = group_cv(x, y, codes, np.ones(len(names)), patient_equal)["prediction"][0]
    fixed_error_bootstrap = np.sqrt(np.average((fixed - y) ** 2, weights=weights))
    assert abs(fixed_error_bootstrap - actual["rmse"][0]) > 1e-4


def test_cluster_weights_keep_visits_together_and_patient_equal_totals():
    codes = np.array([0, 0, 0, 1, 2, 2])
    visit = cluster_weights(codes, 20, np.random.default_rng(9))
    equal = cluster_weights(codes, 20, np.random.default_rng(9), True)
    np.testing.assert_array_equal(visit[:, 0], visit[:, 1])
    np.testing.assert_allclose(equal * np.bincount(codes)[codes], visit)
    np.testing.assert_allclose(equal.sum(1), 3)


def test_weighted_midranks_match_explicit_duplicate_sample_with_ties():
    x = np.array([1., 2., 2., 4., 7.])
    y = np.array([5., 5., 3., 2., 1.])
    weights = np.array([2, 0, 3, 1, 2])
    expected = spearmanr(np.repeat(x, weights), np.repeat(y, weights)).statistic
    actual = weighted_rho_from_kernels(weights, [rank_kernel(x)], rank_kernel(y))[0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_identical_models_have_zero_paired_uncertainty_and_no_identifiers_exported():
    frame = synthetic_frame()
    frame["echonet_lvh"] = frame.f04
    result = analyze(frame, bootstrap_reps=120, cv_bootstrap_reps=120, seed=21)
    for contrast in result["contrasts"]["echonet_lvh"].values():
        assert contrast["status"] == "ok"
        assert contrast["point"] == 0
        assert contrast["ci95"] == [0, 0]
        assert contrast["ci_family"] == [0, 0]
    assert result["models"]["f04"]["partial_r"]["ci95"][0] < result["models"]["f04"]["partial_r"]["ci95"][1]
    assert "synthetic-" not in json.dumps(result)


def test_singular_design_is_reported_without_regularization():
    x = np.ones((12, 2))
    fit = weighted_fit(x, np.arange(12.), np.ones(12))
    assert np.isnan(fit["beta"]).all()
    assert interval([np.nan] * 10)["status"] == "insufficient_valid_bootstrap_draws"
    assert interval([])["status"] == "insufficient_valid_bootstrap_draws"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "zero", "censored"])
def test_invalid_inputs_are_not_silently_dropped(mutation):
    frame = synthetic_frame()
    if mutation == "missing":
        frame.loc[0, "echonet_lvh"] = np.nan
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "zero":
        frame.loc[0, "ntprobnp"] = 0
    else:
        frame["lab_censored"] = "false"
        frame.loc[0, "lab_censored"] = "true"
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_single_visit_requires_explicit_chronology():
    frame = synthetic_frame()
    with pytest.raises(ValueError, match="study_order"):
        analyze_cohorts(frame, sensitivities=["single_visit"], bootstrap_reps=2, cv_bootstrap_reps=2)
    frame["study_order"] = np.arange(len(frame))
    result = analyze_cohorts(frame, sensitivities=["single_visit"], bootstrap_reps=2, cv_bootstrap_reps=2)
    assert result["supplied"]["single_visit"]["visits"] == frame.patient_id.nunique()
