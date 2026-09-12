"""Synthetic checks of the estimand, paired uncertainty, and patient isolation."""

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

import plax_measurement.clinical as clinical

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


def test_historical_families_change_only_their_own_interval_limits(monkeypatch):
    """Family choice must not rerun a different bootstrap or alter an estimand."""
    frame = synthetic_frame()
    frame["e10"] = np.clip(frame.f04 + np.random.default_rng(47).normal(0, 4, len(frame)), 0, 100)
    models = ["f04", "echonet_lvh", "report_lvef", "e10"]
    captured = []
    original = clinical.summarize

    def recording_summary(point, draws, family_size=1):
        captured.append(np.asarray(draws).copy())
        return original(point, draws, family_size)

    monkeypatch.setattr(clinical, "summarize", recording_summary)
    with threadpool_limits(limits=2):
        baseline = analyze(frame, model_columns=models, bootstrap_reps=150,
                           cv_bootstrap_reps=120, seed=918)
        baseline_draws = captured[:]
        for overrides, allowed in [
            ({"f04_family_size": 5}, "f04"),
            ({"base_increment_family_size": 6}, "base"),
        ]:
            captured.clear()
            expanded = analyze(frame, model_columns=models, bootstrap_reps=150,
                               cv_bootstrap_reps=120, seed=918, **overrides)
            assert len(captured) == len(baseline_draws)
            for old, new in zip(baseline_draws, captured):
                np.testing.assert_array_equal(old, new)

            changed = []

            def compare(old, new, path=()):
                assert old.keys() == new.keys()
                for key in old:
                    here = (*path, key)
                    if key == "comparison_families":
                        continue
                    if isinstance(old[key], dict):
                        compare(old[key], new[key], here)
                    elif key == "ci_family":
                        is_f04 = path[0] in {"contrasts", "unadjusted_contrasts"}
                        is_base = path[0] == "models" and path[-1] == "cv_rmse_improvement_vs_base"
                        affected = (allowed == "f04" and is_f04) or (allowed == "base" and is_base)
                        if affected:
                            assert new[key][0] <= old[key][0]
                            assert new[key][1] >= old[key][1]
                            if new[key] != old[key]:
                                changed.append(here)
                        else:
                            assert new[key] == old[key], here
                    else:
                        assert new[key] == old[key], here

            compare(baseline, expanded)
            assert changed, f"The {allowed} family did not affect any nondegenerate interval"
            assert expanded["base_coefficients"] == baseline["base_coefficients"]
            assert expanded["models"]["f04"]["partial_r"] == baseline["models"]["f04"]["partial_r"]
            families = expanded["comparison_families"]
            assert families["f04_contrasts_per_metric"] == (5 if allowed == "f04" else 3)
            assert families["base_increment_contrasts"] == (6 if allowed == "base" else 4)


@pytest.mark.parametrize("family, expected", [(5, [-3.94, 7.94]), (6, [-3.95, 7.95])])
def test_five_and_six_comparison_limits_match_uniform_quantiles(family, expected):
    # Equally spaced uniform draws have known exact linear percentile locations.
    # The 95% marginal interval is unchanged; corrected tails are 0.5% or 1/240.
    result = interval(np.linspace(-4, 8, 2401), family_size=family)
    np.testing.assert_allclose(result["ci95"], [-3.7, 7.7], atol=1e-12)
    np.testing.assert_allclose(result["ci_family"], expected, atol=1e-12)


@pytest.mark.parametrize("overrides", [
    {"f04_family_size": 1}, {"base_increment_family_size": 2},
    {"f04_family_size": 5.0}, {"base_increment_family_size": 6.5},
    {"f04_family_size": True}, {"base_increment_family_size": np.bool_(False)},
])
def test_invalid_family_sizes_are_rejected_before_analysis(overrides):
    with pytest.raises(ValueError, match="must be an integer at least"):
        analyze(synthetic_frame(), bootstrap_reps=1, cv_bootstrap_reps=1, **overrides)


def test_single_f04_keeps_legacy_default_without_creating_comparators():
    result = analyze(synthetic_frame(), model_columns=["f04"],
                     bootstrap_reps=20, cv_bootstrap_reps=20, seed=918)
    families = result["comparison_families"]
    assert families["f04_contrasts_per_metric"] == 1
    assert families["base_increment_contrasts"] == 1
    assert families["selected_f04_contrasts"] == 0
    assert families["selected_base_increments"] == 1
    assert result["contrasts"] == {}
    assert result["models"]["f04"]["partial_r"]["status"] == "ok"


def test_cli_forwards_historical_families_and_records_resolved_scope(tmp_path, monkeypatch):
    """Exercise argparse and JSON metadata without repeating expensive bootstraps."""
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_clinical.py"
    spec = importlib.util.spec_from_file_location("clinical_cli_under_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    frame = synthetic_frame()
    frame["e10"] = frame.f04
    source = tmp_path / "synthetic.csv"
    frame.to_csv(source, index=False)
    destination = tmp_path / "results"
    captured = {}

    def fake_analysis(supplied, **kwargs):
        captured.update(kwargs)
        assert len(supplied) == len(frame)
        return {"synthetic": {"repeated": {"status": "stubbed_for_cli_test"}}}

    monkeypatch.setattr(cli, "analyze_cohorts", fake_analysis)
    monkeypatch.setattr(sys, "argv", [str(script), "--input", str(source),
        "--output-dir", str(destination), "--model-columns", "f04", "echonet_lvh",
        "report_lvef", "e10", "--f04-family-size", "5", "--base-increment-family-size", "6",
        "--bootstrap-reps", "1000", "--cv-bootstrap-reps", "1000", "--seed", "918"])
    cli.main()
    assert captured["f04_family_size"] == 5
    assert captured["base_increment_family_size"] == 6
    assert captured["model_columns"] == ["f04", "echonet_lvh", "report_lvef", "e10"]
    result = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    families = result["design"]["comparison_families"]
    assert families["f04_contrasts_per_metric"] == 5
    assert families["base_increment_contrasts"] == 6
    assert families["selected_f04_contrasts"] == 3
    assert families["selected_base_increments"] == 4
    assert families["f04_family_explicit"] is True
    assert families["base_increment_family_explicit"] is True
    assert "per metric, variant and sensitivity" in families["scope"]
    assert result["analyses"]["synthetic"]["repeated"]["status"] == "stubbed_for_cli_test"
