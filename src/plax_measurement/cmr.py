"""Portable three-reader CMR statistics, adapted from the study analysis scripts.

See docs/cmr.md and docs/cmr_provenance.json for definitions and source hashes.
No annotation databases, imaging files, or participant identifiers are bundled.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260820
BOOTSTRAP_REPS = 10_000
READERS = ("reader01", "reader02", "reader03")
METHODS = ("static", "dynamic", "peak_to_peak")
PROTOCOL_SPACING_CM_PER_PX = 0.15


def prepare_reader_measurements(rows: pd.DataFrame) -> pd.DataFrame:
    """Validate long reader input and reconstruct nominal-scale diameters/EF.

    Each case has three readers by three methods. Distances are absolute display
    y separations; division by the stored vertical resize factor recovers native
    M-mode pixels. Dynamic ED deliberately comes from the same reader's static
    row, matching the source protocol. No EF or diameter clipping is performed.
    """
    required = {
        "case_id", "reader", "method", "lvidd_display_px", "lvids_display_px",
        "resize_ratio_y", "reference_ef",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    data = rows.copy()
    for column in ("case_id", "reader", "method"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must not contain missing or empty values")
        data[column] = data[column].astype(str)
    if set(data.reader) != set(READERS) or set(data.method) != set(METHODS):
        raise ValueError("Expected reader01/reader02/reader03 and static/dynamic/peak_to_peak")
    if data.duplicated(["case_id", "reader", "method"]).any():
        raise ValueError("Duplicate case/reader/method rows")
    if not data.groupby("case_id").size().eq(9).all():
        raise ValueError("Every case must have all nine reader/method rows")
    numeric = ["lvidd_display_px", "lvids_display_px", "resize_ratio_y", "reference_ef"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[numeric].to_numpy(float)).all():
        raise ValueError("Measurements and reference EF must be finite")
    if (data.resize_ratio_y <= 0).any() or (data.lvidd_display_px <= 0).any():
        raise ValueError("Resize factors and ED display distances must be positive")
    if (data.lvids_display_px < 0).any():
        raise ValueError("ES display distances must be nonnegative")
    if not data.groupby("case_id").reference_ef.nunique().eq(1).all():
        raise ValueError("Each case must have one consistent reference EF")

    cases = data.groupby("case_id", sort=True).reference_ef.first().reset_index()
    if len(cases) < 4:
        raise ValueError("At least four complete cases are required")
    cases["source_spacing_cm_per_px"] = PROTOCOL_SPACING_CM_PER_PX
    for reader in READERS:
        for method in METHODS:
            ed_method = "peak_to_peak" if method == "peak_to_peak" else "static"
            ed = data.loc[(data.reader == reader) & (data.method == ed_method)].set_index("case_id").loc[cases.case_id]
            es = data.loc[(data.reader == reader) & (data.method == method)].set_index("case_id").loc[cases.case_id]
            d = ed.lvidd_display_px.to_numpy(float) * PROTOCOL_SPACING_CM_PER_PX / ed.resize_ratio_y.to_numpy(float)
            s = es.lvids_display_px.to_numpy(float) * PROTOCOL_SPACING_CM_PER_PX / es.resize_ratio_y.to_numpy(float)
            cases[f"{reader}_{method}_lvidd_harmonized"] = d
            cases[f"{reader}_{method}_lvids_harmonized"] = s
            cases[f"{reader}_{method}_lvef_harmonized"] = teichholz_ef(d, s)

    for flag in ("same_es_functional", "complete_geometry", "reannotated"):
        if flag not in data:
            continue
        normalized = data[flag].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if normalized.isna().any():
            raise ValueError(f"{flag} must contain only true/false or 1/0")
        data[flag] = normalized
        if not data.groupby("case_id")[flag].nunique().eq(1).all():
            raise ValueError(f"{flag} must be consistent across each case's nine rows")
        cases[flag] = data.groupby("case_id")[flag].first().reindex(cases.case_id).to_numpy(bool)
    return cases


def sensitivity_masks(cases: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recreate source subsets; audit-derived flags must be supplied explicitly."""
    valid = np.ones(len(cases), dtype=bool)
    for reader in READERS:
        for method in METHODS:
            d = cases[f"{reader}_{method}_lvidd_harmonized"].to_numpy(float)
            s = cases[f"{reader}_{method}_lvids_harmonized"].to_numpy(float)
            valid &= (d > s) & (s > 0)
    subsets = {"primary": np.ones(len(cases), dtype=bool)}
    for flag in ("same_es_functional", "complete_geometry"):
        if flag in cases:
            subsets[flag] = cases[flag].to_numpy(bool)
    subsets["physiologic"] = valid
    if "reannotated" in cases:
        subsets["exclude_reannotated"] = ~cases.reannotated.to_numpy(bool)
    return subsets


def raw_error_contrasts(predictions: pd.DataFrame, *, seed: int = SEED,
                        reps: int = BOOTSTRAP_REPS) -> pd.DataFrame:
    """Paired raw-EF delta MAE from the September 10 reannotation update."""
    y = predictions.reference_ef.to_numpy(float)
    dynamic = predictions.consensus_dynamic_raw_ef.to_numpy(float)
    index = np.random.default_rng(seed + 710_000).integers(0, len(predictions), (reps, len(predictions)))
    rows = []
    for method in ("static", "peak_to_peak"):
        delta = abs(dynamic - y) - abs(predictions[f"consensus_{method}_raw_ef"].to_numpy(float) - y)
        boots = delta[index].mean(axis=1)
        low, high = percentile_ci(boots)
        rows.append(dict(metric="raw_delta_mae", contrast="dynamic_minus_" + method,
                         observed=float(delta.mean()), ci_low=low, ci_high=high,
                         p_two_sided=two_sided_bootstrap_p(boots), bootstrap_reps=reps))
    return pd.DataFrame(rows)


def full_sample_calibration_fits(predictions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive full-sample regression; these fits do not generate LOOCV EF."""
    rows = []
    for method in METHODS:
        x = predictions[f"consensus_{method}_raw_ef"]
        if np.ptp(x.to_numpy(float)) == 0:
            rows.append(dict(method=method, slope=np.nan, intercept=np.nan,
                             slope_ci_low=np.nan, slope_ci_high=np.nan, fitted_sd=np.nan))
            continue
        fit = stats.linregress(x, predictions.reference_ef)
        half = stats.t.ppf(.975, len(predictions) - 2) * fit.stderr
        rows.append(dict(method=method, slope=fit.slope, intercept=fit.intercept,
                         slope_ci_low=fit.slope - half, slope_ci_high=fit.slope + half,
                         fitted_sd=float((fit.intercept + fit.slope * x).std(ddof=1))))
    return pd.DataFrame(rows)


def analyze_reader_table(rows: pd.DataFrame, *, seed: int = SEED,
                         reps: int = BOOTSTRAP_REPS) -> dict[str, object]:
    """Run source statistics with per-subset LOOCV and a frozen OOF bootstrap.

    Return tables in memory; this function has no filesystem/database effects.
    Subsets of fewer than four cases are reported as skipped, without estimates.
    """
    if not isinstance(reps, (int, np.integer)) or reps < 1:
        raise ValueError("reps must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cases = prepare_reader_measurements(rows)
    results = {}
    flow = []
    for name, mask in sensitivity_masks(cases).items():
        n = int(mask.sum())
        flow.append(dict(analysis=name, n=n, excluded_n=int(len(cases) - n),
                         status="computed" if n >= 4 else "skipped_fewer_than_four_cases",
                         calibration="refitted LOOCV within this subset"))
        if n < 4:
            continue
        predictions = build_harmonized_predictions(cases, set(cases.loc[mask, "case_id"]))
        agreement, contrasts = manuscript_statistics(predictions, seed=seed, reps=reps)
        reliability = reliability_table(cases, mask, name, 50_000 if name == "physiologic" else 0,
                                        seed=seed, reps=reps)
        # ED is exactly shared; the source update reuses the static ED interval.
        icc_columns = [column for column in reliability if column.startswith("icc_")]
        static_ed = reliability.loc[(reliability.method == "static") & (reliability.metric == "lvidd"), icc_columns]
        reliability.loc[(reliability.method == "dynamic") & (reliability.metric == "lvidd"), icc_columns] = static_ed.to_numpy()
        results[name] = dict(predictions=predictions, agreement=agreement, contrasts=contrasts,
                             raw_error_contrasts=raw_error_contrasts(predictions, seed=seed, reps=reps),
                             reliability=reliability, thresholds=threshold_summary(predictions),
                             calibration_fits=full_sample_calibration_fits(predictions))
    return dict(subsets=results, sample_flow=pd.DataFrame(flow), harmonized=cases)

def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def icc_absolute(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    n, k = data.shape
    grand = data.mean()
    target_means = data.mean(axis=1)
    reader_means = data.mean(axis=0)
    ms_targets = k * np.sum((target_means - grand) ** 2) / (n - 1)
    ms_readers = n * np.sum((reader_means - grand) ** 2) / (k - 1)
    residual = data - target_means[:, None] - reader_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    single_denominator = (
        ms_targets
        + (k - 1) * ms_error
        + k * (ms_readers - ms_error) / n
    )
    average_denominator = ms_targets + (ms_readers - ms_error) / n
    return (
        float((ms_targets - ms_error) / single_denominator),
        float((ms_targets - ms_error) / average_denominator),
    )


def bootstrap_icc(
    values: np.ndarray, rng: np.random.Generator, *, reps: int = BOOTSTRAP_REPS
) -> tuple[tuple[float, float], tuple[float, float]]:
    n = len(values)
    single = np.empty(reps)
    average = np.empty(reps)
    for rep in range(reps):
        index = rng.integers(0, n, n)
        single[rep], average[rep] = icc_absolute(values[index])
    return percentile_ci(single), percentile_ci(average)


def metric_columns(method: str, metric: str) -> list[str]:
    return [f"{reader}_{method}_{metric}_harmonized" for reader in READERS]


def teichholz_volume(diameter: np.ndarray) -> np.ndarray:
    return 7.0 * diameter**3 / (2.4 + diameter)


def teichholz_ef(lvidd: np.ndarray, lvids: np.ndarray) -> np.ndarray:
    edv = teichholz_volume(lvidd)
    esv = teichholz_volume(lvids)
    return 100.0 * (edv - esv) / edv


def interpretation(value: float) -> str:
    if value < 0.5:
        return "poor"
    if value < 0.75:
        return "moderate"
    if value < 0.9:
        return "good"
    return "excellent"


def reliability_table(
    cases: pd.DataFrame, mask: pd.Series, source: str, seed_offset: int, *,
    seed: int = SEED, reps: int = BOOTSTRAP_REPS
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = cases.loc[mask].copy()
    for method_index, method in enumerate(METHODS):
        for metric_index, metric in enumerate(("lvidd", "lvids", "lvef")):
            values = selected[metric_columns(method, metric)].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Missing values in {source}: {method}/{metric}")
            single, average = icc_absolute(values)
            rng = np.random.default_rng(
                seed + seed_offset + method_index * 10_000 + metric_index * 1_000
            )
            single_ci, average_ci = bootstrap_icc(values, rng, reps=reps)
            rows.append(
                {
                    "analysis_source": source,
                    "method": method,
                    "metric": metric,
                    "unit": "percentage points" if metric == "lvef" else "cm",
                    "n": len(selected),
                    "reader01_mean": values[:, 0].mean(),
                    "reader02_mean": values[:, 1].mean(),
                    "reader03_mean": values[:, 2].mean(),
                    "icc_a1_absolute_single": single,
                    "icc_a1_ci_low": single_ci[0],
                    "icc_a1_ci_high": single_ci[1],
                    "icc_a1_interpretation": interpretation(single),
                    "icc_a3_absolute_average": average,
                    "icc_a3_ci_low": average_ci[0],
                    "icc_a3_ci_high": average_ci[1],
                    "bootstrap_reps": reps,
                    "value_definition": (
                        "reader-specific Teichholz LVEF, then arithmetic mean across 3 readers"
                        if metric == "lvef"
                        else (
                            "reader-specific static ED diameter shared by static and dynamic strategies"
                            if method == "dynamic" and metric == "lvidd"
                            else "reader-specific linear dimension"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def fisher_ci(r_value: float, n: int) -> tuple[float, float]:
    if abs(r_value) >= 1:
        return float(r_value), float(r_value)
    if n <= 3:
        return float("nan"), float("nan")
    z = np.arctanh(r_value)
    half_width = 1.959963984540054 / math.sqrt(n - 3)
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def wilson_ci(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = count / n
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / n + z**2 / (4 * n**2)
    ) / denominator
    return center - half, center + half


def agreement_row(
    method: str,
    stage: str,
    reference: np.ndarray,
    prediction: np.ndarray,
    seed_offset: int,
    calibration_slope: float = np.nan,
    calibration_intercept: float = np.nan,
    *, seed: int = SEED, reps: int = BOOTSTRAP_REPS,
) -> dict[str, object]:
    error = prediction - reference
    absolute = np.abs(error)
    bias = float(error.mean())
    sd = float(error.std(ddof=1))
    count = int(np.sum(absolute <= 10))
    within_low, within_high = wilson_ci(count, len(reference))
    r_value = float(np.corrcoef(prediction, reference)[0, 1])
    r_low, r_high = fisher_ci(r_value, len(reference))
    rng = np.random.default_rng(seed + seed_offset)
    index = rng.integers(0, len(reference), size=(reps, len(reference)))
    error_boot = error[index]
    bias_ci = percentile_ci(error_boot.mean(axis=1))
    mae_ci = percentile_ci(np.abs(error_boot).mean(axis=1))
    rmse_ci = percentile_ci(np.sqrt(np.mean(error_boot**2, axis=1)))
    return {
        "method": method,
        "stage": stage,
        "n": len(reference),
        "mean_prediction_ef": float(prediction.mean()),
        "prediction_sd": float(prediction.std(ddof=1)),
        "full_sample_calibration_slope": calibration_slope,
        "full_sample_calibration_intercept": calibration_intercept,
        "bias": bias,
        "bias_ci_low": bias_ci[0],
        "bias_ci_high": bias_ci[1],
        "mae": float(absolute.mean()),
        "mae_ci_low": mae_ci[0],
        "mae_ci_high": mae_ci[1],
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rmse_ci_low": rmse_ci[0],
        "rmse_ci_high": rmse_ci[1],
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "within10_n": count,
        "within10_pct": 100 * count / len(reference),
        "within10_ci_low_pct": 100 * within_low,
        "within10_ci_high_pct": 100 * within_high,
        "pearson_r": r_value,
        "pearson_ci_low": r_low,
        "pearson_ci_high": r_high,
    }


def two_sided_bootstrap_p(values: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0))))


def loocv_linear_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(y, dtype=float)
    for held_out in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[held_out] = False
        design = np.column_stack([x[keep], np.ones(np.sum(keep))])
        slope, intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        predictions[held_out] = slope * x[held_out] + intercept
    return predictions


def build_harmonized_predictions(
    cases: pd.DataFrame, cohort_uids: set[str]
) -> pd.DataFrame:
    selected = cases[cases["case_id"].isin(cohort_uids)].copy()
    selected = selected.sort_values("case_id")
    output = selected[["case_id", "reference_ef"]].copy()
    reference = output["reference_ef"].to_numpy(dtype=float)
    for method in METHODS:
        reader_columns = [
            f"{reader}_{method}_lvef_harmonized" for reader in READERS
        ]
        for column in reader_columns:
            output[column] = selected[column].to_numpy(dtype=float)
        raw = selected[reader_columns].mean(axis=1).to_numpy(dtype=float)
        output[f"consensus_{method}_raw_ef"] = raw
        output[f"consensus_{method}_loocv_ef"] = loocv_linear_predictions(
            raw, reference
        )
    return output


def manuscript_statistics(predictions: pd.DataFrame, *, seed: int = SEED, reps: int = BOOTSTRAP_REPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = predictions["reference_ef"].to_numpy(dtype=float)
    mean_only = (reference.sum() - reference) / (len(reference) - 1)
    arrays: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        raw_values = predictions[f"consensus_{method}_raw_ef"].to_numpy(dtype=float)
        design = np.column_stack([raw_values, np.ones(len(raw_values))])
        slope, intercept = np.linalg.lstsq(design, reference, rcond=None)[0]
        for stage, suffix in (("raw", "raw_ef"), ("loocv", "loocv_ef")):
            values = predictions[f"consensus_{method}_{suffix}"].to_numpy(dtype=float)
            arrays[(method, stage)] = values
            stage_index = 0 if stage == "raw" else 1
            rows.append(
                agreement_row(
                    method,
                    stage,
                    reference,
                    values,
                    600_000 + method_index * 10_000 + stage_index * 1_000,
                    float(slope),
                    float(intercept),
                    seed=seed, reps=reps,
                )
            )
    arrays[("mean_only", "loocv")] = mean_only
    rows.append(
        agreement_row(
            "mean_only", "loocv", reference, mean_only, 690_000, seed=seed, reps=reps
        )
    )
    summary = pd.DataFrame(rows)

    rng = np.random.default_rng(seed + 500_000)
    n = len(reference)
    index = rng.integers(0, n, size=(reps, n))
    contrasts: list[dict[str, object]] = []
    for comparator in ("static", "peak_to_peak"):
        dynamic_r = np.array(
            [
                np.corrcoef(
                    arrays[("dynamic", "raw")][sample], reference[sample]
                )[0, 1]
                for sample in index
            ]
        )
        comparator_r = np.array(
            [
                np.corrcoef(
                    arrays[(comparator, "raw")][sample], reference[sample]
                )[0, 1]
                for sample in index
            ]
        )
        values = dynamic_r - comparator_r
        low, high = percentile_ci(values)
        contrasts.append(
            {
                "metric": "delta_pearson_r",
                "contrast": f"dynamic_minus_{comparator}",
                "observed": (
                    np.corrcoef(arrays[("dynamic", "raw")], reference)[0, 1]
                    - np.corrcoef(arrays[(comparator, "raw")], reference)[0, 1]
                ),
                "ci_low": low,
                "ci_high": high,
                "p_two_sided": two_sided_bootstrap_p(values),
                "bootstrap_reps": reps,
            }
        )

    for comparator in ("static", "peak_to_peak", "mean_only"):
        dynamic_error = np.abs(arrays[("dynamic", "loocv")] - reference)
        comparator_error = np.abs(arrays[(comparator, "loocv")] - reference)
        values = (dynamic_error[index] - comparator_error[index]).mean(axis=1)
        low, high = percentile_ci(values)
        contrasts.append(
            {
                "metric": "delta_mae",
                "contrast": f"dynamic_minus_{comparator}",
                "observed": float(dynamic_error.mean() - comparator_error.mean()),
                "ci_low": low,
                "ci_high": high,
                "p_two_sided": two_sided_bootstrap_p(values),
                "bootstrap_reps": reps,
            }
        )

    mean_sse = np.sum((mean_only - reference) ** 2)
    for method in METHODS:
        method_sse = np.sum((arrays[(method, "loocv")] - reference) ** 2)
        contrasts.append(
            {
                "metric": "cross_validated_r2_vs_mean_only",
                "contrast": method,
                "observed": float(1 - method_sse / mean_sse),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_two_sided": np.nan,
                "bootstrap_reps": 0,
            }
        )
    return summary, pd.DataFrame(contrasts)


def threshold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    reference = predictions["reference_ef"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        prediction = predictions[f"consensus_{method}_loocv_ef"].to_numpy(dtype=float)
        for threshold in (50.0, 40.0):
            actual_positive = reference < threshold
            predicted_positive = prediction < threshold
            tp = int(np.sum(actual_positive & predicted_positive))
            fp = int(np.sum(~actual_positive & predicted_positive))
            fn = int(np.sum(actual_positive & ~predicted_positive))
            tn = int(np.sum(~actual_positive & ~predicted_positive))
            rows.append(
                {
                    "method": method,
                    "threshold_pct": threshold,
                    "reference_positive_n": int(np.sum(actual_positive)),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "sensitivity_pct": (
                        100 * tp / (tp + fn) if tp + fn else np.nan
                    ),
                    "specificity_pct": (
                        100 * tn / (tn + fp) if tn + fp else np.nan
                    ),
                    "accuracy_pct": 100 * (tp + tn) / len(reference),
                    "minimum_calibrated_prediction": float(prediction.min()),
                    "maximum_calibrated_prediction": float(prediction.max()),
                }
            )
    return pd.DataFrame(rows)
