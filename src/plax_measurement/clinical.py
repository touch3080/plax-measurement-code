"""Portable continuous clinical association and patient-cluster validation.

Adapted from the MIT-licensed study scripts identified in
docs/clinical_provenance.json. No clinical rows or application access are bundled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MODELS = ("f04", "echonet_lvh", "report_lvef")
DEFAULT_SEED = 20260911


def checked_solve(xx, xy):
    """Flag unidentified/ill-conditioned designs without regularizing them."""
    eigenvalues = np.linalg.eigvalsh(xx)
    valid = (eigenvalues[..., 0] > 1e-12 * eigenvalues[..., -1]) & (eigenvalues[..., -1] > 0)
    result = np.full(xy.shape, np.nan, dtype=float)
    if valid.any():
        result[valid] = np.linalg.solve(xx[valid], xy[valid, ..., None])[..., 0]
    return result


def weighted_fit(x, y, weights):
    """Batched weighted least squares, including the intercept supplied in x."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.atleast_2d(weights).astype(float)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape[1] != len(y):
        raise ValueError("Incompatible observation axes")
    if not all(np.isfinite(a).all() for a in (x, y, w)) or (w < 0).any() or (w.sum(1) <= 0).any():
        raise ValueError("Nonfinite data or invalid weights")
    xx = np.einsum("bn,ni,nj->bij", w, x, x, optimize=True)
    xy = (w * y) @ x
    beta = checked_solve(xx, xy)
    prediction = beta @ x.T
    sse = np.sum(w * (y - prediction) ** 2, axis=1)
    mean = (w @ y) / w.sum(1)
    sst = np.sum(w * (y - mean[:, None]) ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - sse / sst
    return {"beta": beta, "sse": sse, "sst": sst, "r2": r2}


def _check_codes(codes):
    raw = np.asarray(codes)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Group codes must be a nonempty integer vector")
    groups = int(raw.max()) + 1
    if not np.array_equal(np.unique(raw), np.arange(groups)):
        raise ValueError("Group codes must be contiguous from zero")
    return raw, groups


def cluster_weights(codes, draws, rng, patient_equal=False):
    """Resample patients, carrying all visits and all models together."""
    codes, groups = _check_codes(codes)
    multiplicity = rng.multinomial(groups, np.full(groups, 1 / groups), size=draws)
    weights = multiplicity[:, codes].astype(float)
    if patient_equal:
        weights /= np.bincount(codes)[codes]
    return weights


def interval(samples, family_size=1):
    """Percentile intervals; require at least 95% finite draws."""
    values = np.asarray(samples, dtype=float).ravel()
    if family_size < 1:
        raise ValueError("family_size must be positive")
    valid = values[np.isfinite(values)]
    if not len(values) or len(valid) < .95 * len(values):
        return {"status": "insufficient_valid_bootstrap_draws", "valid_draws": int(len(valid))}
    return {"status": "ok", "valid_draws": int(len(valid)),
            "ci95": np.quantile(valid, [.025, .975]).tolist(),
            "ci_family": np.quantile(valid, [.025 / family_size, 1 - .025 / family_size]).tolist()}


def summarize(point, draws, family_size=1):
    if not np.isfinite(point):
        return {"point": None, "status": "undefined_point"}
    return {"point": float(point), **interval(draws, family_size)}


def weighted_linear(weights, x, y):
    """Source frequency-weighted Pearson correlations and simple regressions."""
    w = np.atleast_2d(weights).astype(float)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x = x[:, None]
    if w.shape[1] != len(x) or y.shape != (len(x),):
        raise ValueError("Incompatible observation axes")
    if not all(np.isfinite(a).all() for a in (w, x, y)) or (w < 0).any() or (w.sum(1) <= 0).any():
        raise ValueError("Nonfinite data or invalid weights")
    w = w / w.sum(axis=1, keepdims=True)
    x0, y0 = x.mean(axis=0), y.mean()
    xc, yc = x - x0, y - y0
    mx, my = w @ xc, w @ yc
    vx = np.maximum(w @ (xc * xc) - mx * mx, 0.)
    vy = np.maximum(w @ (yc * yc) - my * my, 0.)
    cov = w @ (xc * yc[:, None]) - mx * my[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(vx * vy[:, None])
        slope = cov / vx
    return {"r": np.clip(r, -1., 1.), "slope": slope,
            "intercept": my[:, None] + y0 - slope * (mx + x0)}


def rank_kernel(values):
    values = np.asarray(values, float)
    return (values[:, None] < values[None, :]).astype(float) + .5 * (values[:, None] == values[None, :])


def weighted_rho_from_kernels(weights, kernels, target_kernel):
    """Recompute weighted midranks in each draw, including ties and duplicates."""
    weights = np.atleast_2d(weights).astype(float)
    total = weights.sum(axis=1)
    yr = weights @ target_kernel
    yc = yr - (weights * yr).sum(axis=1)[:, None] / total[:, None]
    vy = (weights * yc ** 2).sum(axis=1)
    outputs = []
    for kernel in kernels:
        xr = weights @ kernel
        xc = xr - (weights * xr).sum(axis=1)[:, None] / total[:, None]
        numerator = (weights * xc * yc).sum(axis=1)
        denominator = np.sqrt((weights * xc ** 2).sum(axis=1) * vy)
        with np.errstate(invalid="ignore", divide="ignore"):
            outputs.append(numerator / denominator)
    return np.column_stack(outputs)


def adjusted_batch(designs, y, weights):
    base = weighted_fit(designs["base"], y, weights)
    result = {"base_r2": base["r2"], "base_beta": base["beta"]}
    for name, x in designs.items():
        if name == "base":
            continue
        fit = weighted_fit(x, y, weights)
        with np.errstate(invalid="ignore", divide="ignore"):
            partial_r2 = np.clip((base["sse"] - fit["sse"]) / base["sse"], 0, 1)
            delta_r2 = (base["sse"] - fit["sse"]) / base["sst"]
        result[name] = {"partial_r": np.sign(fit["beta"][:, -1]) * np.sqrt(partial_r2),
                        "ef_beta_per10pp": fit["beta"][:, -1], "r2": fit["r2"],
                        "delta_r2": delta_r2, "partial_r2": partial_r2}
    return result


def group_cv(x, y, codes, multiplicity, patient_equal=False):
    """Refit every fold after removing ALL copies of its held-out patient.

    Multiplicity indexes original patients, not independently sampled rows.
    Patient-equal weights apply both to fitting and to pooled scoring.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    codes, groups = _check_codes(codes)
    m = np.atleast_2d(multiplicity).astype(float)
    if x.ndim != 2 or y.shape != (len(x),) or len(codes) != len(y) or m.shape[1] != groups:
        raise ValueError("Incompatible observation or group axes")
    if not all(np.isfinite(a).all() for a in (x, y, m)) or (m < 0).any() or (m.sum(1) <= 0).any():
        raise ValueError("Nonfinite data or invalid cluster multiplicities")
    counts = np.bincount(codes)
    scale = m / counts if patient_equal else m
    gxx = np.stack([x[codes == g].T @ x[codes == g] for g in range(groups)])
    gxy = np.stack([x[codes == g].T @ y[codes == g] for g in range(groups)])
    total_xx = np.einsum("bg,gij->bij", scale, gxx)
    total_xy = scale @ gxy
    fold_xx = total_xx[:, None] - scale[:, :, None, None] * gxx[None]
    fold_xy = total_xy[:, None] - scale[:, :, None] * gxy[None]
    beta = checked_solve(fold_xx, fold_xy)
    prediction = np.einsum("bni,ni->bn", beta[:, codes], x)
    weights = scale[:, codes]
    valid = np.all(np.isfinite(prediction) | (weights == 0), axis=1)
    error = np.where(weights > 0, prediction - y, 0.)
    mse = np.sum(weights * error ** 2, axis=1) / weights.sum(axis=1)
    mae = np.sum(weights * np.abs(error), axis=1) / weights.sum(axis=1)
    mse[~valid], mae[~valid] = np.nan, np.nan
    return {"prediction": prediction, "rmse": np.sqrt(mse), "mae": mae, "valid": valid}


def cv_batch(designs, y, codes, multiplicity, patient_equal=False):
    fits = {name: group_cv(x, y, codes, multiplicity, patient_equal) for name, x in designs.items()}
    null = group_cv(np.ones((len(y), 1)), y, codes, multiplicity, patient_equal)
    for fit in fits.values():
        with np.errstate(divide="ignore", invalid="ignore"):
            fit["q2_vs_fold_mean"] = 1 - (fit["rmse"] / null["rmse"]) ** 2
        fit["rmse_improvement_vs_base"] = fits["base"]["rmse"] - fit["rmse"]
    fits["fold_mean"] = null
    return fits


def _flag(frame, name):
    values = frame[name]
    if values.isna().any():
        raise ValueError(f"Missing values in flag column {name}")
    mapped = values.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError(f"Flag {name} must contain true/false or 1/0")
    return mapped.astype(bool)


def validate_frame(frame, model_columns=DEFAULT_MODELS):
    """Validate a common complete cohort; never impute or silently drop rows."""
    models = tuple(model_columns)
    if not models or len(set(models)) != len(models) or "f04" not in models:
        raise ValueError("Select distinct model columns including f04")
    reserved = {"patient_id", "study_id", "age_proxy", "log2_creatinine", "ntprobnp", "base", "fold_mean",
                "variant", "report_exact", "age_topcoded", "study_order", "lab_censored", "analysis_eligible"}
    if set(models) & reserved:
        raise ValueError("Model names conflict with reserved columns")
    numeric = ["age_proxy", "log2_creatinine", "ntprobnp", *models]
    required = ["patient_id", "study_id", *numeric]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if not len(frame) or frame[required].isna().any().any():
        raise ValueError("Empty or incomplete clinical input; use shared explicit exclusions")
    for name in ("patient_id", "study_id"):
        if frame[name].astype(str).str.strip().eq("").any():
            raise ValueError("Identifiers must be nonempty")
    key = ["patient_id", "study_id"] + (["variant"] if "variant" in frame else [])
    if frame.duplicated(key).any():
        raise ValueError("Duplicate patient/study pair within a variant")
    values = frame[numeric].to_numpy(float)
    if not np.isfinite(values).all() or (frame.ntprobnp.to_numpy(float) <= 0).any():
        raise ValueError("Finite data and strictly positive NT-proBNP required")
    if ((frame[list(models)].to_numpy(float) < 0) | (frame[list(models)].to_numpy(float) > 100)).any():
        raise ValueError("LVEF values must be percentage points from 0 to 100")
    if "variant" in frame and (frame.variant.isna().any() or frame.variant.astype(str).str.strip().eq("").any()):
        raise ValueError("Variant labels must be nonempty")
    for name in ("report_exact", "age_topcoded", "lab_censored", "analysis_eligible"):
        if name in frame:
            flag = _flag(frame, name)
            if name == "lab_censored" and flag.any():
                raise ValueError("Censored concentrations require an explicit separate model")
            if name == "analysis_eligible" and not flag.all():
                raise ValueError("Ineligible observations require shared explicit exclusions")
    return models


def analyze(frame, bootstrap_reps=20000, cv_bootstrap_reps=2000, seed=DEFAULT_SEED,
            patient_equal=False, model_columns=DEFAULT_MODELS):
    """Analyze one supplied variant/cohort, returning aggregate statistics only."""
    models = validate_frame(frame, model_columns)
    if "variant" in frame and frame.variant.nunique() != 1:
        raise ValueError("analyze requires one variant; use analyze_cohorts for multiple variants")
    if min(bootstrap_reps, cv_bootstrap_reps) < 1:
        raise ValueError("Bootstrap replicate counts must be positive")
    codes, patients = pd.factorize(frame.patient_id.astype(str), sort=True)
    groups = len(patients)
    if groups < 10:
        raise ValueError("At least ten patients required")
    ef = frame[list(models)].to_numpy(float)
    raw = frame.ntprobnp.to_numpy(float)
    y = np.log10(raw)
    cov = frame[["age_proxy", "log2_creatinine"]].to_numpy(float)
    base = np.column_stack([np.ones(len(frame)), cov[:, 0] / 10, cov[:, 1]])
    designs = {"base": base, **{m: np.column_stack([base, ef[:, i] / 10]) for i, m in enumerate(models)}}
    w = np.ones((1, len(frame)))
    if patient_equal:
        w /= np.bincount(codes)[codes]
    kernels = [rank_kernel(ef[:, i]) for i in range(len(models))]
    target = rank_kernel(raw)

    def unadjusted(weights):
        logfit = weighted_linear(weights, ef, y)
        return {"pearson_log10": logfit["r"], "pearson_raw": weighted_linear(weights, ef, raw)["r"],
                "spearman": weighted_rho_from_kernels(weights, kernels, target),
                "slope_log10_per10pp": logfit["slope"] * 10}

    point, continuous = adjusted_batch(designs, y, w), unadjusted(w)
    cv = cv_batch(designs, y, codes, np.ones((1, groups)), patient_equal)
    rng = np.random.default_rng(seed)
    draws, unadjusted_draws = [], []
    for start in range(0, bootstrap_reps, 250):
        weights = cluster_weights(codes, min(250, bootstrap_reps - start), rng, patient_equal)
        draws.append(adjusted_batch(designs, y, weights))
        unadjusted_draws.append(unadjusted(weights))
    rng = np.random.default_rng(seed)
    cv_draws = []
    for start in range(0, cv_bootstrap_reps, 100):
        mult = rng.multinomial(groups, np.full(groups, 1 / groups), size=min(100, cv_bootstrap_reps - start))
        batch = cv_batch(designs, y, codes, mult, patient_equal)
        # Individual predictions stay in memory only, and are not needed for intervals.
        cv_draws.append({name: {key: value for key, value in fit.items() if key not in ("prediction", "valid")}
                         for name, fit in batch.items()})
    result = {"status": "ok", "visits": len(frame), "patients": groups,
              "weighting": "patient_equal" if patient_equal else "visit_equal",
              "base_r2": summarize(point["base_r2"][0], np.concatenate([d["base_r2"] for d in draws])),
              "base_coefficients": {}, "models": {}, "contrasts": {}, "unadjusted": {},
              "unadjusted_contrasts": {}, "comparison_families": {
                  "f04_contrasts_per_metric": max(1, len(models) - 1), "base_increment_contrasts": len(models)}}
    for i, name in enumerate(("intercept", "age_per10years", "log2_creatinine")):
        result["base_coefficients"][name] = summarize(point["base_beta"][0, i], np.concatenate([d["base_beta"][:, i] for d in draws]))
    for name in (*designs, "fold_mean"):
        metrics = {}
        if name in models:
            for metric in point[name]:
                metrics[metric] = summarize(point[name][metric][0], np.concatenate([d[name][metric] for d in draws]))
        for metric in ("rmse", "mae", "q2_vs_fold_mean", "rmse_improvement_vs_base"):
            if metric in cv[name]:
                family = len(models) if metric == "rmse_improvement_vs_base" else 1
                metrics["cv_" + metric] = summarize(cv[name][metric][0], np.concatenate([d[name][metric] for d in cv_draws]), family)
        result["models"][name] = metrics
    family = max(1, len(models) - 1)
    for other in models:
        if other != "f04":
            result["contrasts"][other] = {
                "partial_r_other_minus_f04": summarize(point[other]["partial_r"][0] - point["f04"]["partial_r"][0],
                    np.concatenate([d[other]["partial_r"] - d["f04"]["partial_r"] for d in draws]), family),
                "delta_r2_f04_minus_other": summarize(point["f04"]["delta_r2"][0] - point[other]["delta_r2"][0],
                    np.concatenate([d["f04"]["delta_r2"] - d[other]["delta_r2"] for d in draws]), family),
                "cv_rmse_other_minus_f04": summarize(cv[other]["rmse"][0] - cv["f04"]["rmse"][0],
                    np.concatenate([d[other]["rmse"] - d["f04"]["rmse"] for d in cv_draws]), family)}
    f = models.index("f04")
    for metric, points in continuous.items():
        samples = np.concatenate([d[metric] for d in unadjusted_draws])
        result["unadjusted"][metric] = {name: summarize(points[0, i], samples[:, i]) for i, name in enumerate(models)}
        if metric == "slope_log10_per10pp":
            continue
        result["unadjusted_contrasts"][metric] = {
            name: {"r_other_minus_f04": summarize(points[0, i] - points[0, f], samples[:, i] - samples[:, f], family),
                   "abs_r_f04_minus_other": summarize(abs(points[0, f]) - abs(points[0, i]), abs(samples[:, f]) - abs(samples[:, i]), family)}
            for i, name in enumerate(models) if name != "f04"}
    return result


def analyze_cohorts(frame, sensitivities=(), **kwargs):
    """Separate supplied variants; build only explicitly requested sensitivities."""
    validate_frame(frame, kwargs.get("model_columns", DEFAULT_MODELS))
    supported = {"patient_equal", "exact_report_only", "exclude_topcoded_age", "single_visit"}
    if set(sensitivities) - supported:
        raise ValueError("Unsupported sensitivity")
    grouped = frame.groupby("variant", sort=True) if "variant" in frame else [("supplied", frame)]
    results = {}
    for variant, part in grouped:
        # Retain original order unless an explicit visit-order column is supplied.
        modes = {"repeated": (part, False)}
        for sensitivity in sensitivities:
            if sensitivity == "patient_equal":
                modes[sensitivity] = (part, True)
            elif sensitivity == "exact_report_only":
                if "report_exact" not in part:
                    raise ValueError("exact_report_only requires report_exact")
                modes[sensitivity] = (part[_flag(part, "report_exact")], False)
            elif sensitivity == "exclude_topcoded_age":
                if "age_topcoded" not in part:
                    raise ValueError("exclude_topcoded_age requires age_topcoded")
                modes[sensitivity] = (part[~_flag(part, "age_topcoded")], False)
            else:
                if "study_order" not in part:
                    raise ValueError("single_visit requires numeric study_order; never infer chronology from an ID")
                order = part.study_order.to_numpy(float)
                if not np.isfinite(order).all() or part.duplicated(["patient_id", "study_order"]).any():
                    raise ValueError("study_order must be finite and unique within each patient")
                ordered = part.assign(study_order=order).sort_values("study_order", kind="stable")
                modes[sensitivity] = (ordered.drop_duplicates("patient_id"), False)
        results[str(variant)] = {}
        for mode, (data, equal) in modes.items():
            if data.patient_id.nunique() < 10:
                results[str(variant)][mode] = {"status": "insufficient_patients", "visits": len(data),
                                               "patients": int(data.patient_id.nunique())}
            else:
                results[str(variant)][mode] = analyze(data, patient_equal=equal, **kwargs)
    return results
