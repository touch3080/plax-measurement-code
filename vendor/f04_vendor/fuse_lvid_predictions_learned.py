from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

from .lvid.data import read_csv_rows, write_csv_rows
from .lvid.metrics import line_metrics
from .lvid.utils import ensure_dir, save_json


POINT_KEYS = ("x1_pred", "y1_pred", "x2_pred", "y2_pred")
LABEL_GT_KEYS = ("x1", "y1", "x2", "y2")
PRED_GT_KEYS = ("x1_gt", "y1_gt", "x2_gt", "y2_gt")


@dataclass
class CandidateSpec:
    name: str
    val_csv: Path
    test_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight learned selector/fusion model over existing LVID predictions."
    )
    parser.add_argument("--val-labels", default="prepared_lvid/labels/val.csv")
    parser.add_argument("--test-labels", default="prepared_lvid/labels/test.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("NAME", "VAL_CSV", "TEST_CSV"),
        required=True,
        help="Candidate model name plus validation and test prediction CSV paths.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--holdout-ratio", type=float, default=0.30)
    parser.add_argument("--split-unit", choices=["sample", "video"], default="sample")
    parser.add_argument("--exclude-calc-feature", action="store_true")
    parser.add_argument("--select-by", choices=["objective", "mean", "p95"], default="objective")
    parser.add_argument("--selector-output-name", default="f02_lvid_error_selector.joblib")
    return parser.parse_args()


def parse_candidates(args: argparse.Namespace) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    seen: set[str] = set()
    for name, val_csv, test_csv in args.candidate:
        if name in seen:
            raise ValueError(f"Duplicate candidate name: {name}")
        seen.add(name)
        specs.append(CandidateSpec(name=name, val_csv=Path(val_csv), test_csv=Path(test_csv)))
    if len(specs) < 2:
        raise ValueError("At least two candidate models are required.")
    return specs


def stable_holdout(sample_id: str, ratio: float, seed: int) -> bool:
    payload = f"{seed}:{sample_id}".encode("utf-8")
    value = int(hashlib.sha1(payload).hexdigest()[:12], 16) / float(16**12)
    return value < ratio


def split_meta_items(items: list[dict[str, Any]], ratio: float, seed: int, unit: str):
    if not 0.0 < ratio < 1.0 or unit not in ("sample", "video"):
        raise ValueError("Expected 0 < holdout ratio < 1 and sample/video split unit")
    key = "video_id" if unit == "video" else "sample_id"
    if any(not item.get(key) for item in items):
        raise ValueError(f"Missing {key} in meta split")
    holdout_ids = {item["sample_id"] for item in items if stable_holdout(item[key], ratio, seed)}
    train = [item for item in items if item["sample_id"] not in holdout_ids]
    holdout = [item for item in items if item["sample_id"] in holdout_ids]
    if not train or not holdout:
        raise ValueError("Empty meta-train/holdout split")
    train_videos = {item["video_id"] for item in train}
    holdout_videos = {item["video_id"] for item in holdout}
    shared = sorted(train_videos & holdout_videos)
    if unit == "video" and shared:
        raise AssertionError("Video overlap in grouped split")
    audit = {
        "split_unit": unit, "seed": seed, "holdout_ratio": ratio,
        "meta_train_samples": len(train), "meta_holdout_samples": len(holdout),
        "meta_train_videos": len(train_videos), "meta_holdout_videos": len(holdout_videos),
        "shared_video_count": len(shared), "shared_video_ids": shared,
    }
    return train, holdout, audit


def row_points(row: dict[str, str], keys: tuple[str, str, str, str] = POINT_KEYS) -> np.ndarray:
    return np.array([float(row[k]) for k in keys], dtype=np.float32)


def row_confidence(row: dict[str, str] | None) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def optional_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.linalg.norm(a[0:2] - b[0:2]) + np.linalg.norm(a[2:4] - b[2:4])) / 2.0)


def swap_points(points: np.ndarray) -> np.ndarray:
    return points[[2, 3, 0, 1]].copy()


def align_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    swapped = swap_points(points)
    if endpoint_distance(swapped, reference) < endpoint_distance(points, reference):
        return swapped
    return points.copy()


def line_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[2:4] - points[0:2]))


def midpoint(points: np.ndarray) -> np.ndarray:
    return (points[0:2] + points[2:4]) / 2.0


def angle_rad(points: np.ndarray) -> float:
    vec = points[2:4] - points[0:2]
    return float(math.atan2(float(vec[1]), float(vec[0])))


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    diff = abs(angle_rad(a) - angle_rad(b))
    diff = min(diff, 2 * math.pi - diff)
    return float(math.degrees(diff))


def sample_endpoint_error(pred: np.ndarray, gt: np.ndarray) -> float:
    ordered = endpoint_distance(pred, gt)
    swapped = endpoint_distance(pred, swap_points(gt))
    return min(ordered, swapped)


def per_sample_errors(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    p1 = pred[:, 0:2]
    p2 = pred[:, 2:4]
    g1 = gt[:, 0:2]
    g2 = gt[:, 2:4]
    ordered = (np.linalg.norm(p1 - g1, axis=1) + np.linalg.norm(p2 - g2, axis=1)) / 2.0
    swapped_raw = (np.linalg.norm(p1 - g2, axis=1) + np.linalg.norm(p2 - g1, axis=1)) / 2.0
    endpoint_swap = np.minimum(ordered, swapped_raw)
    midpoint_error = np.linalg.norm(((p1 + p2) / 2.0) - ((g1 + g2) / 2.0), axis=1)
    length_error = np.abs(np.linalg.norm(p2 - p1, axis=1) - np.linalg.norm(g2 - g1, axis=1))
    pred_ang = np.arctan2((p2 - p1)[:, 1], (p2 - p1)[:, 0])
    gt_ang = np.arctan2((g2 - g1)[:, 1], (g2 - g1)[:, 0])
    angle = np.abs(pred_ang - gt_ang)
    angle = np.minimum(angle, 2 * math.pi - angle)
    return {
        "endpoint_error": ordered,
        "endpoint_error_swap": endpoint_swap,
        "midpoint_error": midpoint_error,
        "length_error": length_error,
        "angle_error_deg": np.degrees(angle),
    }


def summarize(pred: np.ndarray, gt: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = line_metrics(pred, gt)
    errors = per_sample_errors(pred, gt)
    swap = errors["endpoint_error_swap"]
    out: dict[str, Any] = {
        "endpoint_error": metrics.endpoint_error,
        "endpoint_error_swap": metrics.endpoint_error_swap,
        "midpoint_error": metrics.midpoint_error,
        "length_error": metrics.length_error,
        "angle_error_deg": metrics.angle_error_deg,
        "samples": int(len(rows)),
        "endpoint_error_swap_median": float(np.median(swap)),
        "endpoint_error_swap_p90": float(np.percentile(swap, 90)),
        "endpoint_error_swap_p95": float(np.percentile(swap, 95)),
        "endpoint_error_swap_max": float(np.max(swap)),
        "by_calc": {},
    }
    calc_arr = np.array([str(r["calc"]) for r in rows])
    for calc in ("LVIDd", "LVIDs"):
        mask = calc_arr == calc
        if not np.any(mask):
            continue
        calc_metrics = line_metrics(pred[mask], gt[mask])
        calc_swap = swap[mask]
        out["by_calc"][calc] = {
            "endpoint_error": calc_metrics.endpoint_error,
            "endpoint_error_swap": calc_metrics.endpoint_error_swap,
            "midpoint_error": calc_metrics.midpoint_error,
            "length_error": calc_metrics.length_error,
            "angle_error_deg": calc_metrics.angle_error_deg,
            "samples": int(np.sum(mask)),
            "endpoint_error_swap_p95": float(np.percentile(calc_swap, 95)),
            "endpoint_error_swap_max": float(np.max(calc_swap)),
        }
    return out


def metric_objective(metrics: dict[str, Any], select_by: str) -> float:
    if select_by == "mean":
        return float(metrics["endpoint_error_swap"])
    if select_by == "p95":
        return float(metrics["endpoint_error_swap_p95"])
    return float(metrics["endpoint_error_swap"]) + 0.25 * float(metrics["endpoint_error_swap_p95"])


def load_prediction_maps(specs: list[CandidateSpec], split: str) -> dict[str, dict[str, dict[str, str]]]:
    maps: dict[str, dict[str, dict[str, str]]] = {}
    for spec in specs:
        path = spec.val_csv if split == "val" else spec.test_csv
        rows = read_csv_rows(path)
        maps[spec.name] = {r["sample_id"]: r for r in rows}
    return maps


def collect_items(
    labels_path: Path,
    specs: list[CandidateSpec],
    pred_maps: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    label_rows = read_csv_rows(labels_path)
    items: list[dict[str, Any]] = []
    reference_name = specs[0].name

    for label in label_rows:
        sid = label["sample_id"]
        reference_row = pred_maps[reference_name].get(sid)
        if reference_row is None:
            raise KeyError(f"Missing {reference_name} prediction for sample_id={sid}")
        reference_points = row_points(reference_row)

        if all(k in reference_row for k in PRED_GT_KEYS):
            gt = row_points(reference_row, PRED_GT_KEYS)
        else:
            gt = row_points(label, LABEL_GT_KEYS)

        preds: dict[str, np.ndarray] = {}
        confs: dict[str, float] = {}
        rows: dict[str, dict[str, str]] = {}
        for spec in specs:
            pred_row = pred_maps[spec.name].get(sid)
            if pred_row is None:
                raise KeyError(f"Missing {spec.name} prediction for sample_id={sid}")
            rows[spec.name] = pred_row
            preds[spec.name] = align_to_reference(row_points(pred_row), reference_points)
            confs[spec.name] = row_confidence(pred_row)

        items.append(
            {
                "sample_id": sid,
                "video_id": reference_row.get("video_id", label.get("video_id", "")),
                "calc": reference_row.get("calc", label.get("calc", "")),
                "frame": reference_row.get("frame", label.get("frame", "")),
                "gt": gt.astype(np.float32),
                "preds": preds,
                "confs": confs,
                "rows": rows,
            }
        )
    return items


def feature_names(candidate_names: list[str], include_calc_feature: bool = True) -> list[str]:
    names = [
        "calc_is_lvids",
        "candidate_confidence",
        "candidate_length",
        "candidate_mid_x",
        "candidate_mid_y",
        "candidate_angle_sin",
        "candidate_angle_cos",
        "candidate_min_x",
        "candidate_max_x",
        "candidate_min_y",
        "candidate_max_y",
        "candidate_edge_margin",
        "candidate_crop_width",
        "candidate_crop_height",
        "candidate_crop_area",
        "consensus_endpoint_distance",
        "consensus_midpoint_distance",
        "mean_pairwise_endpoint_distance",
        "max_pairwise_endpoint_distance",
        "mean_pairwise_angle_diff",
        "max_pairwise_angle_diff",
        "mean_pairwise_length_diff",
        "max_pairwise_length_diff",
    ]
    names += [f"candidate_is_{name}" for name in candidate_names]
    names += [f"{name}_confidence" for name in candidate_names]
    for name in candidate_names:
        names += [
            f"to_{name}_endpoint_distance",
            f"to_{name}_midpoint_distance",
            f"to_{name}_angle_diff",
            f"to_{name}_length_diff",
            f"confidence_minus_{name}",
        ]
    return names if include_calc_feature else names[1:]


def make_candidate_features(
    item: dict[str, Any],
    candidate_name: str,
    candidate_names: list[str],
    include_calc_feature: bool = True,
) -> list[float]:
    preds: dict[str, np.ndarray] = item["preds"]
    confs: dict[str, float] = item["confs"]
    points = preds[candidate_name]
    all_points = np.stack([preds[name] for name in candidate_names], axis=0)
    consensus = np.median(all_points, axis=0)
    mid = midpoint(points)
    length = line_length(points)
    angle = angle_rad(points)
    pairwise_endpoint = [endpoint_distance(points, preds[name]) for name in candidate_names if name != candidate_name]
    pairwise_angle = [angle_diff_deg(points, preds[name]) for name in candidate_names if name != candidate_name]
    pairwise_length = [abs(length - line_length(preds[name])) for name in candidate_names if name != candidate_name]
    row = item["rows"][candidate_name]
    crop_width = optional_float(row, "crop_width", 0.0)
    crop_height = optional_float(row, "crop_height", 0.0)
    values: list[float] = [
        1.0 if item["calc"] == "LVIDs" else 0.0,
        confs[candidate_name],
        length,
        float(mid[0]),
        float(mid[1]),
        math.sin(angle),
        math.cos(angle),
        float(min(points[0], points[2])),
        float(max(points[0], points[2])),
        float(min(points[1], points[3])),
        float(max(points[1], points[3])),
        float(min(points[0], points[2], points[1], points[3], 512 - points[0], 512 - points[2], 384 - points[1], 384 - points[3])),
        crop_width,
        crop_height,
        crop_width * crop_height,
        endpoint_distance(points, consensus),
        float(np.linalg.norm(midpoint(points) - midpoint(consensus))),
        float(np.mean(pairwise_endpoint)) if pairwise_endpoint else 0.0,
        float(np.max(pairwise_endpoint)) if pairwise_endpoint else 0.0,
        float(np.mean(pairwise_angle)) if pairwise_angle else 0.0,
        float(np.max(pairwise_angle)) if pairwise_angle else 0.0,
        float(np.mean(pairwise_length)) if pairwise_length else 0.0,
        float(np.max(pairwise_length)) if pairwise_length else 0.0,
    ]
    values += [1.0 if name == candidate_name else 0.0 for name in candidate_names]
    values += [confs[name] for name in candidate_names]
    for name in candidate_names:
        other = preds[name]
        values += [
            endpoint_distance(points, other),
            float(np.linalg.norm(midpoint(points) - midpoint(other))),
            angle_diff_deg(points, other),
            abs(length - line_length(other)),
            confs[candidate_name] - confs[name],
        ]
    return values if include_calc_feature else values[1:]


def build_training_table(
    items: list[dict[str, Any]],
    candidate_names: list[str],
    keep_sample_ids: set[str] | None = None,
    include_calc_feature: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    targets: list[float] = []
    for item in items:
        if keep_sample_ids is not None and item["sample_id"] not in keep_sample_ids:
            continue
        for candidate_name in candidate_names:
            features.append(make_candidate_features(item, candidate_name, candidate_names, include_calc_feature))
            targets.append(sample_endpoint_error(item["preds"][candidate_name], item["gt"]))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def candidate_error_matrix(items: list[dict[str, Any]], candidate_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [sample_endpoint_error(item["preds"][name], item["gt"]) for name in candidate_names]
            for item in items
        ],
        dtype=np.float32,
    )


def predict_candidate_errors(model: Any, items: list[dict[str, Any]], candidate_names: list[str]) -> np.ndarray:
    # Legacy selectors retain the phase column; new selectors carry their schema.
    include_calc_feature = getattr(model, "lvid_include_calc_feature_", True)
    features: list[list[float]] = []
    for item in items:
        for candidate_name in candidate_names:
            features.append(make_candidate_features(item, candidate_name, candidate_names, include_calc_feature))
    predicted = np.asarray(model.predict(np.asarray(features, dtype=np.float32)), dtype=np.float32)
    return predicted.reshape(len(items), len(candidate_names))


def load_selector(path: str | Path) -> dict[str, Any]:
    selector = joblib.load(path)
    include_calc = selector.get("include_calc_feature", True)
    expected = feature_names(list(selector["candidate_names"]), include_calc)
    if selector["feature_names"] != expected or selector["model"].n_features_in_ != len(expected):
        raise ValueError("Selector feature schema mismatch")
    selector["model"].lvid_include_calc_feature_ = include_calc
    return selector


def make_model_grid(seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "extra_trees_depth6_leaf4",
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees_depth10_leaf3",
            ExtraTreesRegressor(
                n_estimators=600,
                max_depth=10,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees_none_leaf5",
            ExtraTreesRegressor(
                n_estimators=700,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest_depth10_leaf4",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    ]


def strategy_grid() -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = [{"method": "select"}]
    for delta in (0.25, 0.5, 1.0, 2.0, 4.0):
        for max_disagreement in (8.0, 12.0, 16.0, 24.0, 1e9):
            for second_weight in (0.25, 0.40, 0.50):
                strategies.append(
                    {
                        "method": "top2_blend",
                        "delta": delta,
                        "max_disagreement": max_disagreement,
                        "second_weight": second_weight,
                    }
                )
    for temperature in (1.0, 2.0, 4.0, 8.0):
        strategies.append({"method": "softmax_blend", "temperature": temperature})
    return strategies


def apply_strategy_to_items(
    items: list[dict[str, Any]],
    candidate_names: list[str],
    predicted_errors: np.ndarray,
    strategy: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    preds: list[np.ndarray] = []
    decisions: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        errors = predicted_errors[item_idx]
        order = np.argsort(errors)
        best_idx = int(order[0])
        best_name = candidate_names[best_idx]
        best_points = item["preds"][best_name].copy()
        selected = best_points.copy()
        selected_source = best_name
        confidence = item["confs"][best_name]
        second_name = candidate_names[int(order[1])] if len(order) > 1 else ""
        second_predicted_error = float(errors[int(order[1])]) if len(order) > 1 else 0.0
        weights = {name: 0.0 for name in candidate_names}
        weights[best_name] = 1.0

        if strategy["method"] == "top2_blend" and len(order) > 1:
            second_idx = int(order[1])
            second_points = item["preds"][candidate_names[second_idx]]
            disagreement = endpoint_distance(best_points, second_points)
            within_delta = float(errors[second_idx] - errors[best_idx]) <= float(strategy["delta"])
            if within_delta and disagreement <= float(strategy["max_disagreement"]):
                w2 = float(strategy["second_weight"])
                selected = (1.0 - w2) * best_points + w2 * second_points
                selected_source = f"{best_name}+{candidate_names[second_idx]}"
                confidence = (1.0 - w2) * item["confs"][best_name] + w2 * item["confs"][candidate_names[second_idx]]
                weights[best_name] = 1.0 - w2
                weights[candidate_names[second_idx]] = w2
        elif strategy["method"] == "softmax_blend":
            temp = max(float(strategy["temperature"]), 1e-6)
            shifted = errors - np.min(errors)
            raw = np.exp(-shifted / temp)
            raw = raw / np.sum(raw)
            selected = np.zeros(4, dtype=np.float32)
            confidence = 0.0
            active: list[str] = []
            for idx, name in enumerate(candidate_names):
                weight = float(raw[idx])
                selected += weight * item["preds"][name]
                confidence += weight * item["confs"][name]
                weights[name] = weight
                if weight >= 0.15:
                    active.append(name)
            selected_source = "+".join(active) if active else best_name

        preds.append(selected.astype(np.float32))
        decisions.append(
            {
                "selected_source": selected_source,
                "best_source": best_name,
                "second_source": second_name,
                "predicted_error": float(errors[best_idx]),
                "second_predicted_error": second_predicted_error,
                "confidence": float(confidence),
                "weights": weights,
            }
        )
    return np.asarray(preds, dtype=np.float32), decisions


def rows_from_predictions(
    items: list[dict[str, Any]],
    preds: np.ndarray,
    decisions: list[dict[str, Any]],
    candidate_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item, pred, decision in zip(items, preds, decisions):
        row: dict[str, Any] = {
            "sample_id": item["sample_id"],
            "video_id": item["video_id"],
            "calc": item["calc"],
            "frame": item["frame"],
            "x1_pred": float(pred[0]),
            "y1_pred": float(pred[1]),
            "x2_pred": float(pred[2]),
            "y2_pred": float(pred[3]),
            "x1_gt": float(item["gt"][0]),
            "y1_gt": float(item["gt"][1]),
            "x2_gt": float(item["gt"][2]),
            "y2_gt": float(item["gt"][3]),
            "confidence": float(decision["confidence"]),
            "selected_source": decision["selected_source"],
            "best_source": decision["best_source"],
            "second_source": decision["second_source"],
            "predicted_error": float(decision["predicted_error"]),
            "second_predicted_error": float(decision["second_predicted_error"]),
        }
        for name in candidate_names:
            row[f"weight_{name}"] = float(decision["weights"].get(name, 0.0))
            row[f"confidence_{name}"] = float(item["confs"][name])
        rows.append(row)
    return rows


def source_counts(rows: list[dict[str, Any]], field: str = "selected_source") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def evaluate_predictions(items: list[dict[str, Any]], preds: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt = np.asarray([item["gt"] for item in items], dtype=np.float32)
    return summarize(preds, gt, rows)


def candidate_baselines(items: list[dict[str, Any]], candidate_names: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    gt = np.asarray([item["gt"] for item in items], dtype=np.float32)
    minimal_rows = [{"calc": item["calc"]} for item in items]
    for name in candidate_names:
        pred = np.asarray([item["preds"][name] for item in items], dtype=np.float32)
        out[name] = summarize(pred, gt, minimal_rows)
    errors = candidate_error_matrix(items, candidate_names)
    oracle_idx = np.argmin(errors, axis=1)
    oracle_pred = np.asarray([items[i]["preds"][candidate_names[int(oracle_idx[i])]] for i in range(len(items))])
    oracle_rows = [{"calc": item["calc"]} for item in items]
    out["oracle_best_of_candidates"] = summarize(oracle_pred, gt, oracle_rows)
    out["oracle_source_counts"] = {name: int(np.sum(oracle_idx == idx)) for idx, name in enumerate(candidate_names)}
    return out


def save_feature_importance(model: Any, names: list[str], path: Path) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    rows = [
        {"feature": name, "importance": float(score)}
        for name, score in sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
    ]
    write_csv_rows(path, rows, ["feature", "importance"])


def fit_and_score_grid(
    train_items: list[dict[str, Any]],
    holdout_items: list[dict[str, Any]],
    candidate_names: list[str],
    select_by: str,
    seed: int,
    include_calc_feature: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_ids = {item["sample_id"] for item in train_items}
    x_train, y_train = build_training_table(train_items, candidate_names, train_ids, include_calc_feature)
    grid_rows: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for model_name, model in tqdm(make_model_grid(seed), desc="Learned fusion model search", unit="model", dynamic_ncols=True, file=sys.stdout):
        model.fit(x_train, y_train)
        model.lvid_include_calc_feature_ = include_calc_feature
        pred_errors = predict_candidate_errors(model, holdout_items, candidate_names)
        for strategy in strategy_grid():
            pred, decisions = apply_strategy_to_items(holdout_items, candidate_names, pred_errors, strategy)
            rows = rows_from_predictions(holdout_items, pred, decisions, candidate_names)
            metrics = evaluate_predictions(holdout_items, pred, rows)
            record = {
                "model_name": model_name,
                "strategy": strategy,
                "endpoint_error_swap": metrics["endpoint_error_swap"],
                "endpoint_error_swap_p95": metrics["endpoint_error_swap_p95"],
                "endpoint_error_swap_max": metrics["endpoint_error_swap_max"],
                "objective": metric_objective(metrics, select_by),
                "selected_source_counts": source_counts(rows),
            }
            grid_rows.append(record)
            if best_record is None or float(record["objective"]) < float(best_record["objective"]):
                best_record = record

    if best_record is None:
        raise RuntimeError("No learned fusion grid result was produced.")
    return best_record, grid_rows


def train_named_model(model_name: str, seed: int, x: np.ndarray, y: np.ndarray, include_calc_feature: bool = True) -> Any:
    for name, model in make_model_grid(seed):
        if name == model_name:
            model.fit(x, y)
            model.lvid_include_calc_feature_ = include_calc_feature
            return model
    raise KeyError(f"Unknown model name selected by grid: {model_name}")


def write_grid_rows(path: Path, records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "model_name": record["model_name"],
            "strategy": json.dumps(record["strategy"], sort_keys=True),
            "endpoint_error_swap": record["endpoint_error_swap"],
            "endpoint_error_swap_p95": record["endpoint_error_swap_p95"],
            "endpoint_error_swap_max": record["endpoint_error_swap_max"],
            "objective": record["objective"],
            "selected_source_counts": json.dumps(record["selected_source_counts"], sort_keys=True),
        }
        rows.append(row)
    write_csv_rows(
        path,
        rows,
        [
            "model_name",
            "strategy",
            "endpoint_error_swap",
            "endpoint_error_swap_p95",
            "endpoint_error_swap_max",
            "objective",
            "selected_source_counts",
        ],
    )


def apply_and_save_split(
    model: Any,
    strategy: dict[str, Any],
    items: list[dict[str, Any]],
    candidate_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    pred_errors = predict_candidate_errors(model, items, candidate_names)
    preds, decisions = apply_strategy_to_items(items, candidate_names, pred_errors, strategy)
    rows = rows_from_predictions(items, preds, decisions, candidate_names)
    metrics = evaluate_predictions(items, preds, rows)
    metrics["selected_source_counts"] = source_counts(rows)
    metrics["best_source_counts"] = source_counts(rows, "best_source")
    metrics["candidate_baselines"] = candidate_baselines(items, candidate_names)

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_rows(output_dir / "predictions.csv", rows, fieldnames)
    save_json(output_dir / "metrics.json", metrics)
    print({key: value for key, value in metrics.items() if not isinstance(value, dict)})
    print(f"Predictions: {output_dir / 'predictions.csv'}")
    return metrics


def main() -> None:
    args = parse_args()
    specs = parse_candidates(args)
    candidate_names = [spec.name for spec in specs]
    output_dir = ensure_dir(args.output_dir)

    val_maps = load_prediction_maps(specs, "val")
    test_maps = load_prediction_maps(specs, "test")
    val_items = collect_items(Path(args.val_labels), specs, val_maps)
    test_items = collect_items(Path(args.test_labels), specs, test_maps)

    meta_train_items, holdout_items, audit = split_meta_items(val_items, args.holdout_ratio, args.seed, args.split_unit)
    shared_test = {item["video_id"] for item in val_items} & {item["video_id"] for item in test_items}
    if shared_test:
        raise ValueError("Validation/test video overlap")
    include_calc = not args.exclude_calc_feature
    audit.update({"include_calc_feature": include_calc, "val_test_shared_videos": len(shared_test),
                  "val_labels": str(args.val_labels), "test_labels": str(args.test_labels),
                  "candidate_names": candidate_names, "metric_space": "canonical_512x384"})
    save_json(output_dir / "split_audit.json", audit)
    assignments = [{"sample_id": item["sample_id"], "video_id": item["video_id"], "partition": partition}
                   for partition, items in (("meta_train", meta_train_items), ("meta_holdout", holdout_items)) for item in items]
    write_csv_rows(output_dir / "meta_split.csv", assignments, ["sample_id", "video_id", "partition"])

    print(f"Learned fusion candidates: {', '.join(candidate_names)}")
    print(f"Meta-train samples: {len(meta_train_items)}")
    print(f"Meta-holdout samples: {len(holdout_items)}")
    print(f"Test samples: {len(test_items)}")

    best_record, grid_records = fit_and_score_grid(
        meta_train_items,
        holdout_items,
        candidate_names,
        args.select_by,
        args.seed,
        include_calc,
    )
    write_grid_rows(output_dir / "meta_holdout_strategy_search.csv", grid_records)
    save_json(output_dir / "best_selector.json", best_record)
    print(f"Best holdout selector: {best_record}")

    x_all, y_all = build_training_table(val_items, candidate_names, include_calc_feature=include_calc)
    final_model = train_named_model(str(best_record["model_name"]), args.seed, x_all, y_all, include_calc)
    joblib.dump(
        {
            "model": final_model,
            "candidate_names": candidate_names,
            "feature_names": feature_names(candidate_names, include_calc),
            "include_calc_feature": include_calc,
            "schema_version": 2,
            "split_audit": audit,
            "strategy": best_record["strategy"],
            "best_record": best_record,
        },
        output_dir / args.selector_output_name,
    )
    save_feature_importance(final_model, feature_names(candidate_names, include_calc), output_dir / "feature_importance.csv")

    print("\n==== Evaluate learned selector on validation set ====")
    apply_and_save_split(final_model, best_record["strategy"], val_items, candidate_names, output_dir / "eval_val_best")

    print("\n==== Evaluate learned selector on test set ====")
    apply_and_save_split(final_model, best_record["strategy"], test_items, candidate_names, output_dir / "eval_test_best")


if __name__ == "__main__":
    main()
