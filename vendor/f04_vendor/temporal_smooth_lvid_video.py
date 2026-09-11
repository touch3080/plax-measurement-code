from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

from .lvid.utils import draw_lvid_overlay


POINT_KEYS = ("x1", "y1", "x2", "y2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal smoothing and outlier replacement for continuous LVID video predictions.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--overlay-video", default=None)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--distance-mad-threshold", type=float, default=6.0)
    parser.add_argument("--length-mad-threshold", type=float, default=6.0)
    parser.add_argument("--angle-mad-threshold", type=float, default=6.0)
    parser.add_argument("--min-distance-threshold", type=float, default=12.0)
    parser.add_argument("--min-length-threshold", type=float, default=10.0)
    parser.add_argument("--min-angle-threshold", type=float, default=12.0)
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def points_from_row(row: dict[str, str], keys: tuple[str, str, str, str] = POINT_KEYS) -> np.ndarray:
    return np.array([float(row[k]) for k in keys], dtype=np.float32)


def endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.linalg.norm(a[0:2] - b[0:2]) + np.linalg.norm(a[2:4] - b[2:4])) / 2.0)


def orient_continuously(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    out = points.astype(np.float32).copy()
    for i in range(1, len(out)):
        swapped = out[i, [2, 3, 0, 1]]
        if endpoint_distance(swapped, out[i - 1]) < endpoint_distance(out[i], out[i - 1]):
            out[i] = swapped
    return out


def odd_window(window: int) -> int:
    window = max(1, int(window))
    return window if window % 2 == 1 else window + 1


def rolling_median(points: np.ndarray, window: int) -> np.ndarray:
    window = odd_window(window)
    if window <= 1 or len(points) < 3:
        return points.astype(np.float32).copy()
    pad = window // 2
    padded = np.pad(points.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    out = [np.median(padded[i : i + window], axis=0) for i in range(len(points))]
    return np.asarray(out, dtype=np.float32)


def line_length(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, 2:4] - points[:, 0:2], axis=1)


def angle_rad(points: np.ndarray) -> np.ndarray:
    vec = points[:, 2:4] - points[:, 0:2]
    return np.arctan2(vec[:, 1], vec[:, 0])


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(angle_rad(a) - angle_rad(b))
    diff = np.minimum(diff, 2 * math.pi - diff)
    return np.degrees(diff)


def robust_threshold(values: np.ndarray, multiplier: float, minimum: float) -> float:
    values = values.astype(np.float32)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = 1.4826 * mad
    return max(float(minimum), med + float(multiplier) * sigma)


def bidirectional_ema(points: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.01, 1.0))
    if len(points) <= 1 or alpha >= 0.999:
        return points.astype(np.float32).copy()

    forward = points.astype(np.float32).copy()
    for i in range(1, len(forward)):
        forward[i] = alpha * points[i] + (1.0 - alpha) * forward[i - 1]

    backward = points.astype(np.float32).copy()
    for i in range(len(backward) - 2, -1, -1):
        backward[i] = alpha * points[i] + (1.0 - alpha) * backward[i + 1]
    return ((forward + backward) / 2.0).astype(np.float32)


def jump_metrics(points: np.ndarray) -> dict[str, float]:
    if len(points) < 2:
        return {
            "endpoint_jump_mean": 0.0,
            "endpoint_jump_median": 0.0,
            "endpoint_jump_p90": 0.0,
            "endpoint_jump_p95": 0.0,
            "endpoint_jump_max": 0.0,
            "midpoint_jump_mean": 0.0,
            "length_delta_mean": 0.0,
            "angle_delta_mean": 0.0,
        }
    endpoint_jumps = np.array([endpoint_distance(points[i], points[i - 1]) for i in range(1, len(points))], dtype=np.float32)
    mid = (points[:, 0:2] + points[:, 2:4]) / 2.0
    midpoint_jumps = np.linalg.norm(mid[1:] - mid[:-1], axis=1)
    length_delta = np.abs(np.diff(line_length(points)))
    ang = np.unwrap(angle_rad(points))
    angle_delta = np.degrees(np.abs(np.diff(ang)))
    return {
        "endpoint_jump_mean": float(np.mean(endpoint_jumps)),
        "endpoint_jump_median": float(np.median(endpoint_jumps)),
        "endpoint_jump_p90": float(np.percentile(endpoint_jumps, 90)),
        "endpoint_jump_p95": float(np.percentile(endpoint_jumps, 95)),
        "endpoint_jump_max": float(np.max(endpoint_jumps)),
        "midpoint_jump_mean": float(np.mean(midpoint_jumps)),
        "length_delta_mean": float(np.mean(length_delta)),
        "angle_delta_mean": float(np.mean(angle_delta)),
    }


def process_group(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = sorted(rows, key=lambda r: int(float(r["frame_id"])))
    raw = np.stack([points_from_row(r) for r in rows], axis=0)
    raw = orient_continuously(raw)
    median = rolling_median(raw, int(args.median_window))

    distance_dev = np.array([endpoint_distance(raw[i], median[i]) for i in range(len(raw))], dtype=np.float32)
    length_dev = np.abs(line_length(raw) - line_length(median))
    angle_dev = angle_diff_deg(raw, median)
    confidence = np.array([float(r.get("confidence", 1.0) or 0.0) for r in rows], dtype=np.float32)

    distance_thr = robust_threshold(distance_dev, float(args.distance_mad_threshold), float(args.min_distance_threshold))
    length_thr = robust_threshold(length_dev, float(args.length_mad_threshold), float(args.min_length_threshold))
    angle_thr = robust_threshold(angle_dev, float(args.angle_mad_threshold), float(args.min_angle_threshold))

    outlier = (
        (confidence < float(args.min_confidence))
        | (distance_dev > distance_thr)
        | (length_dev > length_thr)
        | (angle_dev > angle_thr)
    )

    replaced = raw.copy()
    replaced[outlier] = median[outlier]
    temporal = bidirectional_ema(replaced, float(args.ema_alpha))

    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        reasons = []
        if confidence[i] < float(args.min_confidence):
            reasons.append("low_conf")
        if distance_dev[i] > distance_thr:
            reasons.append("distance")
        if length_dev[i] > length_thr:
            reasons.append("length")
        if angle_dev[i] > angle_thr:
            reasons.append("angle")
        out = dict(row)
        out["x1_temporal"] = float(temporal[i, 0])
        out["y1_temporal"] = float(temporal[i, 1])
        out["x2_temporal"] = float(temporal[i, 2])
        out["y2_temporal"] = float(temporal[i, 3])
        out["temporal_outlier"] = int(bool(outlier[i]))
        out["temporal_outlier_reason"] = "|".join(reasons)
        out["temporal_distance_dev"] = float(distance_dev[i])
        out["temporal_length_dev"] = float(length_dev[i])
        out["temporal_angle_dev"] = float(angle_dev[i])
        out_rows.append(out)

    metrics = {
        "frames": int(len(rows)),
        "outlier_count": int(np.sum(outlier)),
        "outlier_fraction": float(np.mean(outlier)) if len(outlier) else 0.0,
        "thresholds": {
            "distance": float(distance_thr),
            "length": float(length_thr),
            "angle_deg": float(angle_thr),
            "min_confidence": float(args.min_confidence),
        },
        "raw": jump_metrics(raw),
        "median": jump_metrics(median),
        "temporal": jump_metrics(temporal),
    }
    if all(k in rows[0] for k in ("x1_smooth", "y1_smooth", "x2_smooth", "y2_smooth")):
        existing = np.stack(
            [
                np.array([float(r["x1_smooth"]), float(r["y1_smooth"]), float(r["x2_smooth"]), float(r["y2_smooth"])], dtype=np.float32)
                for r in rows
            ],
            axis=0,
        )
        metrics["existing_smooth"] = jump_metrics(existing)
    return out_rows, metrics


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("video", "")), str(row.get("calc_type", "LVID")))
        groups.setdefault(key, []).append(row)
    return groups


def write_overlay(video_path: str | Path, output_path: str | Path, rows: list[dict[str, Any]]) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    by_frame = {int(float(r["frame_id"])): r for r in rows}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {output_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total if total > 0 else None, desc="Writing temporal overlay", unit="frame", dynamic_ncols=True, file=sys.stdout)
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = by_frame.get(frame_id)
        if row is not None:
            raw = np.array([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float32)
            temporal = np.array(
                [
                    float(row["x1_temporal"]),
                    float(row["y1_temporal"]),
                    float(row["x2_temporal"]),
                    float(row["y2_temporal"]),
                ],
                dtype=np.float32,
            )
            label = f"T01 conf={float(row.get('confidence', 0.0)):.2f} out={int(row.get('temporal_outlier', 0))}"
            overlay = draw_lvid_overlay(frame, pred=raw, pred_color=(0, 0, 255), label=label)
            overlay = draw_lvid_overlay(overlay, pred=temporal, pred_color=(255, 0, 0))
            writer.write(overlay)
        else:
            writer.write(frame)
        frame_id += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    groups = group_rows(rows)

    all_rows: list[dict[str, Any]] = []
    group_metrics: dict[str, Any] = {}
    for key, group in groups.items():
        out_rows, metrics = process_group(group, args)
        all_rows.extend(out_rows)
        group_metrics[f"{key[0]}::{key[1]}"] = metrics

    all_rows.sort(key=lambda r: (str(r.get("video", "")), str(r.get("calc_type", "")), int(float(r["frame_id"]))))
    base_fields = list(rows[0].keys()) if rows else []
    added_fields = [
        "x1_temporal",
        "y1_temporal",
        "x2_temporal",
        "y2_temporal",
        "temporal_outlier",
        "temporal_outlier_reason",
        "temporal_distance_dev",
        "temporal_length_dev",
        "temporal_angle_dev",
    ]
    fieldnames = base_fields + [f for f in added_fields if f not in base_fields]
    write_rows(args.output_csv, all_rows, fieldnames)

    metrics = {
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "groups": group_metrics,
        "parameters": {
            "median_window": int(args.median_window),
            "ema_alpha": float(args.ema_alpha),
            "min_confidence": float(args.min_confidence),
        },
    }
    metrics_json = Path(args.metrics_json) if args.metrics_json else Path(args.output_csv).with_suffix(".metrics.json")
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.video and args.overlay_video:
        write_overlay(args.video, args.overlay_video, all_rows)

    print(f"Wrote: {args.output_csv}")
    print(f"Metrics: {metrics_json}")
    if args.overlay_video:
        print(f"Overlay: {args.overlay_video}")
    print(json.dumps(metrics, indent=2)[:4000])


if __name__ == "__main__":
    main()
