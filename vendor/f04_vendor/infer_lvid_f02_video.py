from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

from .fuse_lvid_predictions_learned import apply_strategy_to_items, endpoint_distance, load_selector, predict_candidate_errors
from .lvid.heatmaps import decode_both_lvid_heatmaps, decode_endpoint_heatmaps
from .lvid.models import build_model
from .lvid.utils import cv_imwrite, load_yaml, resolve_device


CANONICAL_WIDTH = 512
CANONICAL_HEIGHT = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned LVID fusion on one continuous video.")
    parser.add_argument("--selector", default="runs/lvid_f02_learned_fusion/f02_lvid_error_selector.joblib")
    parser.add_argument("--fusion-name", default="F02")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--overlay-video", default=None)
    parser.add_argument("--example-frame", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--yolo-batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--coarse-config", default="configs/lvid_unified_line010_deeplabv3plus_heatmap.yaml")
    parser.add_argument("--coarse-checkpoint", default="runs/lvid_unified_line010_deeplabv3plus_heatmap/checkpoints/best.pt")
    parser.add_argument("--roi-config", default="configs/lvid_roi_deeplabv3plus_heatmap.yaml")
    parser.add_argument("--roi-checkpoint", default="runs/lvid_roi_deeplabv3plus_heatmap/checkpoints/best.pt")
    parser.add_argument("--yolo11s-config", default="configs/lvid_yolo11s_pose.yaml")
    parser.add_argument("--yolo11s-checkpoint", default="runs/lvid_yolo11s_pose/weights/best.pt")
    parser.add_argument("--yolov8s-config", default="configs/lvid_yolov8s_pose.yaml")
    parser.add_argument("--yolov8s-checkpoint", default="runs/lvid_yolov8s_pose/weights/best.pt")
    parser.add_argument("--swin-config", default="configs/lvid_swin_t_fpn_heatmap.yaml")
    parser.add_argument("--swin-checkpoint", default="runs/lvid_swin_t_fpn_heatmap/checkpoints/best.pt")
    parser.add_argument("--swin-s-config", default="configs/lvid_swin_s_fpn_heatmap.yaml")
    parser.add_argument("--swin-s-checkpoint", default="runs/lvid_swin_s_fpn_heatmap/checkpoints/best.pt")
    parser.add_argument("--crop-scale", type=float, default=2.6)
    parser.add_argument("--min-crop-size", type=float, default=128.0)
    parser.add_argument("--max-crop-size", type=float, default=340.0)
    parser.add_argument("--ok-predicted-error", type=float, default=14.0)
    parser.add_argument("--uncertain-predicted-error", type=float, default=20.0)
    parser.add_argument("--ok-disagreement", type=float, default=24.0)
    parser.add_argument("--uncertain-disagreement", type=float, default=40.0)
    parser.add_argument("--ok-confidence", type=float, default=0.45)
    parser.add_argument("--uncertain-confidence", type=float, default=0.30)
    return parser.parse_args()


def load_heatmap_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module]:
    cfg = load_yaml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return cfg, model


def preprocess(frame: np.ndarray, width: int, height: int, grayscale: bool) -> torch.Tensor:
    if grayscale:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        arr = img.astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def compute_crop(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    crop_scale: float,
    min_crop_size: float,
    max_crop_size: float,
) -> tuple[int, int, int, int]:
    p1 = points[0:2]
    p2 = points[2:4]
    center = (p1 + p2) / 2.0
    line_len = float(np.linalg.norm(p2 - p1))
    bbox_w = float(abs(p2[0] - p1[0]))
    bbox_h = float(abs(p2[1] - p1[1]))
    crop_size = max(line_len * crop_scale, bbox_w * 2.0, bbox_h * 2.0, min_crop_size)
    crop_size = min(crop_size, max_crop_size, float(image_width), float(image_height))

    left = float(center[0] - crop_size / 2.0)
    top = float(center[1] - crop_size / 2.0)
    left = min(max(left, 0.0), max(float(image_width) - crop_size, 0.0))
    top = min(max(top, 0.0), max(float(image_height) - crop_size, 0.0))
    x0 = int(round(left))
    y0 = int(round(top))
    x1 = int(round(left + crop_size))
    y1 = int(round(top + crop_size))
    x1 = min(max(x1, x0 + 1), image_width)
    y1 = min(max(y1, y0 + 1), image_height)
    return x0, y0, x1, y1


def roi_to_full(points: np.ndarray, crop: tuple[int, int, int, int], roi_width: int, roi_height: int) -> np.ndarray:
    x0, y0, x1, y1 = crop
    out = points.astype(np.float32).copy()
    out[0::2] = out[0::2] * (float(x1 - x0) / float(roi_width)) + float(x0)
    out[1::2] = out[1::2] * (float(y1 - y0) / float(roi_height)) + float(y0)
    return out


def decode_yolo_result(result) -> tuple[np.ndarray, float]:
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0
    xy = result.keypoints.xy.cpu().numpy()
    if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0
    box_conf = result.boxes.conf.cpu().numpy()
    if result.keypoints.conf is not None:
        kpt_conf = result.keypoints.conf.cpu().numpy()
        score = box_conf * np.mean(kpt_conf[:, :2], axis=1)
    else:
        score = box_conf
    idx = int(np.argmax(score))
    return xy[idx, :2, :].reshape(4).astype(np.float32), float(score[idx])


def canonical_to_original(points: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    out = points.astype(np.float32).copy()
    out[0::2] *= float(orig_w) / float(CANONICAL_WIDTH)
    out[1::2] *= float(orig_h) / float(CANONICAL_HEIGHT)
    return out


def original_to_canonical(points: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    out = points.astype(np.float32).copy()
    out[0::2] *= float(CANONICAL_WIDTH) / float(orig_w)
    out[1::2] *= float(CANONICAL_HEIGHT) / float(orig_h)
    return out


def line_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[2:4] - points[0:2]))


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    av = a[2:4] - a[0:2]
    bv = b[2:4] - b[0:2]
    aa = math.atan2(float(av[1]), float(av[0]))
    ba = math.atan2(float(bv[1]), float(bv[0]))
    diff = abs(aa - ba)
    diff = min(diff, 2 * math.pi - diff)
    return float(math.degrees(diff))


def max_pairwise_disagreement(preds: dict[str, np.ndarray], names: list[str]) -> float:
    values = [endpoint_distance(preds[a], preds[b]) for a, b in itertools.combinations(names, 2)]
    return float(max(values)) if values else 0.0


def mean_pairwise_disagreement(preds: dict[str, np.ndarray], names: list[str]) -> float:
    values = [endpoint_distance(preds[a], preds[b]) for a, b in itertools.combinations(names, 2)]
    return float(np.mean(values)) if values else 0.0


def edge_margin(points: np.ndarray) -> float:
    return float(
        min(
            points[0],
            points[2],
            points[1],
            points[3],
            CANONICAL_WIDTH - points[0],
            CANONICAL_WIDTH - points[2],
            CANONICAL_HEIGHT - points[1],
            CANONICAL_HEIGHT - points[3],
        )
    )


def quality_flag(
    selected: np.ndarray,
    confidence: float,
    predicted_error: float,
    max_disagreement: float,
    args: argparse.Namespace,
) -> tuple[str, str]:
    reasons: list[str] = []
    if confidence < float(args.uncertain_confidence):
        reasons.append("low_confidence")
    if predicted_error > float(args.uncertain_predicted_error):
        reasons.append("high_predicted_error")
    if max_disagreement > float(args.uncertain_disagreement):
        reasons.append("model_disagreement")
    if edge_margin(selected) < -3.0:
        reasons.append("outside_frame")
    if reasons:
        return "reject_or_smooth", "|".join(reasons)

    uncertain: list[str] = []
    if confidence < float(args.ok_confidence):
        uncertain.append("borderline_confidence")
    if predicted_error > float(args.ok_predicted_error):
        uncertain.append("borderline_predicted_error")
    if max_disagreement > float(args.ok_disagreement):
        uncertain.append("borderline_disagreement")
    if edge_margin(selected) < 6.0:
        uncertain.append("near_boundary")
    if uncertain:
        return "uncertain", "|".join(uncertain)
    return "ok", ""


def draw_line(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in points]
    cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.circle(image, (x1, y1), 3, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x2, y2), 3, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, row: dict[str, Any], names: list[str], fusion_name: str) -> np.ndarray:
    overlay = frame.copy()
    colors = {
        "e07": (0, 180, 255),
        "e14": (0, 255, 255),
        "e10_yolo11s": (255, 180, 0),
        "e11_yolov8s": (255, 0, 180),
        "e16_swin_t": (80, 255, 120),
        "e17_swin_s": (140, 220, 80),
    }
    for name in names:
        pts = np.array(
            [
                row[f"{name}_x1"],
                row[f"{name}_y1"],
                row[f"{name}_x2"],
                row[f"{name}_y2"],
            ],
            dtype=np.float32,
        )
        draw_line(overlay, pts, colors.get(name, (160, 160, 160)), name)
    final = np.array([row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32)
    draw_line(overlay, final, (0, 0, 255), fusion_name)
    text = (
        f"{fusion_name} {row['quality_flag']} conf={float(row['confidence']):.2f} "
        f"pred_err={float(row['predicted_error']):.1f} src={row['selected_source']}"
    )
    cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def read_video(video_path: str | Path) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    pbar = tqdm(total=total if total > 0 else None, desc="Reading video", unit="frame", dynamic_ncols=True, file=sys.stdout)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        pbar.update(1)
    pbar.close()
    cap.release()
    return frames, fps, orig_w, orig_h


def run_heatmap_candidates(
    frames: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[tuple[int, int, int, int]]]:
    coarse_cfg, coarse_model = load_heatmap_model(args.coarse_config, args.coarse_checkpoint, device)
    roi_cfg, roi_model = load_heatmap_model(args.roi_config, args.roi_checkpoint, device)
    coarse_w = int(coarse_cfg["data"]["input_width"])
    coarse_h = int(coarse_cfg["data"]["input_height"])
    coarse_gray = bool(coarse_cfg["data"].get("grayscale", True))
    roi_w = int(roi_cfg["data"]["input_width"])
    roi_h = int(roi_cfg["data"]["input_height"])
    roi_gray = bool(roi_cfg["data"].get("grayscale", True))

    e07_pts: list[np.ndarray] = []
    e07_conf: list[float] = []
    e14_pts: list[np.ndarray] = []
    e14_conf: list[float] = []
    all_crops: list[tuple[int, int, int, int]] = []

    with torch.no_grad():
        pbar = tqdm(total=len(frames), desc="Inferring E07/E14 candidates", unit="frame", dynamic_ncols=True, file=sys.stdout)
        for start in range(0, len(frames), int(args.batch_size)):
            end = min(start + int(args.batch_size), len(frames))
            batch_frames = frames[start:end]
            coarse_inputs = [preprocess(frame, coarse_w, coarse_h, coarse_gray) for frame in batch_frames]
            coarse_batch = torch.stack(coarse_inputs, dim=0).to(device)
            coarse_probs = torch.sigmoid(coarse_model(coarse_batch)).cpu().numpy()
            decoded = decode_both_lvid_heatmaps(coarse_probs)
            coarse_pts, coarse_conf = decoded["LVID"]

            resized_gray = [
                cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (coarse_w, coarse_h), interpolation=cv2.INTER_AREA)
                for frame in batch_frames
            ]
            roi_tensors: list[torch.Tensor] = []
            crops: list[tuple[int, int, int, int]] = []
            for local_i, gray in enumerate(resized_gray):
                crop = compute_crop(
                    coarse_pts[local_i],
                    coarse_w,
                    coarse_h,
                    float(args.crop_scale),
                    float(args.min_crop_size),
                    float(args.max_crop_size),
                )
                x0, y0, x1, y1 = crop
                crop_img = gray[y0:y1, x0:x1]
                roi_img = cv2.resize(crop_img, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                if roi_gray:
                    roi_tensors.append(torch.from_numpy(roi_img.astype(np.float32) / 255.0).unsqueeze(0))
                else:
                    rgb = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)
                    roi_tensors.append(torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1))
                crops.append(crop)

            roi_batch = torch.stack(roi_tensors, dim=0).to(device)
            roi_probs = torch.sigmoid(roi_model(roi_batch)).cpu().numpy()
            calc_ids = np.zeros((roi_probs.shape[0],), dtype=np.int64)
            roi_pts, roi_conf = decode_endpoint_heatmaps(roi_probs, calc_ids)

            for local_i in range(len(batch_frames)):
                refined = roi_to_full(roi_pts[local_i], crops[local_i], roi_w, roi_h)
                e07_pts.append(coarse_pts[local_i].astype(np.float32))
                e07_conf.append(float(coarse_conf[local_i]))
                e14_pts.append(refined.astype(np.float32))
                e14_conf.append(float(roi_conf[local_i]))
                all_crops.append(crops[local_i])
                pbar.update(1)
        pbar.close()

    del coarse_model
    del roi_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        {
            "e07": np.asarray(e07_pts, dtype=np.float32),
            "e14": np.asarray(e14_pts, dtype=np.float32),
        },
        {
            "e07": np.asarray(e07_conf, dtype=np.float32),
            "e14": np.asarray(e14_conf, dtype=np.float32),
        },
        all_crops,
    )


def run_full_heatmap_candidate(
    name: str,
    config_path: str,
    checkpoint_path: str,
    frames: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    cfg, model = load_heatmap_model(config_path, checkpoint_path, device)
    width = int(cfg["data"]["input_width"])
    height = int(cfg["data"]["input_height"])
    grayscale = bool(cfg["data"].get("grayscale", True))

    points: list[np.ndarray] = []
    confs: list[float] = []
    with torch.no_grad():
        pbar = tqdm(total=len(frames), desc=f"Inferring {name}", unit="frame", dynamic_ncols=True, file=sys.stdout)
        for start in range(0, len(frames), int(args.batch_size)):
            batch_frames = frames[start : start + int(args.batch_size)]
            inputs = [preprocess(frame, width, height, grayscale) for frame in batch_frames]
            batch = torch.stack(inputs, dim=0).to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            decoded = decode_both_lvid_heatmaps(probs)
            pts, conf = decoded["LVID"]
            points.extend([p.astype(np.float32) for p in pts])
            confs.extend([float(c) for c in conf])
            pbar.update(len(batch_frames))
        pbar.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(points, dtype=np.float32), np.asarray(confs, dtype=np.float32)


def run_yolo_candidate(
    name: str,
    config_path: str,
    checkpoint_path: str,
    frames: list[np.ndarray],
    orig_w: int,
    orig_h: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_yaml(config_path)
    device = str(cfg["train"].get("device", 0))
    model = YOLO(checkpoint_path)
    points: list[np.ndarray] = []
    confs: list[float] = []
    pbar = tqdm(total=len(frames), desc=f"Inferring {name}", unit="frame", dynamic_ncols=True, file=sys.stdout)
    for start in range(0, len(frames), int(args.yolo_batch_size)):
        batch_frames = frames[start : start + int(args.yolo_batch_size)]
        results = model.predict(
            source=batch_frames,
            imgsz=int(args.imgsz),
            batch=int(args.yolo_batch_size),
            device=device,
            conf=float(args.yolo_conf),
            verbose=False,
        )
        for result in results:
            pts_orig, conf = decode_yolo_result(result)
            points.append(original_to_canonical(pts_orig, orig_w, orig_h))
            confs.append(float(conf))
            pbar.update(1)
    pbar.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(points, dtype=np.float32), np.asarray(confs, dtype=np.float32)


def build_items(
    video_stem: str,
    candidate_names: list[str],
    candidate_points: dict[str, np.ndarray],
    candidate_confs: dict[str, np.ndarray],
    crops: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    total = len(next(iter(candidate_points.values())))
    for i in range(total):
        rows: dict[str, dict[str, str]] = {}
        for name in candidate_names:
            rows[name] = {"confidence": str(float(candidate_confs[name][i]))}
        if "e14" in rows:
            x0, y0, x1, y1 = crops[i]
            rows["e14"].update(
                {
                    "crop_left": str(x0),
                    "crop_top": str(y0),
                    "crop_width": str(x1 - x0),
                    "crop_height": str(y1 - y0),
                }
            )
        items.append(
            {
                "sample_id": f"{video_stem}_frame{i}",
                "video_id": video_stem,
                "calc": "LVID",
                "frame": str(i),
                "gt": np.zeros((4,), dtype=np.float32),
                "preds": {name: candidate_points[name][i].astype(np.float32) for name in candidate_names},
                "confs": {name: float(candidate_confs[name][i]) for name in candidate_names},
                "rows": rows,
            }
        )
    return items


def rows_from_fusion(
    video_stem: str,
    items: list[dict[str, Any]],
    candidate_names: list[str],
    fused_canonical: np.ndarray,
    decisions: list[dict[str, Any]],
    orig_w: int,
    orig_h: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (item, pred, decision) in enumerate(zip(items, fused_canonical, decisions)):
        pred_orig = canonical_to_original(pred, orig_w, orig_h)
        max_disagree = max_pairwise_disagreement(item["preds"], candidate_names)
        mean_disagree = mean_pairwise_disagreement(item["preds"], candidate_names)
        flag, reason = quality_flag(
            pred,
            float(decision["confidence"]),
            float(decision["predicted_error"]),
            max_disagree,
            args,
        )
        row: dict[str, Any] = {
            "video": video_stem,
            "frame_id": i,
            "calc_type": "LVID",
            "x1": float(pred_orig[0]),
            "y1": float(pred_orig[1]),
            "x2": float(pred_orig[2]),
            "y2": float(pred_orig[3]),
            "confidence": float(decision["confidence"]),
            "quality_flag": flag,
            "reject_reason": reason,
            "selected_source": decision["selected_source"],
            "best_source": decision["best_source"],
            "second_source": decision["second_source"],
            "predicted_error": float(decision["predicted_error"]),
            "second_predicted_error": float(decision["second_predicted_error"]),
            "candidate_disagreement_mean": mean_disagree,
            "candidate_disagreement_max": max_disagree,
            "line_length": line_length(pred),
            "edge_margin": edge_margin(pred),
        }
        for name in candidate_names:
            pts_orig = canonical_to_original(item["preds"][name], orig_w, orig_h)
            row[f"weight_{name}"] = float(decision["weights"].get(name, 0.0))
            row[f"confidence_{name}"] = float(item["confs"][name])
            row[f"{name}_x1"] = float(pts_orig[0])
            row[f"{name}_y1"] = float(pts_orig[1])
            row[f"{name}_x2"] = float(pts_orig[2])
            row[f"{name}_y2"] = float(pts_orig[3])
            row[f"{name}_length"] = line_length(item["preds"][name])
            row[f"{name}_angle_diff_to_final"] = angle_diff_deg(item["preds"][name], pred)
        rows.append(row)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overlay_video(
    path: str | Path,
    frames: list[np.ndarray],
    rows: list[dict[str, Any]],
    fps: float,
    names: list[str],
    fusion_name: str,
) -> None:
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {path}")
    for frame, row in tqdm(
        zip(frames, rows),
        total=len(rows),
        desc=f"Writing {fusion_name} overlay",
        unit="frame",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        writer.write(draw_overlay(frame, row, names, fusion_name))
    writer.release()


def save_example_frame(path: str | Path, frames: list[np.ndarray], rows: list[dict[str, Any]], names: list[str], fusion_name: str) -> None:
    if not frames or not rows:
        return
    priority = {"reject_or_smooth": 0, "uncertain": 1, "ok": 2}
    best_idx = min(range(len(rows)), key=lambda i: (priority.get(str(rows[i]["quality_flag"]), 3), -float(rows[i]["predicted_error"])))
    cv_imwrite(path, draw_overlay(frames[best_idx], rows[best_idx], names, fusion_name))


def main() -> None:
    args = parse_args()
    selector = load_selector(args.selector)
    selector_model = selector["model"]
    candidate_names = list(selector["candidate_names"])
    strategy = dict(selector["strategy"])

    frames, fps, orig_w, orig_h = read_video(args.video)
    video_stem = Path(args.video).stem
    if not frames:
        raise RuntimeError(f"No frames read from video: {args.video}")

    coarse_cfg = load_yaml(args.coarse_config)
    device = resolve_device(str(coarse_cfg.get("device", "cuda")))
    candidate_points, candidate_confs, crops = run_heatmap_candidates(frames, args, device)

    if "e10_yolo11s" in candidate_names:
        yolo11_pts, yolo11_conf = run_yolo_candidate(
            "e10_yolo11s",
            args.yolo11s_config,
            args.yolo11s_checkpoint,
            frames,
            orig_w,
            orig_h,
            args,
        )
        candidate_points["e10_yolo11s"] = yolo11_pts
        candidate_confs["e10_yolo11s"] = yolo11_conf

    if "e11_yolov8s" in candidate_names:
        yolo8_pts, yolo8_conf = run_yolo_candidate(
            "e11_yolov8s",
            args.yolov8s_config,
            args.yolov8s_checkpoint,
            frames,
            orig_w,
            orig_h,
            args,
        )
        candidate_points["e11_yolov8s"] = yolo8_pts
        candidate_confs["e11_yolov8s"] = yolo8_conf

    if "e16_swin_t" in candidate_names:
        swin_pts, swin_conf = run_full_heatmap_candidate(
            "e16_swin_t",
            args.swin_config,
            args.swin_checkpoint,
            frames,
            args,
            device,
        )
        candidate_points["e16_swin_t"] = swin_pts
        candidate_confs["e16_swin_t"] = swin_conf

    if "e17_swin_s" in candidate_names:
        swin_s_pts, swin_s_conf = run_full_heatmap_candidate(
            "e17_swin_s",
            args.swin_s_config,
            args.swin_s_checkpoint,
            frames,
            args,
            device,
        )
        candidate_points["e17_swin_s"] = swin_s_pts
        candidate_confs["e17_swin_s"] = swin_s_conf

    missing = [name for name in candidate_names if name not in candidate_points]
    if missing:
        raise KeyError(f"Selector expects missing candidates: {missing}")

    items = build_items(video_stem, candidate_names, candidate_points, candidate_confs, crops)
    predicted_errors = predict_candidate_errors(selector_model, items, candidate_names)
    fused, decisions = apply_strategy_to_items(items, candidate_names, predicted_errors, strategy)
    rows = rows_from_fusion(video_stem, items, candidate_names, fused, decisions, orig_w, orig_h, args)
    write_rows(args.output_csv, rows)
    print(f"Wrote: {args.output_csv}")

    if args.overlay_video:
        write_overlay_video(args.overlay_video, frames, rows, fps, candidate_names, str(args.fusion_name))
        print(f"Overlay: {args.overlay_video}")

    if args.example_frame:
        save_example_frame(args.example_frame, frames, rows, candidate_names, str(args.fusion_name))
        print(f"Example frame: {args.example_frame}")

    quality_counts: dict[str, int] = {}
    for row in rows:
        quality_counts[str(row["quality_flag"])] = quality_counts.get(str(row["quality_flag"]), 0) + 1
    print(f"Quality counts: {quality_counts}")


if __name__ == "__main__":
    main()
