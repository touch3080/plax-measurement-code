from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmaps import (
    CALC_TO_ID,
    OFFSET_RADIUS_DEFAULT,
    make_endpoint_heatmaps,
    make_unified_lvid_heatmaps,
    make_unified_lvid_line_heatmaps,
    make_unified_lvid_line_offset_heatmaps,
)
from .utils import cv_imread


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    val = safe_float(value, None)
    if val is None:
        return default
    return int(round(val))


def build_video_index(data_root: str | Path) -> dict[str, Path]:
    data_root = Path(data_root)
    index: dict[str, Path] = {}
    for path in data_root.rglob("*.avi"):
        index[path.stem] = path
    return index


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = points.reshape(2, 2).astype(np.float32)
    ones = np.ones((2, 1), dtype=np.float32)
    hom = np.concatenate([pts, ones], axis=1)
    out = hom @ matrix.T
    return out.reshape(4).astype(np.float32)


def apply_augmentations(
    image: np.ndarray,
    points: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    points = points.astype(np.float32).copy()

    if random.random() < float(cfg.get("affine_prob", 0.0)):
        rotate = random.uniform(-float(cfg.get("rotate_deg", 0.0)), float(cfg.get("rotate_deg", 0.0)))
        scale_delta = float(cfg.get("scale_frac", 0.0))
        scale = random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
        tx = random.uniform(-float(cfg.get("translate_frac", 0.0)), float(cfg.get("translate_frac", 0.0))) * width
        ty = random.uniform(-float(cfg.get("translate_frac", 0.0)), float(cfg.get("translate_frac", 0.0))) * height
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotate, scale).astype(np.float32)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        points = transform_points(points, matrix)
        points[0::2] = np.clip(points[0::2], 0, width - 1)
        points[1::2] = np.clip(points[1::2], 0, height - 1)

    img = image.astype(np.float32)
    contrast = float(cfg.get("contrast", 0.0))
    brightness = float(cfg.get("brightness", 0.0))
    if contrast > 0:
        img *= random.uniform(1.0 - contrast, 1.0 + contrast)
    if brightness > 0:
        img += random.uniform(-brightness, brightness) * 255.0

    gamma_jitter = float(cfg.get("gamma", 0.0))
    if gamma_jitter > 0:
        gamma = random.uniform(1.0 - gamma_jitter, 1.0 + gamma_jitter)
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

    noise_std = float(cfg.get("noise_std", 0.0))
    if noise_std > 0:
        img += np.random.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    if random.random() < float(cfg.get("blur_prob", 0.0)):
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return np.clip(img, 0, 255).astype(np.uint8), points


class LVIDFrameDataset(Dataset):
    def __init__(
        self,
        label_csv: str | Path,
        image_height: int,
        image_width: int,
        sigma: float,
        line_sigma: float | None = None,
        offset_radius: float = OFFSET_RADIUS_DEFAULT,
        grayscale: bool = True,
        augment: bool = False,
        augment_cfg: dict[str, Any] | None = None,
        output_mode: str = "by_calc",
        max_samples: int | None = None,
    ) -> None:
        self.rows = read_csv_rows(label_csv)
        if max_samples is not None:
            self.rows = self.rows[: int(max_samples)]
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.sigma = float(sigma)
        self.line_sigma = float(line_sigma) if line_sigma is not None else None
        self.offset_radius = float(offset_radius)
        self.grayscale = bool(grayscale)
        self.augment = bool(augment)
        self.augment_cfg = augment_cfg or {}
        self.output_mode = output_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        frame_path = Path(row["frame_path"])
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv_imread(frame_path, flag)
        if image is None:
            raise FileNotFoundError(f"Could not read frame image: {frame_path}")
        if not self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points = np.array(
            [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ],
            dtype=np.float32,
        )

        if image.shape[0] != self.image_height or image.shape[1] != self.image_width:
            scale_x = self.image_width / image.shape[1]
            scale_y = self.image_height / image.shape[0]
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            points[0::2] *= scale_x
            points[1::2] *= scale_y

        if self.augment:
            image, points = apply_augmentations(image, points, self.augment_cfg)

        calc_id = CALC_TO_ID[row["calc"]]
        if self.output_mode == "unified":
            target, mask = make_unified_lvid_heatmaps(
                self.image_height,
                self.image_width,
                tuple(float(v) for v in points),
                self.sigma,
            )
        elif self.output_mode == "unified_line":
            target, mask = make_unified_lvid_line_heatmaps(
                self.image_height,
                self.image_width,
                tuple(float(v) for v in points),
                self.sigma,
                self.line_sigma,
            )
        elif self.output_mode == "unified_line_offset":
            target, mask = make_unified_lvid_line_offset_heatmaps(
                self.image_height,
                self.image_width,
                tuple(float(v) for v in points),
                self.sigma,
                self.line_sigma,
                self.offset_radius,
            )
        else:
            target, mask = make_endpoint_heatmaps(
                self.image_height,
                self.image_width,
                calc_id,
                tuple(float(v) for v in points),
                self.sigma,
            )

        if self.grayscale:
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            "image": image_tensor,
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "points": torch.from_numpy(points.astype(np.float32)),
            "calc_id": torch.tensor(calc_id, dtype=torch.long),
            "sample_id": row["sample_id"],
            "calc": row["calc"],
            "video_id": row["video_id"],
            "frame": int(row["frame"]),
            "frame_path": str(frame_path),
        }
