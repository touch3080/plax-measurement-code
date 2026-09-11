from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_csv(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_lvid_overlay(
    image: np.ndarray,
    gt: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    gt_color: tuple[int, int, int] = (0, 255, 0),
    pred_color: tuple[int, int, int] = (0, 0, 255),
    label: str | None = None,
) -> np.ndarray:
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    if gt is not None:
        _draw_line(out, gt, gt_color, "GT")
    if pred is not None:
        _draw_line(out, pred, pred_color, "P")
    if label:
        cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_line(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in points]
    cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.circle(image, (x1, y1), 4, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x2, y2), 4, color, -1, cv2.LINE_AA)
    cv2.putText(image, f"{text}1", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"{text}2", (x2 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def cv_imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read images from Unicode paths on Windows."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def cv_imwrite(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    """Write images to Unicode paths on Windows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def project_root_from_file(file: str | Path, parents: int = 1) -> Path:
    path = Path(file).resolve()
    for _ in range(parents):
        path = path.parent
    return path


def maybe_make_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
