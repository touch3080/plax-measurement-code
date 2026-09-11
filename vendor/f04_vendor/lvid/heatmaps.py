from __future__ import annotations

import math
from typing import Tuple

import numpy as np


CALC_TO_ID = {"LVIDd": 0, "LVIDs": 1}
ID_TO_CALC = {0: "LVIDd", 1: "LVIDs"}
CALC_CHANNELS = {
    0: (0, 1),  # LVIDd p1, p2
    1: (2, 3),  # LVIDs p1, p2
}
OFFSET_RADIUS_DEFAULT = 4.0


def draw_gaussian(heatmap: np.ndarray, x: float, y: float, sigma: float) -> None:
    """Draw a clipped 2D Gaussian into a single heatmap channel."""
    height, width = heatmap.shape
    radius = max(1, int(math.ceil(3 * sigma)))
    cx = int(round(x))
    cy = int(round(y))

    left = max(0, cx - radius)
    right = min(width - 1, cx + radius)
    top = max(0, cy - radius)
    bottom = min(height - 1, cy + radius)
    if left > right or top > bottom:
        return

    xs = np.arange(left, right + 1, dtype=np.float32)
    ys = np.arange(top, bottom + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    heatmap[top : bottom + 1, left : right + 1] = np.maximum(
        heatmap[top : bottom + 1, left : right + 1], patch
    )


def draw_line_gaussian(
    heatmap: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    sigma: float,
) -> None:
    """Draw a soft line heatmap using point-to-segment distance."""
    height, width = heatmap.shape
    radius = max(1, int(math.ceil(3 * sigma)))
    left = max(0, int(math.floor(min(x1, x2))) - radius)
    right = min(width - 1, int(math.ceil(max(x1, x2))) + radius)
    top = max(0, int(math.floor(min(y1, y2))) - radius)
    bottom = min(height - 1, int(math.ceil(max(y1, y2))) + radius)
    if left > right or top > bottom:
        return

    xs = np.arange(left, right + 1, dtype=np.float32)
    ys = np.arange(top, bottom + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    vx = float(x2 - x1)
    vy = float(y2 - y1)
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        draw_gaussian(heatmap, x1, y1, sigma)
        return

    t = ((xx - x1) * vx + (yy - y1) * vy) / denom
    t = np.clip(t, 0.0, 1.0)
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    patch = np.exp(-dist2 / (2 * sigma * sigma))
    heatmap[top : bottom + 1, left : right + 1] = np.maximum(
        heatmap[top : bottom + 1, left : right + 1], patch
    )


def draw_offset_targets(
    offset_x: np.ndarray,
    offset_y: np.ndarray,
    x: float,
    y: float,
    sigma: float,
    offset_radius: float = OFFSET_RADIUS_DEFAULT,
) -> None:
    """Draw normalized endpoint offset targets around one keypoint.

    Offset channels are encoded in [0, 1], where 0.5 means no offset at that
    pixel. Decoding maps them back to [-offset_radius, +offset_radius] pixels.
    """
    height, width = offset_x.shape
    radius = max(1, int(math.ceil(3 * sigma)))
    cx = int(round(x))
    cy = int(round(y))

    left = max(0, cx - radius)
    right = min(width - 1, cx + radius)
    top = max(0, cy - radius)
    bottom = min(height - 1, cy + radius)
    if left > right or top > bottom:
        return

    xs = np.arange(left, right + 1, dtype=np.float32)
    ys = np.arange(top, bottom + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    scale = max(float(offset_radius), 1e-6)
    dx = np.clip(0.5 + (float(x) - xx) / (2.0 * scale), 0.0, 1.0)
    dy = np.clip(0.5 + (float(y) - yy) / (2.0 * scale), 0.0, 1.0)
    offset_x[top : bottom + 1, left : right + 1] = dx
    offset_y[top : bottom + 1, left : right + 1] = dy


def make_endpoint_heatmaps(
    height: int,
    width: int,
    calc_id: int,
    points: Tuple[float, float, float, float],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target heatmaps and active-channel mask for one LVID sample."""
    target = np.zeros((4, height, width), dtype=np.float32)
    mask = np.zeros((4, 1, 1), dtype=np.float32)
    ch1, ch2 = CALC_CHANNELS[int(calc_id)]
    x1, y1, x2, y2 = points
    draw_gaussian(target[ch1], x1, y1, sigma)
    draw_gaussian(target[ch2], x2, y2, sigma)
    mask[ch1, 0, 0] = 1.0
    mask[ch2, 0, 0] = 1.0
    return target, mask


def make_unified_lvid_heatmaps(
    height: int,
    width: int,
    points: Tuple[float, float, float, float],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-channel endpoint heatmaps for a single unified LVID line."""
    target = np.zeros((2, height, width), dtype=np.float32)
    mask = np.ones((2, 1, 1), dtype=np.float32)
    x1, y1, x2, y2 = points
    draw_gaussian(target[0], x1, y1, sigma)
    draw_gaussian(target[1], x2, y2, sigma)
    return target, mask


def make_unified_lvid_line_heatmaps(
    height: int,
    width: int,
    points: Tuple[float, float, float, float],
    sigma: float,
    line_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint plus line heatmaps for a single unified LVID line."""
    target = np.zeros((3, height, width), dtype=np.float32)
    mask = np.ones((3, 1, 1), dtype=np.float32)
    x1, y1, x2, y2 = points
    draw_gaussian(target[0], x1, y1, sigma)
    draw_gaussian(target[1], x2, y2, sigma)
    draw_line_gaussian(target[2], x1, y1, x2, y2, line_sigma if line_sigma is not None else sigma)
    return target, mask


def make_unified_lvid_line_offset_heatmaps(
    height: int,
    width: int,
    points: Tuple[float, float, float, float],
    sigma: float,
    line_sigma: float | None = None,
    offset_radius: float = OFFSET_RADIUS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint, line, and normalized endpoint offset targets.

    Channels:
        0: endpoint 1 heatmap
        1: endpoint 2 heatmap
        2: auxiliary line heatmap
        3: endpoint 1 x-offset, normalized to [0, 1]
        4: endpoint 1 y-offset, normalized to [0, 1]
        5: endpoint 2 x-offset, normalized to [0, 1]
        6: endpoint 2 y-offset, normalized to [0, 1]
    """
    target = np.zeros((7, height, width), dtype=np.float32)
    mask = np.ones((7, 1, 1), dtype=np.float32)
    x1, y1, x2, y2 = points
    draw_gaussian(target[0], x1, y1, sigma)
    draw_gaussian(target[1], x2, y2, sigma)
    draw_line_gaussian(target[2], x1, y1, x2, y2, line_sigma if line_sigma is not None else sigma)
    target[3:7] = 0.5
    draw_offset_targets(target[3], target[4], x1, y1, sigma, offset_radius)
    draw_offset_targets(target[5], target[6], x2, y2, sigma, offset_radius)
    return target, mask


def decode_endpoint_heatmaps(
    heatmaps: np.ndarray,
    calc_ids: np.ndarray,
    offset_radius: float = OFFSET_RADIUS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode endpoint coordinates from heatmaps.

    Args:
        heatmaps: array with shape [B, 4, H, W].
        calc_ids: array with shape [B], 0 for LVIDd and 1 for LVIDs.

    Returns:
        coords: [B, 4] array as x1, y1, x2, y2.
        confidence: [B] mean endpoint max confidence.
    """
    batch, channels, height, width = heatmaps.shape
    coords = np.zeros((batch, 4), dtype=np.float32)
    confidence = np.zeros((batch,), dtype=np.float32)
    for i in range(batch):
        if channels in (2, 3, 7):
            ch1, ch2 = 0, 1
        else:
            ch1, ch2 = CALC_CHANNELS[int(calc_ids[i])]
        vals = []
        for offset, ch in enumerate((ch1, ch2)):
            hm = heatmaps[i, ch]
            flat_idx = int(np.argmax(hm))
            y = flat_idx // width
            x = flat_idx % width
            px = float(x)
            py = float(y)
            if channels >= 7:
                ox_ch, oy_ch = (3, 4) if offset == 0 else (5, 6)
                px += (float(heatmaps[i, ox_ch, y, x]) - 0.5) * 2.0 * float(offset_radius)
                py += (float(heatmaps[i, oy_ch, y, x]) - 0.5) * 2.0 * float(offset_radius)
                px = float(np.clip(px, 0.0, float(width - 1)))
                py = float(np.clip(py, 0.0, float(height - 1)))
            coords[i, offset * 2] = px
            coords[i, offset * 2 + 1] = py
            vals.append(float(hm[y, x]))
        confidence[i] = float(sum(vals) / len(vals))
    return coords, confidence


def decode_both_lvid_heatmaps(heatmaps: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decode LVIDd and LVIDs endpoints from heatmaps for video inference."""
    batch = heatmaps.shape[0]
    if heatmaps.shape[1] in (2, 3, 7):
        calc_ids = np.zeros((batch,), dtype=np.int64)
        return {"LVID": decode_endpoint_heatmaps(heatmaps, calc_ids)}

    output = {}
    for calc_id, calc in ID_TO_CALC.items():
        calc_ids = np.full((batch,), calc_id, dtype=np.int64)
        output[calc] = decode_endpoint_heatmaps(heatmaps, calc_ids)
    return output
