from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricSummary:
    endpoint_error: float
    endpoint_error_swap: float
    midpoint_error: float
    length_error: float
    angle_error_deg: float


def line_metrics(pred: np.ndarray, target: np.ndarray) -> MetricSummary:
    """Compute line metrics for [N, 4] endpoint arrays."""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    p1 = pred[:, 0:2]
    p2 = pred[:, 2:4]
    t1 = target[:, 0:2]
    t2 = target[:, 2:4]

    e1 = np.linalg.norm(p1 - t1, axis=1)
    e2 = np.linalg.norm(p2 - t2, axis=1)
    ordered = (e1 + e2) / 2.0

    se1 = np.linalg.norm(p1 - t2, axis=1)
    se2 = np.linalg.norm(p2 - t1, axis=1)
    swapped = np.minimum(ordered, (se1 + se2) / 2.0)

    midpoint_pred = (p1 + p2) / 2.0
    midpoint_target = (t1 + t2) / 2.0
    midpoint = np.linalg.norm(midpoint_pred - midpoint_target, axis=1)

    pred_len = np.linalg.norm(p2 - p1, axis=1)
    target_len = np.linalg.norm(t2 - t1, axis=1)
    length = np.abs(pred_len - target_len)

    pred_vec = p2 - p1
    target_vec = t2 - t1
    pred_ang = np.arctan2(pred_vec[:, 1], pred_vec[:, 0])
    target_ang = np.arctan2(target_vec[:, 1], target_vec[:, 0])
    angle = np.abs(pred_ang - target_ang)
    angle = np.minimum(angle, 2 * math.pi - angle)
    angle_deg = np.degrees(angle)

    return MetricSummary(
        endpoint_error=float(np.mean(ordered)),
        endpoint_error_swap=float(np.mean(swapped)),
        midpoint_error=float(np.mean(midpoint)),
        length_error=float(np.mean(length)),
        angle_error_deg=float(np.mean(angle_deg)),
    )

