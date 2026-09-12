"""Portable PLAX measurement, original phase rules and offline T01 smoothing.

Functions extracted from the research application retain their numerical rules.
Public wrappers below validate user inputs; no database or image data is read.
See docs/models.md and docs/measurement_provenance.json for scope and provenance.
Original source: MIT, Copyright (c) 2024 Haibo Meng; see repository LICENSE.
"""
from __future__ import annotations

import argparse
import math
from typing import Any
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

POINT_KEYS = ("x1", "y1", "x2", "y2")



def apply_savgol_filter(data, window_length=5, polyorder=2):
    if isinstance(data, list):
        data = np.array(data)
    n, c, _ = data.shape
    
    # Savgol filter requires n >= window_length
    if n < window_length:
        if n > 0: # Only print if there's actually data
            print(f"[DEBUG] Skipping Savgol filter: frames={n}, window={window_length}")
        return data
        
    filtered_data = np.zeros_like(data).astype(np.float32)

    for i in range(c):
        for j in range(2):
            filtered_data[:, i, j] = savgol_filter(data[:, i, j], window_length, polyorder)
    return filtered_data


def Teichholz_volume(lvid):
    """
    Calculate the left ventricular volume using the Teichholz formula. Make sure the unit of lvid is cm!
    """
    return 7 * (lvid ** 3) / (2.4 + lvid)


def estimate_period_fft(data, fs=1):
    signal = np.asarray(data, dtype=float).ravel()
    n = len(signal)
    if n < 3 or not np.isfinite(signal).all() or np.allclose(signal, signal[0]):
        return max(float(n), 1.0)

    signal = signal - np.mean(signal)
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    magnitudes = np.abs(np.fft.rfft(signal))
    if len(magnitudes) <= 1:
        return max(float(n), 1.0)

    magnitudes[0] = 0
    peak_idx = int(np.argmax(magnitudes))
    fundamental_frequency = frequencies[peak_idx]
    if fundamental_frequency <= 0 or magnitudes[peak_idx] <= 0:
        return max(float(n), 1.0)

    period = 1 / fundamental_frequency
    # print('estimate period:', period)
    return max(float(period), 1.0)


def _coerce_finite_curve(data):
    signal = np.asarray(data, dtype=float).ravel()
    if signal.size == 0:
        return signal

    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if not finite.any():
        return np.array([], dtype=float)

    frame_idx = np.arange(signal.size)
    return np.interp(frame_idx, frame_idx[finite], signal[finite])


def _prepare_curve_for_extrema_detection(signal, open_file=None, smooth_sigma=1.0):
    curve = _coerce_finite_curve(signal)
    if curve.size == 0:
        return curve

    if open_file and getattr(open_file, "filt", False):
        curve = butter_lowpass_filter(
            curve,
            cutoff_frequency=getattr(open_file, "cutoff_frequency", 0.2),
        )
    elif smooth_sigma and curve.size >= 5:
        curve = gaussian_filter(curve, sigma=float(smooth_sigma))
    return np.asarray(curve, dtype=float).ravel()


def _find_curve_extrema_indices(
    signal,
    mode="max",
    period=None,
    *,
    distance_period_fraction=0.45,
    prominence_fraction=0.05,
):
    curve = _coerce_finite_curve(signal)
    n = curve.size
    if n <= 2:
        return np.array([], dtype=int)

    amplitude = float(np.ptp(curve))
    if amplitude <= 1e-9:
        return np.array([], dtype=int)

    if period is None:
        period = estimate_period_fft(curve)
    period = max(1.0, min(float(period), float(n)))
    max_distance = max(1.0, float(n - 1))
    distance = int(
        round(
            min(
                max(period * float(distance_period_fraction), 1.0),
                max_distance,
            )
        )
    )
    prominence = max(amplitude * float(prominence_fraction), 1e-9)

    search_curve = curve if mode == "max" else -curve
    peaks = find_peaks(search_curve, distance=max(distance, 1), prominence=prominence)[0]
    indices = [int(i) for i in peaks if 0 < int(i) < n - 1]

    high_level = float(np.percentile(curve, 75))
    low_level = float(np.percentile(curve, 25))
    if not indices:
        interior = curve[1:-1]
        if mode == "max":
            candidates = np.where((interior >= curve[:-2]) & (interior >= curve[2:]) & (interior >= high_level))[0] + 1
            if candidates.size:
                indices.append(int(candidates[np.argmax(curve[candidates])]))
        else:
            candidates = np.where((interior <= curve[:-2]) & (interior <= curve[2:]) & (interior <= low_level))[0] + 1
            if candidates.size:
                indices.append(int(candidates[np.argmin(curve[candidates])]))

    return np.array(sorted(set(indices)), dtype=int)


def _collapse_to_alternating_extrema(signal, ed_candidates, es_candidates):
    """Keep the strongest extremum from each consecutive same-type run."""
    curve = _coerce_finite_curve(signal)
    frame_count = int(curve.size)
    if frame_count == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    events_by_frame = {}
    for marker_type, raw_indices in (("ED", ed_candidates), ("ES", es_candidates)):
        for raw_index in np.atleast_1d(raw_indices):
            try:
                frame_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 < frame_index < frame_count - 1:
                events_by_frame.setdefault(frame_index, set()).add(marker_type)

    # A frame cannot be both a local maximum and minimum. Ignore ambiguous
    # plateau candidates instead of emitting overlapping ED/ES anchors.
    events = [
        (frame_index, next(iter(marker_types)))
        for frame_index, marker_types in events_by_frame.items()
        if len(marker_types) == 1
    ]
    events.sort(key=lambda item: item[0])

    alternating = []
    for frame_index, marker_type in events:
        if not alternating or alternating[-1][1] != marker_type:
            alternating.append((frame_index, marker_type))
            continue

        previous_index = alternating[-1][0]
        current_is_stronger = (
            curve[frame_index] > curve[previous_index]
            if marker_type == "ED"
            else curve[frame_index] < curve[previous_index]
        )
        if current_is_stronger:
            alternating[-1] = (frame_index, marker_type)

    ed_indices = [index for index, marker_type in alternating if marker_type == "ED"]
    es_indices = [index for index, marker_type in alternating if marker_type == "ES"]
    return np.asarray(ed_indices, dtype=int), np.asarray(es_indices, dtype=int)


def _select_plax_volume_markers(signal):
    """Select PLAX volume extrema using the manually validated peak rules."""
    curve = _coerce_finite_curve(signal)
    frame_count = int(curve.size)
    if frame_count <= 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Baseline drift can make FFT report the entire clip as one period. The
    # reviewed PLAX corrections show that this suppresses real short cycles;
    # cap the detection period while still allowing boundary half-cycles.
    estimated_period = estimate_period_fft(curve)
    effective_period = min(float(estimated_period), max(frame_count / 3.0, 1.0))
    extrema_kwargs = {
        "period": effective_period,
        "distance_period_fraction": 0.55,
        "prominence_fraction": 0.05,
    }
    ed_candidates = _find_curve_extrema_indices(curve, mode="max", **extrema_kwargs)
    es_candidates = _find_curve_extrema_indices(curve, mode="min", **extrema_kwargs)
    return _collapse_to_alternating_extrema(curve, ed_candidates, es_candidates)


def sort_landmarks(landmarks, mode='A4C'):
    """
    Sort the landmarks in the order of A4C or PLAX.
    :param landmarks: A NumPy array of shape (n, 2(PLAX) or 3(A4C), 2) containing the landmarks.
    :param mode: The mode to sort the landmarks in. Either 'A4C' or 'PLAX'.
    :return: A NumPy array of shape (n, 2(PLAX) or 3(A4C), 2) with the sorted landmarks.
    """
    if mode == 'A4C':
        # Sort the landmarks in the order of [apical, septal, lateral].
        assert landmarks.shape[1] == 3, "Invalid number of landmarks for A4C mode."
        means = np.mean(landmarks, axis=0)  # (3, 2)
        va_index = int(means[:, 1].argmin())
        remain = [i for i in range(3) if i != va_index]
        if len(remain) != 2:
            remain = [0, 1, 2]
        # Pick aap from remaining two by larger x, and pap as the last one.
        if means[remain[0], 0] >= means[remain[1], 0]:
            aap_index, pap_index = remain[0], remain[1]
        else:
            aap_index, pap_index = remain[1], remain[0]
        return np.array([landmarks[:, va_index], landmarks[:, pap_index], landmarks[:, aap_index]]).transpose(1, 0, 2)
    elif mode == 'PLAX':
        # Sort the landmarks in the order of [lower, upper].
        assert landmarks.shape[1] == 2, "Invalid number of landmarks for PLAX mode."
        upper_index = np.mean(landmarks, axis=0)[:, 1].argmin()
        lower_index = 1 - upper_index
        return np.array([landmarks[:, lower_index], landmarks[:, upper_index]]).transpose(1, 0, 2)
    else:
        raise ValueError("Invalid mode. Only 'A4C' and 'PLAX' are supported.")


def livds2volume(lvids, spacing=1):
    lvids = lvids * spacing
    volumes = 7 * (lvids ** 3) / (2.4 + lvids)
    return volumes


def pair_volume_extrema(volumes, ed_index, es_index, index_base=0):
    """Pair each ED with the following ES before the next ED."""
    raw_volumes = np.asarray(volumes).ravel()
    base = int(index_base)

    def valid_indices(raw_indices):
        values = []
        for raw_index in raw_indices:
            try:
                position = int(raw_index) - base
            except (TypeError, ValueError):
                continue
            if 0 <= position < len(raw_volumes):
                values.append(position)
        return sorted(set(values))

    ed_positions = valid_indices(ed_index)
    es_positions = valid_indices(es_index)
    pairs = []
    for position, ed_position in enumerate(ed_positions):
        next_ed = (
            ed_positions[position + 1]
            if position + 1 < len(ed_positions)
            else len(raw_volumes)
        )
        interval_es = [
            es_position
            for es_position in es_positions
            if ed_position < es_position < next_ed
        ]
        if not interval_es:
            continue

        finite_es = []
        for es_position in interval_es:
            try:
                es_volume = float(raw_volumes[es_position])
            except (TypeError, ValueError):
                continue
            if np.isfinite(es_volume):
                finite_es.append((es_volume, es_position))
        if not finite_es:
            continue

        es_position = min(finite_es)[1]
        try:
            ed_volume = float(raw_volumes[ed_position])
            es_volume = float(raw_volumes[es_position])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ed_volume) or not np.isfinite(es_volume):
            continue
        ef_value = -1.0 if ed_volume == 0 else (ed_volume - es_volume) / ed_volume * 100.0
        pairs.append((ed_position + base, es_position + base, float(ef_value)))
    return pairs


def butter_lowpass_filter(data, cutoff_frequency=0.2, sampling_rate=1, order=4, btype='low', padtype='constant'):
    """
    巴特沃斯低通滤波器 (Butterworth Low-pass Filter)
    用于平滑关键点轨迹，去除抖动。
    
    Args:
        data: 输入数据 (1D 或 2D)
        cutoff: 截止频率
        fs: 采样率 
        order: 滤波器阶数
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # butter/filtfilt requires len(data) > padlen (default 3*max(len(a), len(b)))
    # For a 4th order filter, padlen is 15.
    if len(data) <= 15:
        # print(f"[DEBUG] Skipping Butter filter: length={len(data)} <= 15")
        return data

    nyquist_frequency = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)

    if data.ndim == 1:
        filtered_data = filtfilt(b, a, data, padtype=padtype)
    elif data.ndim == 2:
        filtered_data = np.column_stack([filtfilt(b, a, data[:, i], padtype=padtype) for i in range(data.shape[1])])
    else:
        raise ValueError("Unsupported data dimension. Only 1D or 2D data is supported.")

    return filtered_data


class PLAXLandMark:
    """
    PLAX 切面关键点处理与计算
    计算参数：LVID (左室内径), FS (缩短分数), LVEF (射血分数) 等
    关键点结构：2点 [Lower, Upper] (室间隔, 后壁)
    """

    def __init__(self, points, open_file=None, width=8, sigma=1, **kwargs):
        """
        points is n frames inference result, shape is (n, 2, 2)
        3 means [apical(va), septal(pap), lateral(aap)] in order.
        """
        if isinstance(points, dict):
            points_array = np.array([points[key] for key in sorted(points.keys())])
        elif isinstance(points, list):
            points_array = np.array(points)
        elif isinstance(points, np.ndarray):
            points_array = points
        if points_array.shape[-1] == 3:
            points_array = points_array[:, :, :2]

        points_array = sort_landmarks(points_array, mode='PLAX')

        self.lower = points_array[:, 0]
        self.upper = points_array[:, 1]
        if open_file and open_file.filt:
            self.lower = butter_lowpass_filter(self.lower, cutoff_frequency=open_file.cutoff_frequency)
            self.upper = butter_lowpass_filter(self.upper, cutoff_frequency=open_file.cutoff_frequency)
        self.mid = (self.lower + self.upper) / 2

        self.lowerv, self.upperv = [np.diff(i, axis=0) for i in [self.lower, self.upper]]
        self._lvids_cache = None
        self._jetter_cache = None

        smooth_lvid = gaussian_filter(np.sqrt(np.sum((self.lower - self.upper) ** 2, axis=1)), sigma=sigma)
        self.ed_index = find_peaks(smooth_lvid, width=width)[0].astype(int)
        self.es_index = find_peaks(-smooth_lvid, width=width)[0].astype(int)

    @property
    def lvids(self):
        """
        Calculate the LVID from a series of upper and lower end point.
        """
        if self._lvids_cache is None:
            self._lvids_cache = np.sqrt(np.sum((self.lower - self.upper) ** 2, axis=1))
        return self._lvids_cache

    def set_EDES(self, res, open_file=None, **kwargs):
        detect_curve = _prepare_curve_for_extrema_detection(
            res,
            open_file=open_file,
            smooth_sigma=kwargs.get("smooth_sigma", 0.75),
        )
        self.ed_index, self.es_index = _select_plax_volume_markers(detect_curve)

    def set_ED(self, ED):
        valid_ed = [x for x in np.atleast_1d(ED) if x is not None]
        self.ed_index = np.array(valid_ed).astype(int)

    def set_ES(self, ES):
        valid_es = [x for x in np.atleast_1d(ES) if x is not None]
        self.es_index = np.array(valid_es).astype(int)

    @property
    def jetter(self):
        if self._jetter_cache is None:
            jetter = np.zeros((3, self.lower.shape[0]))
            jetter[0][1:] = np.linalg.norm(self.lowerv, axis=1) ** 2
            jetter[1][1:] = np.linalg.norm(self.upperv, axis=1) ** 2
            jetter[2][1:] = abs(np.diff(self.lvids, axis=0)) ** 2
            self._jetter_cache = jetter
        return self._jetter_cache


def _contiguous_segments(indices):
    values = sorted({int(index) for index in indices})
    if not values:
        return []
    segments = []
    start = previous = values[0]
    for index in values[1:]:
        if index != previous + 1:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return segments


def _is_valid_plax_landmark(raw_points):
    try:
        points = np.asarray(raw_points, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return points.shape == (2, 2) and bool(np.all(np.isfinite(points)))


def build_segmented_plax_curve_payload(
    landmarks,
    frame_validity,
    spacing,
    fps=0.0,
    filt=False,
    cutoff_frequency=0.2,
    marker_indices=None,
):
    """Build PLAX curves without joining or filtering across invalid frame gaps."""
    frame_count = len(landmarks) if landmarks is not None else 0
    raw_validity = frame_validity if frame_validity is not None else []
    validity = [bool(value) for value in raw_validity]
    if frame_count == 0:
        raise RuntimeError("segmented PLAX curve invariant failed: empty video")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("PLAX physical spacing must be finite and positive")
    if len(validity) != frame_count:
        raise RuntimeError(
            "segmented PLAX curve invariant failed: frame-validity count mismatch"
        )

    valid_indices = [
        index
        for index, is_valid in enumerate(validity)
        if is_valid and _is_valid_plax_landmark(landmarks[index])
    ]
    if len(valid_indices) / max(frame_count, 1) < 0.50:
        raise RuntimeError(
            "segmented PLAX curve invariant failed: valid fraction is below 50%"
        )

    valid_set = set(valid_indices)
    excluded_frames = [index + 1 for index in range(frame_count) if index not in valid_set]
    segments = _contiguous_segments(valid_indices)
    lvids = [None] * frame_count
    lv_volumes = [None] * frame_count
    metric_volumes = [None] * frame_count
    jetter = {
        "lower": [None] * frame_count,
        "upper": [None] * frame_count,
        "lvids_diff": [None] * frame_count,
    }
    ed_indices = []
    es_indices = []
    segment_metadata = []
    try:
        safe_fps = float(fps or 0.0)
    except (TypeError, ValueError):
        safe_fps = 0.0
    if not np.isfinite(safe_fps) or safe_fps < 0:
        safe_fps = 0.0
    minimum_metric_frames = max(8, int(np.ceil(safe_fps * 0.5))) if safe_fps > 0 else 8
    filter_context = SimpleNamespace(
        filt=bool(filt),
        cutoff_frequency=float(cutoff_frequency),
    )

    for start, end in segments:
        segment_points = np.asarray(landmarks[start : end + 1], dtype=np.float32)
        plax_marks = PLAXLandMark(segment_points, width=8)
        raw_lvids = np.asarray(plax_marks.lvids, dtype=float) * float(spacing) * 10.0
        raw_volumes = np.asarray(livds2volume(raw_lvids / 10.0), dtype=float)
        display_lvids = raw_lvids
        display_volumes = raw_volumes
        if filt:
            display_lvids = butter_lowpass_filter(
                raw_lvids,
                cutoff_frequency=float(cutoff_frequency),
            )
            display_volumes = butter_lowpass_filter(
                raw_volumes,
                cutoff_frequency=float(cutoff_frequency),
            )

        segment_jitter = np.asarray(plax_marks.jetter, dtype=float)
        for offset, frame_index in enumerate(range(start, end + 1)):
            lvids[frame_index] = float(display_lvids[offset])
            lv_volumes[frame_index] = float(display_volumes[offset])
            metric_volumes[frame_index] = float(raw_volumes[offset])
            for series_index, name in enumerate(("lower", "upper", "lvids_diff")):
                if offset == 0 and start > 0:
                    jetter[name][frame_index] = None
                else:
                    jetter[name][frame_index] = float(segment_jitter[series_index, offset])

        metric_eligible = len(segment_points) >= minimum_metric_frames
        local_ed = []
        local_es = []
        if metric_eligible and marker_indices is None:
            plax_marks.set_EDES(raw_volumes, filter_context)
            local_ed = [int(value) for value in plax_marks.ed_index.tolist()]
            local_es = [int(value) for value in plax_marks.es_index.tolist()]
            ed_indices.extend(start + value for value in local_ed)
            es_indices.extend(start + value for value in local_es)

        segment_metadata.append(
            {
                "start_frame": start + 1,
                "end_frame": end + 1,
                "frame_count": end - start + 1,
                "metric_eligible": metric_eligible,
                "ed_count": len(local_ed),
                "es_count": len(local_es),
            }
        )

    marker_curve = np.asarray(
        [np.nan if value is None else float(value) for value in metric_volumes],
        dtype=float,
    )
    if marker_indices is None:
        ed_array, es_array = _collapse_to_alternating_extrema(
            marker_curve, ed_indices, es_indices,
        )
        ed_indices = ed_array.tolist()
        es_indices = es_array.tolist()
    else:
        ed_indices, es_indices = (list(indices) for indices in marker_indices)
        if any(not isinstance(index, (int, np.integer)) or not 0 <= index < frame_count
               for index in ed_indices + es_indices):
            raise ValueError("PLAX marker index is outside the video")

    ef_values = []
    for ed_index, es_index, ef_value in pair_volume_extrema(
        metric_volumes,
        ed_indices,
        es_indices,
    ):
        if all(index in valid_set for index in range(ed_index, es_index + 1)):
            ef_values.append([int(ed_index), float(ef_value)])

    phase_indices = ed_indices if len(ed_indices) >= 2 else es_indices
    heart_period_frames = [
        current - previous
        for previous, current in zip(phase_indices, phase_indices[1:])
        if current > previous
        and all(index in valid_set for index in range(previous, current + 1))
    ]
    heart_rate_bpm = (
        float(60.0 * safe_fps / np.mean(heart_period_frames))
        if safe_fps > 0 and heart_period_frames
        else 0.0
    )
    return {
        "jetter": jetter,
        "lvids": lvids,
        "lv_volumes": lv_volumes,
        "ef": ef_values,
        "ed_index": ed_indices,
        "es_index": es_indices,
        "heart_rate_bpm": heart_rate_bpm,
        "partial_analysis": True,
        "analysis_status": "partial_success",
        "curve_mode": "segmented",
        "valid_frame_count": len(valid_indices),
        "valid_fraction": len(valid_indices) / max(frame_count, 1),
        "excluded_frames": excluded_frames,
        "missing_frames": excluded_frames,
        "valid_segments": segment_metadata,
        "minimum_metric_segment_frames": minimum_metric_frames,
    }


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


def geometry(ed, es):
    ed, es = np.asarray(ed, float), np.asarray(es, float)
    if ed.shape != (2, 2) or es.shape != (2, 2) or not np.isfinite([ed, es]).all():
        raise ValueError('Invalid line geometry')
    v, w = ed[1] - ed[0], es[1] - es[0]
    ld, ls = np.linalg.norm(v), np.linalg.norm(w)
    if min(ld, ls) <= 0:
        raise ValueError('Coincident endpoints')
    unit = v / ld
    normal = np.array([-unit[1], unit[0]])
    perp = (es - ed.mean(axis=0)) @ normal
    midpoint = abs(float(perp.mean()))
    angle = np.degrees(np.arccos(np.clip(abs(np.dot(v / ld, w / ls)), 0, 1)))
    return dict(ed_length_px=ld, es_length_px=ls,
                midpoint_offset_px=midpoint, midpoint_offset_over_ed=midpoint / ld,
                endpoint_rms_to_fixed_axis_px=float(np.sqrt(np.mean(perp**2))),
                endpoint_rms_over_ed=float(np.sqrt(np.mean(perp**2))) / ld,
                acute_angle_deg=angle)


def teichholz_volume_cm(diameter_cm):
    """Teichholz volume in mL, requiring strictly positive diameters in cm."""
    value = np.asarray(diameter_cm, dtype=float)
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("diameter_cm must be finite and positive")
    return Teichholz_volume(value)


def same_phase_comparison(ed, static_es, dynamic_es, *, spacing_cm=None):
    """Compare supplied ES endpoints on an ED-fixed axis and relocated ES line.

    The caller must verify that both ES lines refer to the same frame and video.
    This function does not infer wall intersections from the ED line.
    Geometry is in original image pixels. A scalar isotropic calibration in
    cm/pixel is required to compute physical diameters and Teichholz EF.
    """
    static = geometry(ed, static_es)
    dynamic = geometry(ed, dynamic_es)
    output = {"static": static, "dynamic": dynamic,
              "delta_lvids_px": float(dynamic["es_length_px"] - static["es_length_px"])}
    output["delta_lvids_over_ed"] = output["delta_lvids_px"] / dynamic["ed_length_px"]
    if spacing_cm is not None:
        if not np.isfinite(spacing_cm) or spacing_cm <= 0:
            raise ValueError("spacing_cm must be finite and positive")
        edv = float(teichholz_volume_cm(dynamic["ed_length_px"] * spacing_cm))
        for name, metrics in (("static", static), ("dynamic", dynamic)):
            esv = float(teichholz_volume_cm(metrics["es_length_px"] * spacing_cm))
            output[name + "_ef_percent"] = 100.0 * (edv - esv) / edv
        output["delta_ef_percentage_points"] = output["dynamic_ef_percent"] - output["static_ef_percent"]
    return output


def validate_endpoint_rows(rows):
    """Require an entire ordered, zero-based, finite, nondegenerate trajectory."""
    if not rows:
        raise ValueError("No endpoint rows supplied")
    ids = [float(row["frame_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("frame_id must be consecutive, ordered and zero-based")
    points = np.asarray([[float(row[key]) for key in POINT_KEYS] for row in rows], dtype=np.float32)
    if not np.isfinite(points).all() or np.any(line_length(points) <= 0):
        raise ValueError("Coordinates must be finite and sampling lines nonzero")
    return points.reshape(-1, 2, 2)


def smooth_t01(rows):
    """Original offline T01: median-5 outliers, bidirectional EMA alpha 0.35.

    Uses future frames. Omitted confidence defaults to 1.0 as in the original;
    supply the original F04 confidence to reproduce its low-confidence gate.
    Threshold minima refer to original-video pixels and angle degrees.
    """
    validate_endpoint_rows(rows)
    confidence = np.asarray([float(r.get("confidence", 1.0) or 0.0) for r in rows])
    if not np.isfinite(confidence).all():
        raise ValueError("confidence must be finite")
    args = SimpleNamespace(median_window=5, ema_alpha=.35, min_confidence=.45,
        distance_mad_threshold=6., length_mad_threshold=6., angle_mad_threshold=6.,
        min_distance_threshold=12., min_length_threshold=10., min_angle_threshold=12.)
    result, metrics = process_group(rows, args)
    points = np.asarray([[r[key + "_temporal"] for key in POINT_KEYS] for r in result])
    if not np.isfinite(points).all() or np.any(line_length(points) <= 0):
        raise ValueError("T01 generated a nonfinite or zero-length line")
    metrics.update(uses_future_frames=True, coordinate_space="original_video_pixels")
    return result, metrics


def analyze_trajectory(rows, *, spacing_cm, fps=0.0, smoothing="none", filt=False,
                       cutoff_frequency=.2):
    """Derive volumes, automatic ED/ES markers and cycle EF from endpoints.

    This is endpoint processing, not a pixel-to-endpoint model. Primary matched
    clinical F04 used SG5/poly2 followed by int16 conversion. T01 is a separate
    sensitivity branch and must not receive a second SG5 pass.
    """
    points = validate_endpoint_rows(rows)
    if not np.isfinite(spacing_cm) or spacing_cm <= 0:
        raise ValueError("spacing_cm must be finite and positive")
    if not np.isfinite(fps) or fps < 0:
        raise ValueError("fps must be finite and nonnegative")
    if filt and (not np.isfinite(cutoff_frequency) or not 0 < cutoff_frequency < .5):
        raise ValueError("cutoff_frequency must be between 0 and 0.5 cycles/frame")
    smoothing_metrics = None
    if smoothing == "matched_savgol5":
        filtered = apply_savgol_filter(points)
        if np.any(filtered < -32768) or np.any(filtered > 32767):
            raise ValueError("SG5 coordinates exceed int16 range")
        points = filtered.astype(np.int16)
    elif smoothing == "T01":
        temporal, smoothing_metrics = smooth_t01(rows)
        points = np.asarray([[r[key + "_temporal"] for key in POINT_KEYS] for r in temporal]).reshape(-1, 2, 2)
    elif smoothing != "none":
        raise ValueError("smoothing must be none, matched_savgol5 or T01")
    if np.any(np.linalg.norm(points[:, 1] - points[:, 0], axis=1) <= 0):
        raise ValueError("Smoothing generated zero-length lines")
    payload = build_segmented_plax_curve_payload(points, [True] * len(points), spacing_cm,
        fps=fps, filt=filt, cutoff_frequency=cutoff_frequency)
    ef = np.asarray([pair[1] for pair in payload["ef"]], dtype=float)
    status = "no_valid_cycle" if not len(ef) else (
        "nonphysical_ef" if np.any(~np.isfinite(ef) | (ef < 0) | (ef > 100)) else "ok")
    return {"status": status, "lvef_percent": float(ef.mean()) if status == "ok" else None,
            "cycle_count": len(ef), "frame_index_base": 0, "spacing_cm_per_pixel": float(spacing_cm),
            "smoothing": smoothing, "smoothing_metrics": smoothing_metrics,
            "phase_filter": {"enabled": bool(filt),
                             "cutoff_cycles_per_frame": float(cutoff_frequency) if filt else None,
                             "policy": "Butterworth order 4, constant padding, bypass segments <=15 frames" if filt else "Gaussian sigma 0.75 for segments >=5 frames"},
            "payload": payload}
