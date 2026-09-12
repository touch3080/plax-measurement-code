"""Original six-candidate F04 fusion with the original offline T01 smoother."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import threading
from types import FunctionType, SimpleNamespace

import cv2
import numpy as np
import torch

from f04_vendor import infer_lvid_f02_video as original
from f04_vendor import temporal_smooth_lvid_video as temporal
from dataclasses import dataclass
from model_bundle import bundle_path, verify_model_bundle


@dataclass
class LVIDVideoResult:
    rows: list[dict]
    metrics: dict

NAMES = ('e07', 'e14', 'e10_yolo11s', 'e11_yolov8s', 'e16_swin_t', 'e17_swin_s')


class F04AnalysisCancelled(RuntimeError):
    """An interrupted video must never publish partial F04/T01 results."""


def validate_video_rows(rows, temporal=False):
    keys = ('x1_temporal', 'y1_temporal', 'x2_temporal', 'y2_temporal') if temporal else ('x1', 'y1', 'x2', 'y2')
    if not rows or [row.get('frame_id') for row in rows] != list(range(len(rows))):
        raise ValueError('F04 requires a complete, consecutive zero-based video')
    for row in rows:
        try:
            points = np.asarray([row[key] for key in keys], dtype=float).reshape(2, 2)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'F04 frame {row["frame_id"]}: missing/invalid {keys} coordinates') from exc
        if not np.isfinite(points).all() or np.linalg.norm(points[0] - points[1]) <= 0:
            raise ValueError(f'F04 frame {row["frame_id"]}: nonfinite or zero-length sampling line')


def bundle_files(bundle_path):
    path = Path(bundle_path).resolve()
    scheme = json.loads(path.read_text(encoding='utf-8-sig'))
    if scheme['default'] != 'original_F04_plus_T01_for_offline_video':
        raise ValueError('Expected original F04 + T01 scheme')
    if [r['experiment_id'] for r in scheme['candidates']] != ['E07', 'E14', 'E10', 'E11', 'E16', 'E17']:
        raise ValueError('Unexpected F04 candidate identity/order')
    relative = [scheme['selector']]
    for spec in scheme['candidates']:
        relative.extend((spec['config'], spec['checkpoint']))
        if 'architecture' in spec:
            relative.append(spec['architecture'])
    paths = [(path.parent / name).resolve() for name in relative]
    if any(not p.is_relative_to(path.parent) for p in paths):
        raise ValueError('F04 bundle path escapes bundle directory')
    return scheme, paths


class F04Pipeline:
    def __init__(self, bundle_config_path=None, device=None):
        if bundle_config_path is None:
            raise ValueError('A user-supplied --bundle path is required')
        self.bundle_path = Path(bundle_config_path)
        self.scheme, paths = bundle_files(self.bundle_path)
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError('; '.join(missing))
        self.root = self.bundle_path.resolve().parent
        if self.scheme.get('asset_format_version') == 1:
            self.asset_manifest = verify_model_bundle(self.root, self.scheme['model_asset_manifest'])
            expected = {bundle_path(self.root, item['path']) for item in self.asset_manifest['files']}
            if not all(p in expected for p in paths):
                raise ValueError('Required F04 asset is not covered by the manifest')
        else:
            self.asset_manifest = None
        self._spec_by_checkpoint = {
            str((self.root / spec['checkpoint']).resolve()): spec for spec in self.scheme['candidates']}
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.selector = original.load_selector(self.root / self.scheme['selector'])
        if tuple(self.selector['candidate_names']) != NAMES:
            raise ValueError('Selector candidate order does not match original F04')
        if self.selector['strategy'] != {'method': 'softmax_blend', 'temperature': 1.0}:
            raise ValueError('Unexpected F04 fusion strategy')
        self.selector['model'].n_jobs = 4
        self._heatmap_models = {}
        self._yolo_models = {}
        self._lock = threading.RLock()
        self._last_metadata = None

    def _load_heatmap(self, config_path, checkpoint_path, device):
        key = str(checkpoint_path)
        if key not in self._heatmap_models:
            cfg = original.load_yaml(config_path)
            model_cfg = deepcopy(cfg['model'])
            # Full checkpoint restores all parameters; do not download ImageNet initialization.
            model_cfg['pretrained'] = False
            model = original.build_model(model_cfg)
            spec = self._spec_by_checkpoint[str(Path(checkpoint_path).resolve())]
            released = spec.get('checkpoint_format') == 'plax_heatmap_state_v1'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=released)
            if released and checkpoint.get('format') != 'plax_heatmap_state_v1':
                raise ValueError('Unexpected released heatmap checkpoint format')
            model.load_state_dict(checkpoint['model_state'], strict=True)
            self._heatmap_models[key] = (cfg, model.to(device).eval())
        return self._heatmap_models[key]

    def _load_yolo(self, checkpoint_path):
        key = str(checkpoint_path)
        if key not in self._yolo_models:
            spec = self._spec_by_checkpoint[str(Path(checkpoint_path).resolve())]
            if spec.get('checkpoint_format') == 'plax_yolo_state_v1':
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if checkpoint.get('format') != 'plax_yolo_state_v1':
                    raise ValueError('Unexpected released YOLO checkpoint format')
                model = original.YOLO(str(bundle_path(self.root, spec['architecture'])), task='pose')
                model.model.load_state_dict(checkpoint['model_state'], strict=True)
                model.model.names = checkpoint['names']
                model.model.kpt_shape = model.model.yaml['kpt_shape']
                model.overrides = dict(checkpoint['inference_args'], model=str(checkpoint_path))
                model.model.args = dict(model.overrides)
                self._yolo_models[key] = model
            else:
                self._yolo_models[key] = original.YOLO(checkpoint_path)
        return self._yolo_models[key]

    def _device_config(self, path):
        cfg = original.load_yaml(path)
        cfg.setdefault('train', {})['device'] = str(self.device)
        return cfg

    def _call_original(self, function, *args, progress=False, progress_callback=None):
        # Per-call globals preserve the original arithmetic without modifying shared modules.
        class CallbackProgress(original.tqdm):
            def __init__(self, *args, **kwargs):
                self.completed = 0
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                self.completed += n
                if progress_callback is not None:
                    progress_callback(self.completed, self.total)
                return super().update(n)

        namespace = dict(function.__globals__)
        namespace.update(load_heatmap_model=self._load_heatmap, YOLO=self._load_yolo,
                         load_yaml=self._device_config,
                         tqdm=partial(CallbackProgress, disable=not progress))
        return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)(*args)

    def predict_frames(self, frames, video_stem='video', batch_size=4, progress=False, progress_callback=None):
        if not frames:
            raise ValueError('F04 requires at least one BGR frame')
        h, w = frames[0].shape[:2]
        if any(f.shape != (h, w, 3) or f.dtype != np.uint8 for f in frames):
            raise ValueError('F04 requires same-size uint8 BGR frames')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        args = SimpleNamespace(batch_size=batch_size, yolo_batch_size=min(4, batch_size),
            imgsz=512, yolo_conf=.01, crop_scale=2.6, min_crop_size=128, max_crop_size=340,
            ok_predicted_error=14., uncertain_predicted_error=20., ok_disagreement=24.,
            uncertain_disagreement=40., ok_confidence=.45, uncertain_confidence=.30)
        specs = self.scheme['candidates']
        args.coarse_config = str(self.root / specs[0]['config'])
        args.coarse_checkpoint = str(self.root / specs[0]['checkpoint'])
        args.roi_config = str(self.root / specs[1]['config'])
        args.roi_checkpoint = str(self.root / specs[1]['checkpoint'])

        def stage_callback(base, weight, stage):
            def report(done, total):
                if progress_callback is not None:
                    progress_callback((base + weight * done / total) / 6, stage)
            report(0, len(frames))
            return report

        with self._lock:
            points, confs, crops = self._call_original(original.run_heatmap_candidates,
                frames, args, self.device, progress=progress,
                progress_callback=stage_callback(0, 2, 'E07 + E14'))
            for candidate_index, (name, spec) in enumerate(zip(NAMES[2:], specs[2:]), start=2):
                inputs = (name, str(self.root / spec['config']), str(self.root / spec['checkpoint']), frames)
                callback = stage_callback(candidate_index, 1, spec['experiment_id'])
                if 'yolo' in name:
                    pred, conf = self._call_original(original.run_yolo_candidate,
                        *inputs, w, h, args, progress=progress, progress_callback=callback)
                else:
                    pred, conf = self._call_original(original.run_full_heatmap_candidate,
                        *inputs, args, self.device, progress=progress, progress_callback=callback)
                points[name], confs[name] = pred, conf
            items = original.build_items(video_stem, list(NAMES), points, confs, crops)
            errors = original.predict_candidate_errors(self.selector['model'], items, list(NAMES))
            fused, decisions = original.apply_strategy_to_items(items, list(NAMES), errors, self.selector['strategy'])
            rows = original.rows_from_fusion(video_stem, items, list(NAMES), fused, decisions, w, h, args)
        return rows

    @staticmethod
    def smooth_video_rows(rows):
        validate_video_rows(rows)
        args = SimpleNamespace(median_window=5, ema_alpha=.35, min_confidence=.45,
            distance_mad_threshold=6., length_mad_threshold=6., angle_mad_threshold=6.,
            min_distance_threshold=12., min_length_threshold=10., min_angle_threshold=12.)
        result, metrics = temporal.process_group(rows, args)
        validate_video_rows(result, temporal=True)
        metrics.update(model='f04', temporal='T01', uses_future_frames=True,
                       coordinate_space='original_video_pixels', phase_feature=0)
        return LVIDVideoResult(result, metrics)

    def predict_video(self, video_path, batch_size=4, progress=False, progress_callback=None, should_cancel=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(video_path)
        rows = []
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        def report(completed, stage):
            if should_cancel is not None and should_cancel():
                raise F04AnalysisCancelled('F04 analysis aborted')
            if progress_callback is not None:
                progress_callback(completed, count, stage)

        try:
            while True:
                report(len(rows), 'decode')
                frames = []
                for _ in range(64):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break
                offset = len(rows)
                kwargs = {}
                if progress_callback is not None or should_cancel is not None:
                    kwargs['progress_callback'] = lambda fraction, stage: report(offset + fraction * len(frames), stage)
                chunk = self.predict_frames(frames, Path(video_path).stem, batch_size, progress, **kwargs)
                if len(chunk) != len(frames):
                    raise RuntimeError('F04 candidate output count does not match decoded frames')
                validate_video_rows(chunk)
                for i, row in enumerate(chunk):
                    row['frame_id'] = offset + i
                rows.extend(chunk)
        finally:
            cap.release()
        if not rows or (count > 0 and len(rows) != count):
            raise RuntimeError(f'Incomplete F04 video decode: {len(rows)}/{count}')
        report(len(rows), 'T01')
        result = self.smooth_video_rows(rows)
        report(len(rows), 'complete')
        result.metrics['fps'] = fps
        return result

    def inference(self, image, contour_num=24):
        row = self.predict_frames([image], batch_size=1)[0]
        line = np.array([row[k] for k in ('x1', 'y1', 'x2', 'y2')], dtype=float).reshape(2, 2)
        self._last_metadata = dict(row, model='f04', temporal='none_single_frame', phase_feature=0)
        return np.zeros((contour_num, 2)), line, 0, None

    def get_last_frame_metadata(self):
        return deepcopy(self._last_metadata)
