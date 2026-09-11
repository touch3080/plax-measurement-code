"""Analyze user-supplied endpoint CSVs; does not perform image inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plax_measurement.measurement import POINT_KEYS, analyze_trajectory, same_phase_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("geometry", "trajectory"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--spacing-cm", type=float, required=name == "trajectory")
    trajectory = sub.choices["trajectory"]
    trajectory.add_argument("--fps", type=float, default=0.)
    trajectory.add_argument("--smoothing", choices=("none", "matched_savgol5", "T01"), default="none")
    trajectory.add_argument("--filter", action="store_true", help="Use original optional Butterworth curve filter")
    trajectory.add_argument("--cutoff-frequency", type=float, default=.2, help="cycles/frame, default 0.2")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if args.mode == "trajectory":
        result = analyze_trajectory(rows, spacing_cm=args.spacing_cm, fps=args.fps,
            smoothing=args.smoothing, filt=args.filter, cutoff_frequency=args.cutoff_frequency)
    else:
        if not rows:
            parser.error("No geometry rows supplied")
        result = []
        for number, row in enumerate(rows, 1):
            frames = [float(row[k]) for k in ("static_es_frame", "dynamic_es_frame")]
            if any(not f.is_integer() or f < 0 for f in frames) or frames[0] != frames[1]:
                parser.error(f"Row {number}: ES frame indices must be identical nonnegative integers")
            lines = [[[float(row[prefix + key]) for key in ("x1", "y1")],
                      [float(row[prefix + key]) for key in ("x2", "y2")]]
                     for prefix in ("ed_", "static_es_", "dynamic_es_")]
            result.append({"row": number, "same_es_frame": int(frames[0]),
                           **same_phase_comparison(*lines, spacing_cm=args.spacing_cm)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.mode} result to {args.output}")


if __name__ == "__main__":
    main()
