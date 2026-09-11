"""Optional F04 video inference with an explicitly supplied, trusted model bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if any((output / name).exists() for name in ("f04_lines.csv", "f04_metrics.json")):
        parser.error("Output files already exist; choose a new output directory")
    from f04_pipeline import F04Pipeline
    import cv2
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    pipeline = F04Pipeline(bundle_config_path=args.bundle, device=args.device)
    result = pipeline.predict_video(args.video, batch_size=args.batch_size, progress=args.progress)
    result.metrics.update(frame_index_base=0,
        scheme_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        torch_version=torch.__version__, device=str(pipeline.device))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "f04_lines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
    (output / "f04_metrics.json").write_text(json.dumps(result.metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(result.rows)} complete frames to {output}")


if __name__ == "__main__":
    main()
