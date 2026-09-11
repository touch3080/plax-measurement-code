#!/usr/bin/env python3
"""Analyze a user-supplied long-format three-reader CMR measurement CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from plax_measurement.cmr import BOOTSTRAP_REPS, SEED, analyze_reader_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Long-format reader CSV; see docs/cmr.md")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS, help="Participant bootstrap repetitions")
    parser.add_argument("--write-predictions", action="store_true",
                        help="Also save harmonized and prediction tables containing input case identifiers")
    args = parser.parse_args()
    rows = pd.read_csv(args.input, dtype={"case_id": str, "reader": str, "method": str})
    results = analyze_reader_table(rows, seed=args.seed, reps=args.reps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results["sample_flow"].to_csv(args.output_dir / "sample_flow.csv", index=False)
    filenames = {
        "agreement": "agreement_summary.csv",
        "contrasts": "bootstrap_contrasts.csv",
        "raw_error_contrasts": "raw_error_bootstrap_contrasts.csv",
        "reliability": "interreader_reliability.csv",
        "thresholds": "exploratory_threshold_summary.csv",
        "calibration_fits": "full_sample_calibration_fits.csv",
    }
    for name, tables in results["subsets"].items():
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for key, filename in filenames.items():
            tables[key].to_csv(folder / filename, index=False)
        if args.write_predictions:
            tables["predictions"].to_csv(folder / "case_predictions.csv", index=False)
    if args.write_predictions:
        results["harmonized"].to_csv(args.output_dir / "reader_measurements_harmonized.csv", index=False)
    manifest = dict(
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        seed=args.seed, bootstrap_reps=args.reps, nominal_cm_per_pixel=0.15,
        calibration_ci="participant bootstrap of frozen out-of-fold predictions; conditional on fitted LOOCV models",
        sensitivity_calibration="LOOCV refitted separately within each selected subset",
        averaging="Teichholz EF per reader, then arithmetic mean of three EF values",
        clipping=False, case_level_outputs_written=args.write_predictions,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        optional_audited_flags_supplied=[flag for flag in ("same_es_functional", "complete_geometry", "reannotated") if flag in rows],
    )
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(results["sample_flow"].to_string(index=False))
    print(f"Saved CMR analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
