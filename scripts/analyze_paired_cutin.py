#!/usr/bin/env python3
"""Reproduce paired cut-in statistics from public aggregate 2x2 counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plax_measurement.cutin import paired_binary_summary, read_aggregate_tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Aggregate four-cell CSV; see docs/paired_cutin.md")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tables = read_aggregate_tables(args.input)
    result = {
        "scope": "Statistics of supplied aggregate paired counts; this computation does not verify the input observations or their provenance.",
        "published_study_context": {
            "design": "post hoc exploratory video-level paired observation",
            "dynamic_label_evidence": "Author-confirmed retrospective manual review on 2026-09-13; explicit historical dynamic zero records were not recovered.",
            "dynamic_path": "F04", "static_path_named_model": None,
            "strict_same_es_endpoint": False,
            "paired_difference_ci_estimated": False,
        },
        "sets": {name: paired_binary_summary(*cells) for name, cells in tables.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "paired_cutin_recomputed.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_sets": len(tables), "output": str(path)}))


if __name__ == "__main__":
    main()
