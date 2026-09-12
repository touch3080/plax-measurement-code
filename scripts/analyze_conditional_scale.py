"""Run conditional A/B scale point diagnostics on authorized local CSV inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from plax_measurement.conditional_scale import diagnose_conditional_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--scale-mode', choices=('fov', 'axes'), default='fov')
    parser.add_argument('--write-case-results', action='store_true',
                        help='Also write local case-level measurements and common-subset selection')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be new or empty')
    before = hashlib.sha256(args.input.read_bytes()).hexdigest()
    try:
        rows = pd.read_csv(args.input, dtype={'case_id': str}, keep_default_na=True,
                           float_precision='round_trip')
        result = diagnose_conditional_scale(rows, args.scale_mode)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != before:
        parser.error('Input changed during analysis')
    result.aggregate['input_sha256'] = before
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'conditional_scale_aggregate.json').write_text(
        json.dumps(result.aggregate, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if args.write_case_results:
        result.measurements.to_csv(args.output_dir / 'conditional_measurements.csv', index=False)
        result.selection.to_csv(args.output_dir / 'diagnostic_selection.csv', index=False)
    print(f"Saved A/B point diagnostics for {result.aggregate['selection']['common_diagnostic_cases']} common cases")


if __name__ == '__main__':
    main()
