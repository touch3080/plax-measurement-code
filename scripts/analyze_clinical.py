"""Analyze a locally supplied clinical CSV and write aggregate results only."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from plax_measurement.clinical import DEFAULT_MODELS, DEFAULT_SEED, analyze_cohorts, resolve_comparison_families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--cv-bootstrap-reps", type=int, default=2000)
    parser.add_argument("--f04-family-size", type=int, help="Cumulative F04 comparisons per metric; defaults to selected methods minus one")
    parser.add_argument("--base-increment-family-size", type=int, help="Cumulative EF-versus-base comparisons; defaults to selected method count")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-columns", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--sensitivities", nargs="*", default=[], choices=[
        "patient_equal", "exact_report_only", "exclude_topcoded_age", "single_visit"])
    args = parser.parse_args()
    if min(args.bootstrap_reps, args.cv_bootstrap_reps) < 1000:
        parser.error("Use at least 1000 replicates for both interval calculations")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.input.resolve() == (args.output_dir / "results.json").resolve():
        parser.error("Input and output must be distinct files")
    try:
        # Preserve local pseudonymous IDs, including leading zeros, only in memory.
        frame = pd.read_csv(args.input, dtype={"patient_id": str, "study_id": str, "variant": str})
        with threadpool_limits(limits=2):
            analyses = analyze_cohorts(frame, sensitivities=args.sensitivities,
                bootstrap_reps=args.bootstrap_reps, cv_bootstrap_reps=args.cv_bootstrap_reps,
                seed=args.seed, model_columns=args.model_columns,
                f04_family_size=args.f04_family_size,
                base_increment_family_size=args.base_increment_family_size)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Clinical analysis failed: {error}\n")
    import plax_measurement.clinical as clinical
    result = {
        "design": {
            "bootstrap_replicates": args.bootstrap_reps,
            "cv_bootstrap_replicates": args.cv_bootstrap_reps,
            "seed": args.seed,
            "model_columns": args.model_columns,
            "comparison_families": resolve_comparison_families(len(args.model_columns), args.f04_family_size, args.base_increment_family_size),
            "sensitivities": args.sensitivities,
            "outcome": "log10(NT-proBNP in pg/mL)",
            "adjustment": "intercept + age_proxy/10 + log2_creatinine; add each LVEF/10 separately",
            "cv_bootstrap": "Resample patients; remove every copy of the held patient and refit every fold",
            "intervals": "95% percentile intervals; Bonferroni within each reported contrast family; >=95% valid draws",
            "invalid_fit": "No ridge or pseudoinverse fallback; crossproduct eigenvalue ratio must exceed 1e-12",
            "clinical_source_sha256": hashlib.sha256(Path(clinical.__file__).read_bytes()).hexdigest(),
            "patient_rows_exported": False,
            "prospective_validation": False,
        },
        "analyses": analyses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote aggregate results.json")


if __name__ == "__main__":
    main()
