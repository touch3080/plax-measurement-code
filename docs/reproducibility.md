# Data and reproducibility

## What is directly reproducible from this repository

`python scripts/build_figures.py` redraws the four main manuscript figures from the included aggregate files. `source_data/manifest.json` records each included file's SHA-256 digest and role. The copied aggregates preserve the frozen Nature draft results; they have not been recomputed or altered for this release.

The CMR, clinical and endpoint-processing commands accept locally supplied inputs. Their documentation specifies the schema and statistical estimands. Source provenance records distinguish original files from portable release adaptations. No identifier-level source table is included, so the complete study estimates cannot be independently regenerated from this repository alone.

## Aggregate data inventory

- `three_reader_cmr_agreement_results.csv`: CMR raw and leave-one-out agreement summaries.
- `three_reader_bootstrap_contrasts.csv`: paired CMR contrasts and mean-only baseline comparisons.
- `three_reader_icc_results.csv`: reader reliability estimates and intervals.
- `three_reader_sensitivity_analysis_results.csv`: aggregate CMR sensitivity estimates.
- `plax_geometry_summary.json`: aggregate endpoint-axis geometry by study set and method.
- `static_cutin_summary.json`: reviewed static cut-in category counts and proportion interval.
- `clinical_association_summary.json`: the primary 45-examination, 39-patient clinical analysis, including 83 videos. This is the full primary cohort, not the smaller E10 common-case sensitivity cohort.

## Environments and validation

The historical CMR reanalysis used Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 and Matplotlib 3.10.8. These are recorded historical versions, not a claim that the portable release was rerun in that environment.

The public release is checked separately using synthetic inputs and source-function comparisons. The tested package versions and checks are recorded in `release_validation.json`. Broad dependency ranges in `pyproject.toml` specify intended compatibility; they do not certify every possible combination. Statistical results may differ in last-digit rounding across library versions.

Primary clinical processing and the T01 temporal sensitivity path are different settings. CMR intervals on leave-one-out calibrated errors resample frozen out-of-fold predictions and are conditional on those fitted predictions; clinical cross-validation bootstrap intervals rerun the full fitting process. Do not interpret either as external clinical validation.

## Data and model access

Dataset access must be obtained through the respective provider. Restricted source records and patient-level derivatives are not redistributed. The corresponding author can answer availability questions for compatible model checkpoints; no checkpoint download or availability commitment is implied by the source release. Inference adapters supplied without the model bundle are source code, not a complete pretrained distribution.

All tests use artificial inputs. Files generated from actual authorized records should remain in local output directories. Public figure outputs are based solely on the aggregate data listed above.
