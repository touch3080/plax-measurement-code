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

The authors adopted the existing inference and analysis environments on
13 September 2026. Exact Python/package versions, complete distribution
snapshots, dependency constraints and effective CMR inference settings are in
[current reproduction environments](current_runtime.md). These environments
support the public release and are distinct from historical CMR training and
inference environments that have not been fully recovered.

The historical CMR reanalysis used Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 and Matplotlib 3.10.8. These are recorded historical versions, not a claim that the portable release was rerun in that environment.

The public release is checked separately using synthetic inputs and source-function comparisons. The tested package versions and checks are recorded in `release_validation.json`. Broad dependency ranges in `pyproject.toml` specify intended compatibility; they do not certify every possible combination. Statistical results may differ in last-digit rounding across library versions.

Version 0.1.1 passed 41 synthetic tests, rechecked the seven frozen aggregate files,
and passed Python compilation checks. The 30-test run, four-figure generation and
visual inspection in v0.1.0 are retained under `historical_validations` in
`release_validation.json`; the figures were not regenerated during this update.

Primary clinical processing and the T01 temporal sensitivity path are different settings. CMR intervals on leave-one-out calibrated errors resample frozen out-of-fold predictions and are conditional on those fitted predictions; clinical cross-validation bootstrap intervals rerun the full fitting process. Do not interpret either as external clinical validation.

## Recorded clinical settings and v0.1.1 audit

[Study settings](study_settings.json) specify the frozen clinical Butterworth
filter at 0.23 cycles/frame and the PLAX peak-distance rule. The generic command
defaults are not substitutes for those recorded settings. The primary adjusted
analysis retains four F04 comparisons and five EF-versus-base comparisons; the
later E10 common-video analysis retains five and six, respectively. Explicit
family parameters preserve those counts when fewer methods are selected for
display. A single-F04 input keeps its legacy default family denominators of one,
while recording zero selected F04-comparator contrasts; it creates no comparator.

The [aggregate audit record](clinical_parameter_audit.json) publishes settings,
source-code hashes, counts and numerical errors without participant rows, video
identifiers, source-machine paths or restricted-input hashes. Local replay
checked 496 successful method/video runs and 1,241 cycles from cached endpoints.
ED/ES indices and all downstream video-level numbers matched exactly; the largest
examination-aggregation difference was 7.11e-15 EF percentage points. Both
previously invalid E10 videos again failed the geometry gate. This checks cached
endpoint processing, not new pixel-to-endpoint inference or physiological timing
accuracy.

The updated statistical API was checked against 404 numeric fields of the frozen
primary adjusted analysis (maximum difference zero) and 66 specified fields from
the E10 common-video analysis (maximum difference 2.50e-16). The latter covers the
three Supplementary Table S6 paired contrasts, the four model partial-r summaries
and the four CV RMSE-improvement-versus-base summaries. It does not claim parity
for the historical E10 archive's unused, family-adjusted absolute CV-error
intervals. Figure 4 uses marginal 95% intervals; its 73 aggregate numeric fields
matched the original adjusted output exactly. Source inputs and frozen results
were unchanged.

The parity record identifies the API hash actually checked. A subsequent
compatibility fix only restores the default denominator for a single selected
F04 method, with a dedicated regression test; it leaves the audited four- and
five-method settings unchanged. No full numerical replay is claimed for a
different set of patients or unrecorded settings.

These local checks required authorized source caches and clinical tables, which
are not distributed. Public users can run the synthetic tests and redraw the
aggregate figures; reproducing the full local replay requires separately
authorized inputs. See [measurement provenance](measurement_provenance.json) and
[clinical provenance](clinical_provenance.json) for the current records and
explicitly retained v0.1.0 validation history.

## Subsequent real-video check of the v0.1.1 algorithms

On 13 September 2026 the public F04 video runner was executed on two prespecified,
authorized clinical videos, totaling 275 frames. Seven original model/selector
assets matched the published SHA-256 values before loading, and six configurations
matched the published templates. All six candidate trajectories, fused raw and
whole-video T01 trajectories matched the frozen local clinical outputs exactly.
The public measurement CLI then reproduced ED/ES indices, cycle counts, volumes
and EF for F04-SG5, F04-T01 and the fresh single E10 candidate: six video/branch
comparisons and 12 cycles, with zero numerical differences.

The [aggregate record](video_inference_audit.json) includes the actual algorithm
hashes and environment, and [direct dependency constraints](inference-constraints.txt)
record the tested Windows/CUDA versions. Restricted videos, trajectories, input
hashes and case results remain local. This check covers the representative clips
on that runtime, not the entire cohort, CPU execution, training or clinical
validity. The earlier source-workstation portability differences on a separate
EchoNet-LVH clip remain documented in the record.

## Data and model access

Dataset access must be obtained through the respective provider. Restricted source records and patient-level derivatives are not redistributed. The authors authorized public F04 and CMR model distribution on 13 September 2026. [Versioned model bundles](model_downloads.md) contain inference-required tensors/configurations with model manifests; optimizer state, logs, embedded source paths and case descriptions were removed. These assets permit model loading with separately authorized inputs, but do not establish complete raw-data/training reproduction, physical CMR calibration or independent validation. Third-party official comparator/pretraining weights are linked to their providers.

All distributed unit tests use artificial inputs. The separately reported local
parity checks used authorized records and publish only aggregate verification
results. Files generated from actual authorized records should remain in local
output directories. Public figure outputs are based solely on the aggregate data
listed above.
