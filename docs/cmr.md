# Portable three-reader CMR analysis

This module preserves the mathematical and statistical implementation used in the
September 10, 2026 reannotation analysis. Its primary source is
`extend_manuscript_statistics.py`; subset refitting and shared-ED interval handling
follow `update_reannotated_statistics.py`. File hashes and the exact scope of the
adaptation are recorded in [cmr_provenance.json](cmr_provenance.json).

The original primary analysis contained **152 participants with manual CMR
measurements**. The source CMR cohort comprised healthy volunteers. This is a
restricted evaluation of measurement strategies, with a pixel-count reference EF
and protocol-derived nominal in-plane scale. It does not establish clinical
interchangeability, diagnostic performance in disease, or transportability of a
calibration model. No participant-level measurements, identifiers, images, or
annotation database are included in this release.

## Run

Install the repository package as described in the main README, then run:

```bash
python scripts/analyze_cmr.py --input /path/to/readers.csv --output-dir /path/to/cmr-output
```

The source defaults are `--seed 20260820 --reps 10000`. A smaller repetition count
is useful for a software smoke test, but is not the manuscript analysis. NumPy,
pandas, and SciPy are the only analysis dependencies. Every subset refits its own
leave-one-out calibration. The CLI writes aggregate tables by default. Add
`--write-predictions` only when local case-level harmonized measurements and
predictions are also wanted; those files carry the input `case_id` values.

## Reader CSV contract

Supply exactly nine rows per case: every combination of three readers and three
methods. All input cases form the primary analysis; there is no hidden cohort
filter. Reproducing the study requires the authorized, frozen primary cohort
export and the associated audit flags, which are not distributed here.

| Column | Definition |
| --- | --- |
| `case_id` | Nonempty opaque case key, used to pair rows and sorted lexicographically before prediction/bootstrap calculations. |
| `reader` | Exactly `reader01`, `reader02`, or `reader03`. |
| `method` | Exactly `static`, `dynamic`, or `peak_to_peak`. |
| `lvidd_display_px` | Absolute vertical separation between the two ED endpoints in displayed M-mode pixels; positive. |
| `lvids_display_px` | Absolute vertical separation between the two ES endpoints in displayed M-mode pixels; nonnegative. |
| `resize_ratio_y` | Positive stored display-to-native vertical resize factor for this method's M-mode measurement. |
| `reference_ef` | Precomputed case-level pixel-count reference EF, in percent; identical on all nine rows. |

These distances represent the source operation `abs(y2 - y1)`, not the Euclidean
distance of a 2D frame line. The resize factor is essential: the source conversion
is `diameter_cm = display_distance_px * 0.15 / resize_ratio_y`. The 0.15 cm/pixel
factor is **nominal protocol scale**, not a verified patient-specific calibration.
The reference EF is an input; the module does not derive reference masks or
validate the upstream pixel counts.

For each reader, static and dynamic strategies share the **static ED observation
and its resize factor**. Dynamic ES uses the dynamic row and its resize factor.
`peak_to_peak` uses its own ED and ES observations. The dynamic row's ED field is
accepted for a uniform schema but is ignored when computing its ED diameter.

All measurements must be finite. Duplicate combinations, incomplete cases,
inconsistent references, invalid factors, and negative ES distances are rejected
without silently removing rows. Primary measurements are not clipped: ES at least
as large as ED can produce zero or negative EF, and zero ES can produce 100% EF.
Such rows remain in the primary analysis and are excluded by the physiologic
sensitivity rule below. The source retained observations rather than repairing
these values.

## Statistics retained

- **Raw EF:** `V(D) = 7 D^3 / (2.4 + D)` and
  `EF = 100 (V(ED) - V(ES)) / V(ED)`, with diameters in nominal cm. Compute EF for
  each reader, then take the arithmetic mean of the three reader EFs. Computing
  EF from averaged diameters gives a different quantity.
- **Calibration:** each leave-one-out fold fits an unconstrained least-squares
  line with intercept from raw consensus EF to reference EF. Its held-out
  prediction is not clipped. Full-sample slope/intercept and a Student-t slope
  interval are descriptive outputs, separate from held-out predictions.
- **Agreement:** signed error is prediction minus reference; report bias, MAE,
  RMSE, and Bland–Altman limits `bias ± 1.96 * sample_SD(error)`. Bias/MAE/RMSE
  intervals use participant bootstrap percentiles (2.5%, 97.5%). The limits of
  agreement themselves are point estimates, not confidence intervals. The
  proportion within 10 percentage points includes equality and has a Wilson
  interval. Pearson correlation has a Fisher-z interval using
  `1.959963984540054 / sqrt(n-3)`.
- **Paired comparisons:** a single participant resample is shared between methods
  for each replicate. Raw delta correlation is dynamic minus static or
  peak-to-peak. Calibrated delta MAE is dynamic minus static, peak-to-peak, or the
  mean-only baseline. Raw delta MAE is also retained. A negative delta MAE favors
  dynamic; a positive delta correlation favors dynamic. The source's two-sided
  bootstrap sign-tail proportion is `min(1, 2*min(P(delta<=0), P(delta>=0)))`.
  There is no plus-one correction, multiple-comparison correction, or nested
  null-model resampling; finite-repetition outputs can equal zero.
- **Mean-only baseline:** each held-out prediction is the mean reference EF of
  the other cases. Cross-validated relative R-squared is
  `1 - SSE(method_LOOCV) / SSE(mean_only_LOOCV)`, not an in-sample regression
  R-squared. The baseline prediction is algebraically inversely related to its
  held-out reference value; its correlation must not be interpreted as a
  predictive model-performance measure.
- **Reliability:** two-way absolute-agreement ICC(A,1) and ICC(A,3) are calculated
  from the cases-by-three-readers matrix separately for ED diameter, ES diameter,
  and reader EF. The source ANOVA mean-square formulas and participant bootstrap
  percentile intervals are retained. With three readers,
  `ICC(A,3) = 3*ICC(A,1)/(1+2*ICC(A,1))` where defined. The identical static/dynamic
  ED matrix shares the exact same bootstrap interval, as in the source update.
- **Exploratory thresholds:** the source's strict `<50%` and `<40%` comparisons
  of LOOCV EF versus reference EF are retained, including counts, sensitivity,
  specificity, and accuracy. Undefined sensitivity/specificity is blank in CSV.
  These tables do not establish clinical utility in a healthy-volunteer cohort.

**The bootstrap of calibrated agreement and delta MAE holds the out-of-fold
predictions fixed.** It is conditional on the fitted LOOCV models. It does not
refit calibration in each bootstrap replicate and does not represent uncertainty
from model selection, an external validation set, or all calibration training
variability. A full nested bootstrap would be a different analysis.

## Sensitivity subsets

The automatic `physiologic` subset requires `ED > ES > 0` for **every reader and
every strategy**, evaluated after nominal-scale reconstruction. Optional columns
below must contain `true`/`false` or `1`/`0`, consistent across all nine rows of a
case. They are externally audited flags, not inferred from the measurement CSV.

| Optional column | Subset semantics retained from the source update |
| --- | --- |
| `same_es_functional` | Every reader has matching static/dynamic source-video content and the same positive ES frame number. |
| `complete_geometry` | Every reader meets functional eligibility, has a valid dynamic ES 2D endpoint cache, and has static ES endpoints aligned with the static ED axis (maximum offset `<0.01` pixels and acute angle `<0.01` degrees). |
| `reannotated` | True for any case included in the reannotation change registry; the `exclude_reannotated` subset retains false cases. The original registry contained five cases. |

Unprovided optional columns produce no corresponding subset. LOOCV is refitted
separately within every selected subset. A subset with fewer than four cases is
reported as skipped in `sample_flow.csv`; at least four primary cases are
required. Undefined statistics from zero variance or degenerate bootstrap draws
can be blank/NaN. The original percentile helper discards nonfinite draws; its
sign-tail calculation uses the full draw array. Inspect degenerate or very small
datasets before interpreting their bootstrap outputs.

## Outputs and provenance

Each computed subset has agreement, paired contrasts, raw-error contrasts, ICC,
exploratory-threshold, and full-sample calibration-fit CSVs. `sample_flow.csv`
reports subset sizes and skip status. `run_manifest.json` records the input hash,
seed, repetitions, package versions, scale, supplied audit flags, and conditional
bootstrap definition. No database is opened or modified. Source seed offsets and
method/metric ordering are preserved; changing case labels can change the sorted
row order and hence finite-bootstrap draws. Exact reproduction also depends on
the numeric export, library versions, and selected cohort.

The public numerical unit tests contain only invented measurements. Development
verification compared this implementation directly with the original source on
12 synthetic cases and 100 bootstrap repetitions: harmonization, reader EF,
consensus/LOOCV predictions, agreement, paired contrasts, ICC estimates/intervals,
raw delta MAE, and thresholds matched exactly numerically. This is software
parity verification, not a repeat of the full participant analysis.

Omitted from this portable release are private-database snapshot and mutation
checks, reannotation identity/operator registries, video-content hashing,
2D endpoint and same-frame audit extraction, coordinate-to-frame scale audits,
stored-versus-recalculated diameter change tables, historical case-by-case
reproduction/change reports, and the separate consensus descriptive-summary CSV.
Audit-dependent subsets can be run using the documented external flags. Their
validity remains the responsibility of the input-data preparation process.
