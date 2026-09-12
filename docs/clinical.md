# Continuous clinical analysis

This release implements the frozen study's continuous NT-proBNP statistical
methods for a user-supplied, locally authorized CSV. It does not contain patient
rows, laboratory records, report text, database connections, video inference,
or clinical pairing logic. It writes aggregate statistics only.

The study's frozen main clinical comparison was **45 examinations in 39
patients**, with F04's matched five-frame Savitzky–Golay processing. Native T01
was a separate sensitivity. These are counts from the existing study, not a
requirement that an arbitrary supplied dataset have the same size. The software
requires at least ten patients for an estimable analysis; smaller sensitivity
subsets receive `insufficient_patients`. This threshold does not establish
adequate statistical power.

## Input contract

Install the package, then supply one row per already-matched examination per
processing variant. LVEF values must refer to the same included examinations
and the same intended video aggregation. Create this CSV under the permissions
applicable to your data; do not commit it to the public repository.

| Column | Meaning |
|---|---|
| `patient_id` | Local pseudonymous grouping key; never used as a predictor or exported |
| `study_id` | Local pseudonymous examination key; unique within patient and variant |
| `age_proxy` | Approximate age in years at the laboratory observation |
| `log2_creatinine` | Already-computed log2 of positive uncensored blood creatinine in mg/dL |
| `ntprobnp` | Positive uncensored NT-proBNP concentration in pg/mL, before log transformation |
| `f04` | F04 LVEF in percentage points, 0–100 |
| `echonet_lvh` | EchoNet-LVH LVEF in percentage points |
| `report_lvef` | Report LVEF in percentage points, following an explicit extraction policy |
| `variant` (optional) | Nonidentifying processing label; each variant is analyzed separately |
| `report_exact` (optional) | Boolean flag identifying an exact numeric report EF |
| `age_topcoded` (optional) | Boolean flag identifying an age proxy based on a topcoded anchor |
| `study_order` (optional) | Numeric chronological order, unique within each patient and variant |
| `lab_censored` (optional) | Boolean censoring flag; any true value is rejected |
| `analysis_eligible` (optional) | Boolean eligibility flag; all supplied rows must be true |

The default model columns are `f04 echonet_lvh report_lvef`. Use
`--model-columns` to select a complete common set including `f04`, for example
`echonet_lvh resnet18 mobile f04 report_lvef`. Additional selected model columns
must also contain finite LVEF percentage values. No model-specific row deletion,
imputation, censoring substitution, transformation offset, or outlier removal
is performed. Missing/ineligible rows require an explicit shared exclusion
before input. `report_exact` and `age_topcoded` accept true/false or 1/0.

The source age proxy was anchor age plus laboratory calendar year minus anchor
year; topcoded anchors are not exact ages. Source report EF used exact numeric
values or midpoints of bounded intervals no wider than 10 percentage points;
threshold reports were excluded. Source creatinine matching used positive
uncensored mg/dL values within 24 hours of the NT-proBNP sample, preferring the
same specimen and then the nearest compatible observation. Preparing and
auditing these inputs remains the data holder's responsibility. This release
does not reimplement record matching, age derivation, report extraction, or
video selection.

## Run

```console
python scripts/analyze_clinical.py --input /path/to/authorized_pairs.csv --output-dir /path/to/local_results --bootstrap-reps 20000 --cv-bootstrap-reps 2000 --seed 20260911
```

For the five-method comparison and the adjusted source sensitivities:

```console
python scripts/analyze_clinical.py --input /path/to/authorized_pairs.csv --output-dir /path/to/local_results --model-columns echonet_lvh resnet18 mobile f04 report_lvef --sensitivities patient_equal exact_report_only exclude_topcoded_age
```

For the later E10 common-video cohort, supply its separately prepared input and
retain the cumulative comparison families even though only four methods remain:

```console
python scripts/analyze_clinical.py --input /path/to/authorized_e10_common_pairs.csv --output-dir /path/to/local_e10_results --model-columns f04 echonet_lvh e10 report_lvef --f04-family-size 5 --base-increment-family-size 6 --bootstrap-reps 20000 --cv-bootstrap-reps 2000 --seed 20260911
```

The E10 analysis used 81 common surviving videos, 44 examinations and 38 patients.
All methods were averaged over those same surviving videos. Simply deleting
examination rows from the primary input does not reconstruct these averages.
The optional complete-video sensitivity used 43 examinations and 37 patients;
its input must be prepared separately because this CLI has no video-success data.

Each supplied variant receives a main `repeated` result. The optional
`single_visit` sensitivity retains the lowest numeric `study_order` for each
patient. The software never treats an identifier or CSV position as a date.
Variants and sensitivities retain separate results; cross-variant paired
differences are not calculated. The CLI requires at least 1,000 replicates for
each bootstrap. The Python API accepts smaller counts for fast synthetic tests;
those tests are not study interval estimates.

The sole output is `results.json`: analysis settings, clinical implementation
SHA256, aggregate sample counts, point estimates, intervals, and estimability
statuses. It contains no input paths, patient/study identifiers, original rows,
individual residuals, or held-out predictions. Supply nonidentifying model and
variant labels because those labels are included in the output.

## Methods retained from the study scripts

The adjusted outcome is `log10(ntprobnp)`. The base weighted least-squares
design is intercept + `age_proxy / 10` + `log2_creatinine`; each extended design
adds one model's LVEF / 10. Adjusted partial Pearson r is the signed square root
of `(SSE_base - SSE_extended) / SSE_base`, with the sign of the EF coefficient.
This is the weighted residual partial correlation. Outputs also include the EF
coefficient per ten percentage points, in-sample R², partial R², and incremental
R². No penalization or model tuning is performed.

The main analysis weights examinations equally. The patient-equal sensitivity
assigns each patient's examinations weights summing to one. Each bootstrap draw
samples patients with replacement and carries all examinations and every model
together. Unadjusted results include weighted Pearson correlations with raw and
log10 NT-proBNP, weighted Spearman correlations, and a descriptive log10 slope.
Spearman midranks are recomputed using each draw's frequency weights, including
ties and repeated patient copies. The rank implementation uses quadratic memory
in examination count and is intended for small research cohorts.

Cross-validation leaves out an entire patient. Every bootstrap replicate
resamples patients and **refits every held-out-patient fold after removing all
copies of that patient**. It does not bootstrap a fixed vector of original
cross-validation errors. Patient-equal analysis applies the same weights to
training and pooled evaluation. Reported RMSE/MAE use log10 concentration units;
Q² compares squared error against predictions from each training fold's mean.
Q² can be negative. `cv_rmse_improvement_vs_base` is base RMSE minus extended
RMSE; positive values mean lower held-out error from adding EF.

Contrasts use differences within the same bootstrap draw. Adjusted outputs
include comparator-minus-F04 partial r and CV RMSE, and F04-minus-comparator
incremental R². Unadjusted outputs include comparator-minus-F04 r and the
difference in absolute correlation. A more negative correlation and a stronger
absolute correlation are distinct comparisons.

Intervals use bootstrap percentiles. `ci95` is marginal 95%; `ci_family` uses
Bonferroni quantiles `[0.025 / K, 1 - 0.025 / K]` for the specified comparison
count K. Defaults remain `M-1` F04 comparisons per metric and `M` base-increment
comparisons for M selected EF methods. Explicit `--f04-family-size` and
`--base-increment-family-size` preserve previously examined comparisons when
fewer methods are displayed. They must be integers no smaller than the number
of corresponding selected contrasts. Both selected and resolved family counts
are recorded in output settings; they do not change point estimates or `ci95`.

| Study analysis | EF methods in its input | F04 family per metric | EF-versus-base family |
|---|---|---:|---:|
| Original 45-examination/39-patient adjusted cohort, including T01 and its adjusted sensitivities | F04, EchoNet-LVH, ResNet18/SimCC, MobileNetV3/SimCC, report LVEF | 4 | 5 |
| Later E10 common-video cohort and its sensitivities | F04, EchoNet-LVH, E10, report LVEF | 5 | 6 |

The E10 cumulative families retain the two historical SimCC methods. Running
its four selected methods with generic defaults would use 3/4 and would not
reproduce the manuscript's Supplementary Table S6 family intervals. Selecting
only the three generic defaults without explicit counts likewise does not
reproduce the primary historical five-method family intervals.

Manuscript Figure 4 and Supplementary Table S5 show **marginal 95% intervals**;
Supplementary Table S6 shows both marginal and five-comparison family intervals
for each of its three metrics separately. These are not one 15-comparison family.
Table S7 shows point estimates. No correction is made across metrics, variants
or sensitivities. The original unadjusted archive additionally contains all-pairs
and thirty-comparison families; those are outside this portable release's scope.
The original E10 archive also attached six-comparison intervals to absolute CV
RMSE/MAE fields. Those auxiliary intervals are not displayed in the manuscript;
this release applies the base family to **RMSE improvement versus base**, while
absolute CV metrics retain marginal intervals. It does not claim to reproduce
every auxiliary interval in that archive.

The crossproduct design must have a minimum/maximum eigenvalue ratio greater
than 1e-12; singular or ill-conditioned fits receive undefined estimates with
no pseudoinverse or ridge fallback. An interval needs at least 95% finite draws;
otherwise its status reports insufficient valid draws. Review statuses at the
metric level even when the analysis container has `status: ok`.

## Scope and verification

This is an exploratory association and internal cross-validation analysis of
a previously selected cohort. It does not establish prospective clinical
prediction, causal effects, within-patient change associations, independent EF
accuracy, interchangeability, or method superiority. NT-proBNP and report EF
are not independent EF gold standards. Correlation and in-sample R² cannot be
substituted for independent measurement agreement. No diagnostic thresholds,
AUC, clinical recommendations, or classifier calibration are included here.

Synthetic tests compare partial r with independent weighted residualization,
compare batched bootstrap CV with explicit weighted refits, change all outcomes
for a held-out repeated patient to check leakage, distinguish refitted CV from
fixed-error resampling, compare weighted tied ranks with expanded observations,
and require zero paired-contrast uncertainty for identical methods. They also
exercise incomplete cohorts and singular fits. Source files and their SHA256
hashes are recorded in `clinical_provenance.json`; attribution is covered by the
repository MIT license.

Before v0.1.0, the portable five-method main analysis was checked locally
against the original frozen adjusted result using 20,000 association and 2,000
CV bootstrap replicates. All 404 numeric fields from that original result
matched exactly in the verification environment. The source input remained
unchanged and no patient rows were written. The restricted source CSV is not
distributed, so this comparison is a recorded local verification, while the
synthetic checks can be run from the public release.

The v0.1.1 provenance audit separately traced the primary 4/5 families and the
E10 5/6 families to their actual frozen scripts and inputs. The earlier
404-field verification concerns the original five-method adjusted result, not
the later E10 analysis. See `study_settings.json` for the recorded processing
and analysis settings and `release_validation.json` for the checks completed
for this release.
