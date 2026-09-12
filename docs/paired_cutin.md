# Static cut-in and transient model-output failures

## Correction in v0.1.5

The authors clarified that six F04 videos contain brief model-output failures
during poor image quality or motion. Each affected video reportedly contains
only one or two isolated frames: comparison with neighboring frames shows an
abrupt deviation from the expected sampling-line position. These distortions
towards the left atrium are classified by the authors as model-quality failures,
separately from cut-in attributable to the sampling mechanism. The proposed
causes include sudden image-quality degradation, respiration and probe movement;
they were not independently adjudicated.

The review covers 100 videos, including the same 96 ES-valid videos. All six
affected videos are in that subset: quality-failure frequencies are 6/100 (6.0%)
and 6/96 (6.25%). These are not dynamic cut-in rates. Exact frame indices and
case-specific counts were not supplied beyond the one-to-two-frame range; no
frame-level error rate or EF effect is estimated. No case identifiers are
included in the public data.

Historical static cut-in counts remain 33/100 and 32/96. Because the retrospective
mechanistic distinction does not establish comparable binary dynamic labels or
a validated frame-eligibility rule, the manuscript no longer reports a paired
McNemar test or static-minus-dynamic cut-in difference. The unqualified zero-event
paired interpretation in v0.1.4 is superseded. That tag is retained for history,
not as the current scientific conclusion. The six videos remain in the review
cohort; they were not silently excluded or declared error-free.

## Generic arithmetic utility

`scripts/analyze_paired_cutin.py` remains available for separately justified
user-supplied aggregate counts. It computes Wilson marginal intervals, an exact
two-sided McNemar test and a rate-difference point estimate. It does not establish
the validity or comparability of input labels, and it is not currently used to
make an empirical CAMEO paired cut-in claim. Each input set requires four explicit
nonnegative cells with columns `analysis_set,static_status,dynamic_status,n`.
Allowed status labels are static_negative/static_positive and
dynamic_negative/dynamic_positive. Missing cells are never imputed.

See `source_data/dynamic_quality_failure_summary.json` for the current descriptive
record and `docs/paired_cutin_provenance.json` for the correction scope. The
existing synthetic tests are arithmetic checks, not study observations. Model
assets and inference algorithms remain unchanged at v0.1.2.
