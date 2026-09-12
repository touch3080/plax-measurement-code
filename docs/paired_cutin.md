# Paired cut-in observations

Version 0.1.4 adds a post hoc exploratory, video-level paired analysis of CAMEO
cut-in observations. It adds no new model, inference run or independent clinical
validation. The static path is the historically recorded fixed line; its named
model is not established. The dynamic path is F04.

Static observations come from existing electronic records. Dynamic negatives
come from the authors' confirmation on 13 September 2026 of manual review of all
100 videos under the same left-atrial/aortic-region definition, including the
frames with static cut-in marks. They are new retrospective review evidence,
not explicit zero labels recovered from the historical database. The number of
dynamic reviewers and whether they were the static reader were not separately
recorded. Blinding and inter-reader reliability are not established.

The endpoint is any left-atrial or aortic-region cut-in in the reviewed video
and sampling path. It is not restricted to the same ES frame: static event marks
may occur outside that frame. Both anatomical regions in one video count as one
positive paired observation; frames are not independent sample units.

## Published results and arithmetic

The existing 96-video ES-valid set is the primary exploratory description:
static 32/96 and dynamic 0/96. The nested full 100-video set is a descriptive
sensitivity analysis: static 33/100 and dynamic 0/100. Selection of the 96-video
set predates this dynamic-label confirmation; the two sets are not independent
replications. The historical static category summary is retained unchanged.

- Each marginal proportion has a two-sided Wilson 95% interval without continuity
  correction. Zero observed events have a nonzero upper confidence limit.
- The paired table has rows static negative/positive and columns dynamic
  negative/positive: `[[n00, n01], [n10, n11]]`.
- The point difference is static minus dynamic, `(n10 - n01) / N * 100`
  percentage points. No paired-difference interval is estimated. Marginal Wilson
  intervals must not be interpreted as an interval for this paired difference.
- Exact two-sided McNemar inference uses
  `binomtest(min(n01, n10), n01 + n10, p=0.5, alternative="two-sided")`.
  With no discordant pairs, the reported P value is 1. P values are unadjusted
  and explicitly exploratory.

The public aggregates permit arithmetic reproduction, not independent
verification of manual observations, anatomical accuracy, EF accuracy or the
complete imaging pipeline. See [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html),
[Wilson intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
and [the exact McNemar definition](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

## Run from aggregate counts

```bash
python scripts/analyze_paired_cutin.py --input source_data/paired_cutin_2x2.csv --output-dir outputs/paired_cutin
```

The output is `paired_cutin_recomputed.json`. It reproduces the statistical
fields of `source_data/paired_cutin_summary.json`; anatomical subcategory counts
and review provenance cannot be inferred from a binary 2x2 table alone. Study
context in the output describes the published observations, not a verification
of an arbitrary user-supplied input.

The input CSV requires `analysis_set,static_status,dynamic_status,n`. Each named
set must have exactly four explicit cells, including zero cells. Accepted status
labels are `static_negative`, `static_positive`, `dynamic_negative` and
`dynamic_positive`. Counts must be nonnegative integers and each table must have
positive total N. Missing cells are not silently filled with zeros. No video
identifiers or participant-level rows are required.

Source and release hashes, arithmetic parity, and synthetic checks are recorded
in [paired cut-in provenance](paired_cutin_provenance.json). Trained model assets
remain at [v0.1.2](https://github.com/touch3080/plax-measurement-code/releases/tag/v0.1.2).
