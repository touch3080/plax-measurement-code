# PLAX measurement code

Code accompanying **Evaluating cardiac artificial intelligence beyond human readouts**, prepared as a Nature Article submission draft. Version **0.1.3**. This repository does not imply journal acceptance.

Correspondence: **Dong Ni (倪东), nidong@szu.edu.cn**.

Repository: [touch3080/plax-measurement-code](https://github.com/touch3080/plax-measurement-code). The manuscript code version is **v0.1.3**; use the versioned release when citing or reproducing this draft. The v0.1.0, v0.1.1 and v0.1.2 tags remain available.

The study distinguishes agreement with human measurements from association with a separately generated functional or physiological comparator. This release contains portable statistical analyses, measurement processing, model source code and the aggregate data needed to redraw the four main figures.

Version 0.1.1 documents the actual clinical phase filter (Butterworth, 0.23 cycles/frame) and adds explicit cumulative comparison-family settings for the E10 analysis. The original primary analysis uses 4/5 comparisons; the E10 common cohort retains 5/6. Measurement algorithms and frozen figure inputs are unchanged. See [recorded study settings](docs/study_settings.json) and [clinical interval scope](docs/clinical.md).

Version 0.1.2 adds publicly downloadable F04 and CMR model bundles, with training metadata removed, file integrity manifests and compatible inference loaders. See [model downloads](docs/model_downloads.md). The original network tensors, measurement algorithms and frozen study results are unchanged.

Version 0.1.3 adds portable conditional CMR A/B axis-scale point diagnostics and a synthetic input example. The source-candidate selection and physical interpretation remain explicit upstream assumptions; see [conditional-scale scope and schema](docs/conditional_scale.md). Model assets remain at v0.1.2.

## Install and verify

Use Python 3.10 or newer in a virtual environment. From the repository root:

For the authors' adopted reproduction runtimes, use the exact versions and
separate inference/analysis environments in [current runtime instructions](docs/current_runtime.md).
The generic installation below retains the package's broader compatibility range.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/build_figures.py
```

The figure command needs no private inputs and writes PDF, SVG and PNG files to `figures/`. Figure 1 is a conceptual illustration; Figures 2–4 use frozen aggregate estimates in `source_data/`. Figure generation does not rerun participant-level inference or statistical analyses.

## Run the analyses

Prepare locally authorized inputs using the schemas in the linked documentation. Keep those inputs and their outputs outside the repository, or in the ignored `local_data/` and `outputs/` directories.

| Component | Entry point | Documentation |
|---|---|---|
| Three-reader CMR agreement, calibration, paired contrasts and reliability | `scripts/analyze_cmr.py` | [CMR input schema and estimands](docs/cmr.md) |
| NT-proBNP associations and leave-one-patient-out validation | `scripts/analyze_clinical.py` | [Clinical input schema and estimands](docs/clinical.md) |
| Endpoint measurement, phase selection and geometry | `scripts/analyze_geometry.py` | [Measurement and model scope](docs/models.md) |
| Conditional CMR A/B axis-scale point diagnostics | `scripts/analyze_conditional_scale.py` | [Input schema and assumption scope](docs/conditional_scale.md) |
| Aggregate manuscript figures | `scripts/build_figures.py` | [Data and reproducibility](docs/reproducibility.md) |

```bash
python scripts/analyze_cmr.py --input local_data/cmr_readers.csv --output-dir outputs/cmr
python scripts/analyze_clinical.py --help
python scripts/analyze_geometry.py --help
```

CMR defaults retain 10,000 bootstrap repetitions. The primary clinical analysis retains 20,000 patient-cluster repetitions for associations and 2,000 repetitions with the full cross-validation fitting procedure rerun. Reduced repetitions are suitable for smoke checks only. See each component's documentation for seed handling, sensitivity definitions and the interpretation of uncertainty intervals.

## Release scope

The Git source tree contains aggregate estimates, code and synthetic tests. Trained F04 and CMR checkpoints are separate [v0.1.2 Release assets](https://github.com/touch3080/plax-measurement-code/releases/tag/v0.1.2). No participant-level records, raw imaging, clinical database or credentials are distributed. Statistical analysis and inference require locally authorized data. The F04 runner was checked on two real clinical videos (275 frames); the released checkpoint tensors are also checked against the originals. See [model downloads and validation](docs/model_downloads.md).

Source provenance files in `docs/` identify the original scripts and their SHA-256 hashes. Portable wrappers replace local database and filesystem assumptions with explicit file arguments. The original historical software environment and this release's test environment are distinct; see [reproducibility notes](docs/reproducibility.md). Synthetic tests establish the checked software properties and source parity, not a fresh independent reproduction of all study results.

Underlying EchoNet-LVH, CMRxRecon, CAMEO and MIMIC datasets remain subject to their providers' access terms. Their source records are not redistributed with the models. Official third-party comparator/pretraining assets must be obtained from their providers. For source-data enquiries, contact the corresponding author above.

## License

See [LICENSE](LICENSE). The original MIT copyright notice is retained for code derived from the landmark3.0 project. Model weights and third-party datasets are not included under that license. See [third-party notices](THIRD_PARTY_NOTICES.md).
