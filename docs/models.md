# Endpoint measurements and optional F04 inference

The lightweight package operates on **user-supplied endpoint coordinates**. It
does not turn an image into endpoints by itself. `vendor/` additionally contains
the original F04 model architecture, six-candidate inference, learned fusion and
T01 source, with a portable bundle adapter. Trained F04 and CMR assets are available
as separate [versioned downloads](model_downloads.md); patient data, annotation
databases and DICOM files are not included. The checkpoint-based
video runner was checked on two authorized real clinical videos (275 frames) with
the original, hash-verified model bundle on 13 September 2026. All six candidate
trajectories, fused raw/T01 endpoints and the checked SG5/T01 phase/EF outputs
matched the frozen local clinical outputs exactly. This is a representative
runtime check; see [scope and environment](video_inference_audit.json).

## Measurement inputs

All coordinates are original-video pixels. Both ES lines in a same-phase
comparison must belong to the **same video and ES frame**. The ED-fixed ES
endpoints are supplied by the user from the intersections on the ED-fixed axis;
the code does not fabricate those intersections by projecting a relocated line.

`scripts/analyze_geometry.py geometry` reads these CSV columns:

```text
static_es_frame,dynamic_es_frame,ed_x1,ed_y1,ed_x2,ed_y2,static_es_x1,static_es_y1,static_es_x2,static_es_y2,dynamic_es_x1,dynamic_es_y1,dynamic_es_x2,dynamic_es_y2
20,20,0,0,100,0,20,0,80,0,30,10,70,10
```

```bash
python scripts/analyze_geometry.py geometry --input endpoints.csv --output geometry.json --spacing-cm 0.05
```

The example is synthetic. The geometry function reproduces the perpendicular
distance of the ES midpoint from the ED infinite axis, its ratio to ED length,
endpoint RMS distance, the acute undirected angle and ES diameter differences.
The offset is not the Euclidean displacement of the two midpoints. Scalar
calibration is in **centimetres per pixel**, not millimetres per pixel; it must be
obtained from the source acquisition. Anisotropic pixel spacing requires prior
coordinate transformation and is not silently approximated by this API.

With calibration, Teichholz volume is `7 * D**3 / (2.4 + D)` in mL for `D` in cm.
EF is `100 * (EDV - ESV) / EDV`. Geometry-only processing can omit calibration.
The geometry command verifies identical nonnegative integer ES frame numbers;
the caller remains responsible for video identity and correctness of annotation.

## Trajectories, preprocessing and phase rules

```text
frame_id,x1,y1,x2,y2,confidence
0,10,20,90,20,0.9
1,10,20,89,20,0.8
```

```bash
python scripts/analyze_geometry.py trajectory --input f04_lines.csv --output trajectory.json --spacing-cm 0.05 --fps 30 --smoothing matched_savgol5 --filter --cutoff-frequency 0.23
python scripts/analyze_geometry.py trajectory --input f04_lines.csv --output trajectory_t01.json --spacing-cm 0.05 --fps 30 --smoothing T01 --filter --cutoff-frequency 0.23
```

The CLI requires a complete, ordered trajectory with consecutive zero-based
`frame_id` and finite nondegenerate lines. `confidence` is optional, but reproducing
F04 T01 requires the actual fused confidence. Omission defaults to 1.0 as in the
original smoother. Output marker indices are zero-based; the inherited payload's
`excluded_frames` and segment start/end fields are one-based.

The two preprocessing branches have different roles:

- `matched_savgol5`: original clinical primary comparison convention, five-frame
  second-order Savitzky–Golay coordinate smoothing followed by int16 quantization.
  Sequences shorter than five frames skip SG filtering but still quantize.
- `T01`: original full-video offline smoother, endpoint orientation continuity,
  five-frame coordinate median, confidence threshold 0.45, robust outlier gates
  using median + 6 × 1.4826 × MAD (minimum thresholds: endpoint distance 12 px,
  length 10 px, angle 12 degrees), then bidirectional EMA with alpha 0.35. It uses
  future frames. Do not apply a second SG5 pass to this sensitivity branch.
- `none`: use already prepared endpoints without additional coordinate filtering.

Each branch consumes the raw `x1,y1,x2,y2` columns. When reading F04 exports,
`T01` recomputes from these raw columns; it does not smooth the exported
`*_temporal` columns again. Pass a separate CSV with prepared points as the raw
columns when intentionally selecting `none`.

The frozen clinical runs enabled the fourth-order Butterworth curve filter at
**0.23 cycles/frame**, applied forward and backward with constant boundary
padding; segments of at most 15 frames bypass this filter. Both commands above
select that recorded setting. Replace the illustrative calibration and frame
rate with the actual video's values. The CLI's generic default leaves this
filter off and then uses Gaussian sigma 0.75; that Gaussian branch was not used
for the frozen clinical results. The helper defaults sigma 1.0 and peak-distance
fraction 0.45 were not the effective PLAX study settings.

After coordinate preprocessing, original Teichholz volumes drive automatic
phase detection. Peaks require prominence of 5% of the detection-curve range.
The effective period is `min(FFT period, segment_length / 3)`; peak distance is
0.55 times that effective period, rounded and bounded to valid frame distances.
Boundary extrema are excluded, same-type runs collapse to the
strongest extremum, and each ED pairs to the minimum following ES before the
next ED. EF is evaluated on the **unfiltered metric volumes**, not the smoothed
display/detection curve. Eligible automatic-phase segments require at least
`max(8, ceil(fps * 0.5))` frames. The original segmented API is included and never
pairs EF across invalid frame gaps; the CLI intentionally requires complete inputs.
The returned clip EF is the mean of valid cycle EF values. If any cycle lies
outside 0–100%, the summary reports `nonphysical_ef` rather than clipping it.

## Optional checkpoint-based F04 runner

F04 predicts two LVID endpoints; it is not a direct EF regression model. The six
candidates are ordered E07 (coarse DeepLabV3+), E14 (ROI DeepLabV3+), E10
(YOLO11s pose), E11 (YOLOv8s pose), E16 (Swin-T FPN) and E17 (Swin-S FPN).
The ExtraTrees selector uses 600 trees, maximum depth 10, minimum leaf size 3
and softmax blending at temperature 1.0. The legacy phase feature is 0 (`LVID`);
no ED/ES reference labels are supplied to video inference.

The complete original `.py` algorithm modules are under `vendor/f04_vendor/`.
`vendor/f04_pipeline.py` replaces application-specific imports/path resolution
with a required bundle path and a local result dataclass. Candidate inference,
fusion, chunk size 64, T01-after-concatenation and numerical parameters are kept.
`vendor/run_f04.py` is a new standalone runner without a database dependency.

Download the [F04 v0.1.2 bundle](https://github.com/touch3080/plax-measurement-code/releases/download/v0.1.2/f04-models-v0.1.2.zip),
verify its hash in [the release manifest](model_assets_v0.1.2.json), and extract it.
It contains minimal inference configurations, YOLO architecture definitions,
six tensor-only checkpoints, the ExtraTrees/joblib selector and `FINAL_SCHEME.json`.
The loader checks each file against `MODEL_MANIFEST.json` before deserialization.
Removing training metadata changes file hashes while preserving all network
tensors; original and published identities are recorded in the manifest.
The older `vendor/bundle_template/` documents the legacy author-container layout;
use the downloaded bundle's own scheme for the v0.1.2 assets.

In an isolated environment with dependencies compatible with the supplied
checkpoint, install PyTorch, torchvision, Ultralytics, OpenCV, NumPy, SciPy,
scikit-learn, joblib, PyYAML and tqdm. The successful local check used Python
3.10.20, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, Ultralytics 8.4.19 and
scikit-learn 1.7.2 on Windows with an RTX 5060 Laptop GPU. The [tested direct
dependency constraints](inference-constraints.txt) record the other versions;
they are not a complete transitive lockfile or a test of arbitrary platforms.
Only load trusted PyTorch/joblib files: these formats can execute code when loaded.

```bash
python vendor/run_f04.py --bundle /path/to/bundle/FINAL_SCHEME.json --video /path/to/video.avi --output-dir /path/to/new-results --device cuda:0 --batch-size 4
```

Full heatmap checkpoint loading disables ImageNet pretrained initialization.
The output contains original and T01 endpoint columns; pass the original columns
to the measurement CLI with the appropriate branch. Calibration, clip selection,
reference annotation, training, and cohort assembly remain separate steps.
The CUDA command above matches the tested configuration. CPU execution was not
tested in this audit. The two selected clips exercised multiple 64-frame chunks;
their F04-SG5, F04-T01 and separately extracted fresh E10-SG5 trajectories produced
six video/branch comparisons and 12 cycles with identical ED/ES markers and zero
volume or EF differences. No full-cohort rerun or training reproduction is implied.

The reference here is the frozen output generated on this local runtime. An
earlier comparison to the original source workstation on a separate EchoNet-LVH
video retained nonzero differences (up to 1.0685 native pixels for raw F04 and
0.3828 pixels for T01). Exact agreement in this local audit does not establish
exact cross-platform numerical equivalence.

## Attribution and provenance

The source repository carries MIT, Copyright (c) 2024 Haibo Meng; a copy is in
`vendor/LICENSE.landmark3.0` and must accompany redistribution. No separate
license or copyright notice was present in the inspected `f04_vendor` files.
Third-party packages retain their own licenses; importing Ultralytics does not
make its package or model assets covered by this repository's MIT license.
The released E10/E11 state dictionaries retain the original checkpoints'
AGPL-3.0 designation and include its license text. See [model notices](model_downloads.md).

`measurement_provenance.json` records original source hashes, extracted symbol
hashes, copied-file identity, release-file hashes and required asset hashes.
Only relative source labels are published; the manifest contains no private
filesystem locations. The extraction preserves the original functions, while
new wrappers add validation, explicit units and portable CLI inputs. The release
contains no claim that synthetic checks establish clinical performance or that
the optional model runner reproduces full training.
