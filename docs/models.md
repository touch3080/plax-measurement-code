# Endpoint measurements and optional F04 inference

The lightweight package operates on **user-supplied endpoint coordinates**. It
does not turn an image into endpoints by itself. `vendor/` additionally contains
the original F04 model architecture, six-candidate inference, learned fusion and
T01 source, with a portable bundle adapter. No model weights, fusion checkpoint,
patient data, annotation database or DICOM files are included. The optional
checkpoint-based video runner has not been validated in this release against a
real video or downloaded checkpoint. Synthetic tests validate endpoint processing.

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
python scripts/analyze_geometry.py trajectory --input f04_lines.csv --output trajectory.json --spacing-cm 0.05 --fps 30 --smoothing matched_savgol5
python scripts/analyze_geometry.py trajectory --input f04_lines.csv --output trajectory_t01.json --spacing-cm 0.05 --fps 30 --smoothing T01
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

After coordinate preprocessing, original Teichholz volumes drive automatic
phase detection. The default detection curve uses Gaussian sigma 0.75; optional
`--filter` selects the original fourth-order Butterworth rule, with cutoff in
cycles/frame and no filtering for segments at most 15 frames. Peaks require 5%
prominence and distance 0.55 times the estimated FFT period, capped at one third
of segment length. Boundary extrema are excluded, same-type runs collapse to the
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

The bundle template includes six original YAML configurations and
`FINAL_SCHEME.json`. Obtain the exact six checkpoints and ExtraTrees/joblib
selector from the authors (corresponding author Dong Ni, nidong@szu.edu.cn).
There is no publicly verified model-download URL in this release; requesting
assets does not guarantee that they can be distributed under the applicable
terms. Put supplied assets at the paths relative to `FINAL_SCHEME.json` and
verify each SHA-256 against `measurement_provenance.json` before use. No automatic
download, model substitution or checkpoint regeneration is provided.

In an isolated environment with dependencies compatible with the supplied
checkpoint, install PyTorch, torchvision, Ultralytics, OpenCV, NumPy, SciPy,
scikit-learn, joblib, PyYAML and tqdm. Training-compatible versions must be
confirmed by the model provider; a validated inference lockfile is not supplied.
Only load trusted PyTorch/joblib files: these formats can execute code when loaded.

```bash
python vendor/run_f04.py --bundle /path/to/bundle/FINAL_SCHEME.json --video /path/to/video.avi --output-dir /path/to/new-results --device cpu
```

Full heatmap checkpoint loading disables ImageNet pretrained initialization.
The output contains original and T01 endpoint columns; pass the original columns
to the measurement CLI with the appropriate branch. Calibration, clip selection,
reference annotation, training, and cohort assembly remain separate steps.

## Attribution and provenance

The source repository carries MIT, Copyright (c) 2024 Haibo Meng; a copy is in
`vendor/LICENSE.landmark3.0` and must accompany redistribution. No separate
license or copyright notice was present in the inspected `f04_vendor` files.
Third-party packages retain their own licenses; importing Ultralytics does not
make its package or model assets covered by this repository's MIT license.

`measurement_provenance.json` records original source hashes, extracted symbol
hashes, copied-file identity, release-file hashes and required asset hashes.
Only relative source labels are published; the manifest contains no private
filesystem locations. The extraction preserves the original functions, while
new wrappers add validation, explicit units and portable CLI inputs. The release
contains no claim that synthetic checks establish clinical performance or that
the optional model runner reproduces full training.
