# Trained model downloads — v0.1.2

The authors authorized public release of the F04 and CMR trained assets. Download
the ZIP files from [GitHub Release v0.1.2](https://github.com/touch3080/plax-measurement-code/releases/tag/v0.1.2).
Archive SHA-256 values, individual files and original-to-published identities are
recorded in [model_assets_v0.1.2.json](model_assets_v0.1.2.json).

| Bundle | Contents | Download |
|---|---|---|
| F04 | E07/E14 heatmaps, E10/E11 pose models, E16/E17 Swin heatmaps, ExtraTrees selector, minimal configurations and YOLO architectures | [f04-models-v0.1.2.zip](https://github.com/touch3080/plax-measurement-code/releases/download/v0.1.2/f04-models-v0.1.2.zip) |
| CMR | Fold-0 nnU-Net best network, 2D plan, numerical label map and inference metadata | [cmr-model-v0.1.2.zip](https://github.com/touch3080/plax-measurement-code/releases/download/v0.1.2/cmr-model-v0.1.2.zip) |

Extract outside the source repository. The inference wrappers verify the files
against the bundle's `MODEL_MANIFEST.json` before deserializing model data.
The archive digest is checked separately against the versioned public manifest.

## F04

Install the [adopted inference dependencies](inference-constraints.txt), including
the matching CUDA build of PyTorch, using the [current runtime instructions](current_runtime.md). Then run:

```bash
python vendor/run_f04.py --bundle /models/f04-v0.1.2/FINAL_SCHEME.json --video /data/authorized-video.avi --output-dir /results/new-f04 --device cuda:0 --batch-size 4
python scripts/analyze_geometry.py trajectory --input /results/new-f04/f04_lines.csv --output /results/new-f04/ef.json --spacing-cm ACTUAL_CM_PER_PIXEL --fps ACTUAL_FPS --smoothing matched_savgol5 --filter --cutoff-frequency 0.23
```

Replace calibration and frame rate with the actual acquisition values. Use
`--smoothing T01` for the separate temporal sensitivity branch; do not apply a
second SG5 filter. The YOLO state dictionaries are reconstructed from packaged
architectures without fetching generic pretrained weights. The selector remains
a trusted, hash-verified joblib object.

## CMR

The adopted environment uses `nnunetv2==2.6.4`,
`dynamic-network-architectures==0.4.3` and `SimpleITK==2.5.4` with Python 3.10.20
and PyTorch 2.10.0+cu128. The exact dependency snapshots and effective inference
settings are in [current runtime instructions](current_runtime.md).
Prepared cine-SAX NIfTI files must use nnU-Net channel names
such as `case_0000.nii.gz`.

```bash
python vendor/run_cmr_segmentation.py --model-dir /models/cmr-v0.1.2 --input-dir /data/prepared-nifti --output-dir /results/new-cmr-masks --device cuda
```

Label 0 is background and label 1 is LV. Labels 2/3 retain numerical output but
are named generically because their archived anatomical descriptions conflict.
The preprocessing plan's unit spacing is the historical prepared-data convention;
it does not certify physical millimetres per pixel. This package supplies the
model and a prepared-NIfTI runner, not a verified raw-MAT preprocessing chain.
Training/evaluation overlap and nonindependent segmentation reference remain
limitations. The recovered checkpoint's identity is known; no historical
inference-time file hash is claimed.

## Sanitization and validation

The published files remove optimizer/scheduler state, logs, training metrics,
source-machine paths and case descriptions. Every network tensor is checked
against the trusted original; only container metadata and loading format change.
Detailed aggregate validation is in [model_release_validation.json](model_release_validation.json).
Frozen manuscript results are not replaced by this release.

## Third-party sources and notices

The original E10/E11 checkpoint metadata specifies **AGPL-3.0**; the F04 archive
retains that designation and includes its license text. The upstream
[Ultralytics license](https://www.ultralytics.com/license) remains applicable.
Original author architecture/inference source retains its MIT copyright notice;
it does not override model or dependency terms. [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
is obtained from its original provider under Apache-2.0.

The official [EchoNet-LVH code and model resources](https://github.com/echonet/lvh)
and [EchoNet-LVH dataset website](https://echonet.github.io/lvh/) remain the sources
for that comparator. Generic [torchvision pretrained model resources](https://docs.pytorch.org/vision/stable/models.html)
and [Ultralytics model resources](https://docs.ultralytics.com/models/)
must be obtained under their providers' terms. Those independent pretrained
assets and third-party datasets are not republished in these bundles.
