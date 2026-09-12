# Current reproduction environments

The authors adopted the existing, tested environments below on 13 September
2026 for running the public code. These are current reproduction environments;
they are not a reconstruction of the software originally used for CMR training
or historical cohort inference. No model tensors, measurement algorithms or
frozen research results were changed by this environment adoption. Source code
remains v0.1.3 and the model bundles remain v0.1.2.

## Recorded versions

| Component | Model inference and PLAX processing | Portable statistical analyses |
|---|---|---|
| Python | 3.10.20 | 3.13.12 |
| NumPy | 2.2.6 | 2.4.6 |
| pandas | 2.3.3 | 3.0.3 |
| SciPy | 1.15.3 | 1.17.1 |
| Matplotlib | 3.10.8 | 3.10.9 |
| PyTorch | 2.10.0+cu128 | Not required |
| torchvision | 0.25.0+cu128 | Not required |
| nnunetv2 | 2.6.4 | Not required |
| dynamic-network-architectures | 0.4.3 | Not required |
| SimpleITK | 2.5.4 | Not required |
| Ultralytics | 8.4.19 | Not required |
| OpenCV | 4.13.0.92 | Not required |

Both environments were inspected on Windows build 26200. The inference
environment uses an NVIDIA GeForce RTX 5060 Laptop GPU, NVIDIA driver 592.82,
PyTorch CUDA runtime 12.8 and cuDNN version integer 91002. The CUDA runtime
reported by PyTorch does not identify a separately installed CUDA toolkit.
Detailed package versions, snapshot hashes and checks are recorded in
[current_runtime.json](current_runtime.json).

## Install the recorded dependencies

Use separate environments with the Python versions above. From the repository
root, in the Python 3.10.20 inference environment:

```bash
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r docs/environments/inference-requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

In the Python 3.13.12 analysis environment:

```bash
python -m pip install -r docs/environments/analysis-requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

Each requirements file uses the corresponding complete installed-distribution
snapshot as constraints. The snapshots record exact versions of additional
installed packages without requiring every additional package to be installed.
The repository's editable installation is omitted from the snapshot and is
installed explicitly above. Source-machine paths and local build URLs are
removed; package names and versions are retained. These are version pins, not
hashed wheel archives or a claim that every platform has been tested. This
adoption checks the existing environments; a fresh installation on another
machine has not been performed.

## CMR inference settings in the adopted runtime

The unchanged `vendor/run_cmr_segmentation.py` entry uses fold 0 and
`checkpoint_best.pth`, four PyTorch CPU threads, one preprocessing process and
one segmentation-export process. It requires an empty output directory, does
not overwrite results and does not save probability maps. The validated command
uses `--device cuda`, which enables `perform_everything_on_device`; the CLI's
default remains CPU and CPU inference was not validated in this adoption.

The installed nnunetv2 2.6.4 constructor supplies `tile_step_size=0.5`,
`use_gaussian=True` and `use_mirroring=True`. The released model allows mirroring
on axes `(0, 1)`. These effective settings are now recorded with the exact
dependency version, rather than inferred for an unavailable historical runtime.
No postprocessing or ensemble across folds is added by the wrapper. The prepared
NIfTI input interface and the raw-MAT reconstruction limitations remain as
described in [model downloads](model_downloads.md).

## Verification scope

Both current environments pass `python -m pip check`. The unchanged CMR CLI was
also checked on the previously used synthetic prepared NIfTI with the released
model, using forced weights-only checkpoint loading. The shape and spacing are
preserved and the resulting numerical mask matches the preceding synthetic
check. This is an interface check, not an assessment of segmentation accuracy or
physical calibration.

The preceding [model release checks](model_release_validation.json) cover
network-tensor identity and F04 inference on two authorized videos, and
`release_validation.json` records the 61-test v0.1.3 analysis check. Those earlier
checks are retained, not represented as new full-cohort inference or training.
The current environment decision does not settle the historical CMR label
conversion, raw-MAT processing, dataset overlap or reference-independence issues.
