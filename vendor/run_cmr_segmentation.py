"""Run the released CMR nnU-Net model on user-supplied prepared NIfTI images."""
from __future__ import annotations

import argparse
from pathlib import Path
from model_bundle import bundle_path, verify_model_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Prepared nnU-Net NIfTI files named <case>_0000.nii.gz')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not args.input_dir.is_dir() or not any(args.input_dir.glob('*_0000.nii.gz')):
        parser.error('Input directory must contain prepared *_0000.nii.gz files')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty')
    manifest = verify_model_bundle(args.model_dir)
    if manifest.get('model_family') != 'CMR nnUNet fold 0':
        parser.error('Expected the released CMR model bundle')
    verified = {bundle_path(args.model_dir, item['path']) for item in manifest['files']}
    required = ('plans.json', 'dataset.json', 'fold_0/checkpoint_best.pth')
    if not all(bundle_path(args.model_dir, name) in verified for name in required):
        parser.error('Required CMR files must be covered by the manifest')
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(4)
    predictor = nnUNetPredictor(device=torch.device(args.device),
                               perform_everything_on_device=args.device == 'cuda',
                               verbose=False, verbose_preprocessing=False,
                               allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(str(args.model_dir), use_folds=(0,),
                                                   checkpoint_name='checkpoint_best.pth')
    predictor.predict_from_files(str(args.input_dir), str(args.output_dir),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=1,
                                 num_processes_segmentation_export=1)


if __name__ == '__main__':
    main()
