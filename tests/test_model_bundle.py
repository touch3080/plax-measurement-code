"""Manifest checks protect the boundary before any model deserialization."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release_model_bundle', Path(__file__).parents[1] / 'vendor/model_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def make_manifest(root, entries):
    (root / 'MODEL_MANIFEST.json').write_text(json.dumps({'format_version': 1, 'files': entries}))


def test_verified_bundle_then_tamper(tmp_path):
    path = tmp_path / 'weights.bin'
    path.write_bytes(b'original')
    make_manifest(tmp_path, [{'path': 'weights.bin', 'bytes': 8,
                             'sha256': hashlib.sha256(b'original').hexdigest()}])
    assert bundle.verify_model_bundle(tmp_path)['format_version'] == 1
    path.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='SHA-256'):
        bundle.verify_model_bundle(tmp_path)


def test_manifest_cannot_escape_root(tmp_path):
    make_manifest(tmp_path, [{'path': '../outside.pt', 'bytes': 1, 'sha256': 'unused'}])
    with pytest.raises(ValueError, match='escapes'):
        bundle.verify_model_bundle(tmp_path)


def test_duplicate_and_missing_files_rejected(tmp_path):
    entry = {'path': 'weights.bin', 'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
    make_manifest(tmp_path, [entry])
    with pytest.raises(ValueError, match='Missing'):
        bundle.verify_model_bundle(tmp_path)
    (tmp_path / 'weights.bin').write_bytes(b'x')
    make_manifest(tmp_path, [entry, entry])
    with pytest.raises(ValueError, match='Duplicate'):
        bundle.verify_model_bundle(tmp_path)
