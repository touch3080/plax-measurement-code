"""Check released model files before checkpoint deserialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def bundle_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Model path escapes bundle directory')
    return path


def verify_model_bundle(root, manifest_name='MODEL_MANIFEST.json'):
    """Verify every declared file; the download archive hash is published separately."""
    root = Path(root).resolve()
    manifest = json.loads(bundle_path(root, manifest_name).read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or not manifest.get('files'):
        raise ValueError('Unsupported or empty model manifest')
    seen = set()
    for entry in manifest['files']:
        path = bundle_path(root, entry['path'])
        if path in seen:
            raise ValueError('Duplicate model manifest path')
        seen.add(path)
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Missing model file or wrong size: {entry["path"]}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError(f'Model file SHA-256 mismatch: {entry["path"]}')
    return manifest
