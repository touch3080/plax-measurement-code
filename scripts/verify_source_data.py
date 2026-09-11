#!/usr/bin/env python3
"""Verify the frozen aggregate data against the release manifest."""
import hashlib
import json
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "source_data"
manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    path = source / entry["file"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry["sha256"]:
        raise SystemExit(f"Aggregate data checksum mismatch: {entry['file']}")
print(f"Verified {len(manifest['files'])} frozen aggregate files.")
