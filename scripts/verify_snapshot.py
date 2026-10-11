"""Validate archived source bytes and derived-label member contents."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT / "provenance/source_manifest.json").read_text(encoding="utf-8"))
    for entry in entries:
        data = (ROOT / entry["packaged_path"]).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != entry["packaged_sha256"]:
            raise ValueError(f"Source checksum mismatch: {entry['packaged_path']}")
        if entry["transformation"].startswith("none") and actual != entry["source_sha256"]:
            raise ValueError(f"Original bytes changed: {entry['packaged_path']}")
    labels = json.loads((ROOT / "provenance/derived_label_manifest.json").read_text(encoding="utf-8"))
    for entry in labels:
        path = ROOT / entry["packaged_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["packaged_sha256"]:
            raise ValueError(f"Label archive changed: {path.name}")
        with ZipFile(path) as archive:
            if archive.namelist() != ["y_pred_last.npy"]:
                raise ValueError(f"Unexpected bundled NPZ members: {path.name}")
            if hashlib.sha256(archive.read("y_pred_last.npy")).hexdigest() != entry["source_member_sha256"]:
                raise ValueError(f"Original labels changed: {path.name}")
    print(f"Verified {len(entries)} source/table checksums and {len(labels)} unchanged label sequences.")


if __name__ == "__main__":
    main()
