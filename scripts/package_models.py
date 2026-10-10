#!/usr/bin/env python3
"""Pack models/onnx/* into release archives and write the committed manifest.

    python scripts/package_models.py --base-url https://github.com/yugh88/VIT_YughJuneja_Hackathon/releases/download/models-v1

Writes ``dist/models/<name>.tar.gz`` (upload these as GitHub Release assets) and
``data/reference/models_manifest.json`` with each archive's size and SHA-256.
All four models are default downloads: the NLI model is used for non-English sentiment (D78),
not only as the zero-shot event fallback, so it can no longer be fetched lazily.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAZY: set[str] = set()  # nothing is lazy since D78 (NLI sentiment blend)
ORDER = ["sentiment-en", "sentiment-multi", "event-multi", "zeroshot-nli"]
EXTRAS = {"event-multi": ("credit_specialist.joblib",)}  # Credit Event specialist head (D63)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tag", default="models-v1")
    args = ap.parse_args()
    import shutil

    if shutil.disk_usage(ROOT).free / 1e9 - 1.2 < 15:
        raise SystemExit("package-models: would leave < 15 GB free (project disk rule); free space first")
    out = ROOT / "dist" / "models"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ORDER:
        src = ROOT / "models" / "onnx" / name
        if not (src / "model.int8.onnx").exists():
            print(f"skip {name}: not built")
            continue
        archive = out / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            for f in ("model.int8.onnx", "tokenizer.json", "sentinel.json", *EXTRAS.get(name, ())):
                tar.add(src / f, arcname=f)
        files.append(
            {
                "name": name,
                "file": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": sha256_of(archive),
                "lazy": name in LAZY,
            }
        )
        print(f"{archive.name}: {archive.stat().st_size / 1e6:.0f} MB")
    manifest = {
        "tag": args.tag,
        "base_url": args.base_url,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "default_download_bytes": sum(f["bytes"] for f in files if not f["lazy"]),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    path = ROOT / "data" / "reference" / "models_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(
        f"wrote {path}: default download {manifest['default_download_bytes'] / 1e6:.0f} MB, "
        f"all {manifest['total_bytes'] / 1e6:.0f} MB"
    )


if __name__ == "__main__":
    main()
