"""Download the fine-tuned ONNX models from this repo's GitHub Release (keyless).

    python -m engine.models_download             # default set (everything not marked lazy)
    python -m engine.models_download --all       # also the lazy zero-shot fallback
    python -m engine.models_download --only zeroshot-nli

``data/reference/models_manifest.json`` (committed) lists every archive with its size
and SHA-256. Downloads resume from a ``.part`` file via HTTP Range, show a progress
bar, are verified against the checksum and only then unpacked into ``models/onnx/``.
Set ``SENTINEL_MODELS_BASE_URL`` to use a mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from engine.config import REPO_ROOT, get_settings

log = logging.getLogger(__name__)
MANIFEST = REPO_ROOT / "data" / "reference" / "models_manifest.json"
MODELS_DIR = REPO_ROOT / "models"


class ChecksumError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bar(done: int, total: int, started: float, name: str) -> None:
    frac = done / total if total else 0.0
    rate = done / max(time.time() - started, 1e-6) / 1e6
    sys.stderr.write(
        f"\r  {name:<28} [{'#' * int(30 * frac):<30}] {frac:6.1%} {done / 1e6:7.1f}/{total / 1e6:.1f} MB "
        f"{rate:5.1f} MB/s"
    )
    sys.stderr.flush()


def download(
    url: str, dest: Path, sha256: str, size: int, client: httpx.Client | None = None, progress: bool = True
) -> Path:
    """Resumable download with checksum verification. Returns the verified file."""
    if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == sha256:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    own = client is None
    client = client or httpx.Client(
        follow_redirects=True, timeout=60, headers={"User-Agent": get_settings().user_agent}
    )
    try:
        started = last_draw = time.time()
        if have < size:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 200 and have:  # server ignored Range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                with open(part, "ab") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
                        have += len(chunk)
                        if progress and (time.time() - last_draw > 0.25 or have >= size):
                            _bar(have, size, started, dest.name)
                            last_draw = time.time()
            if progress:
                sys.stderr.write("\n")
    finally:
        if own:
            client.close()
    digest = sha256_of(part)
    if digest != sha256:
        part.unlink(missing_ok=True)
        raise ChecksumError(
            f"{dest.name}: sha256 {digest[:12]}... != expected {sha256[:12]}... (file removed)"
        )
    part.replace(dest)
    return dest


def install(
    entry: dict[str, Any], base_url: str, models_dir: Path = MODELS_DIR, progress: bool = True
) -> Path:
    target = models_dir / "onnx" / entry["name"]
    if (
        (target / "model.int8.onnx").exists()
        and (target / ".sha256").exists()
        and (target / ".sha256").read_text().strip() == entry["sha256"]
    ):
        return target
    archive = models_dir / "downloads" / entry["file"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    download(
        f"{base_url.rstrip('/')}/{entry['file']}", archive, entry["sha256"], entry["bytes"], progress=progress
    )
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target, filter="data")
    (target / ".sha256").write_text(entry["sha256"])
    archive.unlink()
    return target


def base_url(manifest: dict[str, Any]) -> str:
    return os.environ.get("SENTINEL_MODELS_BASE_URL") or str(manifest["base_url"])


def ensure(names: list[str] | None = None, include_lazy: bool = False, progress: bool = True) -> list[Path]:
    m = load_manifest()
    out = []
    for e in m["files"]:
        if names is not None and e["name"] not in names:
            continue
        if names is None and e.get("lazy") and not include_lazy:
            continue
        out.append(install(e, base_url(m), progress=progress))
    return out


_lazy_threads: dict[str, threading.Thread] = {}


def fetch_lazy_in_background(name: str) -> None:
    """Start downloading a lazy model once, without blocking the caller."""
    if name in _lazy_threads and _lazy_threads[name].is_alive():
        return

    def run() -> None:
        try:
            ensure([name], progress=False)
            log.info("lazy model %s downloaded", name)
        except Exception as exc:  # offline demo, release not published yet, ...
            log.warning("lazy model %s unavailable: %s", name, exc)

    t = threading.Thread(target=run, name=f"download-{name}", daemon=True)
    _lazy_threads[name] = t
    t.start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include lazy models (zero-shot fallback)")
    ap.add_argument("--only", nargs="*", help="model names to fetch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    m = load_manifest()
    chosen = [
        e
        for e in m["files"]
        if (args.only and e["name"] in args.only) or (not args.only and (args.all or not e.get("lazy")))
    ]
    total = sum(e["bytes"] for e in chosen)
    print(f"Sentinel models: {len(chosen)} archives, {total / 1e6:.0f} MB from {base_url(m)}")
    ensure([e["name"] for e in chosen], progress=True)
    print("models ready in", MODELS_DIR / "onnx")


if __name__ == "__main__":
    main()
