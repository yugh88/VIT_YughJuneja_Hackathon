#!/usr/bin/env python3
"""Fail (exit 1) if any tracked text file contains something that looks like a secret.

The project promises to use keyless sources only; CI runs this on every push.
Patterns cover the common credential formats plus generic ``api_key = "..."``
assignments. Add ``# keyscan: ignore`` to a line to whitelist a false positive.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "Private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Generic secret assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|secret|access[_-]?token|auth[_-]?token|password)\b\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{16,}['\"]"
    ),
}
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".gz",
    ".zip",
    ".parquet",
    ".onnx",
    ".bin",
    ".ico",
    ".woff",
    ".woff2",
}
SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "data/raw", "__pycache__"}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [ROOT / p for p in out.splitlines() if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p for p in ROOT.rglob("*") if p.is_file()]


def scan(paths: list[Path]) -> list[str]:
    hits = []
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in SKIP_SUFFIXES or any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        if p.name == "check_no_keys.py":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if "keyscan: ignore" in line:
                continue
            for label, rx in PATTERNS.items():
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {label}")
    return hits


def main() -> int:
    files = tracked_files()
    hits = scan(files)
    if hits:
        print("Possible secrets found:\n  " + "\n  ".join(hits))
        return 1
    print(f"keyscan: {len(files)} files checked, no API keys found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
