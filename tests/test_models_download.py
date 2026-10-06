"""Model download: resumes a partial file over HTTP Range and refuses a bad checksum."""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from engine.models_download import ChecksumError, download

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB
SEEN_RANGES: list[str | None] = []


class RangeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        SEEN_RANGES.append(rng)
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/m.tar.gz"
    srv.shutdown()


def test_download_resumes_and_verifies_checksum(server: str, tmp_path: Path) -> None:
    good = hashlib.sha256(PAYLOAD).hexdigest()
    dest = tmp_path / "m.tar.gz"
    dest.with_suffix(".gz.part").write_bytes(PAYLOAD[:300_000])  # an interrupted earlier download
    SEEN_RANGES.clear()
    out = download(server, dest, good, len(PAYLOAD), progress=False)
    assert out.read_bytes() == PAYLOAD
    assert SEEN_RANGES == ["bytes=300000-"]  # only the missing tail was fetched
    assert not dest.with_suffix(".gz.part").exists()

    bad = tmp_path / "bad.tar.gz"
    with pytest.raises(ChecksumError):
        download(server, bad, "0" * 64, len(PAYLOAD), progress=False)
    assert not bad.exists() and not bad.with_suffix(".gz.part").exists()
