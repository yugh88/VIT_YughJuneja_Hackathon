"""Small, dependency-light text helpers used by connectors."""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_html(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"(?i)<br\s*/?>|</p>", " ", s)
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s))).strip()


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def parse_dt(value: str | None) -> datetime | None:
    """Parse RFC 2822, ISO 8601 or GDELT 'YYYYMMDDHHMMSS' timestamps into aware UTC."""
    if not value:
        return None
    v = value.strip()
    try:
        if len(v) == 14 and v.isdigit():
            return datetime.strptime(v, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if len(v) == 16 and v[8] == "T" and v.endswith("Z"):  # GDELT DOC: 20261004T081500Z
            return datetime.strptime(v, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if v[:4].isdigit() and "-" in v[:8]:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        else:
            dt = parsedate_to_datetime(v)
    except (ValueError, TypeError):
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
