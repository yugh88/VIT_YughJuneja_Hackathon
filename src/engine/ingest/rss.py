"""RSS/Atom connectors: financial news, central banks, Reddit and SEC EDGAR 8-Ks.

All feeds are fetched without keys. Feeds in one connector are polled concurrently
(bounded) and a single bad feed is logged and skipped rather than failing the
whole poll; the poll only counts as failed if *every* feed fails.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx

from engine.ingest.base import Source
from engine.ingest.text import domain_of, parse_dt, strip_html
from engine.schema import RawDoc, SourceType

log = logging.getLogger(__name__)


def _entry_time(e: Any) -> datetime:
    for key in ("published", "updated", "created"):
        dt = parse_dt(e.get(key))
        if dt:
            return dt
    return datetime.now(UTC)


class RssSource(Source):
    """A list of feeds; each feed item becomes one RawDoc.

    Feed config keys: ``url`` (required), ``outlet``, ``language``, ``source_type``,
    ``tags`` (free metadata such as the ticker a Google News query is about).
    """

    source_type = SourceType.NEWS
    poll_interval_s = 180.0

    def __init__(
        self,
        name: str = "rss",
        feeds: list[dict[str, Any]] | None = None,
        concurrency: int = 6,
        max_items_per_feed: int = 40,
        **kw: Any,
    ):
        super().__init__(name, **kw)
        self.feeds = feeds or []
        self.concurrency = concurrency
        self.max_items = max_items_per_feed

    def parse_feed(self, content: bytes, feed: dict[str, Any]) -> list[RawDoc]:
        parsed = feedparser.parse(content)
        stype = SourceType(feed.get("source_type", self.source_type.value))
        docs = []
        for e in parsed.entries[: self.max_items]:
            link = e.get("link", "")
            title = strip_html(e.get("title"))
            if not title:
                continue
            summary = strip_html(e.get("summary") or e.get("description"))
            if summary == title or summary.startswith(title):
                summary = summary[len(title) :].strip(" -")
            outlet = feed.get("outlet") or domain_of(link)
            # Google News wraps the publisher in <source>; keep the real outlet.
            src = e.get("source")
            if isinstance(src, dict) and src.get("title"):
                outlet = (src.get("href") and domain_of(src["href"])) or src["title"]
            docs.append(
                RawDoc(
                    doc_id=RawDoc.make_id(self.name, link or title),
                    source=self.name,
                    source_type=stype,
                    outlet=outlet,
                    url=link,
                    title=title,
                    text=summary[:2000],
                    language=feed.get("language"),
                    author=e.get("author"),
                    published_at=_entry_time(e),
                    meta={
                        "feed": feed.get("id") or feed["url"],
                        **({"tags": feed["tags"]} if feed.get("tags") else {}),
                    },
                )
            )
        return docs

    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        sem = asyncio.Semaphore(self.concurrency)
        errors: list[str] = []

        async def one(feed: dict[str, Any]) -> list[RawDoc]:
            async with sem:
                try:
                    r = (await client.get(feed["url"])).raise_for_status()
                    return self.parse_feed(r.content, feed)
                except Exception as exc:
                    errors.append(f"{feed['url'][:60]}: {type(exc).__name__}")
                    return []

        results = await asyncio.gather(*(one(f) for f in self.feeds))
        if self.feeds and len(errors) == len(self.feeds):
            raise ConnectionError(f"all {len(errors)} feeds failed, e.g. {errors[0]}")
        if errors:
            log.info("%s: %d/%d feeds failed this poll", self.name, len(errors), len(self.feeds))
        return [d for docs in results for d in docs]


class RedditRssSource(RssSource):
    """Reddit's public ``.json`` listings now answer 403 without OAuth, but the
    ``.rss`` listings are still public, so we read those."""

    source_type = SourceType.SOCIAL
    poll_interval_s = 120.0

    def __init__(self, name: str = "reddit", subreddits: list[str] | None = None, **kw: Any):
        feeds = [
            {
                "url": f"https://www.reddit.com/r/{s}/new/.rss?limit=50",
                "outlet": f"reddit.com/r/{s}",
                "source_type": "social",
                "id": f"r/{s}",
            }
            for s in (subreddits or ["stocks"])
        ]
        super().__init__(name, feeds=feeds, concurrency=2, **kw)


_ITEM = re.compile(r"Item\s+(\d\.\d\d)")
_EDGAR_TITLE = re.compile(r"^(?P<form>[\w\-/]+)\s+-\s+(?P<company>.+?)\s+\((?P<cik>\d+)\)")

# 8-K item numbers -> the event class they evidence (weak labels for the event model).
EDGAR_ITEM_EVENTS = {
    "1.01": "Merger/Acquisition",
    "1.02": "Merger/Acquisition",
    "1.03": "Credit Event",
    "1.05": "Cyber/Technology Risk",
    "2.01": "Merger/Acquisition",
    "2.02": "Earnings/Guidance",
    "2.03": "Credit Event",
    "2.04": "Credit Event",
    "2.05": "Operational/Supply Chain",
    "2.06": "Credit Event",
    "3.01": "Regulatory/Legal",
    "4.01": "Regulatory/Legal",
    "4.02": "Earnings/Guidance",
    "5.01": "Management Change",
    "5.02": "Management Change",
    "7.01": "Other/Non-event",
    "8.01": "Other/Non-event",
}


class EdgarSource(RssSource):
    """SEC EDGAR "latest filings" Atom feed for 8-Ks. SEC needs only a descriptive
    User-Agent header, which the shared HTTP client sends."""

    source_type = SourceType.FILING
    poll_interval_s = 600.0
    URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=100&output=atom"

    def __init__(self, name: str = "edgar_8k", **kw: Any):
        super().__init__(
            name,
            feeds=[
                {
                    "url": self.URL,
                    "outlet": "sec.gov",
                    "source_type": "filing",
                    "language": "en",
                    "id": "edgar-8k",
                }
            ],
            max_items_per_feed=100,
            **kw,
        )

    def parse_feed(self, content: bytes, feed: dict[str, Any]) -> list[RawDoc]:
        out = []
        for d in super().parse_feed(content, feed):
            m = _EDGAR_TITLE.match(d.title)
            items = sorted(set(_ITEM.findall(d.text)))
            company = m.group("company").strip() if m else d.title
            events = sorted({EDGAR_ITEM_EVENTS[i] for i in items if i in EDGAR_ITEM_EVENTS})
            item_txt = "; ".join(f"Item {i}" for i in items)
            out.append(
                d.model_copy(
                    update={
                        "title": f"{company} files 8-K" + (f" ({item_txt})" if item_txt else ""),
                        "meta": {
                            **d.meta,
                            "form": m.group("form") if m else "8-K",
                            "company": company,
                            "cik": m.group("cik") if m else None,
                            "items": items,
                            "weak_events": events,
                        },
                    }
                )
            )
        return out
