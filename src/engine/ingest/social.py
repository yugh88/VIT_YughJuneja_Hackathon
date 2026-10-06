"""Keyless social connectors: Mastodon public hashtag timelines and Hacker News (Algolia)."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from engine.ingest.base import Source
from engine.ingest.text import parse_dt, strip_html
from engine.schema import RawDoc, SourceType

log = logging.getLogger(__name__)


class MastodonSource(Source):
    """``GET /api/v1/timelines/tag/:tag`` is public on mastodon.social (no token)."""

    source_type = SourceType.SOCIAL
    poll_interval_s = 60.0

    def __init__(
        self,
        name: str = "mastodon",
        instance: str = "mastodon.social",
        tags: list[str] | None = None,
        limit: int = 40,
        **kw: Any,
    ):
        super().__init__(name, **kw)
        self.instance = instance
        self.tags = tags or ["stocks"]
        self.limit = limit
        self._since: dict[str, str] = {}

    def parse(self, statuses: list[dict[str, Any]], tag: str) -> list[RawDoc]:
        docs = []
        for st in statuses:
            if st.get("reblog"):
                continue
            text = strip_html(st.get("content"))
            if len(text) < 20:
                continue
            acct = (st.get("account") or {}).get("acct", "unknown")
            docs.append(
                RawDoc(
                    doc_id=RawDoc.make_id("mastodon", st.get("uri") or st.get("url") or st["id"]),
                    source=self.name,
                    source_type=SourceType.SOCIAL,
                    outlet=f"{self.instance}/@{acct}",
                    url=st.get("url") or "",
                    title="",
                    text=text[:2000],
                    language=st.get("language"),
                    author=acct,
                    published_at=parse_dt(st.get("created_at")) or datetime.now(UTC),
                    meta={
                        "tag": tag,
                        "reblogs": st.get("reblogs_count", 0),
                        "favourites": st.get("favourites_count", 0),
                        "followers": (st.get("account") or {}).get("followers_count", 0),
                    },
                )
            )
        return docs

    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        out: list[RawDoc] = []
        failures = 0
        for tag in self.tags:
            params: dict[str, Any] = {"limit": self.limit}
            if tag in self._since:
                params["since_id"] = self._since[tag]
            try:
                r = (
                    await client.get(f"https://{self.instance}/api/v1/timelines/tag/{tag}", params=params)
                ).raise_for_status()
            except httpx.HTTPError:
                failures += 1
                continue
            statuses = r.json()
            if statuses:
                self._since[tag] = statuses[0]["id"]
            out += self.parse(statuses, tag)
            await asyncio.sleep(0.5)
        if failures == len(self.tags):
            raise ConnectionError(f"mastodon: all {failures} tag timelines failed")
        return out


class HackerNewsSource(Source):
    """Algolia HN search API, newest stories matching finance keywords."""

    source_type = SourceType.SOCIAL
    poll_interval_s = 120.0
    API = "https://hn.algolia.com/api/v1/search_by_date"

    def __init__(self, name: str = "hackernews", queries: list[str] | None = None, **kw: Any):
        super().__init__(name, **kw)
        self.queries = queries or ["stocks"]

    def parse(self, payload: dict[str, Any], query: str) -> list[RawDoc]:
        docs = []
        for h in payload.get("hits", []):
            title = (h.get("title") or h.get("story_title") or "").strip()
            if not title:
                continue
            url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            docs.append(
                RawDoc(
                    doc_id=RawDoc.make_id("hackernews", str(h.get("objectID"))),
                    source=self.name,
                    source_type=SourceType.SOCIAL,
                    outlet="news.ycombinator.com",
                    url=url,
                    title=title,
                    text=strip_html(h.get("story_text"))[:2000],
                    language="en",
                    author=h.get("author"),
                    published_at=parse_dt(h.get("created_at")) or datetime.now(UTC),
                    meta={
                        "query": query,
                        "points": h.get("points") or 0,
                        "comments": h.get("num_comments") or 0,
                    },
                )
            )
        return docs

    async def fetch(self, client: httpx.AsyncClient) -> list[RawDoc]:
        out: list[RawDoc] = []
        for q in self.queries:
            r = (
                await client.get(self.API, params={"query": q, "tags": "story", "hitsPerPage": 30})
            ).raise_for_status()
            out += self.parse(r.json(), q)
        return out
