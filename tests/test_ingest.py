"""Connector parsers on real recorded payloads, and replay pacing."""

from __future__ import annotations

from conftest import FIXTURES, make_doc, write_twin
from engine.ingest.gdelt import parse_events, parse_gkg
from engine.ingest.replay import ReplayReader
from engine.ingest.rss import EdgarSource, RssSource
from engine.ingest.social import MastodonSource


def test_gdelt_gkg_parse_filters_joins_and_unescapes() -> None:
    events = parse_events((FIXTURES / "events_sample.csv").read_bytes())
    docs = parse_gkg((FIXTURES / "gkg_sample.csv").read_bytes(), events)
    # 6 fixture rows: 4 carry finance/risk themes, 2 do not and must be dropped
    assert len(docs) == 4
    assert all(d.title and d.url.startswith("http") for d in docs)
    joined = [d for d in docs if "gdelt_event" in d.meta]
    assert len(joined) == 2, "GKG rows whose URL is in the Events export get CAMEO/Goldstein fields"
    ev = joined[0].meta["gdelt_event"]
    assert {"cameo_root", "goldstein", "num_mentions", "num_sources"} <= ev.keys()
    assert docs[0].meta["gdelt_tone"]["word_count"] > 0
    assert len({d.doc_id for d in docs}) == 4


def test_gdelt_translated_rows_keep_source_language() -> None:
    raw = (FIXTURES / "gkg_sample.csv").read_text().splitlines()
    cols = raw[1].split("\t")
    cols[25] = "srclc:jpn;eng:GT-JPN 1.0"
    cols[26] = (
        "<PAGE_TITLE>&#x30C8;&#x30E8;&#x30BF;&#x682A;&#x304C;&#x6025;&#x843D;&#x3001;&#x5E02;&#x5834;</PAGE_TITLE>"
    )
    docs = parse_gkg("\t".join(cols).encode(), translated=True)
    assert docs[0].language == "ja"
    assert docs[0].title.startswith("トヨタ株"), "HTML entities in GDELT titles are decoded"


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Nvidia shares jump after record data-centre sales - Reuters</title>
<link>https://news.google.com/rss/articles/abc</link><pubDate>Fri, 03 Oct 2026 08:14:00 GMT</pubDate>
<source url="https://www.reuters.com">Reuters</source></item>
<item><title></title><link>https://x/empty</link></item>
</channel></rss>"""


def test_rss_parse_uses_publisher_outlet_and_skips_empty() -> None:
    src = RssSource(feeds=[])
    docs = src.parse_feed(
        RSS, {"url": "https://news.google.com/rss/search?q=NVDA", "language": "en", "tags": ["NVDA"]}
    )
    assert len(docs) == 1
    d = docs[0]
    assert d.outlet == "reuters.com"
    assert d.published_at.isoformat() == "2026-10-03T08:14:00+00:00"
    assert d.meta["tags"] == ["NVDA"]


ATOM = b"""<?xml version="1.0" encoding="ISO-8859-1" ?><feed xmlns="http://www.w3.org/2005/Atom"><title>Latest</title>
<entry><title>8-K - Acme Bancorp (0000123456) (Filer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/123456/x.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-10-03 &lt;b&gt;AccNo:&lt;/b&gt; 1 &lt;b&gt;Size:&lt;/b&gt; 1 MB
&lt;br&gt;Item 1.03: Bankruptcy or Receivership&lt;br&gt;Item 5.02: Departure of Directors</summary>
<updated>2026-10-03T16:05:01-04:00</updated></entry></feed>"""


def test_edgar_items_become_weak_event_labels() -> None:
    docs = EdgarSource().parse_feed(ATOM, {"url": EdgarSource.URL, "source_type": "filing"})
    assert len(docs) == 1
    m = docs[0].meta
    assert m["company"] == "Acme Bancorp" and m["cik"] == "0000123456"
    assert m["items"] == ["1.03", "5.02"]
    assert m["weak_events"] == ["Credit Event", "Management Change"]
    assert docs[0].published_at.hour == 20  # converted to UTC


def test_mastodon_skips_boosts_and_short_posts() -> None:
    statuses = [
        {
            "id": "1",
            "uri": "u1",
            "url": "https://m/1",
            "content": "<p>$TSLA guidance cut again, margins under pressure</p>",
            "language": "en",
            "created_at": "2026-10-03T08:00:00.000Z",
            "account": {"acct": "a", "followers_count": 5},
        },
        {"id": "2", "uri": "u2", "content": "<p>boost</p>", "reblog": {"id": "9"}, "account": {"acct": "b"}},
        {"id": "3", "uri": "u3", "content": "<p>short</p>", "account": {"acct": "c"}},
    ]
    docs = MastodonSource().parse(statuses, "stocks")
    assert [d.text for d in docs] == ["$TSLA guidance cut again, margins under pressure"]


def test_replay_reader_clamps_gaps_and_loops_with_fresh_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_twin(tmp_path, "rss", [make_doc(i, minutes=60) for i in range(3)])
    reader = ReplayReader.for_source(tmp_path, "rss", speed=60.0, max_gap_s=0.5, loop=True)
    assert reader is not None
    it = iter(reader)
    pairs = [next(it) for _ in range(5)]
    assert pairs[0][0] == 0.0
    assert all(delay <= 0.5 for delay, _ in pairs), "60-minute gaps at 60x would be 60 s; clamped to 0.5 s"
    ids = [d.doc_id for _, d in pairs]
    assert ids[3] == ids[0] + "~r1", "second lap gets new ids so idempotent stores keep them"
    stamped = reader.stamp(pairs[0][1])
    assert stamped.replayed and "original_published_at" in stamped.meta


def test_replay_spreads_same_timestamp_batches() -> None:
    from engine.ingest.replay import spread_batches

    # four GDELT docs stamped with one file time, next file 900 s later
    assert spread_batches([0, 0, 0, 0, 900]) == [0, 225, 450, 675, 900]
    assert spread_batches([5.0, 7.0]) == [5.0, 7.0]
