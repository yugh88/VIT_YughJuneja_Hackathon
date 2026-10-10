"""API contract and the Phase 1 exit criterion: replayed docs stream into the API."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from engine.api.app import create_app
from engine.config import Settings
from engine.schema import Signal


@pytest.fixture
def client(replay_settings: Settings):  # type: ignore[no-untyped-def]
    with TestClient(create_app(replay_settings)) as c:
        yield c


def _wait_for_docs(client: TestClient, n: int, timeout: float = 15.0) -> dict:  # type: ignore[type-arg]
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/v1/raw", params={"limit": 100}).json()
        if body["count"] >= n:
            return body  # type: ignore[no-any-return]
        time.sleep(0.05)
    raise AssertionError(f"fewer than {n} docs arrived within {timeout}s")


def test_replayed_docs_stream_into_api(client: TestClient) -> None:
    body = _wait_for_docs(client, 30)
    sources = {d["source"] for d in body["items"]}
    assert sources == {"rss", "tweets"}
    assert all(d["replayed"] for d in body["items"])

    health = client.get("/v1/health").json()
    assert health["mode"] == "replay"
    by_name = {s["name"]: s for s in health["sources"]}
    assert by_name["tweets"]["configured_mode"] == "replay-only"
    assert by_name["rss"]["active_mode"] == "replay" and by_name["rss"]["docs_total"] >= 10


def test_raw_filters_and_bad_input_is_4xx(client: TestClient) -> None:
    _wait_for_docs(client, 10)
    only = client.get("/v1/raw", params={"source": "tweets"}).json()
    assert only["count"] > 0 and {d["source"] for d in only["items"]} == {"tweets"}
    assert client.get("/v1/raw", params={"limit": 0}).status_code == 422
    assert client.get("/v1/raw", params={"limit": "lots"}).status_code == 422
    assert client.post("/v1/sources/nope/kill").status_code == 404


def test_metrics_and_openapi(client: TestClient) -> None:
    _wait_for_docs(client, 5)
    m = client.get("/v1/metrics")
    assert m.status_code == 200 and "sentinel_docs_ingested_total" in m.text
    spec = client.get("/openapi.json").json()
    assert {"/v1/health", "/v1/raw", "/v1/metrics"} <= set(spec["paths"])


def _wait_for(client: TestClient, path: str, pred, timeout: float = 30.0):  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(path)
        if pred(r.json()):
            return r.json()
        time.sleep(0.1)
    raise AssertionError(f"{path} condition not met in {timeout}s: {r.json()}")


def test_analyze_returns_valid_signals_in_three_languages(client: TestClient) -> None:
    """Phase 2 exit criterion."""
    _wait_for(client, "/v1/health", lambda h: h["pipeline"]["status"] == "ready")
    cases = {
        "en": ("JPMorgan shares slump after a surprise loss and a dividend cut", "JPM"),
        "ja": ("トヨタ自動車、大規模リコールで株価が急落", "CORP:TOYOTA"),
        "hi": ("रिलायंस इंडस्ट्रीज़ का मुनाफ़ा बढ़ा, शेयरों में तेज़ी", "RELIANCE.NS"),
    }
    for lang, (text, entity) in cases.items():
        body = client.post("/v1/analyze", json={"text": text}).json()
        assert body["language"]["code"] == lang
        assert [s["entity"]["id"] for s in body["signals"]] == [entity]
        Signal.model_validate(body["signals"][0])  # contract: response parses as the schema


def test_analyze_rejects_bad_input(client: TestClient) -> None:
    assert client.post("/v1/analyze", json={"text": ""}).status_code == 422
    assert client.post("/v1/analyze", json={"nope": 1}).status_code == 422
    assert client.post("/v1/analyze", json={"text": "x" * 20_001}).status_code in {413, 503}


def test_stream_signals_aggregates_and_websocket(client: TestClient) -> None:
    body = _wait_for(client, "/v1/signals?limit=5", lambda b: b["count"] > 0)
    assert all(Signal.model_validate(s) for s in body["items"])
    ents = client.get("/v1/entities").json()
    eid = ents["items"][0]["id"]
    agg = client.get(f"/v1/entities/{eid}/aggregate").json()
    assert -1.0 <= agg["score"] <= 1.0 and agg["n_signals"] >= 1 and agg["top_drivers"]
    assert client.get("/v1/entities/NOPE/aggregate").status_code == 404
    assert client.get("/v1/signals", params={"cursor": -1}).status_code == 422
    idx = client.get("/v1/modules/index").json()  # Module A
    assert (
        len(idx["weights"]) == 20 and abs(sum(w["weight"] for w in idx["weights"]) - 1) < 1e-4
    )  # 5-dp display rounding
    stress = client.get("/v1/modules/stress").json()  # Module B
    assert stress["book"]["positions"] == 300
    run = client.post(
        "/v1/modules/stress/run", json={"event": "Macroeconomic", "impact": 8, "country": "US"}
    ).json()
    assert run["loss"] > 0 and run["monte_carlo"]["var_99"] >= run["monte_carlo"]["mean_loss"]
    with client.websocket_connect("/v1/stream") as ws:
        # Publish one fresh document once the socket is open, so a push arrives whatever the replay's state.
        from conftest import make_doc
        from engine.bus import TOPIC_RAW

        doc = make_doc(9999, "rss", title="Microsoft shares jump after record quarterly profit")
        client.portal.call(client.app.state.sentinel.bus.publish, TOPIC_RAW, doc)  # type: ignore[union-attr,attr-defined]
        Signal.model_validate_json(ws.receive_text())


def test_explanation_tokens_split_words_cjk_and_strip_punctuation() -> None:
    from engine.api.app import _tokens

    s = "Moody's cuts Ukraine, outlook negative."
    assert [s[a:b] for a, b in _tokens(s)] == ["Moody's", "cuts", "Ukraine", "outlook", "negative"]
    j = "トヨタ自動車、稼働停止"
    assert (
        all(b - a <= 2 for a, b in _tokens(j))
        and "".join(j[a:b] for a, b in _tokens(j)) == "トヨタ自動車稼働停止"
    )
