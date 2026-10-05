"""FastAPI application: REST + WebSocket + Prometheus, and the built web UI at ``/``.

Run with ``uvicorn engine.api.app:app`` (``make demo`` does this in replay mode).
OpenAPI docs are generated at ``/docs``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from engine import __version__
from engine.api.ratelimit import RateLimitMiddleware
from engine.api.state import AppState
from engine.bus import TOPIC_RAW, TOPIC_SIGNALS
from engine.config import REPO_ROOT, Settings, get_settings
from engine.schema import EventClass, RawDoc, Signal

log = logging.getLogger(__name__)
WEB_DIST = REPO_ROOT / "src" / "web" / "dist"


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="Any language, up to 20 KB")
    language: str | None = Field(None, description="Optional ISO 639-1 hint, e.g. 'ja'")


def _state(request: Request) -> AppState:
    return request.app.state.sentinel


def create_app(settings: Settings | None = None, start_background: bool = True) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState.build(settings)
        app.state.sentinel = state
        if start_background:
            await state.start()
        try:
            yield
        finally:
            await state.stop()

    app = FastAPI(
        title="Sentinel Risk Signal Engine",
        version=__version__,
        description="Keyless, multilingual risk signals from news and social text: sentiment [-1, 1], "
        "event class and impact 1-10, plus an index rebalancer and a bank-book stress tester.",
        lifespan=lifespan,
    )
    app.add_middleware(RateLimitMiddleware, rate=settings.rate_limit_rps)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error", "type": type(exc).__name__})

    # ---------------------------------------------------------------- health
    @app.get("/v1/health", tags=["system"])
    async def health(request: Request) -> dict[str, Any]:
        st = _state(request)
        sources = st.ingest.health()
        return {
            "status": "ok",
            "version": __version__,
            "mode": st.ingest.mode.value,
            "uptime_s": round(time.time() - st.started_at, 1),
            "sources": sources,
            "sources_up": sum(1 for s in sources if s["docs_per_min"] > 0 or s["active_mode"] == "live"),
            "docs_total": st.monitor.total,
            "docs_per_min": sum(s["docs_per_min"] for s in sources),
            "languages_seen": len(st.pipeline.stats.by_language)
            if st.pipeline
            else len(st.monitor.languages),
            "pipeline": _pipeline_health(st),
            "bus": {
                "published": dict(st.bus.published),
                "dropped": dict(st.bus.dropped),
                "lag": st.bus.lag(),
            },
        }

    def _pipeline_health(st: AppState) -> dict[str, Any]:
        if st.pipeline is None:
            return {
                "status": "loading" if st.pipeline_error is None else "failed",
                "error": st.pipeline_error,
            }
        ps = st.pipeline.stats
        return {
            "status": "ready",
            "modes": st.pipeline.modes,
            "models": st.pipeline.model_versions,
            "docs": ps.docs,
            "signals": ps.signals,
            "duplicates": ps.duplicates,
            "duplicate_ratio": round(ps.duplicates / ps.docs, 4) if ps.docs else 0.0,
            "unlinked": ps.unlinked,
            "store": {
                "signals": len(st.store),
                "duplicates_ignored": st.store.duplicates,
                "written": st.store.written,
            },
            "bot_suppressed": st.aggregator.bot_suppressed,
            "dead_letters": st.dead_letters,
            "languages": dict(sorted(ps.by_language.items(), key=lambda kv: -kv[1])),
        }

    # ---------------------------------------------------------------- signals
    @app.post("/v1/analyze", tags=["signals"])
    async def analyze(body: AnalyzeRequest, request: Request) -> dict[str, Any]:
        """Score any text, in any language, and return the full signal(s) plus model diagnostics."""
        st = _state(request)
        if len(body.text.encode("utf-8")) > settings.analyze_max_bytes:
            raise HTTPException(413, f"text exceeds {settings.analyze_max_bytes} bytes")
        if st.pipeline is None:
            raise HTTPException(
                503,
                "NLP pipeline is still loading" if st.pipeline_error is None else "NLP pipeline unavailable",
            )
        tr = await asyncio.to_thread(st.pipeline.analyze, body.text, body.language)
        return {
            "language": {
                "code": tr.lang.lang,
                "confidence": round(tr.lang.confidence, 3),
                "method": tr.lang.method,
            },
            "entities": [
                {
                    "id": m.entity.id,
                    "name": m.entity.name,
                    "alias": m.alias,
                    "start": m.start,
                    "end": m.end,
                    "method": m.method,
                }
                for m in tr.links.mentions
            ],
            "event": None
            if tr.event is None
            else {
                "primary": tr.event.primary,
                "confidence": tr.event.confidence,
                "method": tr.event.method,
                "probabilities": tr.event.probs,
            },
            "sentiment": {
                eid: {
                    "score": r.score,
                    "label": r.label,
                    "confidence": r.confidence,
                    "probabilities": dict(zip(("negative", "neutral", "positive"), r.probs, strict=True)),
                    "model": r.model,
                    "sentence": tr.sentences.get(eid),
                }
                for eid, r in tr.sentiments.items()
            },
            "signals": [s.model_dump(mode="json") for s in tr.signals],
            "dropped": tr.dropped,
            "modes": st.pipeline.modes,
        }

    @app.get("/v1/signals", tags=["signals"])
    async def signals(
        request: Request,
        entity: str | None = None,
        event: EventClass | None = None,
        min_impact: Annotated[int, Query(ge=1, le=10)] = 1,
        language: str | None = None,
        since: Annotated[
            float | None, Query(description="Unix seconds; only signals published after")
        ] = None,
        cursor: Annotated[int | None, Query(ge=0, description="next_cursor from the previous page")] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> dict[str, Any]:
        """Signals, newest first, filtered and cursor-paginated."""
        page = _state(request).store.query(entity, event, min_impact, language, since, cursor, limit)
        return {
            "count": len(page.items),
            "next_cursor": page.next_cursor,
            "items": [s.model_dump(mode="json") for s in page.items],
        }

    @app.get("/v1/entities", tags=["signals"])
    async def entities(request: Request, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> dict[str, Any]:
        """Entities with signals so far and their current aggregate sentiment S_e(t)."""
        st = _state(request)
        now = time.time()
        ents = sorted(st.aggregator.entities.values(), key=lambda e: -e.n_signals)[:limit]
        rows = [
            {
                "id": e.entity_id,
                "name": e.name,
                "sector": e.sector,
                "country": e.country,
                "score": round(st.aggregator.score(e.entity_id, now), 4),
                "n_signals": e.n_signals,
                "last_signal_at": e.last_t,
            }
            for e in ents
        ]
        return {"count": len(st.aggregator.entities), "items": rows, "time_scale": st.time_scale}

    @app.get("/v1/entities/{entity_id}/aggregate", tags=["signals"])
    async def entity_aggregate(entity_id: str, request: Request) -> dict[str, Any]:
        """Current S_e(t), its 24 h trend and the signals driving it."""
        snap = _state(request).aggregator.snapshot(entity_id, time.time())
        if snap is None:
            raise HTTPException(404, f"no signals yet for {entity_id!r}")
        return snap

    @app.websocket("/v1/stream")
    async def stream_signals(ws: WebSocket, entity: str | None = None, min_impact: int = 1) -> None:
        """Push every new signal (optionally filtered) as JSON."""
        await ws.accept()
        st: AppState = ws.app.state.sentinel
        q = st.bus.subscribe(TOPIC_SIGNALS, maxsize=1000)
        try:
            while True:
                sig: Signal = await q.get()
                if (entity and sig.entity.id != entity) or sig.impact.score < min_impact:
                    continue
                await ws.send_text(sig.model_dump_json())
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            st.bus.unsubscribe(TOPIC_SIGNALS, q)
            with contextlib.suppress(Exception):
                await ws.close()

    @app.get("/v1/metrics", tags=["system"], response_class=Response)
    async def metrics(request: Request) -> Response:
        return Response(generate_latest(_state(request).metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/sources/{name}/kill", tags=["system"])
    async def kill_source(name: str, request: Request) -> dict[str, Any]:
        """Resilience demo: make a live source fail so its breaker opens and its replay twin takes over."""
        if not _state(request).ingest.set_killed(name, True):
            raise HTTPException(404, f"unknown source {name!r}")
        return {"source": name, "killed": True}

    @app.post("/v1/sources/{name}/revive", tags=["system"])
    async def revive_source(name: str, request: Request) -> dict[str, Any]:
        if not _state(request).ingest.set_killed(name, False):
            raise HTTPException(404, f"unknown source {name!r}")
        return {"source": name, "killed": False}

    # ---------------------------------------------------------------- raw docs
    @app.get("/v1/raw", tags=["ingest"])
    async def raw_docs(
        request: Request,
        source: str | None = None,
        language: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> dict[str, Any]:
        """Most recent raw documents as ingested (before NLP), newest first."""
        st = _state(request)
        out: list[RawDoc] = []
        for d in reversed(st.monitor.recent):
            if source and d.source != source:
                continue
            if language and d.language != language:
                continue
            out.append(d)
            if len(out) >= limit:
                break
        return {
            "count": len(out),
            "items": [d.model_dump(mode="json") for d in out],
            "languages": dict(st.monitor.languages.most_common()),
        }

    @app.websocket("/v1/stream/raw")
    async def stream_raw(ws: WebSocket, source: str | None = None) -> None:
        await ws.accept()
        st: AppState = ws.app.state.sentinel
        q = st.bus.subscribe(TOPIC_RAW, maxsize=500)
        try:
            while True:
                doc: RawDoc = await q.get()
                if source and doc.source != source:
                    continue
                await ws.send_text(doc.model_dump_json())
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            st.bus.unsubscribe(TOPIC_RAW, q)
            with contextlib.suppress(Exception):
                await ws.close()

    from engine.api.modules_api import router as modules_router

    app.include_router(modules_router)

    # ---------------------------------------------------------------- web UI
    if (WEB_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        async def root() -> dict[str, str]:
            return {
                "name": "Sentinel",
                "docs": "/docs",
                "health": "/v1/health",
                "note": "web UI not built yet: run `make web`",
            }

    return app


app = create_app()
