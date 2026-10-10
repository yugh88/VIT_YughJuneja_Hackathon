"""REST endpoints for Module A (index rebalancer) and Module B (stress testing)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from engine.config import REPO_ROOT
from engine.schema import EventClass

router = APIRouter(prefix="/v1/modules")
EVAL = REPO_ROOT / "docs" / "eval"


def _st(request: Request) -> Any:
    return request.app.state.sentinel


def _module(request: Request, name: str) -> Any:
    m = getattr(_st(request), name)
    if m is None:
        raise HTTPException(503, f"{name} module unavailable (reference data missing)")
    return m


# ------------------------------------------------------------------ Module A
@router.get("/index", tags=["module A"])
async def index_state(
    request: Request, history: Annotated[int, Query(ge=1, le=5000)] = 500
) -> dict[str, Any]:
    """Current weights vs base, sentiment per name, weights over time and the rebalance log."""
    rb = _module(request, "rebalancer")
    out = rb.state(time.time())
    out["history"] = [{"t": t, "weights": w} for t, w in list(rb.history)[-history:]]
    out["log"] = [e.__dict__ for e in list(rb.log)[-100:]][::-1]
    return out


@router.post("/index/kappa", tags=["module A"])
async def set_kappa(request: Request, value: Annotated[float, Query(ge=0.0, le=3.0)]) -> dict[str, Any]:
    rb = _module(request, "rebalancer")
    rb.kappa = value
    entry = rb.rebalance(time.time(), f"kappa set to {value}")
    return {"kappa": value, "rebalanced": entry is not None}


@router.post("/index/{action}", tags=["module A"])
async def pause_resume(request: Request, action: str) -> dict[str, Any]:
    if action not in {"pause", "resume"}:
        raise HTTPException(404, "use pause or resume")
    rb = _module(request, "rebalancer")
    rb.paused = action == "pause"
    return {"paused": rb.paused}


@router.get("/index/backtest", tags=["module A"])
async def backtest() -> dict[str, Any]:
    p = EVAL / "backtest.json"
    if not p.exists():
        raise HTTPException(404, "run `python -m modules.rebalancer.backtest` first")
    return json.loads(p.read_text())


# ------------------------------------------------------------------ Module B
class ScenarioRequest(BaseModel):
    event: EventClass = EventClass.GEOPOLITICAL
    impact: float = Field(9, ge=1, le=10)
    country: str | None = Field(None, description="ISO-2 country to target (1.5x), its region gets 1.2x")
    sector: str | None = Field(None, description="GICS sector to target (1.3x)")
    entity_id: str | None = Field(None, description="Named obligor (Credit Event: -3 notches)")


def _summary(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("id", "cluster", "fired_at", "trigger", "loss", "delta_cet1_bps", "delta_rwa",
                                  "delta_expected_loss", "corroboration")} | {"var_99": r["monte_carlo"]["var_99"],
                                                                             "scenario": r["scenario"]["name"]}  # fmt: skip


@router.get("/stress", tags=["module B"])
async def stress_state(request: Request) -> dict[str, Any]:
    """Book summary, stress tests run so far, and recent trigger decisions (fired or why not)."""
    se = _module(request, "stress")
    from modules.stress.pricing import load_book

    book = load_book()
    return {
        "book": {"positions": len(book), "exposure": float(book["exposure"].sum()),
                 "by_asset_class": book.groupby("asset_class")["exposure"].sum().round(0).to_dict(),
                 "counterparties": int(book["cp_id"].nunique()), "countries": int(book["country"].nunique()),
                 "named_entities": sorted(book["entity_id"].dropna().unique().tolist())},
        "rule": {"events": ["Geopolitical", "Macroeconomic", "Credit Event"], "impact_gt": se.min_impact,
                 "corroboration": f">= {se.min_outlets} outlets or burst z >= {se.min_burst} within 6 h, "
                                  "plus >= 1 tier-1 outlet",
                 "credit_gate": "Credit Event needs confidence >= 0.7 and a distress term",
                 "cluster": "(event, country) or (Credit Event, obligor); merged within 6 h",
                 "severity": "impact x corroboration (max(outlets, burst z), capped at 20)",
                 "time_scale": se.time_scale},
        "alert_queue": [{k: v for k, v in a.items() if k != "first_t"} for a in se.alert_queue()],
        "results": [_summary(r) for r in list(se.results)][::-1],
        "decisions": list(se.decisions)[-50:][::-1],
    }  # fmt: skip


@router.get("/stress/results/{result_id}", tags=["module B"])
async def stress_result(result_id: str, request: Request) -> dict[str, Any]:
    for r in _module(request, "stress").results:
        if r["id"] == result_id:
            return r
    raise HTTPException(404, "unknown result")


@router.post("/stress/run", tags=["module B"])
async def stress_run(body: ScenarioRequest, request: Request) -> dict[str, Any]:
    """Custom scenario (the dashboard's sliders): full before/after, waterfall, capital and Monte Carlo."""
    se = _module(request, "stress")
    return await asyncio.to_thread(
        se.run,
        None,
        "custom",
        None,
        event=body.event.value,
        impact=body.impact,
        country=body.country,
        sector=body.sector,
        entity_id=body.entity_id,
    )


@router.get("/stress/replay/2022-02-24", tags=["module B"])
async def replay_2022(refresh: bool = False) -> dict[str, Any]:
    """24 Feb 2022 through the full system (recorded GDELT, real models, real trigger rule)."""
    p = EVAL / "replay_2022-02-24.json"
    if refresh or not p.exists():
        from modules.stress.replay import run

        out = await asyncio.to_thread(run)
        out.pop("signals_snapshot", None)
        p.write_text(json.dumps(out, indent=1))
        return out
    return json.loads(p.read_text())


# ------------------------------------------------------------------ stored evaluation results
eval_router = APIRouter(prefix="/v1/eval", tags=["evaluation"])
EVAL_FILES = {
    "eval",
    "impact",
    "credit",
    "bench",
    "backtest",
    "replay_2022-02-24",
    "replay_2022-02-24_signals",
    "credit_experiments",
    "v2_summary",
}


@eval_router.get("/{name}")
async def eval_file(name: str) -> Any:
    """Results written by `make eval`, `make bench`, the impact/credit scripts and the backtest
    (read-only; the dashboard's Model Lab renders these files as they are)."""
    if name not in EVAL_FILES:
        raise HTTPException(404, f"unknown result file; one of {sorted(EVAL_FILES)}")
    p = EVAL / f"{name}.json"
    gz = EVAL / f"{name}.json.gz"  # large results are stored compressed
    if gz.exists():
        import gzip

        with gzip.open(gz, "rt") as fh:
            return json.loads(fh.read())
    if not p.exists():
        raise HTTPException(404, f"{p.name} not generated yet")
    return json.loads(p.read_text())
