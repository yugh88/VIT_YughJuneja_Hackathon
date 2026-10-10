"""Historical replay: run 24 Feb 2022 (Russia's invasion of Ukraine) through the whole system.

    python -m modules.stress.replay            # prints the trigger timeline

Recorded GDELT windows (English + translingual, 00:00-12:00 UTC; data/replay/scenario_2022-02-24)
go through the shipped NLP pipeline with their original timestamps, then into Module B's
trigger. Nothing is scripted: whether and when a stress test fires is decided by the models,
the corroboration rule and the debounce exactly as in the live system.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from typing import Any

from engine.config import get_settings
from engine.ingest.replay import load_docs, twin_paths
from modules.stress.engine import StressEngine

log = logging.getLogger("scenario_replay")
NAME = "scenario_2022-02-24"


def run(
    n_draws: int = 2_000, batch: int = 64, start_hour: float = 0.0, end_hour: float = 12.0
) -> dict[str, Any]:
    from engine.nlp.pipeline import Pipeline

    docs = load_docs(twin_paths(get_settings().replay_dir, NAME))
    docs = [d for d in docs if start_hour <= d.published_at.hour + d.published_at.minute / 60 < end_hour]
    pipe = Pipeline(zero_shot=False)
    engine = StressEngine(n_draws=n_draws)
    t0 = time.time()
    n_signals = 0
    timeline: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    snapshot: list[dict[str, Any]] = []  # compact signals for the dashboard's 24 Feb 2022 view
    for i in range(0, len(docs), batch):
        for tr in pipe.process_batch(docs[i : i + batch]):
            for sig in tr.signals:
                n_signals += 1
                snapshot.append({"id": sig.signal_id, "doc": sig.doc_id, "t": sig.published_at.isoformat(),
                                 "outlet": sig.source.outlet, "src": sig.source.name, "type": sig.source.type.value,
                                 "lang": sig.language, "eid": sig.entity.id, "name": sig.entity.name,
                                 "country": sig.entity.country, "sector": sig.entity.sector,
                                 "s": round(sig.sentiment.score, 3), "ev": sig.event.primary.value,
                                 "evc": round(sig.event.confidence, 3), "imp": sig.impact.score, "lo": sig.impact.low,
                                 "hi": sig.impact.high, "drv": sig.impact.drivers, "title": sig.evidence.title,
                                 "sent": sig.evidence.sentence, "url": sig.evidence.url,
                                 "synd": sig.syndication_count})  # fmt: skip
                res = engine.on_signal(sig)
                if res is not None:
                    results[res["id"]] = {k: v for k, v in res.items() if k not in ("fired_at", "runtime_s")}
                    timeline.append({"time_utc": sig.published_at.strftime("%Y-%m-%d %H:%M"), "cluster": res["cluster"],
                                     "headline": res["trigger"]["title"], "impact": res["trigger"]["impact"],
                                     "corroboration": res["corroboration"], "loss_musd": round(res["loss"] / 1e6, 1),
                                     "severity": round(sig.impact.score * res["corroboration"]["score"], 1),
                                     "delta_cet1_bps": res["delta_cet1_bps"],
                                     "var99_musd": round(res["monte_carlo"]["var_99"] / 1e6, 1)})  # fmt: skip
    near = [d for d in engine.decisions if not d["fired"] and d["impact"] > 7]
    reasons = ("not corroborated", "credit gate", "merged")
    queue = [{k: v for k, v in a.items() if k != "first_t"} for a in engine.alert_queue()]
    out = {"docs": len(docs), "signals": n_signals, "seconds": round(time.time() - t0, 1),
           "trigger_version": "v2 (D66): country/obligor clusters merged within 6 h, tier-1 outlet, credit gate",
           "trigger_events_considered": len(engine.decisions), "stress_tests_fired": len(timeline),
           "first_trigger_utc": timeline[0]["time_utc"] if timeline else None, "timeline": timeline,
           "alert_queue": queue, "results": results,
           "languages": dict(sorted(pipe.stats.by_language.items(), key=lambda kv: -kv[1])),
           "blocked_high_impact": {r.split()[0] if r != "not corroborated" else "not_corroborated":
                                   sum(r in d["reason"] for d in near) for r in reasons}}  # fmt: skip
    out["signals_snapshot"] = snapshot
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = run()
    snapshot = out.pop("signals_snapshot")
    with gzip.open(
        get_settings().data_dir.parent / "docs" / "eval" / "replay_2022-02-24_signals.json.gz", "wt"
    ) as fh:
        fh.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))
    (get_settings().data_dir.parent / "docs" / "eval" / "replay_2022-02-24.json").write_text(
        json.dumps(out, separators=(",", ":"))  # compact, like backtest.json
    )
    print(json.dumps({k: v for k, v in out.items() if k != "timeline"}, indent=1))
    for t in out["timeline"][:15]:
        print(t["time_utc"], t["cluster"], t["impact"], t["corroboration"], "|", t["headline"][:80])


if __name__ == "__main__":
    main()
