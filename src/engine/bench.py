"""`make bench`: measured latency and throughput of the full NLP pipeline (and, from
Phase 4, the Monte Carlo stress engine). Results go to ``docs/eval/bench.json`` and
the model card.

    python -m engine.bench [--docs 1500]

* single-document latency: one doc per call through all seven stages (p50/p95);
* throughput grid: ONNX Runtime intra-op threads x batch size, the batch being what
  the stream worker's dynamic batching (up to N docs or 100 ms) hands the pipeline;
* session reuse: cold start (sessions created) vs warm calls;
* Module B: one full stress run with 10,000 Monte Carlo draws over the 300-position book.
Documents are real replayed headlines/posts from ``data/replay``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import platform
import statistics
import subprocess
import time
from typing import Any

from engine.config import REPO_ROOT, get_settings
from engine.ingest.replay import load_docs, twin_paths
from engine.schema import RawDoc

log = logging.getLogger("bench")
OUT = REPO_ROOT / "docs" / "eval" / "bench.json"


def corpus(n: int) -> list[RawDoc]:
    """A balanced mix: n/8 docs from each recorded source, interleaved."""
    s = get_settings()
    names = ("gdelt_gkg", "rss", "tweets", "mastodon", "edgar_8k", "hackernews", "reddit", "central_banks")
    per = [load_docs(twin_paths(s.replay_dir, name))[: n // len(names)] for name in names]
    return [d for group in zip(*per, strict=False) for d in group][:n]


def machine() -> str:
    chip = platform.processor() or platform.machine()
    with contextlib.suppress(FileNotFoundError):
        chip = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            or chip
        )
    return f"{chip}, {os.cpu_count()} logical cores, {platform.system()}"


def build_pipeline(threads: int) -> Any:
    os.environ["SENTINEL_ONNX_THREADS"] = str(threads)
    get_settings.cache_clear()
    from engine.nlp.pipeline import Pipeline

    t0 = time.perf_counter()
    p = Pipeline(zero_shot=False, dedupe=False)
    return p, time.perf_counter() - t0


def monte_carlo_bench() -> dict[str, Any]:
    """Module B: 10,000 correlated draws x 300 positions, full revaluation (plan: well under 1 s)."""
    from modules.stress.pricing import scenario_for, stress

    sc = scenario_for("Geopolitical", 9, "RU", None, "SOV:RU")
    stress(sc, n_draws=10_000)  # warm-up (loads book, covariance)
    times = []
    for seed in range(5):
        t0 = time.perf_counter()
        stress(sc, n_draws=10_000, seed=seed)
        times.append(time.perf_counter() - t0)
    return {
        "draws": 10_000,
        "positions": 300,
        "median_s": round(statistics.median(times), 3),
        "max_s": round(max(times), 3),
    }


def run(n_docs: int) -> dict[str, Any]:
    from training.model_card import provenance, render_model_card

    docs = corpus(n_docs)
    grid = []
    single: dict[str, Any] = {}
    for threads in (1, 2, 4, 8):
        pipe, cold = build_pipeline(threads)
        # dedupe off: repeated passes over one corpus must not be dropped as duplicates
        pipe.process_batch(docs[:64], dedupe=False)  # warm-up
        for batch in (1, 8, 32, 64):
            sample = docs if batch > 1 else docs[:120]
            t0 = time.perf_counter()
            signals = 0
            for i in range(0, len(sample), batch):
                signals += sum(
                    len(t.signals) for t in pipe.process_batch(sample[i : i + batch], dedupe=False)
                )
            dt = time.perf_counter() - t0
            grid.append(
                {
                    "threads": threads,
                    "batch": batch,
                    "docs_per_s": round(len(sample) / dt, 1),
                    "signals_per_s": round(signals / dt, 1),
                }
            )
            log.info("threads=%d batch=%d: %.1f docs/s", threads, batch, len(sample) / dt)
        if threads == 4:
            lat = []
            for d in docs[:240]:
                t0 = time.perf_counter()
                pipe.process_batch([d], dedupe=False)
                lat.append((time.perf_counter() - t0) * 1000)
            q = statistics.quantiles(lat, n=100)
            single = {"p50_ms": round(q[49], 1), "p95_ms": round(q[94], 1), "cold_start_s": round(cold, 2)}
    best = max(grid, key=lambda r: r["docs_per_s"])
    four = max((r for r in grid if r["threads"] == 4), key=lambda r: r["docs_per_s"])
    summary = {
        "single-doc latency p50 / p95, 4 threads (target p95 <= 150 ms)": f"{single['p50_ms']} / {single['p95_ms']} ms",
        "throughput, 4 threads, batch 32 (target >= 150 docs/s)": f"{next(r['docs_per_s'] for r in grid if r['threads'] == 4 and r['batch'] == 32)} docs/s",
        "best throughput (T used in the README scale tables)": f"{best['docs_per_s']} docs/s (threads={best['threads']}, batch={best['batch']})",
        "best at 4 threads": f"{four['docs_per_s']} docs/s (batch={four['batch']})",
        "cold start (all sessions created), 4 threads": f"{single['cold_start_s']} s",
        "corpus": f"{len(docs)} replayed docs (GDELT, RSS, tweets, Mastodon, EDGAR, HN, Reddit, central banks)",
    }
    out = {
        "provenance": provenance(),
        "machine": machine(),
        "summary": summary,
        "grid": grid,
        "single_doc": single,
        "best_docs_per_s": best["docs_per_s"],
        "best_config": {"threads": best["threads"], "batch": best["batch"]},
    }
    mc = monte_carlo_bench()
    summary["Module B Monte Carlo (10,000 draws x 300 positions, full stress run)"] = (
        f"{mc['median_s']} s median"
    )
    out["monte_carlo"] = mc
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    render_model_card()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=480)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    out = run(args.docs)
    for k, v in out["summary"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
