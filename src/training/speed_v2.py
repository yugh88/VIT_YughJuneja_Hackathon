"""Speed measurements for the score-improvement round (DECISIONS D75); CPU, single process.

    python -m training.speed_v2       # -> docs/eval/v2/speed.json

Configurations on the benchmark corpus (360 replayed docs, all sources, batch 32):
* before: no logits cache, no NLI sentiment blend (the pipeline before this round);
* now:    the shipped pipeline (cache on, NLI blend for non-English sentiment);
* now + CoreML: SENTINEL_ONNX_PROVIDER=coreml (Mac only; CPU stays the default for everyone else).
Also: how often the zero-shot event fallback fires (event confidence < 0.55), and CoreML accuracy on
the dev sets compared with CPU (must stay within 0.005).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from typing import Any

from engine.config import REPO_ROOT, get_settings

log = logging.getLogger("speed_v2")
OUT = REPO_ROOT / "docs" / "eval" / "v2" / "speed.json"


def _pipeline(threads: int, provider: str, nli: bool, cache: bool) -> Any:
    os.environ["SENTINEL_ONNX_THREADS"] = str(threads)
    os.environ["SENTINEL_ONNX_PROVIDER"] = provider
    get_settings.cache_clear()
    from engine.nlp import onnx_model
    from engine.nlp.pipeline import Pipeline

    onnx_model.OnnxClassifier.CACHE_SIZE = 50_000 if cache else 0
    p = Pipeline(zero_shot=False, dedupe=False)
    if not nli:
        p.sentiment._nli = None
    return p


def measure(
    threads: int, provider: str = "", nli: bool = True, cache: bool = True, passes: int = 2
) -> dict[str, Any]:
    from engine.bench import corpus

    docs = corpus(360)
    p = _pipeline(threads, provider, nli, cache)
    p.process_batch(docs[:32], dedupe=False)  # warm-up
    t0 = time.perf_counter()
    for _ in range(passes):  # a second pass shows what the cache saves on repeated text
        for i in range(0, len(docs), 32):
            p.process_batch(docs[i : i + 32], dedupe=False)
    dt = time.perf_counter() - t0
    lat = []
    for d in docs[:200]:
        t1 = time.perf_counter()
        p.process_batch([d], dedupe=False)
        lat.append((time.perf_counter() - t1) * 1000)
    q = statistics.quantiles(lat, n=100)
    first = None
    if passes > 1:  # throughput of the first pass alone (no repeats), measured separately
        p2 = _pipeline(threads, provider, nli, cache)
        p2.process_batch(docs[:32], dedupe=False)
        t2 = time.perf_counter()
        for i in range(0, len(docs), 32):
            p2.process_batch(docs[i : i + 32], dedupe=False)
        first = round(len(docs) / (time.perf_counter() - t2), 1)
    return {"threads": threads, "provider": provider or "cpu", "nli_blend": nli, "cache": cache,
            "docs_per_s_unique": first, "docs_per_s_with_repeats": round(passes * len(docs) / dt, 1),
            "p50_ms": round(q[49], 1), "p95_ms": round(q[94], 1)}  # fmt: skip


def zero_shot_rate() -> dict[str, Any]:
    from engine.bench import corpus

    p = _pipeline(4, "", True, True)
    traces = p.process_batch(corpus(360), dedupe=False)
    ev = [t.event for t in traces if t.event is not None]
    low = sum(1 for e in ev if e.confidence < 0.55)
    return {"docs_with_event": len(ev), "below_0.55": low, "share": round(low / max(len(ev), 1), 4)}


def coreml_accuracy() -> dict[str, Any]:
    from training.eval_v2 import CLASSES, SENT, Native, credit_metrics, load, native_metrics

    out: dict[str, Any] = {}
    for prov in ("", "coreml"):
        os.environ["SENTINEL_ONNX_PROVIDER"] = prov
        get_settings.cache_clear()
        d = load("native", "dev")
        nat = Native()
        ps, pe, _, _ = nat.probs(d["headline"].tolist())
        m = native_metrics(d, [SENT[i] for i in ps.argmax(1)], [CLASSES[i] for i in pe.argmax(1)])
        c = load("credit", "dev")
        _, _, ev, conf = nat.probs(c["headline"].tolist(), lang="en")
        cm = credit_metrics(c, ev, conf)
        out[prov or "cpu"] = {"native_sentiment_f1": m["sentiment_macro_f1"], "native_event_f1": m["event_macro_f1"],
                              "credit_f1": round(cm["f1"], 4)}  # fmt: skip
    os.environ["SENTINEL_ONNX_PROVIDER"] = ""
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    res: dict[str, Any] = {
        "spec": "DECISIONS D75",
        "corpus": "360 replayed docs, all sources, batch 32",
        "runs": [],
    }
    for threads in (4, 8):
        res["runs"].append(
            measure(threads, nli=False, cache=False) | {"config": "before (no cache, no NLI blend)"}
        )
        res["runs"].append(measure(threads) | {"config": "now (cache, NLI blend)"})
    try:
        res["runs"].append(measure(8, provider="coreml") | {"config": "now + CoreML (Mac only)"})
        res["coreml_accuracy_dev"] = coreml_accuracy()
    except Exception as e:  # CoreML provider missing or failing: report, keep the CPU default
        res["coreml_error"] = repr(e)
    res["zero_shot_fallback"] = zero_shot_rate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
