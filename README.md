# Sentinel: Global Risk Signal Engine

Keyless, multilingual risk signals from news and social text. Each linked company or
country gets a **sentiment score in [-1, 1]**, an **event class** (12 classes) and an
**impact score from 1 to 10**, learned from how markets actually reacted to past news. The
signals drive two downstream modules: **(A)** a tactical rebalancer for a 20-stock S&P 100
index, and **(B)** an event-driven stress test of a synthetic wholesale banking book.

Built for the S&P Global x CRISIL *Code to Connect* 2026 case study.

> **Status: Phase 4 of 5.** The engine and both modules work today: the sentiment-tilted
> index (with backtest and placebo test) and event-driven stress testing of a synthetic
> bank book (with Monte Carlo and a 24 Feb 2022 replay). Results are in the generated
> [model card](docs/model_card.md). The dashboard is next. See [Build status](#build-status).

## Quickstart

```bash
git clone https://github.com/yugh88/VIT_YughJuneja_Hackathon.git
cd VIT_YughJuneja_Hackathon
docker compose up              # or: make setup && make demo
# open http://localhost:8000/docs   (API)   ·   http://localhost:8000/v1/health   (sources, models)
```

**Runtime:** Python 3.11 (Node 20 only for the dashboard build, which Docker does for you).
All dependencies are pinned. **No API keys are needed anywhere.** The demo runs **fully
offline** from recorded replays in `data/replay/`.

**First run downloads 811 MB of models** (three fine-tuned ONNX models, from this repo's
GitHub Release). The download resumes if interrupted, shows a progress bar and is
verified against SHA-256 checksums committed in `data/reference/models_manifest.json`. The
zero-shot event fallback (another 301 MB) is fetched lazily, only if a low-confidence event
needs it. Without models every stage falls back (lexicon, keyword rules, heuristic impact)
and says so in `/v1/health`.

| Command | What it does |
| --- | --- |
| `make demo` | Offline: replays every recorded source at 30x into the API on :8000 |
| `make live` | Live keyless sources; each falls back to its replay twin if it fails |
| `make models` | Download the fine-tuned models (811 MB) + language ID |
| `make test` | Offline test suite (40 tests; `make test-all` adds the 24 Feb 2022 replay) |
| `make backtest`, `make replay-2022` | Module A backtest; Module B historical replay |
| `make eval` | Score every shipped model on held-out data and regenerate `docs/model_card.md` |
| `make bench` | Measure latency and throughput into the model card |
| `make lint` | ruff, mypy and the API-key scan |
| `make train` / `make impact` | Re-train the NLP models / the impact model from public data |

## Results (measured; full details in the [model card](docs/model_card.md))

| What | Result | Target |
| --- | --- | --- |
| Sentiment, Twitter financial news test (uncontaminated) | macro-F1 **0.851**, ECE 0.030 | >= 0.85, ECE <= 0.05 |
| Sentiment, SEntFiN held-out (entity-level) | macro-F1 **0.899** | - |
| Sentiment, FiQA-2018 continuous scores | MSE **0.070** (FiQA-faithful map) / 0.152 (published full-range map) | <= 0.10 |
| Sentiment, 8 languages: **proxy** (machine-translated PhraseBank test) | mean macro-F1 0.828 | >= 0.75 |
| Sentiment, 10 languages, native human labels | pending: 300 headlines prepared for labelling | >= 0.75 |
| Gain over Loughran-McDonald + VADER lexicons | +0.40 macro-F1 (Twitter), +0.39 (SEntFiN) | report |
| Event class, topic test (9 of 12 classes have test data; train/test near-duplicates removed) | macro-F1 **0.818** | >= 0.80 |
| Credit Event, frozen topic test (only 11 examples, ~1/3 mislabelled) | F1 0.37 (95% CI 0.11-0.58), precision 0.31 | F1 >= 0.65, P >= 0.80 (missed) |
| Credit Event, SEC 8-K filings from 2025 (independent, 2,347 credit) | F1 0.9996 text-only; 8-K text is templated, so this only validates the filing path | - |
| Impact vs realised abnormal return, 2024+ out of sample (kind-adjusted z) | Spearman rho **0.074** (95% CI 0.062-0.086); heuristic baseline 0.010 | >= 0.25 (missed) |
| Impact >= 8: share in top decile of realised moves | **0.138** vs base rate 0.104 (lift 1.32x) | >= 2x (missed) |
| Impact monotonicity: mean realised z, score 1 -> 10 | 1.19 -> 1.88; flat from 1 to 9, clear jump at 10 | rising (partly) |
| Single-document latency, 4 threads | p95 253 ms | <= 150 ms (missed) |
| Throughput, mixed news + social, CPU | **14.2 docs/s** best (8 threads); 10-11 docs/s at 4 threads | >= 150 docs/s (missed) |

Financial PhraseBank scores (all-agree accuracy 0.991) are left out of the table on purpose:
the English model's base, ProsusAI/finbert, was itself trained on PhraseBank, so those
numbers are likely optimistic. They appear in the model card with that caveat attached.

Sentiment scale (open decision, `docs/DECISIONS.md` D38): the published map lets clearly strong
headlines reach |score| >= 0.7 (40/40 English check headlines; neutral ones stay <= 0.2). It
costs FiQA MSE (0.152 vs 0.070), because FiQA's human raters score strong news around 0.4-0.6.
`SENTINEL_SENTIMENT_SCALE=human` switches to the FiQA-faithful map.

## Module A: sentiment-tilted index (20 S&P 100 names)

Weights start from float-adjusted market caps (capped at 10%) and tilt with sentiment:
`w = b * clip(exp(kappa * S), 0.5, 2.0)`, then limits are applied: 15% per name, 1% floor, 35% per
sector. Rebalancing happens on an impact >= 5 signal or every 15 minutes, with hysteresis
(only |dS| > 0.10 counts) and a 10% turnover cap. `GET /v1/modules/index` returns current weights,
weights over time and a rebalance log naming the signal behind each move.

**Backtest, 2020-2026** (`make backtest`; 10 bp per side, no look-ahead): the tilt does **not**
add value on our historical signals. Information ratio is -1.48 versus the cap-weighted index on
0.4% tracking error, about what costs explain. It beats only 76% of 100 date-shuffled placebos;
a real effect needs 95%. We report this as it is.

## Module B: event-driven stress testing

A seeded synthetic book: $10 bn, 300 positions (loans, revolvers, IG/HY/sovereign bonds, swaps, FX
forwards, CDS, options), 120 counterparties, 25 countries, sector mix from the provided
transactions dataset. A qualifying signal re-prices it as follows:
- **Trigger:** Geopolitical, Macroeconomic or Credit Event, *and* impact > 7, *and*
  corroboration (>= 3 outlets or a mention burst within 6 h); one test per story cluster
  per 6 h.
- **Shocks:** the event class's shock is measured from a real anchor episode (Ukraine 2022,
  the 2022 tightening cycle, SVB 2023), then scaled by impact and targeted by country, region
  and sector.
- **Repricing:** bonds by duration and convexity, loans by Vasicek-stressed PD, derivatives by
  DV01, delta, gamma and vega.
- **Capital:** Basel IRB risk weights give dRWA and dCET1.
- **Monte Carlo:** 10,000 correlated draws (0.27 s) give 99% VaR and 97.5% ES.

Try `POST /v1/modules/stress/run` with `{"event": "Geopolitical", "impact": 9, "country": "RU"}`, or
`GET /v1/modules/stress/replay/2022-02-24` to watch 24 Feb 2022 flow through the whole system.

## Sources (all keyless)

| Source | Type | Replay twin |
| --- | --- | --- |
| GDELT 2.0 GKG + Events, English and translingual (65 languages) | News | 6 h recorded |
| GDELT DOC 2.0 API | News | Covered by the GKG replay (the API rate-limits with 429; backs off 15 min) |
| Google News (per ticker and a business section in 9 languages), Yahoo Finance, Economic Times, Mint | News | Recorded |
| Fed, ECB, RBI press releases | Central bank | Recorded |
| SEC EDGAR 8-K feed (item codes become weak event labels) | Filing | Recorded |
| Mastodon hashtags, Reddit RSS, Hacker News | Social | Recorded |
| Historical finance tweets (Hugging Face), replayed as a stream | Social | 6,000 tweets, 13 Jul 2020 |

Each source runs behind a circuit breaker with exponential back-off (1 s to 60 s). While
a source is down, its replay twin keeps the stream going. Try it: `POST /v1/sources/rss/kill`.

## NLP engine

Every document goes through seven stages, each a separate module:

| Stage | How | Module |
| --- | --- | --- |
| Normalise | Unicode NFKC; strip URLs, handles and publisher suffixes; split sentences in any script | `nlp/normalise.py` |
| Language ID | fastText `lid.176` (176 languages) | `nlp/langid.py` |
| De-duplicate | MinHash LSH (128 perms, 5-shingles, Jaccard >= 0.8); wire copies collapse into a `syndication_count` | `nlp/dedupe.py` |
| Entity linking | Aho-Corasick over ~9.7k aliases of 284 entities (S&P 100, Nifty 50, 60 banks, 66 sovereigns) in 20+ languages from Wikidata; cashtags; context words for names like "Apple" | `nlp/entities.py` |
| Sentiment in [-1, 1] | Entity-aware `<entity> [SEP] <sentence>`. English uses fine-tuned FinBERT; other languages use fine-tuned XLM-R (translate-train, 8 languages) | `nlp/sentiment.py` |
| Event class (12) | Fine-tuned XLM-R. Below 0.55 confidence it blends with zero-shot NLI. SEC 8-K item codes and GDELT CAMEO codes override weak guesses | `nlp/events.py` |
| Impact 1-10 | Gradient boosting on 25 features, trained on 2019-2023 market reactions (see below) | `impact/` |

### Impact score: learned from what markets did

For each historical (company, headline, time) we measure the **abnormal return** over the
next two trading days (market-model beta from the prior 120 days), scaled by the stock's own
60-day residual volatility: `z = |sum of abnormal returns| / sigma60`. Countries use their index and
VIX moves. Because those two label definitions differ in scale (country labels run higher), the
target is centred within each entity type. Without that adjustment rho looks better (0.150), but
half of it came from "country vs company", not from reading the news. Training data: 2,000 GDELT windows (one per weekday, 2019-2026), giving 74,529 labelled
signals scored by the shipped pipeline, with prices from yfinance. **No look-ahead**: news after
the close maps to the next open, every feature uses data dated before the reaction day, and the
split is by time (train 2019-2023, test 2024 onward).
Each signal shows a range (the interquartile range of what actually happened to past signals
with the same score) and its top drivers.

### Per-entity aggregate (what Module A consumes)

`S_e(t)` = decayed, weighted mean of the entity's signals (weight = confidence x credibility x
novelty; half-life 12 h for news, 2 h for social) with a neutral prior, so thin evidence
shrinks toward 0 and stale scores decay instead of freezing. Social accounts posting the
same cashtag more than 20 times an hour get zero weight.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/analyze` | Score any text in any language: signals plus language, entities, per-class probabilities |
| `GET /v1/signals?entity=&event=&min_impact=&language=&since=&cursor=&limit=` | Signals, newest first, cursor-paginated |
| `GET /v1/entities` | Entities seen so far with their current `S_e(t)` |
| `GET /v1/entities/{id}/aggregate` | Current `S_e(t)`, 24 h trend and the signals driving it |
| `WS /v1/stream?entity=&min_impact=` | Push stream of new signals |
| `GET /v1/health`, `GET /v1/metrics` | Sources, breakers, models and modes; Prometheus metrics |
| `GET /v1/modules/index`, `POST /v1/modules/index/kappa`, `/pause`, `/resume`, `GET /v1/modules/index/backtest` | Module A |
| `GET /v1/modules/stress`, `POST /v1/modules/stress/run`, `GET /v1/modules/stress/results/{id}`, `GET /v1/modules/stress/replay/2022-02-24` | Module B |
| `GET /v1/raw`, `WS /v1/stream/raw` | Raw documents before NLP |
| `POST /v1/sources/{name}/kill`, `/revive` | Resilience demo |

**Files:** every signal is appended to `data/output/signals.jsonl`, and each completed hour is
also written to `data/output/parquet/`. Signal ids are idempotent, so replays never double-count.
Query either file with DuckDB:
`duckdb -c "select entity.id, avg(sentiment.score) from 'data/output/signals.jsonl' group by 1"`.

## Scale

Measured single-worker throughput **T = 14.2 docs/s** (mixed news + social, CPU, 8 ONNX
threads; `make bench`). Peak load is taken as 10x the daily average.

| Daily volume | Average | Peak | CPU workers at T = 14.2 |
| --- | --- | --- | --- |
| 1 M docs/day | 11.6 docs/s | 116 docs/s | 9 |
| 10 M docs/day | 116 docs/s | 1,157 docs/s | 82 |
| 50 M docs/day | 579 docs/s | 5,787 docs/s | 408 |

T is low because the models run in FP32. Full INT8 would be about 3.5x faster on CPU, but cost
2-3 accuracy points on our data (D30), so accuracy won. Levers for scale, all measured or
planned: GPU workers, INT8 behind an accuracy gate, and routing short headlines separately from
long social posts, which cost several times more. Already in place: near-duplicates (13% of the
replayed stream) and documents naming no tracked entity (about 64% in the demo) exit before any
model runs.

## Repository layout

```
src/engine/      ingest/ nlp/ impact/ aggregate/ api/     the signal engine
src/modules/     rebalancer/ stress/                      Module A and Module B (Phase 4)
src/training/    data, translation, weak labels, training, calibration, impact, evaluation
src/web/         React dashboard (Phase 5)
data/            replay/ reference/ eval/ train_extra/ output/ and SOURCES.md
docs/            model card, DECISIONS.md, attribution, deck
tests/           pytest suite
```

## Build status

- [x] **Phase 1**: skeleton, schema, config, Makefile, Docker, CI, data download, replay recorder, 9 connectors with circuit breakers
- [x] **Phase 2**: normalise, language ID, de-duplication, entity linking, sentiment and event models (ONNX)
- [x] **Phase 3**: learned impact model, aggregation, signal WebSocket, JSONL/Parquet sink, model release + downloader, `make eval`, `make bench`
- [x] **Phase 4**: Module A (rebalancer, backtest, placebo) and Module B (synthetic book, calibrated shocks, IRB capital, Monte Carlo, trigger, 24 Feb 2022 replay); Credit Event work
- [ ] **Phase 5**: dashboard, Demo Mode, final docs

## Data and licence

Public or synthetic data only, and no real client data. Every dataset is cited with its
licence and measured row counts in [`data/SOURCES.md`](data/SOURCES.md); models and
libraries in [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md). Code is MIT licensed; see
[`LICENSE`](LICENSE).
