# Sentinel: Global Risk Signal Engine - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Yugh Juneja
**Contact:** `<college email>`
**College / Campus:** VIT
**Demo video:** `<link>` ([recording script](docs/DEMO_SCRIPT.md))
**Slide Deck Link:** [docs/presentation.pdf](docs/presentation.pdf) (7 slides; also [PPTX with speaker notes](docs/presentation.pptx))

> A keyless, multilingual **AI/NLP Risk Engine** that reads news and social posts and outputs, for every
> company or country mentioned, a **Sentiment Score (-1 to +1)**, an **Event Classification (12 classes)** and an
> **Impact Score (1 to 10)** learned from real market reactions. Both downstream modules are built:
> **Module A** (sentiment-tilted index of 20 S&P 100 stocks) and **Module B** (event-driven stress test of a
> $10 bn synthetic wholesale bank book), with a 6-screen dashboard that runs fully offline.

<p align="center">
  <a href="docs/diagrams/architecture.png"><img src="docs/architecture.png" alt="Sentinel architecture: sources, ingestion, NLP risk engine, signal output, consumers" width="100%"></a>
  <br><sub>Click to open full size. All diagrams: <a href="#diagrams">Diagrams</a></sub>
</p>

| Read this first | |
| --- | --- |
| 🚀 Run it | [Quickstart](#4-quickstart--installation): `docker compose up` then open http://localhost:8000 |
| 📊 Results | [Key results](#5-key-results--domain-impact) · [what we tried and what failed](#what-we-tried-and-what-failed) · [model card](docs/model_card.md) |
| 🧭 Design | [Architecture](#2-architecture--tech-stack) · [diagrams](#diagrams) · [decision log (83 entries)](docs/DECISIONS.md) |
| 🗂 Data | [Dataset used](#3-dataset-used) · [data/SOURCES.md](data/SOURCES.md) · [sample output](data/sample_output/) |

### The brief, point by point

| Requirement in the case study | What Sentinel delivers | Where |
| --- | --- | --- |
| Ingest text from **≥ 2 sources** (e.g. news + Twitter/X) | **9 keyless sources**: GDELT (65 languages), Google News, Yahoo, Economic Times, Mint, SEC 8-K, Fed/ECB/RBI, Mastodon, Reddit, Hacker News, finance tweets | `src/engine/ingest/` |
| **Sentiment Score** -1.0 to 1.0 | Per entity; fine-tuned FinBERT (English) + XLM-R (other languages); macro-F1 **0.851** | `src/engine/nlp/sentiment.py` |
| **Event Classification** (Geopolitical, Macro, Credit, M&A, Product Launch, …) | 12 classes, including all 5 named in the brief; macro-F1 **0.818** | `src/engine/nlp/events.py` |
| **Impact Score** 1–10 | Gradient boosting **trained on 74,964 news → abnormal-return pairs**, with a likely range and drivers | `src/engine/impact/` |
| Output via **API or file** | REST + WebSocket (FastAPI), `signals.jsonl`, hourly Parquet | `src/engine/api/` |
| **≥ 1 downstream module** | **Both**: Module A rebalancer **and** Module B stress test, each with a dashboard | `src/modules/`, `src/web/` |
| Module A: weights up on positive, down on negative sentiment; dashboard of weights over time | ✓ Stacked weights (today, 7 days, 30 days), risk limits, backtest with placebo test | *Index rebalancer* screen |
| Module B: loans/bonds/derivatives book; trigger on high-impact events (e.g. Geopolitical > 7); shocks; before/after dashboard | ✓ 300 positions, trigger = class + impact > 7 + corroboration; calibrated shocks; value before → after, ΔCET1, VaR | *Stress test* screen |
| Deliverables: public repo, demo ≤ 5 min, ≤ 7 slides | Repo · [demo script](docs/DEMO_SCRIPT.md) (built-in 5-min guided tour) · [7-slide deck](docs/presentation.pdf) | |

---

## 1. Project Overview / Problem Statement & Approach

**Problem.** Market-moving risk reaches a bank as text: news wires, regulatory filings, central-bank releases and social
posts, in dozens of languages, far faster than analysts can read them. A risk desk needs four answers within minutes of a
story: *who* is affected, *how* the news reads, *what kind* of event it is, and *how big* it could be for the book.
Keyword alerts and finance lexicons miss context and ignore non-English sources. A single headline should never move
capital on its own.

**Approach.** Sentinel is a streaming pipeline with five layers (diagram above):
1. **Ingest keylessly and never go dark.** Nine connectors, each behind a circuit breaker that fails over to a recorded replay.
2. **Read like an analyst.** Cheap stages first (normalise, language ID across 176 languages, spam filter, MinHash de-duplication, entity linking over 9.7k aliases), then fine-tuned transformers scoring sentiment *per entity* and classifying the event.
3. **Learn impact from the market.** The Impact Score is not a hand-written rule. It is trained on how much stocks and indices actually moved after 74,964 historical news items, with a strict time split.
4. **Act on it.** Module A tilts a 20-stock index within hard limits. Module B fires a corroborated stress test that re-prices a synthetic bank book, computes capital impact and runs a 10,000-path Monte Carlo.
5. **Test honestly.** Frozen and time-split test sets, placebo tests and baselines. Every target and every miss is published, in the [model card](docs/model_card.md) and on the dashboard's *Track record* screen.

**Outcome in one line:** on the replayed news of **24 Feb 2022**, 11,520 articles became 3,107 signals and **23 stress
tests**. The first tests fired at 00:00 UTC from Russian-language headlines, and 8 of the top 9 alerts were war clusters,
each sized in dollars and CET1 basis points.

## 2. Architecture & Tech Stack

### Diagrams

Hand-drawn, Excalidraw-style diagrams, each clickable. Editable `.excalidraw` sources are in
[`docs/diagrams/source/`](docs/diagrams/source/); open them at [excalidraw.com](https://excalidraw.com).

| Diagram | What it shows |
| --- | --- |
| [**System architecture**](docs/diagrams/architecture.png) | Sources → ingestion → 8-stage NLP engine → signal output → consumers |
| [**One headline through the engine**](docs/diagrams/signal-walkthrough.png) | Real output for a Hindi headline at every stage, the JSON produced, and how each module treats it |
| [**Module A flow**](docs/diagrams/module-a-flow.png) | Sentiment → surprise → rebalance decision → tilt → risk limits → weights |
| [**Module B flow**](docs/diagrams/module-b-flow.png) | Trigger gates (class, impact > 7, corroboration, credit gate) → shocks → repricing → capital → dashboard |
| [**How the Impact Score is learned**](docs/diagrams/impact-training.png) | GDELT history → price labels (abnormal returns) → features → model → time-split evaluation |
| [**Dashboard wireframe**](docs/diagrams/dashboard-wireframe.png) | The 6 screens and what each one answers |

### The engine, stage by stage

| # | Stage | How | Code |
| --- | --- | --- | --- |
| 1 | Normalise | Unicode NFKC; strip URLs, handles, publisher suffixes; sentence split in any script | `nlp/normalise.py` |
| 2 | Language ID | fastText `lid.176`; short texts fall back to the source's language | `nlp/langid.py` |
| 3 | Quality filter | Drops promotional social posts (cashtag spam, promo phrases) and counts them | `nlp/quality.py` |
| 4 | De-duplicate | MinHash LSH (128 perms, Jaccard ≥ 0.8); wire copies → `syndication_count` | `nlp/dedupe.py` |
| 5 | Entity linking | Aho-Corasick over 9.7k aliases of 284 entities (S&P 100, Nifty 50, 60 banks, 66 sovereigns) in 20+ languages from Wikidata | `nlp/entities.py` |
| 6 | **Sentiment Score** | `<entity> [SEP] <sentence>` into fine-tuned FinBERT (EN) / XLM-R (8 languages), calibrated to [-1, 1] | `nlp/sentiment.py` |
| 7 | **Event Classification** | Fine-tuned XLM-R, 12 classes; below 0.55 confidence it blends with zero-shot NLI; SEC 8-K items and GDELT CAMEO codes override weak guesses | `nlp/events.py` |
| 8 | **Impact Score** | Gradient boosting on 25 features, with a likely range (IQR) and signed drivers | `impact/` |

Every model stage has a fallback (lexicon, keyword rules, heuristic impact), and `/v1/health` reports which mode is active.
Per-entity aggregate `S_e(t)` = decayed, weighted mean of the entity's signals (half-life 12 h for news, 2 h for social) with a
neutral prior.

<details>
<summary><b>Example output: one RiskSignal (real, abridged)</b></summary>

```json
{
  "entity":    {"id": "TATAMOTORS.NS", "name": "Tata Motors", "country": "IN", "sector": "Consumer Discretionary"},
  "language":  "hi",
  "sentiment": {"score": 0.797, "label": "positive", "confidence": 0.929},
  "event":     {"primary": "Earnings/Guidance", "confidence": 0.988},
  "impact":    {"score": 8, "low": 4, "high": 8, "drivers": ["abs_sentiment=0.80", "sentiment=0.80"]},
  "evidence":  {"sentence": "टाटा मोटर्स का मुनाफा दोगुना हुआ"},
  "novelty": 1.0, "syndication_count": 1,
  "model_versions": {"sentiment": "sentiment-multi-1.0", "event": "event-multi-2.0-credit (model)", "impact": "hgb-impact-20261005"}
}
```
200 more signals: [`data/sample_output/signals_sample.jsonl`](data/sample_output/signals_sample.jsonl).
</details>

### Tech stack and why

| Layer | Choice | Why |
| --- | --- | --- |
| Language / API | Python 3.11, FastAPI, Uvicorn, WebSockets | Async ingestion and push streams in one process |
| NLP models | Hugging Face fine-tunes of `ProsusAI/finbert` and `xlm-roberta-base`; mDeBERTa-XNLI zero-shot fallback | Finance-domain English plus cross-lingual transfer |
| Inference | ONNX Runtime on CPU (FP32 compute, fp16 storage) | No GPU needed; full INT8 cost 2–3 accuracy points, so it was rejected |
| Impact model | scikit-learn HistGradientBoosting | Fast and robust on mixed numeric and categorical features |
| Text tooling | fastText `lid.176`, datasketch MinHash, pyahocorasick | Fast, language-agnostic pre-filtering |
| Storage | JSONL + Parquet (query with DuckDB) | Zero-ops files that any tool can read |
| Dashboard | React 18 + TypeScript + ECharts, fonts and map bundled | Works offline and is served by the API at `/` |
| Ops | Docker, Makefile, GitHub Actions CI (ruff, mypy, pytest, web build, API-key scan) | One-command run, reproducible checks |

### API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/analyze` (`"explain": true` adds word importance) | Score any text in any language |
| `GET /v1/signals?entity=&event=&min_impact=&language=&since=&cursor=&limit=` | Signals, newest first, cursor-paginated |
| `WS /v1/stream?entity=&min_impact=` | Push stream of new signals |
| `GET /v1/entities`, `GET /v1/entities/{id}/aggregate` | Current `S_e(t)`, 24 h trend and the signals behind it |
| `GET /v1/modules/index` (+ `/kappa`, `/pause`, `/resume`, `/backtest`) | Module A |
| `GET /v1/modules/stress`, `POST /v1/modules/stress/run`, `GET /v1/modules/stress/replay/2022-02-24` | Module B |
| `GET /v1/health`, `GET /v1/metrics`, `POST /v1/sources/{name}/kill` and `/revive` | Health, Prometheus metrics, resilience demo |

Interactive docs: http://localhost:8000/docs. Every signal is also appended to `data/output/signals.jsonl` (idempotent ids)
and rolled into hourly Parquet: `duckdb -c "select entity.id, avg(sentiment.score) from 'data/output/signals.jsonl' group by 1"`.

<details>
<summary><b>Module A: tactical index rebalancing (details)</b></summary>

Weights start from float-adjusted market caps (capped at 10%) and tilt with sentiment:
`w = base × clip(exp(κ·S), 0.5, 2.0)`, with κ = 0.5 and a slider on the dashboard. Limits: ≤ 15% per name, ≥ 1% floor,
≤ 35% per sector, ≤ 10% turnover per rebalance. A rebalance happens on an impact ≥ 5 signal or every 15 minutes, with
hysteresis. Two versions were written down **before** the backtest: v1 tilts on sentiment; v2 tilts on the sentiment
*surprise* (S minus the stock's own 30-day mean), using only signals with impact ≥ 6 and hysteresis 0.15. v2 won on 2020–2023
validation and is the live default. [Flow diagram](docs/diagrams/module-a-flow.png).
</details>

<details>
<summary><b>Module B: strategic stress testing (details)</b></summary>

- **Book:** seeded synthetic, $10 bn, 300 positions (loans, revolvers, IG/HY/sovereign bonds, swaps, FX forwards, CDS, options), 120 counterparties, 25 countries, sector mix from the provided transactions dataset.
- **Trigger:** Geopolitical, Macroeconomic or Credit Event, **and** impact > 7, **and** corroboration (≥ 3 outlets or a mention burst within 6 h, with ≥ 1 tier-1 outlet). Credit events also need confidence ≥ 0.7 and a distress term. One test per (country or obligor, event) per 6 h.
- **Shocks:** measured from real anchor windows (Russia–Ukraine 2022, the 2022 tightening cycle, SVB 2023), scaled by impact and targeted by country, region and sector.
- **Repricing:** bonds by duration and convexity; loans by Vasicek-stressed PD; derivatives by DV01, delta, gamma and vega. **Capital:** Basel IRB risk weights give ΔRWA and ΔCET1. **Monte Carlo:** 10,000 correlated draws give 99% VaR and 97.5% ES in 0.21 s.

[Flow diagram](docs/diagrams/module-b-flow.png).
</details>

### Dashboard (6 screens)

| | |
| --- | --- |
| [![Home](docs/screenshots/dashboard/1_home.png)](docs/screenshots/dashboard/1_home.png) **Home**: where risk is building, overall risk level, alerts | [![Stress test](docs/screenshots/dashboard/3b_stress_test_replay_2022-02-24.png)](docs/screenshots/dashboard/3b_stress_test_replay_2022-02-24.png) **Stress test (Module B)**: value before → after, CET1, VaR, loss by asset class (24 Feb 2022 replay) |
| [![Index rebalancer](docs/screenshots/dashboard/4_index_rebalancer.png)](docs/screenshots/dashboard/4_index_rebalancer.png) **Index rebalancer (Module A)**: weights over time, limits, backtest verdict | [![Try it](docs/screenshots/dashboard/5_try_it_hindi.png)](docs/screenshots/dashboard/5_try_it_hindi.png) **Try it**: any headline in any language, with the words that drove the score |

Also: [Live news](docs/screenshots/dashboard/2_live_news.png) · [Track record](docs/screenshots/dashboard/6_track_record.png) ·
[Stress test, live view](docs/screenshots/dashboard/3_stress_test.png) · [wireframe](docs/diagrams/dashboard-wireframe.png).
**Start guided tour** in the header runs a 5-minute walkthrough using only recorded data.

## 3. Dataset Used

All data is **public or synthetic**. No real client data is used, and nothing comes from S&P Global or CRISIL.
Every dataset is cited with its licence and measured row counts in [`data/SOURCES.md`](data/SOURCES.md).

| Data | Nature | Used for | In repo |
| --- | --- | --- | --- |
| Financial PhraseBank, Twitter Financial News (sentiment + topic), FiQA-2018, SEntFiN 1.0 | Public (Hugging Face) | Training and testing sentiment and event models | Downloaded by `make download` |
| GDELT 2.0 GKG + Events, 2019–2026 (2,000 random 15-min windows) + yfinance prices | Public | Impact-model labels (abnormal returns) | Regenerated by `make impact` |
| Live feeds: GDELT, Google News, Yahoo, ET, Mint, Fed/ECB/RBI, SEC 8-K, Mastodon, Reddit, HN | Public, keyless | Real-time ingestion | **Recorded replays** in `data/replay/` (7.5 MB) |
| 24 Feb 2022 news (11,520 GDELT articles, 00:00–12:00 UTC) | Public | Historical replay for Module B | `data/replay/scenario_2022-02-24/` |
| Stock Market Tweets (Apr–Jul 2020) | Public (CC BY 4.0) | Stands in for X/Twitter, which has no keyless API | `data/replay/tweets/` |
| Bank book: 300 positions, 120 counterparties | **Synthetic**, seeded (seed 42) | Module B portfolio | `data/portfolio/` |
| Financial Transactions dataset (13.3 M card transactions) | Public (case-study sample data) | Sector mix of the synthetic book | `data/reference/sector_mix.json` |
| Entity gazetteer (284 entities, 9.7k aliases) | Wikidata (CC0) + hand-written seed | Entity linking | `data/reference/` |
| Evaluation sets (credit 150, multilingual 300 headlines) | Public headlines, annotated blind in two passes | Held-out tests | `data/eval/` |
| Sample engine output | Produced by this repo | Shows the output format without running anything | [`data/sample_output/`](data/sample_output/) |

**Assumptions:**
- The synthetic book is representative of a wholesale bank's mix. Retail card spend is used as a proxy for the sector mix of wholesale exposures.
- Shocks scale linearly with the Impact Score from the anchor episode's measured moves.
- Recorded tweets stand in for live X data.
- Evaluation labels come from a blind, two-pass annotation; a human spot-check of 20 rows per set is pending.
- Large training downloads (`data/raw/`, ~205 MB) are not committed, because they are regenerable from the cited sources.

## 4. Quickstart & Installation

**Runtime:** Python 3.11 and Node 20 (dashboard build only). Tested on macOS (Apple M3 Pro) and Ubuntu (CI).
**No API keys are needed anywhere.** The default demo is **fully offline** and replays recorded sources.

**Option A: Docker (recommended)**
```bash
git clone https://github.com/yugh88/VIT_YughJuneja_Hackathon.git
cd VIT_YughJuneja_Hackathon
docker compose up
# open http://localhost:8000 and press "Start guided tour"   ·   API docs: http://localhost:8000/docs
```

**Option B: local Python**
```bash
git clone https://github.com/yugh88/VIT_YughJuneja_Hackathon.git
cd VIT_YughJuneja_Hackathon
make setup        # venv + pip install -r requirements.txt + models (811 MB, checksummed, resumable) + dashboard build
make demo         # offline replay at 30x on http://localhost:8000
```

The first run downloads three fine-tuned ONNX models (811 MB) from this repo's GitHub Release `models-v1`. Each file is
verified against the SHA-256 in [`data/reference/models_manifest.json`](data/reference/models_manifest.json). If the
download is not possible, every stage falls back and says so in `/v1/health`.

| Command | What it does |
| --- | --- |
| `make demo` | Offline: replays every recorded source at 30x into the API and dashboard on :8000 |
| `make live` | Live keyless sources; each falls back to its replay twin if it fails |
| `make test` | Offline test suite (45 tests); `make lint` runs ruff, mypy and the API-key scan |
| `make backtest` · `make replay-2022` | Module A backtest · Module B replay of 24 Feb 2022 |
| `make eval` · `make bench` | Re-score every model on held-out data and regenerate the [model card](docs/model_card.md) · latency and throughput |
| `make train` · `make impact` | Re-train the NLP models · the impact model, from public data |
| `make deck` · `make diagrams` | Rebuild [the deck](docs/presentation.pdf) · [the diagrams](docs/diagrams/) |

Resilience demo: `curl -X POST localhost:8000/v1/sources/rss/kill`. The health page shows the replay twin taking over.

## 5. Key Results & Domain Impact

All numbers are measured on **held-out data** and regenerated by `make eval`, `make bench` and `make backtest`. Full
tables with confidence intervals: [model card](docs/model_card.md).

### Engine quality vs targets

| Metric (test set) | Result | Baseline | Target | |
| --- | --- | --- | --- | --- |
| Sentiment macro-F1, Twitter financial news (n = 2,388, uncontaminated) | **0.851** · ECE 0.030 | 0.452 (LM + VADER lexicons) | ≥ 0.85 | ✅ |
| Entity-level sentiment macro-F1, SEntFiN held-out (n = 784) | **0.899** | 0.512 | – | ✅ |
| Event classification macro-F1, 9 classes (n = 4,117, near-duplicates removed) | **0.818** · accuracy 0.887 | – | ≥ 0.80 | ✅ |
| Credit Event, 150 hard 2024+ headlines | **F1 0.80** · recall 1.00 · precision 0.67 | – | F1 ≥ 0.65 · P ≥ 0.80 | ✅ / ❌ |
| Impact vs realised abnormal move, 2024+ out of sample (n = 30,153) | **Spearman ρ 0.115** (95% CI 0.104–0.127) | 0.043 (hand-written heuristic) | ≥ 0.25 | ❌ (2.7× baseline) |
| Impact ≥ 8: share of top-decile real moves | **16.8%** vs 10.4% base rate (lift 1.61×) | – | ≥ 2× | ❌ |
| Impact 10 vs 1: mean realised move | **2.61σ vs 1.13σ** (2.3×) | – | rising | partly (flat from 1 to 8) |
| Native multilingual sentiment, 300 headlines in 10 languages | macro-F1 **0.53** (CI 0.46–0.60) | translated proxy said 0.83 | ≥ 0.75 | ❌ |
| Latency p95 / throughput (CPU, 4 threads) | 254 ms / **10–13 docs/s** | – | 150 ms / 150 docs/s | ❌ |

### Downstream modules

| Module | Result |
| --- | --- |
| **B · Stress test**, 24 Feb 2022 replay | 11,520 articles → 3,107 signals → **23 stress tests**, the first at 00:00 UTC. 8 of the top 9 alerts are war clusters (Ukraine, Russia, oil > $100, ruble). Gates blocked 26 uncorroborated signals and merged 146 into open alerts |
| **B · Scenarios** (seeded $10 bn book) | Macro, US, impact 8: **−$321 m** ($10.0 bn → $9.68 bn), ΔCET1 −743 bp, 99% VaR $428 m · Geopolitical, Russia, impact 9: −$72 m, ΔCET1 −437 bp · Credit, SVB, impact 10: −$22 m, ΔEL $30 m · Monte Carlo 10,000 draws in **0.21 s** |
| **A · Index rebalancer** (backtest 2024 → Aug 2026, 10 bp per side, run once) | Risk limits held on **1,673 of 1,673** rebalance days. Net IR −1.43 (v2) / −1.31 (v1); beats 33% / 74% of date-shuffled placebos (bar: 95%). **No proven edge after costs**, and the dashboard says so |

### Efficiency vs a naive approach

| | Naive: score every document | Sentinel |
| --- | --- | --- |
| Documents reaching the transformer models | 100% | **≤ 36%**: about 64% name no tracked entity and exit early; 13% are near-duplicates collapsed into one signal |
| Stress tests on 24 Feb 2022 | one per high-impact headline (hundreds) | **23**, one per corroborated (country or obligor, event) cluster per 6 h |
| Sentiment quality | lexicon macro-F1 0.45 | **0.85** (+0.40) |
| Impact ranking | heuristic ρ 0.043 | **0.115** (2.7×) |

### What we tried and what failed

| Tried | Outcome | Decision |
| --- | --- | --- |
| Full INT8 quantisation for speed | Prediction agreement only 0.89 (FinBERT); accuracy 0.87 → 0.85; XLM-R 0.854 → 0.825 | Rejected: kept FP32 compute with fp16 storage (files halved, agreement 1.00) |
| ONNX Runtime graph fusion | 24.8 → 24.8 docs/s, no gain | Not adopted |
| One sentiment scale for both FiQA and strong headlines | 0 of 3,465 candidate maps met both requirements | Ship two scores: `score` (full range) and `score_human` (FiQA MSE 0.070) |
| Impact model without entity-type adjustment | ρ looked better (0.150), but half of it was country-vs-company, not the news | Adjusted the target per entity type; report 0.115 |
| Impact v1 (no earnings feature) | ρ 0.074 | Added the earnings calendar → 0.115; disclosed that ρ is 0.030 outside earnings windows |
| Event test scores before cleaning | 0.830, inflated by train/test near-duplicates | Removed 1,104 leaking rows; honest score 0.818 |
| Multilingual evaluation on machine-translated text | Proxy macro-F1 0.83 | Native headlines gave 0.53; we report 0.53 |
| Credit Event on the original test split | F1 0.37 (n = 11, about 1/3 of labels wrong) | Built a 150-headline hard set: F1 0.80; precision 0.67 is still below target |
| "Price-move" rule v1 (any move word → event) | Broke 6 validation rows ("surging inflation") | v2 needs a stock subject: 0 validation changes |
| Social-post impact cap v1 (any news within 6 h confirms) | A car-chase story "confirmed" JPMorgan *Chase* tweets | Only the same near-duplicate story from news confirms |
| Module A v1 (sentiment) and v2 (surprise), both pre-registered | Neither beats 95% of placebos | Shipped as a transparent, controlled consumer, not as alpha |

### Domain impact

- **Early, multilingual warning.** On 24 Feb 2022 the first stress tests fired from Russian-language headlines at 00:00 UTC. An English-only keyword system would not have seen them.
- **A capital number, not just a red flag.** Each alert carries loss, ΔEL, ΔRWA, ΔCET1, 99% VaR and 97.5% ES, which a risk committee can act on.
- **Explainable for model-risk review.** Every score has its evidence sentence, the words that drove the sentiment (occlusion), signed impact drivers, a likely range and model versions.
- **Noise control.** Corroboration, tier-1 outlets and 6-hour clustering mean one tweet cannot trigger a capital event.
- **Cheap and portable.** Keyless sources, CPU-only, runs offline. The same signal feeds a portfolio tool (Module A) and a risk tool (Module B).

### Limitations & next steps

Native-language quality (0.53 vs 0.75 target), impact correlation, CPU throughput and credit precision are below target.
Module A has no edge after costs. Next steps:
1. Human-labelled native-language data.
2. Static INT8 or GPU workers behind an accuracy gate. Today 10 M docs/day needs about 90–116 CPU workers.
3. Intraday impact labels.
4. More hard negatives for credit precision.
5. Real exposure feeds in place of the synthetic book.

---

## Repository layout

```
├── README.md · LICENSE (MIT) · requirements*.txt · pyproject.toml · Makefile · Dockerfile · docker-compose.yml
├── src/
│   ├── engine/      ingest/ nlp/ impact/ aggregate/ api/        the AI/NLP Risk Engine
│   ├── modules/     rebalancer/ (Module A)   stress/ (Module B)
│   ├── training/    data download, weak labels, fine-tuning, calibration, impact dataset, evaluation, model card
│   └── web/         React + TypeScript + ECharts dashboard
├── data/            replay/ reference/ portfolio/ eval/ sample_output/ train_extra/  ·  SOURCES.md
├── docs/            presentation.pdf/.pptx · architecture.png · diagrams/ · screenshots/ · model_card.md
│                    DECISIONS.md · DEMO_SCRIPT.md · ATTRIBUTION.md · eval/ (raw result JSON) · deck/ (deck source)
└── tests/           pytest suite (45 offline tests)
```

## Documentation index

| Document | Contents |
| --- | --- |
| [docs/presentation.pdf](docs/presentation.pdf) | 7-slide deck: problem, system design, implementation, results, domain impact, limitations |
| [docs/model_card.md](docs/model_card.md) | Every metric with confidence intervals, per-class and per-language tables (generated) |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 83 logged design decisions and experiments, with evidence |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | Shot-by-shot script for the demo video and the 5-minute live demo |
| [docs/deck/speaker_notes.md](docs/deck/speaker_notes.md) | Pitch script and likely jury questions with answers |
| [data/SOURCES.md](data/SOURCES.md) · [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) | Data sources, licences and citations · models and libraries |

## Data and licence

Public or synthetic data only, and no confidential or client data. Code is MIT licensed; see [`LICENSE`](LICENSE).
Third-party datasets keep their own licences (listed in `data/SOURCES.md`). Financial PhraseBank is CC BY-NC-SA, so it
is used for non-commercial research here.
