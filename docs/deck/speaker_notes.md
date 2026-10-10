# Speaker notes: live pitch, about 5 minutes

One section per slide. `make deck` copies each section into the PPTX speaker notes.

## 1
(20 s) I'm Yugh Juneja from VIT. This is Sentinel, an AI/NLP risk engine. It reads news and social posts in many languages and turns each story into three numbers per company or country: a sentiment score, an event class and an impact score. Those numbers drive both downstream modules from the brief, a sentiment-tilted index and a bank stress test. Four numbers to remember: nine keyless sources, 0.85 sentiment F1, 23 automatic stress tests on the morning Russia invaded Ukraine, and a full Monte Carlo in a fifth of a second.

## 2
(40 s) The problem in my own words: a risk desk must answer four questions minutes after a story breaks. Who is affected, how does it read, what kind of event is it, and how big is it for our book. Keyword alerts can't do this, and they ignore non-English sources. So the approach has four steps. Ingest from free sources that never go dark. Read each sentence per entity with fine-tuned transformers. Learn the impact score from how markets actually moved after past news. Then act on it, and test honestly. The bottom row maps every requirement in the brief to what we built; we built both modules, not one.

## 3
(45 s) Here's the system, left to right. Nine connectors, each behind a circuit breaker that fails over to a recorded replay, which is why the whole demo runs offline. The core engine has eight stages. The first five are cheap: normalise, language ID across 176 languages, a spam filter, near-duplicate removal, and entity linking over 9,700 aliases. Only documents about a tracked company or country reach the three models, which produce the three required outputs in yellow. The output is one JSON RiskSignal, served over REST and WebSocket and written to JSONL and Parquet. Module A subscribes to sentiment, Module B to event plus impact, and the dashboard reads everything.

## 4
(40 s) Each choice is backed by a number. The fine-tuned models beat finance lexicons by 40 F1 points. The impact model ranks real market moves 2.7 times better than a hand-written rule. Cheap filters mean at least 64% of documents never reach a model. We kept FP32 because INT8 cost accuracy. The trigger needs corroboration from several outlets, so 3,107 signals became 23 stress tests, not 3,107. Shocks are calibrated on real episodes: Ukraine 2022, the 2022 rate hikes and SVB.

## 5
(40 s) Results, all on held-out data. Sentiment macro-F1 is 0.851 on a test set the base model never saw, against 0.45 for lexicons. Entity-level sentiment is 0.90 and event classification 0.82, both above target. Credit events: 100% recall on 150 deliberately hard headlines. On the right, the honest picture of the impact score: on 30,000 out-of-sample news items, a score of 10 was followed by moves 2.3 times bigger than a score of 1. Scores 1 to 8 are nearly flat, and I'll come back to that.

## 6
(50 s) Why it matters. This is real recorded news from 24 February 2022, replayed through the real models. 11,520 articles became 3,107 signals and 23 stress tests, and 8 of the top 9 alerts are war clusters. The first tests fired at midnight UTC from Russian-language headlines, which an English-only system would have missed. Each alert gives the risk committee a capital number, not just a red flag: loss, expected loss, change in risk-weighted assets, CET1 impact, VaR and expected shortfall. Every score carries its evidence and drivers, so it can be explained in a model-risk review. It runs keyless on a CPU.

## 7
(35 s) Finally, what didn't work. We met the English sentiment, event and credit-F1 targets. We missed native multilingual quality, impact correlation, latency and throughput, and the index module shows no edge after costs. We say so on the dashboard. The most important lesson: a machine-translated test said 0.83 for multilingual; real native headlines said 0.53. We report 0.53. Next steps: human-labelled native data, INT8 or GPU behind an accuracy gate, intraday impact labels, and real exposure feeds. Thank you. Happy to go into any part of the code.

## Likely jury questions (keep ready)

- **Why is impact ρ only 0.115?** Same-day headlines explain little of next-day returns in the literature. Ours is clearly non-zero (95% CI 0.104–0.127) and 2.7× the heuristic. Most of it comes from the earnings calendar; outside earnings windows it is 0.030. That is stated in the model card.
- **Isn't the bank book fake?** It is synthetic and seeded (seed 42), as the brief allows: 300 positions, 120 counterparties, 25 countries, with the sector mix taken from the provided transactions dataset. Shocks come from measured market moves in real anchor windows.
- **Why not an LLM?** Cost, latency and determinism. Fine-tuned encoders give calibrated probabilities and need no API key. Every score is reproducible and explainable by occlusion.
- **How do you avoid look-ahead?** News after the close maps to the next open. Every feature uses data dated before the reaction day. The split is by time (train 2019–2023, test 2024+), and the test set was touched once.
- **What if a source goes down?** Its circuit breaker opens and the replay twin keeps the stream going. Live demo: `POST /v1/sources/rss/kill`.
- **How would it scale to 10 M docs/day?** About 116 docs/s on average and 1,157 at peak. That is roughly 90–116 CPU workers today, fewer with INT8 or GPU behind an accuracy gate. Ingestion and models scale horizontally.
