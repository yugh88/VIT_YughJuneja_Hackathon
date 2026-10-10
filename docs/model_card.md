# Model card: Sentinel

Every number below is produced by a script in this repo (`make eval`, `make bench`, `python -m training.train_impact`). This file is generated; do not edit it by hand.

## 1. Sentiment (uncontaminated test sets first)

_Generated 2026-10-05T13:37:24Z · commit `6cf9e26` (uncommitted changes) · source `02c34ed0c3790413`_

| Test set | Model | Macro-F1 | Accuracy | ECE | n |
| --- | --- | --- | --- | --- | --- |
| Twitter financial news (test) | sentiment-en | 0.8513 | 0.8844 | 0.0302 | 2388 |
| Twitter financial news (test) | sentiment-multi | 0.8401 | 0.874 | 0.0532 | 2388 |
| SEntFiN 1.0 held-out (entity-level) | sentiment-en | 0.8986 | 0.898 | 0.014 | 784 |
| SEntFiN 1.0 held-out (entity-level) | sentiment-multi | 0.8918 | 0.8916 | 0.0519 | 784 |

**FiQA-2018 continuous scores (test, n=234), mean squared error**

| Model | `sentiment.score` (range map) | `sentiment.score_human` (human map) | Target |
| --- | --- | --- | --- |
| sentiment-en | 0.1522 | 0.0704 | <= 0.10 |
| sentiment-multi | 0.1843 | 0.0685 | <= 0.10 |

Every signal carries both: `sentiment.score` (range map; drives Module A and the UI) and `sentiment.score_human` (FiQA-faithful map). The trade-off in one sentence: FiQA's raters score even strong news around 0.4-0.6, so the full-range score costs FiQA MSE (0.15 vs 0.07) in exchange for letting clearly strong headlines reach |score| >= 0.7 (min 0.711, median 0.778 on the 40-headline check; neutral max 0.126). No single map met both (candidates: {'tried': 3465, 'pass_check': 609, 'pass_check_and_budget': 0}); docs/DECISIONS.md D38.

**Multilingual: PROXY result.** PhraseBank test split machine-translated (NLLB-200) into 8 languages; a proxy, not native text. Native human-labelled results replace this once `data/eval/multilingual_to_label.csv` is labelled.

| Language | Macro-F1 | Accuracy | n |
| --- | --- | --- | --- |
| ar | 0.8118 | 0.8447 | 483 |
| de | 0.8375 | 0.8592 | 483 |
| es | 0.8205 | 0.8447 | 483 |
| fr | 0.8209 | 0.853 | 483 |
| hi | 0.8601 | 0.8675 | 483 |
| ja | 0.8298 | 0.8468 | 483 |
| pt | 0.8286 | 0.8489 | 483 |
| zh | 0.8161 | 0.8323 | 483 |
| **mean (8 languages)** | 0.8282 |  |  |

**Multilingual, native headlines (300, 10 languages): AI-annotated, blind, two-pass (event κ = 0.9952, sentiment κ = 0.9939), human spot-check pending.** Not human-labelled. Both passes were made by the same assistant in one session, so κ measures self-consistency, not agreement between independent annotators.

Sentiment macro-F1 0.531 (95% CI [0.4611, 0.5969]); event macro-F1 0.3681 (95% CI [0.2533, 0.4387]), event accuracy 0.52; gap vs English Twitter sentiment 0.3389.

| Language | n | Sentiment macro-F1 | Event macro-F1 |
| --- | --- | --- | --- |
| ar | 30 | 0.5849 | 0.248 |
| de | 30 | 0.5017 | 0.1106 |
| es | 30 | 0.4519 | 0.2583 |
| fr | 30 | 0.685 | 0.3555 |
| hi | 30 | 0.3784 | 0.1159 |
| it | 30 | 0.4683 | 0.1205 |
| ja | 30 | 0.5764 | 0.1869 |
| pt | 30 | 0.4364 | 0.1732 |
| ru | 30 | 0.3557 | 0.1926 |
| zh | 30 | 0.6854 | 0.243 |

**Multilingual, native human labels:** labelled, mean macro-F1 0.5124, gap vs English 0.3389

**Lexicon baselines (same test sets)**

| Test set | LM | VADER | LM + VADER | Engine fallback lexicon | sentiment-en | Gain over LM + VADER |
| --- | --- | --- | --- | --- | --- | --- |
| twitter_test | 0.4467 | 0.447 | 0.4522 | 0.552 | 0.8513 | +0.3991 |
| sentfin_test | 0.4598 | 0.5033 | 0.512 | 0.5116 | 0.8986 | +0.3866 |

**Financial PhraseBank (CONTAMINATED: read with the caveat).** ProsusAI/finbert, the base of sentiment-en, was itself fine-tuned on Financial PhraseBank; these scores are likely optimistic.

| Split | Macro-F1 | Accuracy | n |
| --- | --- | --- | --- |
| all annotators agree (test) | 0.9879 | 0.991 | 223 |
| full 50%-agree (test) | 0.8642 | 0.8778 | 483 |

## 2. Event classification

_Generated 2026-10-05T13:37:24Z · commit `6cf9e26` (uncommitted changes) · source `02c34ed0c3790413`_

Test: twitter-financial-news-topic validation split, topics mapped to our classes (n=4117). Macro-F1 over the 9 classes present: **0.8183**, accuracy 0.8873, ECE 0.041.

No clean test data exists for: Operational/Supply Chain, ESG/Climate, Cyber/Technology Risk. They are trained from weak labels only; the native multilingual set will measure them.

| Class | F1 |
| --- | --- |
| Geopolitical | 0.8819 |
| Macroeconomic | 0.8844 |
| Credit Event | 0.3704 |
| Merger/Acquisition | 0.8069 |
| Product Launch | 0.8826 |
| Earnings/Guidance | 0.9318 |
| Regulatory/Legal | 0.7699 |
| Management Change | 0.9375 |
| Other/Non-event | 0.8992 |

Leakage check (training-only text found in eval sets): {'hindi_idioms_in_eval': 0, 'weak_labels_in_eval': 0}

## 3. Impact score (learned from realised abnormal returns)

_Generated 2026-10-05T10:45:08Z · model `hgb-impact-20261005`_

**Headline:** on the 2024+ test years the learned impact score ranks realised moves with Spearman rho = 0.1154 (95% CI 0.1041-0.1266, which excludes zero), and signals scored >= 8 land in the top decile of realised moves 1.612x as often as chance. News-to-return predictability is known to be weak (most price moves are not explained by same-day headlines), so a small but clearly non-zero rho is the honest result; the target of 0.25 is missed.

**Ablation: where the signal comes from** (test years):

| Variant | Spearman rho |
| --- | --- |
| Learned model (all features) | 0.1154 |
| Same model, no `earnings_window` feature | 0.0834 |
| Learned model, only the 91% of signals outside earnings windows | 0.0302 |
| Heuristic baseline | 0.0434 |

Much of the rho is the earnings calendar, not the headline: outside earnings windows rho is 0.0302. Earnings dates are scheduled weeks ahead, so the feature is known at publication time (no look-ahead).

**Where the signal comes from:** much of this rho is the earnings calendar, not the headline. With the same model and no `earnings_window` feature, test rho is 0.0834; on the 91% of test signals outside an earnings window it is 0.0302. Earnings dates are scheduled weeks ahead, so the feature is known at publication time (no look-ahead).

Label: z = |sum of market-model abnormal returns over d0, d0+1| / sigma60 (residual); sovereigns: max(|index move|, |VIX change|) / sigma60; target and metrics use z adjusted within entity kind (log1p z centred on each kind's 2019-2023 median). Split: train 2019-01-01..2023-12-31, test 2024-01-01.. (hyper-parameters tuned on 2023). Rows: {'all': 91120, 'labelled': 74964, 'train': 44811, 'test': 30153, 'test_entities': 266}.

| Metric (test years) | Learned model | Heuristic baseline | Target |
| --- | --- | --- | --- |
| Spearman rho, predicted vs realised z | 0.1154 (95% CI [0.1041, 0.1266], p=7.7e-90) | 0.0434 | >= 0.25 |
| Spearman rho per entity-day | 0.0919 | 0.0322 |  |
| Spearman rho vs raw z (no kind adjustment) | 0.1316 | 0.0228 |  |
| Precision at impact >= 8 (realised z in top decile) | 0.1678 (n=8880) | 0.2179 (n=312) | >= 2x base rate (0.1041) |
| Lift over base rate | 1.612 | 2.094 | >= 2.0 |

**Monotonicity: mean realised z by predicted score bucket** (chart data)

| Score | n | Mean z | Median z | Share in top decile |
| --- | --- | --- | --- | --- |
| 1 | 2572 | 1.127 | 0.904 | 0.066 |
| 2 | 2711 | 1.126 | 0.847 | 0.077 |
| 3 | 2958 | 1.145 | 0.881 | 0.076 |
| 4 | 3171 | 1.176 | 0.877 | 0.084 |
| 5 | 3163 | 1.167 | 0.881 | 0.083 |
| 6 | 3349 | 1.154 | 0.861 | 0.077 |
| 7 | 3349 | 1.19 | 0.881 | 0.077 |
| 8 | 3240 | 1.217 | 0.912 | 0.081 |
| 9 | 2906 | 1.377 | 0.964 | 0.097 |
| 10 | 2734 | 2.608 | 1.679 | 0.347 |

Top features by permutation importance (drop in test rho): earnings_window 0.0775, burst_z 0.0089, sector 0.0065, log_dollar_volume_20d 0.0056, country_tier 0.0047, day_of_week 0.0035, log_syndication 0.0034, event 0.0031

## 5. Credit Event class (focused work)

Decision rule chosen on validation only (n=3050, 34 Credit; 1052 rule-labelled GDELT and templated EDGAR rows excluded): `{'threshold': 0.55, 'specialist_weight': 0.5, 'backstop': True}`.

| Validation | Macro-F1 | Credit P | Credit R | Credit F1 |
| --- | --- | --- | --- | --- |
| plain classifier | 0.9352 | 0.9394 | 0.9118 | 0.9254 |
| chosen rule | 0.938 | 0.9429 | 0.9706 | 0.9565 |

Test sets, touched once at the end:

| Test set | Variant | Macro-F1 | Credit P | Credit R | Credit F1 (95% CI) | n Credit |
| --- | --- | --- | --- | --- | --- | --- |
| frozen topic test | system | 0.8128 | 0.25 | 0.4545 | 0.3226 [0.0909, 0.5185] | 11 |
| frozen topic test | model_only_argmax | 0.8183 | 0.3125 | 0.4545 | 0.3704 [0.1111, 0.5833] | 11 |
| EDGAR 8-K filed >= 2025 (independent) | system_with_item_override |  | 1.0 | 0.9996 | 0.9998 [0.9994, 1.0] | 2347 |
| EDGAR 8-K filed >= 2025 (independent) | text_model_only |  | 1.0 | 0.9991 | 0.9996 [0.9989, 1.0] | 2347 |
| credit_labelled.csv (AI-annotated) | system | 0.4883 | 0.6667 | 1.0 | 0.8 [0.6792, 0.898] | 24 |
| credit_labelled.csv (AI-annotated) | model_only_argmax | 0.4806 | 0.6571 | 0.9583 | 0.7797 [0.6538, 0.8852] | 24 |

Precision of Credit predictions at confidence >= 0.8, frozen test: 0.3636 (n=11).

`credit_labelled.csv`: 150 English headlines from 2024 on, never used in training or tuning (~60 distress-worded candidates + ~90 lookalikes: affirmations, upgrades, stable outlooks, credit cards, tax/carbon credits, AML fines, loan growth). **AI-annotated, blind, two-pass (κ = 0.9764), human spot-check pending**; not human-labelled. It is deliberately hard, so macro-F1 over all classes is not representative. Module B's credit gate (Credit, confidence >= 0.7, distress term): precision 0.7419, recall 0.9583 (n=31). Target F1 >= 0.65 is met on this set; precision >= 0.80 is not. Most false positives are corporate 'restructuring' (layoffs).

Hard-negative rules (affirmed / stable-positive outlook / upgrade -> not Credit; AML -> Regulatory/Legal), validation only: 0 changes on the objective rows, 2 on all 4102 validation rows (Credit F1 0.9884 -> 0.9922); kept: True.

## 6. Module A: sentiment-tilted index backtest

_Generated 2026-10-05T13:27:42Z · commit `6cf9e26` (uncommitted changes) · source `1e1f0233e295e235`_

Two versions specified before running (DECISIONS D67); chosen on validation (2020-01-01..2023-12-31) by net IR: **v2_surprise**. Test (2024-01-01..2026-08-31) run once for both, costs 10 bp per side, turnover cap 10%.

| Version | Period | Net IR | Gross IR | Avg daily turnover | Placebos beaten | Passes |
| --- | --- | --- | --- | --- | --- | --- |
| v1_sentiment | validation | -1.2836 | -0.1152 | 0.0239 |  |  |
| v1_sentiment | test | -1.3056 | -0.1392 | 0.029 | 74% | **no** |
| v2_surprise | validation | -0.665 | 0.082 | 0.0174 |  |  |
| v2_surprise | test | -1.4262 | -0.6425 | 0.0218 | 33% | **no** |

**Verdict:** Neither version adds value: on the 2024+ test period neither has a positive net information ratio that beats 95% of date-shuffled placebos. The tilt is shown for transparency, not as an edge.

## 7. Module B: event-driven stress testing

Synthetic book (seed 42): 300 positions, $10.0 bn, 120 corporate counterparties, 25 countries. Shocks calibrated from measured anchor-window moves (data/reference/shocks.yaml).

| Scenario | Loss | dEL | RWA before -> after | dCET1 (losses / RWA) | 99% VaR | 97.5% ES |
| --- | --- | --- | --- | --- | --- | --- |
| Geopolitical (impact 9) [RU, SOV:RU] | $72m | $14m | 5.92 -> 8.37 bn | -437.1 bp (-122.3 / -350.5) | $175m | $177m |
| Macroeconomic (impact 8) [US] | $321m | $46m | 5.92 -> 8.53 bn | -743.1 bp (-542.3 / -366.5) | $428m | $430m |
| Credit Event (impact 10) [US, BANK:SIVB] | $22m | $30m | 5.92 -> 9.10 bn | -442.8 bp (-37.0 / -418.7) | $138m | $140m |

**24 Feb 2022 replay** (recorded GDELT, real models, real trigger rule): 11520 docs -> 3107 signals -> 23 stress tests fired; first at 2022-02-24 00:00 UTC. High-impact signals blocked: {'not_corroborated': 26, 'credit': 1, 'merged': 146}.

Trigger: v2 (D66): country/obligor clusters merged within 6 h, tier-1 outlet, credit gate. Ranked alert queue (severity = impact x corroboration), top 10:

| Severity | Cluster | First (UTC) | Merged signals | Top headline |
| --- | --- | --- | --- | --- |
| 200.0 | Geopolitical|UA | 00:00 | 210 | The Daily Herald - Ukraine declares state of emergency, announces comp |
| 200.0 | Geopolitical|RU | 00:00 | 206 | UN Security Council holds emergency meeting over Russia-Ukraine crisis |
| 200.0 | Geopolitical|RU | 06:00 | 229 | Bitcoin Plunges Into 'Extreme Fear' Abyss As Russia Launches Full-Scal |
| 200.0 | Geopolitical|UA | 06:00 | 215 | Putin threatens West with 'consequences greater than any you have face |
| 180.0 | Geopolitical|US | 00:00 | 31 | AP sources: Yemen's Houthis seize another US Embassy staffer |
| 180.0 | Macroeconomic|UA | 01:30 | 20 | Ukraine situation in danger of spiralling into major crisis: India at  |
| 180.0 | Macroeconomic|RU | 06:30 | 32 | Oil tops $100, gas, grains, metals spike as Russia invades Ukraine |
| 180.0 | Macroeconomic|UA | 07:45 | 21 | Oil tops $100, gas, grains, metals spike as Russia invades Ukraine |
| 104.0 | Geopolitical|US | 08:15 | 9 | Russia-Ukraine war hits market: Ruble at record low against US Dollar, |
| 80.0 | Macroeconomic|US | 01:30 | 5 | Wisconsin Dubbed One of the Least Friendly States in America |

| Time (UTC) | Cluster | Impact | Corroboration | Trigger headline | Loss | dCET1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2022-02-24 00:00 | Geopolitical|US | 9 | 8 outlets, burst 0.0 | Посол США в Афинах сообщил об отправке в скором времени 12 вертолетов  | $129.7m | -632.9 bp |
| 2022-02-24 00:00 | Geopolitical|UA | 9 | 13 outlets, burst 0.0 | Украина готова говорить с Россией в разных форматах, заявил Зеленский | $60.3m | -412.1 bp |
| 2022-02-24 00:00 | Geopolitical|RU | 9 | 9 outlets, burst 0.0 | Зеленский заявил, что 23 февраля инициировал разговор с Путиным и не п | $72.5m | -437.1 bp |
| 2022-02-24 01:30 | Macroeconomic|US | 8 | 6 outlets, burst 0.0 | Les enchères pour des éoliennes en mer aux Etats-unis dépassent 1,5 mi | $321.2m | -743.1 bp |
| 2022-02-24 01:30 | Macroeconomic|UA | 9 | 16 outlets, burst 0.0 | Consejo de Seguridad de la ONU a otra reunión urgente sobre Ucrania |  | $381.7m | -833.2 bp |
| 2022-02-24 02:00 | Geopolitical|CN | 8 | 4 outlets, burst 4.7 | China envoy calls for diplomatic solution to Ukraine crisis | $58.6m | -380.6 bp |
| 2022-02-24 04:15 | Macroeconomic|CN | 9 | 6 outlets, burst 3.28 | Китай призвал все стороны вокруг Украины проявить сдержанность | $399.5m | -867.0 bp |
| 2022-02-24 06:00 | Geopolitical|RU | 9 | 185 outlets, burst 1.26 | Russia launches attacks on key Ukrainian cities | $72.5m | -437.1 bp |

## 8. Latency and throughput

_Generated 2026-10-05T11:13:19Z · commit `070e956` (uncommitted changes) · source `a957f99e1892e1d8`_

Machine: Apple M3 Pro, 11 logical cores, Darwin

| Measure | Value |
| --- | --- |
| single-doc latency p50 / p95, 4 threads (target p95 <= 150 ms) | 1.1 / 254.0 ms |
| throughput, 4 threads, batch 32 (target >= 150 docs/s) | 10.0 docs/s |
| best throughput (T used in the README scale tables) | 13.4 docs/s (threads=8, batch=64) |
| best at 4 threads | 10.6 docs/s (batch=1) |
| cold start (all sessions created), 4 threads | 3.51 s |
| corpus | 360 replayed docs (GDELT, RSS, tweets, Mastodon, EDGAR, HN, Reddit, central banks) |
| Module B Monte Carlo (10,000 draws x 300 positions, full stress run) | 0.207 s median |

Throughput grid (docs/s through the full pipeline, sentiment + event + impact):

| ONNX threads | batch | docs/s |
| --- | --- | --- |
| 1 | 1 | 3.3 |
| 1 | 8 | 2.9 |
| 1 | 32 | 2.6 |
| 1 | 64 | 2.5 |
| 2 | 1 | 6.1 |
| 2 | 8 | 5.6 |
| 2 | 32 | 5.1 |
| 2 | 64 | 5.0 |
| 4 | 1 | 10.6 |
| 4 | 8 | 10.4 |
| 4 | 32 | 10.0 |
| 4 | 64 | 9.8 |
| 8 | 1 | 12.1 |
| 8 | 8 | 13.2 |
| 8 | 32 | 13.3 |
| 8 | 64 | 13.4 |

