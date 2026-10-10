# Demo video script

Two cuts from one recording session:

- **Main video, ~8 min** (the guidelines allow up to 10). Upload to YouTube as **Unlisted** and link it in the README.
- **Live-demo cut, ≤ 5 min** (the problem statement's limit). Use segments 2, 3, 4 and 6 only, or press **Start guided tour**, which runs about 5 minutes on its own.

Before recording: `docker compose up` (or `make demo`) has been running for about 2 minutes, so the replay has filled the screens. The browser is at 1440 × 900, zoom 100%, with a clean profile and no bookmarks bar.

| # | Time | Screen | What to show | What to say |
|---|------|--------|--------------|-------------|
| 1 | 0:00–0:30 | Slide 1 (deck) | Title and the four numbers | "Sentinel turns news and social text in many languages into sentiment, event and impact signals, and drives two modules: an index rebalancer and a bank stress test." |
| 2 | 0:30–1:00 | Terminal | `git clone …`, `docker compose up`, the log shows sources starting; open `http://localhost:8000` | "One command, no API keys. Offline it replays recorded sources; `make live` uses the live feeds." |
| 3 | 1:00–2:00 | **Home** | Counters, the world map (hover Ukraine and the US), overall risk level, alerts | "Stories read, signals found, languages. Each dot is a country; bigger means more stories, colour is the risk level." |
| 4 | 2:00–3:00 | **Live news** | Filter by event, then the *Impact 7+* switch; open one story; point at the spam counter | "Every story is scored: Sentiment −1 to +1, event class, Impact 1 to 10, plus the companies it affects. Promotional posts are dropped and counted." |
| 5 | 3:00–4:00 | **Try it** | Click the *Hindi* example, then type `Apple unveils new iPhone with record pre-orders`, then `tata ki recent news btao` | "Any language. Here it finds Tata Motors in a Hindi headline: +1.00, Earnings, Impact 7. These are the words that drove the score, and these are the plain reasons for the impact score. It also answers company questions in Hinglish." |
| 6 | 4:00–5:30 | **Stress test** → header pill **Replay 24 Feb 2022** → *Replay* button | Alerts queue; portfolio value before → after; CET1; VaR; loss by asset class; shocks and anchor episode | "This is the real news flow from the morning of the invasion. 11,520 articles became 3,107 signals and 23 stress tests, and the top alerts are the war. For each one: $10 bn → $9.94 bn, CET1 12.0% → 9.8% after tax, VaR $160 m, and which asset classes lose." |
| 7 | 5:30–6:00 | **Stress test** → *Build your own scenario* | Credit Event, impact 10, SVB → run | "A custom scenario reprices 300 positions and runs 10,000 Monte Carlo paths in about 0.2 s." |
| 8 | 6:00–7:00 | **Index rebalancer** | Weights over time (7 and 30 days); move the tilt slider; largest changes; risk limits; verdict | "Positive sentiment adds weight, negative removes it, within 15% per stock, 35% per sector and 10% turnover. The backtest shows no edge after costs, and the page says so." |
| 9 | 7:00–7:30 | **Track record** | Credit detection, bigger moves at higher scores, where it falls short | "Every number is from held-out data, including the misses." |
| 10 | 7:30–8:00 | Terminal or `http://localhost:8000/docs` | `curl -X POST localhost:8000/v1/analyze -d '{"text":"…"}'`; `POST /v1/sources/rss/kill` and the health page shows the replay twin taking over | "The same signals are available over REST, WebSocket and JSONL files. When a source dies, its replay twin keeps the stream going." |

## Checklist after upload

- [ ] YouTube visibility **Unlisted** (not Private), and it plays in an incognito window
- [ ] Link pasted into `README.md` (the **Demo Video** line at the top)
- [ ] Captions on (auto-generated is fine) for accessibility
