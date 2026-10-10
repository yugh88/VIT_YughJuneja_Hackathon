import { useEffect, useState } from "react";
import { dispatch, type Action } from "../api";
import type { View } from "../data";

interface Step { hash: string; view?: View; title: string; text: string; seconds: number; action?: Action }

// Guided ~5-minute tour. Every step uses recorded data and the models on disk: no internet needed.
const STEPS: Step[] = [
  { hash: "home", view: "replay", seconds: 40, title: "1 · What could move markets", text: "This is the morning of 24 Feb 2022, recorded and run through Sentinel: 11,520 stories in dozens of languages, the countries where risk is building, and the alerts that need attention." },
  { hash: "news", view: "replay", seconds: 30, title: "2 · Every story, scored", text: "Each story gets a Sentiment Score (−1 to +1), an Event Classification and an Impact Score from 1 to 10, learned from how prices reacted to past news." },
  { hash: "try", seconds: 35, title: "3 · Any language", action: { type: "analyze-example", arg: "Hindi" }, text: "A Hindi headline is read in Hindi: the company is found, the words that drove the sentiment are highlighted, and the Impact Score explains itself." },
  { hash: "stress", seconds: 55, title: "4 · Replay the invasion", action: { type: "replay-2022" }, text: "The same morning on a 45-second clock. A stress test fires only when a serious story is confirmed by several outlets including a major one. Watch the alerts arrive." },
  { hash: "stress", seconds: 35, title: "5 · What it costs a bank", text: "Each alert re-prices a sample $10 bn portfolio: value before and after, loss by asset class, expected loss, the capital ratio and the 99% Value at Risk." },
  { hash: "stress", seconds: 25, title: "6 · Build your own scenario", action: { type: "custom-scenario" }, text: "The same engine on a scenario you choose: here a severe Credit Event in US banks." },
  { hash: "index", view: "live", seconds: 35, title: "7 · The index, honestly", text: "Positive news raises a stock's weight within risk limits. Tested on 2024+ data it did not beat the cap-weighted index, and the page says so plainly." },
  { hash: "track", seconds: 40, title: "8 · Track record", text: "What worked, measured on news the system had never seen, and where it falls short. The full report for analysts is one click away." },
];

export default function Tour({ go, setView, onExit }: { go: (h: string) => void; setView: (v: View) => void; onExit: () => void }) {
  const [i, setI] = useState(0);
  const [t, setT] = useState(0);
  const [paused, setPaused] = useState(false);
  const step = STEPS[i];

  useEffect(() => {
    if (step.view) setView(step.view);
    go(step.hash);
    setT(0);
    if (step.action) window.setTimeout(() => dispatch(step.action!), 500);
  }, [i]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (paused) return;
    const id = window.setInterval(() => setT((x) => x + 0.25), 250);
    return () => window.clearInterval(id);
  }, [paused]);

  useEffect(() => {
    if (t >= step.seconds) (i < STEPS.length - 1 ? setI(i + 1) : onExit());
  }, [t]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="tour" role="dialog" aria-label="Guided tour">
      <div className="row between"><b>{step.title}</b><span className="muted small">{i + 1} / {STEPS.length}</span></div>
      <div className="progress"><span style={{ width: `${Math.min(100, (t / step.seconds) * 100)}%` }} /></div>
      <p style={{ margin: "6px 0 12px" }}>{step.text}</p>
      <div className="row">
        <button className="btn" onClick={() => setI(Math.max(0, i - 1))} disabled={i === 0}>Back</button>
        <button className="btn" onClick={() => setPaused(!paused)}>{paused ? "Resume" : "Pause"}</button>
        <button className="btn primary" onClick={() => (i < STEPS.length - 1 ? setI(i + 1) : onExit())}>{i < STEPS.length - 1 ? "Next" : "Finish"}</button>
        <span className="spacer" />
        <button className="link" onClick={onExit}>Exit</button>
      </div>
    </div>
  );
}
