import { useCallback, useEffect, useState } from "react";
import Tour from "./components/Tour";
import { DataProvider, servingRecorded, useData } from "./data";
import Home from "./screens/Home";
import ModelLab from "./screens/ModelLab";
import News from "./screens/News";
import Rebalancer from "./screens/Rebalancer";
import StressTest from "./screens/StressTest";
import System from "./screens/System";
import TrackRecord from "./screens/TrackRecord";
import TryIt from "./screens/TryIt";

const NAV: [string, string][] = [
  ["home", "Home"], ["news", "Live news"], ["stress", "Stress test"], ["index", "Index rebalancer"], ["try", "Try it"], ["track", "Track record"],
];

function parse(): [string, string | undefined] {
  const h = location.hash.slice(1);
  const i = h.indexOf("/");
  return i < 0 ? [h || "home", undefined] : [h.slice(0, i), h.slice(i + 1)];
}

function Shell() {
  const { data, view, setView, health } = useData();
  const [[screen, param], setRoute] = useState(parse);
  const [tour, setTour] = useState(false);
  const [, tick] = useState(0);

  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener("hashchange", on);
    const id = window.setInterval(() => tick((x) => x + 1), 20000); // refresh "updated N min ago"
    return () => { window.removeEventListener("hashchange", on); window.clearInterval(id); };
  }, []);
  const go = useCallback((h: string) => {
    if (location.hash.slice(1) !== h) location.hash = h;
    else setRoute(parse());
    window.scrollTo({ top: 0 });
  }, []);

  const mins = Math.max(0, Math.round((Date.now() - data.updatedAt) / 60000));
  const recorded = servingRecorded(health);
  return (
    <>
      <header className="header">
        <div className="logo">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 2.5 4 5.5v6c0 4.6 3.3 8.8 8 10 4.7-1.2 8-5.4 8-10v-6l-8-3Z" stroke="#38bdf8" strokeWidth="2" strokeLinejoin="round" />
            <path d="m8.5 12 2.5 2.5 4.5-5" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Sentinel
        </div>
        <nav className="nav" aria-label="pages">
          {NAV.map(([id, label]) => (
            <button key={id} className={screen === id ? "active" : ""} aria-current={screen === id ? "page" : undefined} onClick={() => go(id)}>{label}</button>
          ))}
        </nav>
        <span className="spacer" />
        <button
          className="status"
          onClick={() => setView(view === "live" ? "replay" : "live")}
          title={view === "live" ? "Switch the Home, Live news and Stress test pages to the recorded morning of 24 Feb 2022" : "Back to the live engine"}
          style={{ cursor: "pointer" }}
        >
          <span className={`dot ${view === "replay" || recorded ? "replay" : ""}`} />
          {view === "replay" ? "Replaying 24 Feb 2022" : recorded ? "Replay mode · recorded sources" : `Live · updated ${mins === 0 ? "just now" : `${mins} min ago`}`}
        </button>
        <button className="tour-btn" onClick={() => setTour(true)}>Start guided tour</button>
      </header>
      {data.error && !health && (
        <div style={{ background: "#fff4e5", padding: "10px 28px", fontWeight: 600 }}>The risk engine is not reachable. Start it with <code>make demo</code>.</div>
      )}
      {screen === "home" && <Home go={go} />}
      {screen === "news" && <News go={go} />}
      {screen === "stress" && <StressTest param={param} go={go} />}
      {screen === "index" && <Rebalancer go={go} />}
      {screen === "try" && <TryIt go={go} param={param} />}
      {screen === "track" && <TrackRecord go={go} />}
      {screen === "lab" && <ModelLab theme="light" />}
      {screen === "system" && <System health={health} refresh={() => undefined} />}
      {tour && <Tour go={go} setView={setView} onExit={() => setTour(false)} />}
    </>
  );
}

export default function App() {
  return (
    <DataProvider>
      <Shell />
    </DataProvider>
  );
}
