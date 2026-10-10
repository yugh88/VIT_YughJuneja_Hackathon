// Plain-language labels. Nothing here invents a number: each function only renames or
// explains a value the engine produced.

export const LANG: Record<string, string> = {
  en: "English", hi: "Hindi", ja: "Japanese", zh: "Chinese", de: "German", fr: "French", es: "Spanish",
  ar: "Arabic", pt: "Portuguese", ru: "Russian", it: "Italian", ko: "Korean", tr: "Turkish", nl: "Dutch",
  pl: "Polish", uk: "Ukrainian", id: "Indonesian", vi: "Vietnamese", th: "Thai", fa: "Persian", he: "Hebrew",
  bn: "Bengali", ta: "Tamil", mr: "Marathi", ur: "Urdu", sv: "Swedish", el: "Greek", ro: "Romanian", cs: "Czech",
  hu: "Hungarian", fi: "Finnish", da: "Danish", no: "Norwegian", ms: "Malay", sr: "Serbian", bg: "Bulgarian",
  hr: "Croatian", sk: "Slovak", ca: "Catalan", az: "Azerbaijani", ka: "Georgian", hy: "Armenian", kk: "Kazakh", ml: "Malayalam", te: "Telugu", kn: "Kannada", gu: "Gujarati", pa: "Punjabi",
  ne: "Nepali", si: "Sinhala", sw: "Swahili", tl: "Tagalog", my: "Burmese", km: "Khmer", lt: "Lithuanian", lv: "Latvian",
  et: "Estonian", sl: "Slovenian", mk: "Macedonian", sq: "Albanian", be: "Belarusian", uz: "Uzbek", mn: "Mongolian",
  am: "Amharic", ha: "Hausa", so: "Somali", ps: "Pashto", is: "Icelandic", ga: "Irish", af: "Afrikaans", eu: "Basque", gl: "Galician",
};
export const langName = (c: string) => (c === "und" ? "Language unclear" : LANG[c] ?? c.toUpperCase());

const OUTLETS: Record<string, string> = {
  "reuters.com": "Reuters", "bloomberg.com": "Bloomberg", "ft.com": "Financial Times", "wsj.com": "WSJ",
  "apnews.com": "AP", "bbc.co.uk": "BBC", "bbc.com": "BBC", "cnbc.com": "CNBC", "cnn.com": "CNN",
  "nikkei.com": "Nikkei", "asia.nikkei.com": "Nikkei Asia", "economictimes.indiatimes.com": "Economic Times",
  "livemint.com": "Mint", "business-standard.com": "Business Standard", "aljazeera.com": "Al Jazeera",
  "theguardian.com": "The Guardian", "nytimes.com": "New York Times", "washingtonpost.com": "Washington Post",
  "dw.com": "DW", "france24.com": "France 24", "euronews.com": "Euronews", "tass.ru": "TASS", "ria.ru": "RIA",
  "scmp.com": "SCMP", "sec.gov": "SEC filing", "rbi.org.in": "Reserve Bank of India", "federalreserve.gov": "Federal Reserve",
  "ecb.europa.eu": "ECB", "news.ycombinator.com": "Hacker News", "twitter.com": "Twitter/X",
};

export function outletName(outlet: string, source: string): string {
  const o = (outlet || "").toLowerCase().replace(/^www\./, "");
  if (OUTLETS[o]) return OUTLETS[o];
  if (o.startsWith("reddit.com")) return "Reddit";
  if (source === "mastodon") return "Mastodon";
  if (source === "edgar_8k") return "SEC filing";
  return o || source;
}

export const EVENT_SHORT: Record<string, string> = {
  "Earnings/Guidance": "Earnings",
  "Other/Non-event": "Other",
  "Operational/Supply Chain": "Operations",
  "Cyber/Technology Risk": "Cyber",
  "Regulatory/Legal": "Regulatory",
  "Merger/Acquisition": "Merger/Acquisition",
  "Management Change": "Management",
  "ESG/Climate": "ESG/Climate",
};
export const eventShort = (e: string) => EVENT_SHORT[e] ?? e;

/** Chip colour per event class (orange = risk events, amber = macro, blue = corporate). */
export function eventTone(e: string): string {
  if (e === "Geopolitical") return "orange";
  if (e === "Macroeconomic") return "amber";
  if (e === "Credit Event") return "";
  if (e === "Other/Non-event") return "";
  return "blue";
}

export const ASSET_LABELS: Record<string, string> = {
  term_loan: "Corporate loans", revolver: "Credit lines", bond_ig: "Investment-grade bonds", bond_hy: "High-yield bonds",
  sovereign_bond: "Government bonds", irs: "Interest-rate swaps", fx_forward: "FX forwards", cds: "Credit default swaps",
  equity_option: "Equity options",
};

/** Shock chips for the stress screen: factor moves in the units the engine uses (data/reference/shocks.yaml). */
export function shockChips(f: Record<string, number>, emFx?: number): { label: string; tone: string }[] {
  const pctv = (v: number) => `${v > 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(0)}%`;
  const bp = (v: number) => `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(0)} bp`;
  const out: { label: string; tone: string; mag: number }[] = [
    { label: `Equity ${pctv(f.equity ?? 0)}`, tone: (f.equity ?? 0) < 0 ? "orange" : "blue", mag: Math.abs(f.equity ?? 0) * 100 },
    { label: `Banks ${pctv(f.financials ?? 0)}`, tone: (f.financials ?? 0) < 0 ? "orange" : "blue", mag: Math.abs(f.financials ?? 0) * 100 },
    { label: `Oil ${pctv(f.oil ?? 0)}`, tone: "orange", mag: Math.abs(f.oil ?? 0) * 100 },
    { label: `High-yield spreads ${bp(f.hy ?? 0)}`, tone: (f.hy ?? 0) > 0 ? "orange" : "blue", mag: Math.abs(f.hy ?? 0) / 10 },
    { label: `IG spreads ${bp(f.ig ?? 0)}`, tone: (f.ig ?? 0) > 0 ? "orange" : "blue", mag: Math.abs(f.ig ?? 0) / 10 },
    { label: `EM spreads ${bp(f.em_spread ?? 0)}`, tone: (f.em_spread ?? 0) > 0 ? "orange" : "blue", mag: Math.abs(f.em_spread ?? 0) / 10 },
    { label: `Rates ${bp(f.rates ?? 0)}`, tone: "blue", mag: Math.abs(f.rates ?? 0) / 10 },
    { label: `USD ${pctv(f.usd ?? 0)}`, tone: "", mag: Math.abs(f.usd ?? 0) * 100 },
    { label: `VIX ${(f.vix ?? 0) > 0 ? "+" : "−"}${Math.abs(f.vix ?? 0).toFixed(1)} pts`, tone: (f.vix ?? 0) > 0 ? "orange" : "blue", mag: Math.abs(f.vix ?? 0) / 2 },
  ];
  if (emFx) out.push({ label: `EM FX ${pctv(-Math.abs(emFx))}`, tone: "orange", mag: Math.abs(emFx) * 100 });
  return out.filter((c) => c.mag >= 0.5).sort((a, b) => b.mag - a.mag).slice(0, 7);
}

/** "Why Impact Score N?": the impact model's top drivers (occlusion vs the training median) in words. */
export function driverText(d: string): string {
  const [k, raw] = d.split("=");
  const v = Number(raw);
  switch (k) {
    case "event": return `${raw} news has historically moved prices more than routine news.`;
    case "earnings_window": return "The company reports earnings within a day, when prices move more.";
    case "burst_z": return `Mentions of this name are bursting (${raw} standard deviations above its normal rate).`;
    case "outlets_window": return `Several outlets reported it in the same window (${raw}).`;
    case "log_syndication": return `Repeated by about ${Math.round(Math.expm1(v)) + 1} outlets.`;
    case "novelty": return v >= 0.8 ? "The story is new, not a repeat of earlier news." : "Partly repeats earlier news.";
    case "abs_sentiment": return `Strong tone in the text (|sentiment| ${raw}).`;
    case "sentiment": return `Sentiment of ${raw} for this name.`;
    case "vol_20d": return "The stock or market has been volatile in the last 20 days.";
    case "log_dollar_volume_20d": return "Trading volume in this name is unusual for its size.";
    case "country_tier": return "The country's market type (developed, emerging, frontier) matters for the size of moves.";
    case "sector": return `Sector: ${raw}.`;
    case "kind": return `Kind of entity: ${raw}.`;
    case "market_open": return v ? "Published while the market was open." : "Published outside market hours; the move shows at the next open.";
    case "day_of_week": return "Day of the week (moves differ across weekdays).";
    case "severity_hits": return `Contains severe words (${raw}), such as "default", "war" or "collapse".`;
    case "credibility": return `Source credibility ${raw}.`;
    case "event_confidence": return `The event class is clear (confidence ${raw}).`;
    case "confidence": return `The sentiment reading is confident (${raw}).`;
    case "goldstein": case "abs_goldstein": return `GDELT rates the event's intensity at ${raw} on the Goldstein scale.`;
    case "log_num_mentions": case "log_num_sources": return "GDELT counts many mentions or sources for this event.";
    case "avg_tone": return `GDELT measured an average tone of ${raw}.`;
    case "source_type": return `Source type: ${raw}.`;
    default: return d;
  }
}

export const SECTOR_SHORT = (s?: string | null) => (s ?? "").replace("Information Technology", "Technology");

// ------------------------------------------------------------------ Try it: plain-language helpers

const HINGLISH = new Set(["hai", "hain", "ka", "ki", "ke", "ko", "mein", "kya", "hua", "nahi", "aur", "se", "par", "badha", "badhi",
  "ghata", "ghati", "munafa", "nuksan", "nuksaan", "batao", "btao", "bataiye", "kar", "raha", "rahi", "gaya", "gayi", "bazaar",
  "girawat", "tezi", "mandi", "accha", "bahut", "khabar", "abhi", "aaj", "wala", "wali"]);

const NON_LATIN = new Set(["ja", "zh", "ko", "ar", "hi", "ru", "el", "he", "th", "fa", "uk", "bg", "sr", "mk", "be", "kk", "ka", "hy", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur", "ne", "si", "my", "km", "am", "ps"]);

/** Language label that never shows a low-confidence guess as fact. */
export function languageLabel(text: string, code: string, conf: number): string {
  const latin = /[a-z]/i.test(text) && !/[ऀ-ॿ]/.test(text);
  const words = text.toLowerCase().match(/[a-z]+/g) ?? [];
  if (latin && words.some((w) => HINGLISH.has(w))) return "Hinglish (Hindi in Latin script)";
  if (conf < 0.6) return "Language unclear";
  // Latin-script text "detected" as a language written in another script is a guess, not a reading.
  if (latin && NON_LATIN.has(code)) return "Language unclear";
  return `${langName(code)} (${Math.round(conf * 100)}% sure)`;
}

const EVENT_UP: Record<string, string> = {
  "Earnings/Guidance": "Earnings news usually moves this kind of stock.",
  "Credit Event": "Defaults and downgrades usually hit prices hard.",
  Geopolitical: "War, sanctions and diplomacy news can move whole markets.",
  Macroeconomic: "Interest-rate, inflation and growth news moves broad markets.",
  "Merger/Acquisition": "Deal news often moves the shares involved.",
  "Regulatory/Legal": "Fines, rulings and probes often move the company's shares.",
  "Management Change": "Leadership changes can move the shares.",
  "Product Launch": "Product news can move the shares when it is big enough.",
  "Operational/Supply Chain": "Strikes, recalls and outages can hit output and the shares.",
  "Cyber/Technology Risk": "Cyber attacks and outages can hit the shares.",
  "ESG/Climate": "Climate and disaster news can hit the companies involved.",
  "Other/Non-event": "General news.",
};

interface Effect { feature: string; value: string; effect: number }

/** One effect of the impact model, in words. `up` = it raised the score. */
function effectText(e: Effect, up: boolean, sentiment: number): string | null {
  const v = Number(e.value);
  const tone = `${sentiment >= 0 ? "positive" : "negative"} tone (Sentiment Score ${sentiment >= 0 ? "+" : "−"}${Math.abs(sentiment).toFixed(2)})`;
  switch (e.feature) {
    case "event":
      return up ? EVENT_UP[e.value] ?? `${e.value} news tends to move prices.` : `${eventShort(e.value)} news rarely moves prices much.`;
    case "abs_sentiment":
    case "sentiment":
      return up ? `Strong ${tone}.` : `Mild ${tone}, so a smaller reaction is expected.`;
    case "log_syndication": {
      const n = Math.round(Math.expm1(v)) + 1;
      return up ? `Reported by ${n} outlets.` : `Only ${n === 1 ? "one outlet has" : `${n} outlets have`} reported it so far.`;
    }
    case "outlets_window": return up ? `Reported by ${Math.round(v)} outlets within the hour.` : "Few outlets covered it in the same hour.";
    case "burst_z": return up ? "Mentions of this name jumped far above normal." : "No unusual jump in mentions of this name.";
    case "novelty": return up ? "A new story, not a repeat of earlier news." : "Repeats earlier news, so markets may already have priced it in.";
    case "earnings_window": return v >= 1 ? (up ? "The company reports results around now, when prices move more." : null) : up ? null : "Not near an earnings date, when moves are usually smaller.";
    case "vol_20d": return up ? "The stock has been swinging more than usual lately." : "The stock has been calm lately.";
    case "log_dollar_volume_20d": return up ? "A heavily traded stock, so news spreads into the price fast." : "Lighter trading in this name.";
    case "country_tier": return up ? "Shares in this market tend to react strongly to news." : "Shares in this market tend to react calmly to news.";
    case "sector": return up ? `${e.value} stocks have reacted strongly to news like this.` : `${e.value} stocks usually react calmly to news like this.`;
    case "kind": return up ? "Companies of this kind react strongly to news like this." : "Companies of this kind usually react calmly.";
    case "market_open": return v >= 1 ? "Published while markets were open, so the reaction is immediate." : "Published outside market hours; any move shows at the next open.";
    case "day_of_week": return up ? "News on this weekday has moved prices more." : "News on this weekday has moved prices less.";
    case "credibility": return up ? "Comes from a well-established source." : "Comes from a less established source.";
    case "confidence": case "event_confidence": return up ? "The reading is clear-cut." : "The reading is uncertain, which keeps the score lower.";
    case "severity_hits": return up ? "Uses severe words such as default, war or collapse." : null;
    case "source_type": return e.value === "social" ? (up ? null : "Social posts usually move prices less than news.") : null;
    case "goldstein": case "abs_goldstein": case "log_num_mentions": case "log_num_sources": case "avg_tone": case "has_gdelt_event":
      return up ? "News agencies rated the event as significant." : "News agencies rated the event as minor.";
    default: return null;
  }
}

/** 2-3 plain reasons for an Impact Score, strongest first, from the model's own effects. */
export function impactReasons(impact: { effects?: Effect[]; drivers: string[] }, sentiment: number): string[] {
  const out: string[] = [];
  const effects = impact.effects ?? [];
  for (const e of effects.filter((x) => x.effect > 0)) {
    const t = effectText(e, true, sentiment);
    if (t && !out.includes(t)) out.push(t);
    if (out.length === 3) return out;
  }
  for (const e of effects.filter((x) => x.effect < 0)) {
    if (out.length >= 2) break;
    const t = effectText(e, false, sentiment);
    if (t && !out.includes(t)) out.push(t);
  }
  if (!effects.length) {
    for (const d of impact.drivers) {
      const [feature, value] = d.split("=");
      const t = effectText({ feature, value, effect: 1 }, true, sentiment);
      if (t && !out.includes(t)) out.push(t);
    }
  }
  return out.slice(0, 3);
}

/** The 20 stocks of the Module A index. */
export const INDEX_TICKERS = new Set(["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "UNH", "PFE", "PG", "KO", "WMT", "CAT", "BA"]);
