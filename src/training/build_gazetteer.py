"""Build the multilingual entity gazetteer from Wikidata (keyless).

    python -m training.build_gazetteer

Reads ``data/reference/entities_seed.csv`` (S&P 100, Nifty 50, global banks, key
corporates, sovereigns), resolves each to a Wikidata item (by exchange ticker for
equities, ISO code for sovereigns, name search otherwise), pulls labels, aliases
and demonyms in ~20 languages, and writes ``data/reference/gazetteer.json.gz``.
The output is committed, so the engine never needs Wikidata at runtime.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from engine.config import get_settings

log = logging.getLogger("gazetteer")
SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
LANGS = [
    "en", "de", "fr", "es", "pt", "it", "nl", "ja", "zh", "zh-hans", "zh-hant", "ko", "hi",
    "ar", "ru", "uk", "tr", "vi", "id", "pl", "he", "fa", "th", "bn", "mr", "ta",
]  # fmt: skip
EXCHANGES = {
    "US": {"New York Stock Exchange", "Nasdaq", "NASDAQ"},
    "IN": {"National Stock Exchange of India", "Bombay Stock Exchange", "BSE"},
}
# Hand-checked QIDs for names the automatic lookups miss (verified on wikidata.org, 2026-10-04).
OVERRIDES = {
    "ADANIPORTS.NS": "Q16058076", "APOLLOHOSP.NS": "Q4780321", "INDUSINDBK.NS": "Q2040323",
    "NESTLEIND.NS": "Q104845552", "NTPC.NS": "Q1967805", "POWERGRID.NS": "Q3400886",
    "SHRIRAMFIN.NS": "Q17074288", "TATAMOTORS.NS": "Q188514", "CORP:EVERGRANDE": "Q5417145",
}  # fmt: skip
SEARCH_HINTS = {
    "bank": ("bank", "financial", "company", "business"),
    "corporate": ("company", "business", "group", "manufacturer", "conglomerate"),
}


def _sparql(client: httpx.Client, query: str) -> list[dict[str, Any]]:
    r = client.get(SPARQL, params={"query": query, "format": "json"}, timeout=120)
    r.raise_for_status()
    return r.json()["results"]["bindings"]  # type: ignore[no-any-return]


def resolve_equities(client: httpx.Client, rows: list[dict[str, str]]) -> dict[str, str]:
    tickers = " ".join(f'"{r["ticker"]}"' for r in rows)
    q = f"""SELECT ?item ?ticker ?exLabel WHERE {{
      ?item p:P414 ?st . ?st ps:P414 ?ex ; pq:P249 ?ticker . VALUES ?ticker {{ {tickers} }}
      FILTER NOT EXISTS {{ ?st pq:P582 ?end }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}"""
    found: dict[tuple[str, str], str] = {}
    for b in _sparql(client, q):
        found.setdefault((b["ticker"]["value"], b["exLabel"]["value"]), b["item"]["value"].rsplit("/", 1)[1])
    out = {}
    for r in rows:
        for ex in EXCHANGES[r["exchange"]]:
            if (r["ticker"], ex) in found:
                out[r["id"]] = found[(r["ticker"], ex)]
                break
    return out


def resolve_sovereigns(client: httpx.Client, rows: list[dict[str, str]]) -> dict[str, str]:
    codes = " ".join(f'"{r["country"]}"' for r in rows)
    q = f"SELECT ?item ?code WHERE {{ ?item wdt:P297 ?code . VALUES ?code {{ {codes} }} }}"
    by_code = {b["code"]["value"]: b["item"]["value"].rsplit("/", 1)[1] for b in _sparql(client, q)}
    return {r["id"]: by_code[r["country"]] for r in rows if r["country"] in by_code}


def resolve_by_search(client: httpx.Client, rows: list[dict[str, str]]) -> dict[str, str]:
    out = {}
    for r in rows:
        resp = client.get(
            API,
            params={
                "action": "wbsearchentities",
                "search": r["name"],
                "language": "en",
                "format": "json",
                "limit": 7,
                "type": "item",
            },
        )
        hits = resp.json().get("search", [])
        hints = SEARCH_HINTS.get(r["kind"], ("company",))
        pick = next((h for h in hits if any(k in h.get("description", "").lower() for k in hints)), None)
        if pick:
            out[r["id"]] = pick["id"]
        else:
            log.warning("no confident Wikidata match for %s (%s)", r["id"], r["name"])
        time.sleep(0.1)
    return out


def fetch_entities(client: httpx.Client, qids: list[str]) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    for i in range(0, len(qids), 50):
        resp = client.get(
            API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(qids[i : i + 50]),
                "props": "labels|aliases|claims",
                "languages": "|".join(LANGS),
                "format": "json",
            },
            timeout=120,
        )
        data.update(resp.json()["entities"])
        time.sleep(0.2)
    return data


def names_from(entity: dict[str, Any], kind: str) -> dict[str, list[str]]:
    names: dict[str, set[str]] = {}
    for lang, v in entity.get("labels", {}).items():
        names.setdefault(lang.split("-")[0], set()).add(v["value"])
    for lang, vs in entity.get("aliases", {}).items():
        for v in vs:
            names.setdefault(lang.split("-")[0], set()).add(v["value"])
    if kind == "sovereign":  # demonyms (P1549), e.g. "Russian", "russe", "ロシアの"
        for c in entity.get("claims", {}).get("P1549", []):
            dv = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(dv, dict) and dv.get("language", "").split("-")[0] in {
                lg.split("-")[0] for lg in LANGS
            }:
                names.setdefault(dv["language"].split("-")[0], set()).add(dv["text"])
    return {k: sorted(v) for k, v in names.items()}


def build(seed: Path, out: Path) -> dict[str, Any]:
    with open(seed, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    headers = {"User-Agent": get_settings().user_agent, "Accept": "application/sparql-results+json"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=60) as client:
        qid: dict[str, str] = dict(OVERRIDES)
        qid |= resolve_equities(client, [r for r in rows if r["kind"] == "equity"])
        qid |= resolve_sovereigns(client, [r for r in rows if r["kind"] == "sovereign"])
        qid |= resolve_by_search(
            client,
            [
                r
                for r in rows
                if r["id"] not in qid and (r["kind"] in {"bank", "corporate"} or r["kind"] == "equity")
            ],
        )
        ents = fetch_entities(client, sorted(set(qid.values())))
    entities = []
    for r in rows:
        sector, _, region = r["sector"].partition("|")
        names = names_from(ents.get(qid.get(r["id"], ""), {}), r["kind"])
        names.setdefault("en", [])
        if r["name"] not in names["en"]:
            names["en"].insert(0, r["name"])
        entities.append(
            {
                "id": r["id"],
                "kind": r["kind"],
                "name": r["name"],
                "ticker": r["ticker"] or None,
                "country": r["country"],
                "sector": sector,
                "region": region or None,
                "ambiguous": r["ambiguous"] == "1",
                "wikidata": qid.get(r["id"]),
                "names": names,
            }
        )
    unresolved = [e["id"] for e in entities if not e["wikidata"]]
    doc = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Wikidata (CC0)",
        "languages": LANGS,
        "unresolved": unresolved,
        "entities": entities,
    }
    out.write_bytes(gzip.compress(json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"), 9))
    n_alias = sum(len(v) for e in entities for v in e["names"].values())
    log.info("wrote %s: %d entities, %d names, unresolved=%s", out, len(entities), n_alias, unresolved)
    return doc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ref = get_settings().reference_dir
    build(ref / "entities_seed.csv", ref / "gazetteer.json.gz")


if __name__ == "__main__":
    main()
