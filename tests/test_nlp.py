"""Pipeline stages: each test guards a failure that would embarrass us in a demo."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_doc
from engine.nlp.dedupe import Deduplicator
from engine.nlp.entities import get_linker
from engine.nlp.langid import MODEL_PATH, LanguageIdentifier
from engine.nlp.normalise import normalise
from engine.schema import EventClass


def ids(text: str, meta: dict | None = None) -> list[str]:  # type: ignore[type-arg]
    return [e.id for e in get_linker().link(text, meta).entities()]


def test_normalise_strips_urls_handles_and_publisher_suffix() -> None:
    c = normalise("Nvidia jumps on record sales - Reuters", "Read more https://t.co/x @trader99 now.")
    assert c.sentences[0] == "Nvidia jumps on record sales"
    assert "http" not in c.text and "@trader99" not in c.text


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Apple shares rise after strong iPhone sales", "en"),
        ("トヨタ株が急落、リコール拡大で", "ja"),
        ("रिलायंस इंडस्ट्रीज़ के शेयर गिरे", "hi"),
        pytest.param(
            "Die Aktie von Siemens fällt deutlich",
            "de",
            # the script heuristic cannot separate Latin-script languages without the fastText model
            marks=pytest.mark.skipif(not MODEL_PATH.exists(), reason="models/lid.176.ftz not downloaded"),
        ),
    ],
)
def test_language_id(text: str, lang: str) -> None:
    assert LanguageIdentifier().detect(text).lang == lang


def test_entity_linking_is_multilingual() -> None:
    assert ids("トヨタ株が急落、アップルの供給網に懸念") == ["CORP:TOYOTA", "AAPL"]
    assert ids("रिलायंस इंडस्ट्रीज़ के शेयर गिरे") == ["RELIANCE.NS"]
    assert ids("Les actions de BNP Paribas reculent") == ["BANK:BNP"]
    assert ids("俄罗斯与乌克兰谈判") == ["SOV:RU", "SOV:UA"]


def test_entity_linking_tricky_names() -> None:
    assert ids("I baked an apple pie with my family") == []  # "Apple pie" != AAPL
    assert ids("Apple shares jump after record iPhone sales") == ["AAPL"]
    assert ids("Bank of China raises lending rates") == ["BANK:BOC"]  # longest match, not China
    assert ids("Us and them: a story") == []  # case-sensitive short alias "US"
    assert ids("US inflation surprises the Fed") == ["SOV:US"]
    assert ids("Investors are fed up with volatility") == []  # "Fed" is case-sensitive
    assert ids("$TSLA guidance cut; NVDA rallies") == ["TSLA", "NVDA"]  # cashtag counted once


def test_entity_fallbacks_from_metadata() -> None:
    assert ids("Shares slip in early trade", {"tags": ["MSFT"]}) == ["MSFT"]
    assert ids("Protesters clash with police", {"gdelt_event": {"action_fips": "UP"}}) == ["SOV:UA"]


def test_dedupe_collapses_syndicated_copies() -> None:
    d = Deduplicator()
    a = "Oil prices jump as OPEC agrees surprise production cut of one million barrels per day"
    r1 = d.check("a", a)
    r2 = d.check("b", a + " - Reuters")
    r3 = d.check("c", "Tech stocks slide as bond yields climb to a sixteen year high on Fed worries")
    assert not r1.is_duplicate and r2.is_duplicate and r2.cluster_id == "a" and r2.syndication_count == 2
    assert not r3.is_duplicate


def test_pipeline_signals_respect_schema_and_ranges(pipeline) -> None:  # type: ignore[no-untyped-def]
    tr = pipeline.analyze("Silicon Valley Bank collapses as depositors flee; regulators seize the lender")
    assert tr.signals, tr.dropped
    s = tr.signals[0]
    assert s.entity.id == "BANK:SIVB"
    assert -1.0 <= s.sentiment.score <= 1.0 and s.sentiment.label in {"negative", "neutral", "positive"}
    assert s.event.primary in set(EventClass)
    assert 1 <= s.impact.low <= s.impact.score <= s.impact.high <= 10


def test_pipeline_drops_duplicates_and_unlinked(pipeline) -> None:  # type: ignore[no-untyped-def]
    docs = [
        make_doc(1, title="Microsoft agrees to acquire gaming studio for $2 billion in cash"),
        make_doc(2, title="Microsoft agrees to acquire gaming studio for $2 billion in cash - Bloomberg"),
        make_doc(3, title="Local bakery wins award for the best sourdough in town this year"),
    ]
    traces = pipeline.process_batch(docs)
    assert [t.dropped for t in traces] == [None, "duplicate", "no entity"]


def test_local_news_with_a_dateline_is_not_a_sovereign_signal(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Regression: 'Kansas small-business fraud' once became 'US Credit Event, impact 8'."""
    kansas = make_doc(
        90, title="Kansas candle-making business accused of fraud to get a loan",
        meta={"gdelt_event": {"cameo_root": "18", "goldstein": -9.5, "num_sources": 1, "action_fips": "US"},
              "gdelt_locations": [{"type": 3, "fips": "US", "lat": 38.5, "lon": -98.0}]},
    )  # fmt: skip
    war = make_doc(
        91,
        title="Artillery barrage hits the front line as fighting intensifies",
        meta={"gdelt_event": {"cameo_root": "19", "goldstein": -10.0, "num_sources": 6, "action_fips": "UP"}},
    )
    tk, tw = pipeline.process_batch([kansas, war], dedupe=False)
    assert not tk.signals and tk.dropped == "no entity"
    if tw.event and tw.event.primary == "Geopolitical":  # with corroboration, the dateline does count
        assert [s.entity.id for s in tw.signals] == ["SOV:UA"]


def test_sentiment_range_check(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Clearly strong headlines reach |score| >= 0.7, neutral ones stay <= 0.2 (data/eval/sentiment_range_check.csv)."""
    import pandas as pd

    if pipeline.sentiment.mode != "model":
        pytest.skip("fine-tuned sentiment models not installed")
    df = pd.read_csv(Path(__file__).parents[1] / "data" / "eval" / "sentiment_range_check.csv")
    traces = pipeline.process_batch(
        [make_doc(1000 + i, title=t) for i, t in enumerate(df["text"])], dedupe=False, zero_shot=False
    )
    bad = []
    for (_, row), tr in zip(df.iterrows(), traces, strict=True):
        if tr.signals:
            s = tr.signals[0].sentiment.score
        else:  # no named entity (e.g. "Moody's upgrades the company"): document-level score
            s = pipeline.sentiment.score_batch([("", row["text"], "en", 1.0)])[0].score
        ok = (
            abs(s) >= 0.7 and (s > 0) == (row["expect"] == "strong_pos")
            if row["expect"] != "neutral"
            else abs(s) <= 0.2
        )
        if not ok:
            bad.append((row["expect"], round(s, 2), row["text"][:60]))
    assert not bad, bad


def test_try_it_lookup_detects_company_questions_and_ambiguity() -> None:
    from engine.nlp.lookup import lookup

    r = lookup("tata ki recent news btao")
    assert r["mode"] == "company" and r["intent"] and r["ambiguous"]
    assert {"Tata Motors", "Tata Steel", "Tata Consultancy Services"} <= {m["name"] for m in r["matches"]}
    assert lookup("TCS")["resolved"] == "TCS.NS"
    assert lookup("infosys ka kya hua")["resolved"] == "INFY.NS"
    assert lookup("relaince")["resolved"] == "RELIANCE.NS"  # misspelt romanised name
    assert lookup("टाटा मोटर्स की ताजा खबर बताओ")["resolved"] == "TATAMOTORS.NS"
    for headline in ("tata motors ka munafa badha", "Moody's stuft die Kreditwürdigkeit der Commerzbank herab",
                     "トヨタ自動車、半導体不足で国内工場の稼働を一時停止", "asdfgh qwerty"):  # fmt: skip
        assert lookup(headline)["mode"] == "headline", headline


def test_promotional_posts_are_filtered_and_short_texts_use_source_language() -> None:
    from engine.nlp.quality import is_promotional, price_move, real_word_count

    spam = "$SNAP $CSCO $UBER $TXN $NIO $IFF Get instant updates and free trading analysis click here"
    assert is_promotional(spam) == "6 cashtags"
    assert is_promotional("$TSLA #EV #stocks #tesla #moon") is not None  # tags are most of the words
    assert is_promotional("Join our Discord group for daily picks") is not None  # promo phrase
    assert is_promotional("$AAPL beats estimates as iPhone sales jump in China") is None
    assert is_promotional("Twilio $TWLO is set to join the S&P 500 on October 6") is None  # not "join our"
    assert is_promotional("You can click links to other Markdown files") is None
    assert (
        real_word_count("$VXRT 🚀🚀 #moon 1.25") == 0
        and real_word_count("トヨタ自動車、半導体不足で国内工場") >= 5
    )
    assert price_move("Tesla shares plunge after recall of 2 million cars", [(0, 5)])[:2] == (
        True,
        "recall of 2 million cars",
    )
    assert not price_move("Profit slumps as fees fall", [])[0]  # the move is about profit, not the stock


@pytest.mark.parametrize(
    ("text", "primary", "expected"),
    [  # invented headlines (none from any evaluation set)
        (
            "Acme Corp announces 500 job cuts as part of restructuring plan",
            "Credit Event",
            "Operational/Supply Chain",
        ),
        ("Broker X cuts Acme price target to $40, reiterates hold", "Credit Event", "Other/Non-event"),
        ("Fitch revises Ruritania's outlook to positive", "Credit Event", "Other/Non-event"),
        ("Moody's downgrades Ruritania to B2 with stable outlook", "Credit Event", None),
        ("Tribunal dismisses insolvency plea against Acme Infra", "Credit Event", "Regulatory/Legal"),
        ("Corporate insolvencies rise to a record high in March", "Credit Event", "Macroeconomic"),
        (
            "Inspectors downgrade the town's hospital to 'requires improvement'",
            "Credit Event",
            "Other/Non-event",
        ),
        ("Acme Retail appoints liquidators after winding-up notice", "Other/Non-event", "Credit Event"),
        ("S&P revises Ruritania outlook from stable to negative", "Macroeconomic", "Credit Event"),
        ("Acme files for Chapter 11 bankruptcy protection", "Credit Event", None),
    ],
)
def test_credit_rules_v2(text: str, primary: str, expected: str | None) -> None:
    from engine.nlp.credit import credit_rule_v2

    assert credit_rule_v2(text, primary) == expected


def test_nli_handles_long_premise() -> None:
    from engine.nlp.onnx_model import OnnxClassifier
    from engine.nlp.pipeline import ONNX_DIR

    d = ONNX_DIR / "zeroshot-nli"
    if not (d / "model.int8.onnx").exists():
        pytest.skip("NLI model not downloaded")
    zs = OnnxClassifier(d)
    long_text = "The central bank raised rates again. " * 200
    assert zs.logits([long_text], ["This is bad news."]).shape == (1, len(zs.labels))


def test_hinglish_detection_and_transliteration() -> None:
    from engine.nlp.hinglish import is_hinglish, to_devanagari

    assert is_hinglish("Acme Motors ka munafa badha")  # invented examples
    assert not is_hinglish("Le gouvernement se prépare par avance")  # French "se", "par"
    assert not is_hinglish("Acme shares rise after strong results")
    assert not is_hinglish("एक्मे का मुनाफा बढ़ा")  # already Devanagari
    out = to_devanagari("Acme ka munafa badha")
    assert out.startswith("Acme ") and any("ऀ" <= ch <= "ॿ" for ch in out)
