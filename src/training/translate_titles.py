"""English versions of the recorded non-English headlines, for display only.

    python -m training.translate_titles        # about 10k titles, once, offline (NLLB-200 on MPS/GPU/CPU)

Scoring never uses these: the models read the original text. The dashboard shows the English
title so a reader can follow a French or Russian story. Running NLLB (600M) at serve time
would make Docker and setup much heavier (D-log: translate-test), so the translations are
made once here and shipped as ``data/reference/title_translations.json.gz`` ({title: english}).
Live headlines that are not in the file keep their original text.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import time

from engine.config import REPO_ROOT
from engine.nlp.hinglish import is_hinglish, to_devanagari
from engine.nlp.langid import LanguageIdentifier
from engine.nlp.translations import PATH

log = logging.getLogger("translate_titles")
# fastText lid.176 code -> NLLB-200 code, for the languages that occur in the recordings.
NLLB_SRC = {
    "ar": "arb_Arab", "az": "azj_Latn", "bg": "bul_Cyrl", "bn": "ben_Beng", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek", "es": "spa_Latn", "et": "est_Latn",
    "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva",
    "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda", "ko": "kor_Hang", "lt": "lit_Latn",
    "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn",
    "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "si": "sin_Sinh", "sk": "slk_Latn", "sl": "slv_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu",
    "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn",
    "vi": "vie_Latn", "zh": "zho_Hans",
}  # fmt: skip
MIN_CONF = 0.5  # below this fastText is guessing, unless the source tagged the same language


def recorded_titles() -> dict[str, str | None]:
    """{title: the source's own language tag} for every recorded document."""
    titles: dict[str, str | None] = {}
    for f in sorted(glob.glob(str(REPO_ROOT / "data" / "replay" / "*" / "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if t := (d.get("title") or "").strip():
                    titles.setdefault(t, d.get("language"))
    return dict(sorted(titles.items()))


def to_translate(titles: dict[str, str | None]) -> dict[str, list[tuple[str, str]]]:
    """Group (title, text to translate) by source language; English and unsure titles are skipped."""
    lid = LanguageIdentifier()
    by_lang: dict[str, list[tuple[str, str]]] = {}
    for t, hint in titles.items():
        if is_hinglish(t):
            by_lang.setdefault("hi", []).append((t, to_devanagari(t)))
            continue
        r = lid.detect(t)
        if r.lang != "en" and r.lang in NLLB_SRC and (r.confidence >= MIN_CONF or r.lang == hint):
            by_lang.setdefault(r.lang, []).append((t, t))
    return by_lang


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import torch

    from training.translate import Translator

    out: dict[str, str] = {}
    if PATH.exists():  # incremental: only titles not translated yet
        with gzip.open(PATH, "rt") as fh:
            out = json.loads(fh.read())
    by_lang = {
        lang: todo
        for lang, items in to_translate(recorded_titles()).items()
        if (todo := [it for it in items if it[0] not in out])
    }
    log.info("%d new titles in %d languages", sum(map(len, by_lang.values())), len(by_lang))
    tr = Translator()
    eng = tr.tok.convert_tokens_to_ids("eng_Latn")
    t0 = time.time()
    with torch.inference_mode():
        for lang, items in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
            tr.tok.src_lang = NLLB_SRC[lang]
            items.sort(key=lambda it: len(it[1]))  # length-bucketing
            for b in range(0, len(items), args.batch):
                chunk = items[b : b + args.batch]
                enc = tr.tok(
                    [s for _, s in chunk], return_tensors="pt", padding=True, truncation=True, max_length=96
                )
                enc = {k: v.to(tr.dev) for k, v in enc.items()}
                gen = tr.model.generate(**enc, forced_bos_token_id=eng, max_new_tokens=96, num_beams=2)
                for (title, _), en in zip(
                    chunk, tr.tok.batch_decode(gen, skip_special_tokens=True), strict=True
                ):
                    if en.strip() and en.strip() != title:
                        out[title] = en.strip()
            log.info("%s: %d titles (%.0fs so far)", lang, len(items), time.time() - t0)
    with gzip.open(PATH, "wt") as fh:
        fh.write(json.dumps(dict(sorted(out.items())), ensure_ascii=False, separators=(",", ":")))
    log.info("wrote %d translations to %s", len(out), PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
