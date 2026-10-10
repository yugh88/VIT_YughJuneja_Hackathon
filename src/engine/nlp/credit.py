"""Unambiguous credit-event phrases in 13 languages.

One definition, two uses: weak labels for training (``training.credit_labels``) and a
high-precision backstop at inference (``engine.nlp.events``). Phrases are chosen for
precision, not recall: each names a default, bankruptcy/insolvency filing, missed
payment, rating downgrade, distressed restructuring or covenant breach outright.
"""

from __future__ import annotations

import re

_PHRASES = {
    "en": r"files? for (chapter 11|chapter 7|bankruptcy|insolvency)|filed for (chapter 11|bankruptcy|insolvency)|"
          r"bankruptcy (protection|filing|proceedings)|chapter 11|"
          r"defaults? on (its |a |the )?(\$?[\d.,]+ ?(bn|billion|million|m) )?(debt|bonds?|loans?|notes|payments?|obligations)|"
          r"(sovereign|debt|bond|loan|payment) default|"
          r"miss(es|ed)? (a |an |its )?(coupon|interest|bond|debt|loan|principal) payments?|"
          r"(downgrades?d?|cuts?) .{0,40}(credit rating|to junk|rating to (b|c|junk)|outlook to negative)|"
          r"credit rating (downgrade|cut)|downgraded to junk|junk status|"
          r"(debt|distressed) restructuring|restructur(e|es|ing) (its |their )?debts?|distressed (debt )?exchange|"
          r"insolvency (proceedings|filing|administrator)|enters? (administration|receivership|liquidation)|"
          r"placed (in|into) (administration|receivership)|"
          r"(breach(es|ed)?|violat(es|ed|ion of)) (of )?(its )?(debt |loan |financial )?covenants?|covenant (breach|waiver)|"
          r"acceleration of (the )?(debt|notes|loans)|event of default",
    "de": r"insolvenz(antrag|verfahren)?|meldet insolvenz an|zahlungsunfähig\w*|zahlungsausfall|herabstuf\w*|"
          r"stuft .{0,30} herab|umschuldung|schuldenschnitt|kreditausfall|gläubigerschutz",
    "fr": r"faillite|redressement judiciaire|liquidation judiciaire|défaut de paiement|cessation de paiements|"
          r"abaisse (la )?note|dégradation de (la )?note|restructuration de (la )?dette|procédure de sauvegarde",
    "es": r"quiebra|concurso de acreedores|suspensión de pagos|impago|rebaja (la |su )?calificación|"
          r"reestructuración de (la )?deuda|cesación de pagos|default de (la )?deuda",
    "pt": r"falência|recuperação judicial|calote|inadimplência|rebaixa (a |sua )?nota|rebaixamento de (rating|nota)|"
          r"reestruturação da dívida|default da dívida",
    "it": r"fallimento|insolvenza|default del debito|declassa\w*|taglio del rating|ristrutturazione del debito|"
          r"concordato preventivo",
    "ru": r"банкротств\w*|дефолт\w*|понизил[аио]? (кредитный )?рейтинг|реструктуризаци[яи] долга|неплатежеспособн\w*|"
          r"просрочк[аи] (по )?(выплат|платеж)",
    "tr": r"iflas\w*|konkordato|temerrüt\w*|not indirim\w*|borç yapılandırma\w*",
    "zh": r"违约|債務違約|破产|破產|债务重组|債務重組|下调.{0,6}评级|评级下调|清盘|爆雷|无力偿还|展期兑付",
    "ja": r"債務不履行|デフォルト|経営破綻|民事再生|会社更生法|格下げ|破産|経営破たん",
    "ko": r"부도|파산|신용등급 강등|디폴트|법정관리|채무불이행|기업회생",
    "hi": r"दिवालिया|डिफ़ॉल्ट|डिफॉल्ट|रेटिंग घटा\w*|रेटिंग में कटौती|ऋण पुनर्गठन|कर्ज़? (चुकाने|भुगतान) में चूक",
    "ar": r"إفلاس|تعثر عن السداد|التخلف عن السداد|خفض التصنيف|تخفيض التصنيف|إعادة هيكلة الديون",
}  # fmt: skip
# Phrases that look like credit language but are not credit events.
_EXCLUDE = re.compile(
    r"moral bankruptcy|morally bankrupt|default (setting|browser|option|mode|password|app)|by default|"
    r"chapter 11 of (the|this) (book|novel|series)|spiritual|bankrupt of ideas",
    re.I,
)
CREDIT_RX = {lang: re.compile(rf"(?<!\w)(?:{p})", re.I) for lang, p in _PHRASES.items()}
_ANY = re.compile("|".join(f"(?:{p})" for p in _PHRASES.values()), re.I)


def credit_phrase(text: str, lang: str | None = None) -> str | None:
    """The unambiguous credit phrase found in ``text``, or None."""
    if _EXCLUDE.search(text):
        return None
    rx = CREDIT_RX.get(lang or "") if lang else None
    m = (rx or _ANY).search(text) or (_ANY.search(text) if rx else None)
    return m.group(0) if m else None


# ---------------------------------------------------------------- hard negatives and distress terms
# Credit-lookalikes that are *not* credit events (LABELLING.md): rating affirmations, stable/positive
# outlooks and upgrades are never Credit; anti-money-laundering stories are Regulatory/Legal. Both
# yield to an explicit distress term in the same headline ("affirmed ... but lowered to SD").
_NOT_CREDIT = re.compile(
    r"\baffirm(s|ed|ing)?\b|\breaffirm\w*|\boutlook\b.{0,25}\b(stable|positive)\b|\b(stable|positive) outlook\b|"
    r"\bupgrad(e|es|ed|ing)\b|"
    r"bestätigt .{0,20}(rating|note)|hochgestuft|stuft .{0,30} hoch|"
    r"confirme (la |sa )?note|relève (la |sa )?note|perspective (stable|positive)|"
    r"confirma (la |su )?calificación|(eleva|mejora) (la |su )?calificación|perspectiva (estable|positiva)|"
    r"(eleva|melhora) (a |sua )?nota|perspectiva (estável|positiva)|"
    r"(alza|migliora|promuove) .{0,15}rating|outlook (stabile|positivo)|"
    r"подтвердил[аио]? .{0,15}рейтинг|повысил[аио]? .{0,15}рейтинг|прогноз .{0,10}(стабильн|позитивн)|"
    r"上调.{0,6}评级|评级上调|维持.{0,6}评级|展望(稳定|正面)|格上げ|見通し.{0,4}(安定的|ポジティブ)|"
    r"रेटिंग (बढ़ा|बरकरार)|رفع التصنيف|تثبيت التصنيف",
    re.I,
)
_AML = re.compile(
    r"money[- ]laundering|\banti-money\b|\baml\b|"
    r"geldwäsche|blanchiment|lavado de (dinero|activos)|blanqueo de capitales|lavagem de dinheiro|riciclaggio|"
    r"отмывани\w* (денег|средств)|洗钱|洗錢|マネーロンダリング|資金洗浄|मनी लॉन्ड्रिंग|غسل الأموال",
    re.I,
)
# The distress terms Module B requires for a Credit Event trigger (user spec) plus their direct synonyms.
DISTRESS = re.compile(
    r"downgrad\w*|\bdefault(s|ed|ing)?\b|selective default|miss(es|ed)? (a |an |its )?(coupon|interest|bond|debt|loan|principal)? ?payments?|"
    r"restructur\w*|insolven\w*|bankrupt\w*|chapter 11|covenant (breach|waiver)|breach\w* .{0,20}covenants?|"
    r"\bnclt\b|\bibc\b|liquidat\w*|receivership|enters? administration|"
    r"(lower(s|ed)?|cut(s)?|slash(es|ed)?) .{0,40}rating|rating .{0,10}(lowered|cut)|to junk|"
    r"lowered to .?(sd|d|rd)\b",
    re.I,
)


def distress_term(text: str) -> str | None:
    """An explicit distress term (English list above, or any unambiguous credit phrase in 13 languages)."""
    if _EXCLUDE.search(text):
        return None
    m = DISTRESS.search(text)
    return m.group(0) if m else credit_phrase(text)


def hard_negative(text: str) -> str | None:
    """'aml' -> Regulatory/Legal; 'not_credit' -> must not be Credit; None -> no rule applies."""
    if distress_term(text):
        return None
    if _AML.search(text):
        return "aml"
    if _NOT_CREDIT.search(text):
        return "not_credit"
    return None


# ---------------------------------------------------------------- v2 credit rules (DECISIONS D74)
# Selected on credit_v2_dev only. Each returns the class a Credit prediction should become, or forces
# Credit for an explicit event the classifier missed. English only (the credit sets are English).
_AGENCY = re.compile(
    r"\bs&p\b|standard (&|and) poor|moody'?s|\bfitch\b|\bcrisil\b|\bicra\b|\bcare ratings\b|\bcare\b.{0,3}rating|"
    r"india ratings|\bdbrs\b|\bkbra\b|scope ratings|ram ratings|am best|acuit[eé]|\br&i\b|\bjcr\b",
    re.I,
)
_CREDIT_WORDS = re.compile(
    r"credit rating|\bratings?\b|\boutlook\b|\bnotch\w*|\bbonds?\b|\bdebt\b|\bdefault\w*|\bjunk\b|investment.grade|\bidr\b",
    re.I,
)
_DEBT_WORDS = re.compile(
    r"\bdebts?\b|\bcreditors?\b|\bbonds?\b|\bbondholders?\b|\blenders?\b|\bloans?\b|\bnotes\b|\bcoupons?\b|\bdefault\w*|"
    r"insolven\w*|bankrupt\w*|chapter 11|liquidat\w*|\badministration\b|receivership|winding.up",
    re.I,
)
_ANALYST = re.compile(
    r"price target|\bpt\b|\bto (sell|neutral|hold|underperform|underweight|equal.weight|market perform|reduce|in.line)\b|"
    r"(stock|shares?) (rating|price)|analyst",
    re.I,
)
_JOBS = re.compile(
    r"\blayoffs?\b|\bjob cuts?\b|\bjobs\b|\bworkforce\b|\bheadcount\b|\bemployees\b|\bstaff\b|\bpositions\b|\bworkers\b",
    re.I,
)
_PROCEDURAL = re.compile(
    r"\b(nclt|nclat|tribunal|court|bench|judge)\b.{0,80}\b(dismiss\w*|reject\w*|stays?|stayed|defers?|deferred|set(s)? aside|"
    r"quash\w*|clarif\w*|relief|questions?|seeks responses)\b|laws? and regulations|\bclarification\b",
    re.I,
)
_AGGREGATE = re.compile(
    r"\b(filings|petitions|insolvencies|defaults|bankruptcies|applications)\b.{0,40}\b(rise|rises|rising|hit|high|surg\w*|"
    r"record|fall|fell|drop\w*|settled)\b|\b(rise|surge|jump|wave) (in|of) (corporate )?(insolvencies|bankruptcies|defaults)\b",
    re.I,
)
_RECOVERY = re.compile(
    r"\b(removes?|lifts?|exits?|emerges? from|cured?) .{0,25}default|debt crisis recovery", re.I
)
_TALK = re.compile(
    r"\b(could|may|might|risk of|at risk|fears? of|warns? of|drifting|signals?|threat of)\b.{0,40}\b(default|downgrad\w*|junk|bankrupt\w*)|"
    r"\?\s*$",
    re.I,
)
_OUTLOOK_UP = re.compile(
    r"\boutlook\b.{0,45}\bto\s*'?(stable|positive)\b|\b(stable|positive) outlook\b(?!.{0,15}\b(cut|downgrad\w*|lower\w*)\b)",
    re.I,
)
_RATING_CUT = re.compile(
    r"(downgrad\w*|cuts?|lower\w*|slash\w*) .{0,40}\bto\s*'?[a-c]{1,3}[+-]?\d?\b|downgraded to|cut to junk",
    re.I,
)
_FORCE_CREDIT = re.compile(
    r"appoints? liquidators?|winding.up (order|notice|petition) against|\boutlook\b.{0,45}\bto\s*'?negative\b|"
    r"negative (outlook|watch)|creditwatch negative|(placed|put) on (review|watch) for (a )?downgrade",
    re.I,
)


def credit_rule_v2(text: str, primary: str) -> str | None:
    """Class to use instead of ``primary`` (None = keep it). Applied after the classifier."""
    credit = "Credit Event"
    if primary == credit:
        if _AGGREGATE.search(text):
            return "Macroeconomic"
        if _PROCEDURAL.search(text):
            return "Regulatory/Legal"
        if _RECOVERY.search(text) or _TALK.search(text):
            return "Other/Non-event"
        if _ANALYST.search(text) and not _AGENCY.search(text):
            return "Other/Non-event"
        if re.search(r"restructur\w*", text, re.I) and _JOBS.search(text) and not _DEBT_WORDS.search(text):
            return "Operational/Supply Chain"
        if _OUTLOOK_UP.search(text) and not _RATING_CUT.search(text) and not _FORCE_CREDIT.search(text):
            return "Other/Non-event"
        if re.search(r"downgrad\w*", text, re.I) and not (_AGENCY.search(text) or _CREDIT_WORDS.search(text)):
            return "Other/Non-event"
        return None
    if _FORCE_CREDIT.search(text) and not (_AGGREGATE.search(text) or _TALK.search(text)):
        return credit
    return None
