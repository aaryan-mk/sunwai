"""Indic number normalisation: digits to spoken words for clear quantities."""

from __future__ import annotations

import logging
import os
from typing import Dict


import collections

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# v3: number normalisation for TTS (spoken form only). Replaces v1/v2.
# ---------------------------------------------------------------------------
# review r1/r2 fixes: (1) a digit run is classified first; only clear QUANTITIES become words, identifiers (OTP/PIN/phone/codes,
# leading zeros, 5+ digits, separators, version/time/date shapes, years, 4-digit numbers without a unit) are left exactly as written;
# (2) no Gemini, no punctuation rewriting; (3) the spoken form never changes tgt_text (display, subtitles, fit guard input): it lives in
# seg["num_text"] with seg["num_src"] = the tgt_text it was made from (a refit/edit/resynth voids it); it is separate from the
# expression stage's tts_text; (4) the stage runs AFTER the fit and, when the spoken form would exceed the fit budget, the line is
# re-fitted with a smaller budget through the guarded rewrite, else it is NOT normalised; (5) bounded, locked cache.
# Kill switches: TTS_NORMALIZE=0 (everything), TTS_NORMALIZE_NUMBERS=0.
_TN_VER = "g091v3"


_TN_NUM_LANGS = ("hi", "mr", "en", "ta", "te", "bn")


_TN_PCT = {"hi": "प्रतिशत", "mr": "टक्के", "bn": "শতাংশ", "ta": "சதவீதம்", "te": "శాతం", "en": "percent"}


_TN_CACHE_MAX = 512


_TN_CACHE: "collections.OrderedDict" = __import__("collections").OrderedDict()


_TN_LOCK = __import__("threading").Lock()


_TN_INV: Dict[str, Dict[int, str]] = {}


_TN_ID_CUES = frozenset("""otp pin code id no no. number num phone mobile call dial account acct a/c room flight gate route version ver v model ticket order invoice zip pincode ref tel
ओटीपी पिन कोड नंबर नम्बर फोन मोबाइल खाता कमरा फ्लाइट ऑर्डर बिल क्रमांक आईडी
எண் கோட் ஓடிபி போன் மொபைல் அறை
নম্বর কোড ওটিপি ফোন মোবাইল""".split())


_TN_UNIT_AFTER = frozenset("""रुपये रुपए डॉलर लोग लोगों छात्र छात्रों मिनट घंटे घंटा सेकंड किलोमीटर किलो मीटर साल सालों वर्ष दिन दिनों बच्चे बच्चों आदमी किताबें किताब कॉपियाँ अंक वोट कॉपी ग्राम प्रतिशत बजे महीने महीनों हफ्ते सप्ताह देश देशों शहर शहरों गुना बार लीटर डिग्री सेंटीमीटर किलोग्राम पन्ने पृष्ठ सदस्य सदस्यों खिलाड़ी गोल मंज़िल मंजिल कमरे कुर्सियाँ सीटें
people students dollars rupees minutes hours seconds years days weeks months km kg meters metres miles percent users times words pages points members countries cities books children men women players goals floors rooms
ரூபாய் நிமிடம் நிமிட மணி நாட்கள் நாள் வருடம் ஆண்டு ஆண்டுகள் பேர் மக்கள் மாணவர்கள் கிலோமீட்டர் வயது வயதில் மாதங்கள் வாரங்கள் நாடுகள் புத்தகங்கள்
টাকা মিনিট ঘণ্টা দিন বছর জন লোক কিলোমিটার""".split())


_TN_CURRENCY_CODES = frozenset("usd inr eur gbp rs rs. aed".split())


_TN_SKIP_ONE = frozenset({"ta", "te"})              # 'one' before a noun is a different word (ஒரு / ఒక), table has the standalone form only


def _tn_flag(name: str, default: bool = True) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "off", "no", "n")


def _tn_inv(lang: str) -> Dict[int, str]:
    """value -> first number word of fit_stage._NUMV[lang] (insertion order); only int values."""
    inv = _TN_INV.get(lang)
    if inv is None:
        from sunwai.pipeline.fitting.fit_stage import _NUMV
        inv = {}
        for w, v in (_NUMV.get(lang) or {}).items():
            if isinstance(v, int):
                inv.setdefault(v, w)
        if lang == "hi" and "पाँच" in (_NUMV.get("hi") or {}):
            inv[5] = "पाँच"                      # the spelling people expect (table lists पांच first)
        _TN_INV[lang] = inv
    return inv


def _tn_int_words(n: int, lang: str):
    """Deterministic integer -> words, or None when the language table cannot express it exactly (caller then leaves the digits)."""
    inv = _tn_inv(lang)
    if n < 0 or not inv:
        return None
    if lang == "en":
        def lt1000(m):
            out = []
            if m >= 100:
                out += [inv[m // 100], inv[100]]; m %= 100
            if m >= 20:
                t10 = inv[m // 10 * 10]; m %= 10
                out.append(t10 + ("-" + inv[m] if m else ""))
            elif m or not out:
                out.append(inv[m])
            return out
        if n == 0:
            return inv[0]
        parts = []
        for scale, name in ((10**9, 10**9), (10**6, 10**6), (10**3, 10**3)):
            q, n = divmod(n, scale)
            if q:
                if q >= 1000:
                    return None
                parts += lt1000(q) + [inv[name]]
        if n:
            parts += lt1000(n)
        try:
            return " ".join(parts)
        except Exception:
            return None
    if lang in ("hi", "mr"):
        if n >= 10**9:
            return None
        parts = []
        try:
            for scale in (10**7, 10**5, 10**3):
                q, n = divmod(n, scale)
                if q:
                    parts += [inv[q], inv[scale]]          # q < 100 here, 0..99 are in the table
            if n >= 100:
                h, n = divmod(n, 100)
                if lang == "mr" and (h > 1 or n):
                    return None                              # mr 'she' hundreds forms are not in the table
                parts += [inv[100]] if lang == "mr" else [inv[h], inv[100]]
            if n or not parts:
                parts.append(inv[n])
        except KeyError:
            return None
        return " ".join(parts)
    return inv.get(n)                                        # ta / te / bn: only the exact table words


def _tn_words_to_int(tokens, lang):
    """Inverse (for tests / fallback verification): list of number-word tokens -> int, or None."""
    from sunwai.pipeline.fitting.fit_stage import _NUMV, _nfc
    tab = _NUMV.get(lang) or {}
    total = cur = 0
    tokens = [x for t in tokens for x in t.split("-") if x]
    for t in tokens:
        v = tab.get(_nfc(t).casefold()) if lang == "en" else tab.get(_nfc(t))
        if not isinstance(v, int):
            return None
        if v < 100:
            cur += v
        elif v == 100:
            cur = (cur or 1) * 100
        else:
            total += (cur or 1) * v; cur = 0
    return total + cur


def _tn_digits(tok: str) -> str:
    import unicodedata as _u
    return "".join(str(_u.decimal(c)) if c.isdecimal() else c for c in tok)


_TN_DOTTED_CUES = frozenset("no ref acct tel ph ver".split())


def _tn_cue_before(text: str, s: int) -> bool:
    """An identifier cue (OTP, PIN, code, phone, नंबर, No., ...) among the 3 words before the run, inside the same sentence/clause."""
    import regex as _rx
    if s > 0 and text[s - 1] == "#":
        return True
    pre = text[max(0, s - 240):s]
    cut = 0
    for m in _rx.finditer(r"[,;!?।\n\u2026]|\.{2,}|([\p{L}\p{M}]+)\.[\"'\u201d\u2019)\]]*(?=\s)", pre):
        if m.group(1) is None or m.group(1).casefold() not in _TN_DOTTED_CUES:
            cut = m.end()
    toks = _rx.findall(r"[\p{L}\p{M}/.]+", pre[cut:])[-3:]
    return any(t.casefold().strip(".") in _TN_ID_CUES for t in toks)


def _tn_next_word(text: str, e: int) -> str:
    import regex as _rx
    m = _rx.match(r"\s*([\p{L}\p{M}]+)", text[e:])
    return m.group(1).casefold() if m else ""


def _tn_quantity_words(text: str, m, lang: str):
    """Words for one digit run when it is a clear quantity, else None (the run stays exactly as written)."""
    import regex as _rx
    raw = m.group(1)
    if m.group(2) is not None:
        return None                                   # decimals: not handled
    d = _tn_digits(raw)
    s, e = m.start(1), m.end()
    prev = text[s - 1] if s > 0 else ""
    nxt = text[m.end(1)] if m.end(1) < len(text) else ""
    if (prev and prev in "-/:+@#_.%$\u20b9\u20ac\u00a3'\u2019\u066c\u066b") or (nxt and nxt in "-/:@#_'\u2019\u066c\u066b") or (nxt == "." and _rx.match(r"\.\p{Nd}", text[m.end(1):])):
        return None                                   # phone / time / date / version / code / currency-symbol shapes
    if _rx.search(r"\p{Nd}[ \u00a0\u202f]$", text[max(0, s - 2):s]) or _rx.match(r"[ \u00a0\u202f]\p{Nd}", text[m.end(1):]):
        return None                                   # space-separated digit groups ("1 500", "98765 43210")
    nw = _tn_next_word(text, m.end(1))
    if _tn_cue_before(text, s) or nw in _TN_ID_CUES or nw in _TN_CURRENCY_CODES:
        return None
    grouped = "," in d
    plain = d.replace(",", "")
    if grouped and not (_rx.fullmatch(r"\d{1,3}(,\d{3})+", d) or _rx.fullmatch(r"\d{1,2}(,\d{2})*,\d{3}", d)):
        return None
    if len(plain) > 1 and plain.startswith("0"):
        return None                                   # leading zero = identifier
    n = int(plain)
    unit = nw in _TN_UNIT_AFTER or bool(m.group(3) == "%")
    if not unit:
        return None                                   # conservative: a bare number could be a seat / bus / address / code; only numbers with a unit or % are quantities
    if not grouped and len(plain) >= 5:
        return None                                   # 5+ digits without grouping: phone / account / pin
    if lang in _TN_SKIP_ONE and n == 1:
        return None
    w = _tn_int_words(n, lang)
    if w is None:
        return None
    if m.group(3) == "%":
        pw = _TN_PCT.get(lang)
        if not pw:
            return None
        w = w + " " + pw
    return w


def _tn_numbers(text: str, lang: str) -> str:
    import regex as _rx
    if not _tn_inv(lang):
        return text
    pat = _rx.compile(r"(?<![\p{L}\p{M}\p{Nd}])(?<!\p{Nd}[,.])(\p{Nd}+(?:,\p{Nd}{2,3})*)(?:\.(\p{Nd}+))?(?![\p{L}\p{M}\p{Nd}])(?![,.]\p{Nd})(?:\s?(%)(?![\p{L}\p{M}\p{Nd}]))?")
    out, pos = [], 0
    for m in pat.finditer(text):
        w = _tn_quantity_words(text, m, lang)
        if w is None:
            continue
        out.append(text[pos:m.start()]); out.append(w); pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def normalize_numbers(text: str, lang: str) -> str:
    """Spoken form of a display line: clear quantities written as words, everything else untouched. Never raises."""
    if not text or len(text) > 2000 or lang not in _TN_NUM_LANGS or not _tn_flag("TTS_NORMALIZE", True) or not _tn_flag("TTS_NORMALIZE_NUMBERS", True):
        return text
    ck = (_TN_VER, text, lang)
    with _TN_LOCK:
        hit = _TN_CACHE.get(ck)
        if hit is not None:
            _TN_CACHE.move_to_end(ck)
            return hit
    try:
        out = _tn_numbers(text, lang)
    except Exception as e:
        log.warning("normalize_numbers failed (%s): keeping the text", str(e)[:100])
        out = text
    with _TN_LOCK:
        _TN_CACHE[ck] = out
        while len(_TN_CACHE) > _TN_CACHE_MAX:
            _TN_CACHE.popitem(last=False)
    return out
