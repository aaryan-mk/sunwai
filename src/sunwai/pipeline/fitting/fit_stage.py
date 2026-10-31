"""Duration-aware rewrite stage: shorten each translated line with Gemini so its TTS fits the slot.

Budget = voice rate (graphemes per second of TTS audio, measured on indic-speak) x slot duration.
The MT output is kept in seg["mt_text"]; seg["tgt_text"] becomes the fitted line. Cached in fit.json.
"""
import json
import logging
import os
import time
from pathlib import Path

import regex
import requests

log = logging.getLogger("dub_pipeline")

# measured 2026-10-27 on indic-speak; other voices fall back to the language default
VOICE_RATE = {"hi": 5.4, "ta": 6.86}
DEFAULT_RATE = 6.0
SLACK = 1.0          # budget = rate * slot * SLACK; the TTS stage absorbs up to atempo 1.25
from sunwai.paths import DATA_DIR  # noqa: E402
NAMES = {"hi": "Hindi", "ta": "Tamil", "mr": "Marathi", "ur": "Urdu", "bn": "Bengali", "te": "Telugu"}
_G = regex.compile(r"\X")

# target-script guard. Gemini fit/MT occasionally return text in the WRONG Indic script (te->ta job 24dae2ffe9d5:
# 4 of 18 Tamil lines were Telugu/Malayalam/Gujarati). Scripts accepted per target language; unmapped languages
# (en, sat, ...) are not checked.
_SCRIPTS = {
    "hi": ("Devanagari",), "mr": ("Devanagari",), "ne": ("Devanagari",), "sa": ("Devanagari",), "kok": ("Devanagari",),
    "mai": ("Devanagari",), "doi": ("Devanagari",), "brx": ("Devanagari",),
    "bn": ("Bengali",), "as": ("Bengali",), "mni": ("Bengali", "Meetei_Mayek"),
    "ta": ("Tamil",), "te": ("Telugu",), "gu": ("Gujarati",), "kn": ("Kannada",), "ml": ("Malayalam",),
    "sat": ("Ol_Chiki",), "pa": ("Gurmukhi",), "or": ("Oriya",), "ur": ("Arabic",), "ks": ("Arabic", "Devanagari"), "sd": ("Arabic", "Devanagari"),
}
_SCRIPT_RE = {}
_LM = regex.compile(r"[\p{L}\p{M}]")
_LATIN_WORD = regex.compile(r"\p{Script=Latin}{2,}")
_NEUTRAL = regex.compile(r"[\p{Script=Latin}\p{Script=Common}\p{Script=Inherited}]")


def script_share(text, lang):
    """Share of the letters/marks in `text` that belong to the target language's script, ignoring digits, punctuation,
    whitespace and Latin. None when the language is not checked or the text has no such letters."""
    names = _SCRIPTS.get(lang)
    if not names:
        return None
    rx = _SCRIPT_RE.get(lang)
    if rx is None:
        rx = _SCRIPT_RE[lang] = regex.compile("[" + "".join(r"\p{Script=%s}" % n for n in names) + "]")
    inn = tot = lat = 0
    for ch in text or "":
        if not _LM.match(ch):
            continue
        if _NEUTRAL.match(ch):
            if regex.match(r"\p{Script=Latin}", ch):
                lat += 1
            continue
        tot += 1
        if rx.match(ch):
            inn += 1
    # Latin is allowed for names/acronyms (a bare "Sachin Tendulkar" is a correct line), but a SENTENCE in Latin
    # (>= 5 Latin words and more Latin than target-script letters) is an English/romanised answer: share 0.
    if lat > tot and len(_LATIN_WORD.findall(text or "")) >= 5:
        return 0.0
    return (inn / tot) if tot else None


def script_ok(text, lang, thr=None, strict=False):
    """True unless the text is demonstrably in another script: share < SCRIPT_MIN_SHARE (default 0.9), or, with
    strict=True (used for the Gemini fit, whose fallback to the MT line is free), ANY letter of a foreign Indic/other
    script (the te->ta job had lines at 0.96 and 0.98 with a stray Malayalam word / Bengali letter)."""
    if thr is None:
        try:
            thr = float(os.environ.get("SCRIPT_MIN_SHARE", "").strip() or "0.9")
            if not (0.0 < thr <= 1.0):          # NaN fails this too
                raise ValueError
        except ValueError:
            thr = 0.9
    if strict:
        thr = 1.0
    sh = script_share(text, lang)
    return sh is None or sh >= thr


def _atomic_json(path, obj, **kw):
    """tmp + re-parse + os.replace: fit.json / segments.json are never visible half-written."""
    path = str(path)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, **kw)
    with open(tmp, encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp, path)


def graphemes(text):
    return sum(1 for g in _G.findall(text) if regex.search(r"[\p{L}\p{N}]", g))


def _llm_call(prompt, key=None):
    """name kept for callers; provider (gemini|openrouter) chosen by llm_client (LLM_PROVIDER). An explicit key is used for provider=gemini only."""
    from sunwai.clients import llm as llm_client
    return llm_client.generate_text(prompt, "fit", timeout=60, retries=5, key=key)


def _prompt(src, tgt, budget, name, src_name=None):
    """src_name: the actual source language; None = unknown / unconfident LID, so the prompt does not claim English."""
    sn = src_name or "source"
    return (f"You are a film dubbing translator. Rewrite the {name} line so it can be spoken in the same time as the {sn}-language line.\n"
            f"Target length: about {budget} {name} letters (graphemes; spaces and punctuation not counted). "
            f"Keep every piece of information from the {sn}-language line; shorten by choosing shorter words and phrasing and dropping "
            f"fillers, repetitions and politeness padding, never by dropping or changing facts, names, numbers (same value), negations or units. "
            f"Keep digits exactly as digits and spelled-out numbers spelled out. "
            f"Natural spoken {name}. Output only the {name} line.\n{sn[:1].upper() + sn[1:]}: {src}\nCurrent {name}: {tgt}")


# ---------------------------------------------------------------------------------------------------------------
# meaning-preservation guard. The fit prompt asks for shorter text, but acceptance used to be script share +
# word count + grapheme reduction only. A rewrite is now also checked against its MT input (and the source line) for
# numbers, Latin tokens, named entities (capitalised source words found again by a script-agnostic consonant skeleton),
# negation and units. Deterministic and partial by design: it catches clear losses, not every paraphrase.
# Kill switch: FIT_MEANING_GUARD=0.
# ---------------------------------------------------------------------------------------------------------------
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

SRC_NAMES = dict(NAMES, en="English", ml="Malayalam", kn="Kannada", gu="Gujarati", pa="Punjabi", or_="Odia", ne="Nepali",
                 sa="Sanskrit", as_="Assamese")
SRC_NAMES.update({"or": "Odia", "as": "Assamese"})


def guard_enabled():
    return os.environ.get("FIT_MEANING_GUARD", "1").strip().lower() not in ("0", "false", "off", "no")


def source_lang_code(out_dir):
    """(code, confident) of the job's spoken-language ID from lid.json (chunk dir, then the job dir next to SPEAKER_MAP).
    ('', False) when unknown."""
    for d in (out_dir, os.path.dirname(os.environ.get("SPEAKER_MAP", "")) or None):
        if not d:
            continue
        try:
            j = json.load(open(os.path.join(str(d), "lid.json"), encoding="utf-8"))
            code = (j.get("label") or "").split(":")[0].strip()
            if code:
                return code, float(j.get("prob") or 0) >= 0.7
        except Exception:
            continue
    return "", False


def confident_source_lang(out_dir):
    """Source-language code only when the LID is confident (prob >= 0.7); else None (generic checks)."""
    code, conf = source_lang_code(out_dir)
    return code if (code and conf) else None


def source_lang_name(out_dir):
    """Display name of the source language, or None when unknown / not confident (the prompt then says 'source')."""
    code, conf = source_lang_code(out_dir)
    return SRC_NAMES.get(code) if (code and conf) else None


def _nfc(w):
    """NFC + drop ZWNJ/ZWJ (U+200C/U+200D), so table keys and text tokens are spelled the same way (r+ZWJ+ya in Marathi etc.)."""
    return unicodedata.normalize("NFC", w).replace("\u200c", "").replace("\u200d", "").replace("\u0931", "\u0930")


_DIG = regex.compile(r"\p{Nd}+(?:[.,]\p{Nd}+)*")
_LAT_TOK = regex.compile(r"\p{Script=Latin}[\p{Script=Latin}\p{M}'’-]*\p{Script=Latin}|\p{Script=Latin}")
_TOK = regex.compile(r"[\p{L}\p{M}\p{Nd}]+")

# number words (partial: units, tens, powers) per target language; a digit run in the MT line may legitimately be rewritten
# into words (the prompt asks for it), so words count as "number present". Languages without a list must keep digits.
_NUMW = {
    "hi": "दो तीन चार पांच पाँच छह छः छे सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हज़ार हजार लाख करोड़ करोड मिलियन बिलियन शून्य पहला दूसरा तीसरा चौथा",
    "mr": "दोन तीन चार पाच सहा सात आठ नऊ दहा अकरा बारा वीस तीस चाळीस पन्नास साठ सत्तर ऐंशी नव्वद शंभर हजार लाख कोटी दशलक्ष अब्ज शून्य",
    "bn": "দুই দুটি তিন চার পাঁচ ছয় সাত আট দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ বিশ ত্রিশ চল্লিশ পঞ্চাশ ষাট সত্তর আশি নব্বই শত একশো শো হাজার লক্ষ লাখ কোটি মিলিয়ন বিলিয়ন শূন্য",
    "ta": "இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது பத்து பதினொன்று பன்னிரண்டு இருபது முப்பது நாற்பது ஐம்பது அறுபது எழுபது எண்பது தொண்ணூறு நூறு ஆயிரம் இலட்சம் லட்சம் கோடி மில்லியன் பில்லியன் பூஜ்ஜியம்",
    "te": "రెండు మూడు నాలుగు ఐదు ఆరు ఏడు ఎనిమిది తొమ్మిది పది పదకొండు పన్నెండు ఇరవై ముప్పై నలభై యాభై అరవై డెబ్బై ఎనభై తొంభై వంద వందల వేల వేయి లక్ష లక్షల కోటి మిలియన్ బిలియన్ సున్నా",
    "kn": "ಎರಡು ಮೂರು ನಾಲ್ಕು ಐದು ಆರು ಏಳು ಎಂಟು ಒಂಬತ್ತು ಹತ್ತು ಇಪ್ಪತ್ತು ಮೂವತ್ತು ನಲವತ್ತು ಐವತ್ತು ಅರವತ್ತು ಎಪ್ಪತ್ತು ಎಂಬತ್ತು ತೊಂಬತ್ತು ನೂರು ಸಾವಿರ ಲಕ್ಷ ಕೋಟಿ",
    "ml": "രണ്ട് മൂന്ന് നാല് അഞ്ച് ആറ് ഏഴ് എട്ട് ഒമ്പത് പത്ത് ഇരുപത് മുപ്പത് നാല്പത് അമ്പത് അറുപത് എഴുപത് എൺപത് തൊണ്ണൂറ് നൂറ് ആയിരം ലക്ഷം കോടി",
    "gu": "બે ત્રણ ચાર પાંચ છ સાત આઠ નવ દસ વીસ ત્રીસ ચાલીસ પચાસ સાઠ સિત્તેર એંસી નેવું સો હજાર લાખ કરોડ",
    "pa": "ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌਂ ਦਸ ਵੀਹ ਤੀਹ ਚਾਲੀ ਪੰਜਾਹ ਸੌ ਹਜ਼ਾਰ ਲੱਖ ਕਰੋੜ",
    "ur": "دو تین چار پانچ چھ سات آٹھ نو دس بیس تیس چالیس پچاس ساٹھ ستر اسی نوے سو ہزار لاکھ کروڑ ملین بلین",
}
_NUMW = {k: frozenset(_nfc(w) for w in v.split()) for k, v in _NUMW.items()}
_NUMW["mai"] = _NUMW["ne"] = _NUMW["hi"]

# negation (matched as whole tokens; for the agglutinative languages also as token substrings). Presence is compared, not counts.
_NEG_TOK = {
    "hi": "नहीं नही न मत बिना कभी_नहीं नहीं_तो ना", "mr": "नाही नका नये नसे नसून न ना", "ne": "छैन होइन नगर्नुहोस् न",
    "bn": "না নেই নয় নি নাই কখনো_না ছাড়া", "gu": "નથી ના નહીં નહિ", "pa": "ਨਹੀਂ ਨਾ ਨਹੀ", "ur": "نہیں نہ مت بغیر",
    "ta": "இல்லை வேண்டாம் அல்ல இல்லாமல் ஆனால்_இல்லை இல்லாத", "te": "కాదు లేదు లేరు లేవు వద్దు లేకుండా కాని",
    "kn": "ಅಲ್ಲ ಇಲ್ಲ ಬೇಡ ಇಲ್ಲದೆ", "ml": "അല്ല ഇല്ല വേണ്ട ഇല്ലാതെ",
}
_NEG_SUB = {"ta": ["இல்ல", "வேண்டா", "அல்ல", "மாட்டே", "மாட்டோ", "மாட்டா", "மாட்டீ"], "te": ["లేద", "కాద", "వద్ద", "లేక"],
            "kn": ["ಇಲ್ಲ", "ಅಲ್ಲ", "ಬೇಡ"], "ml": ["ഇല്ല", "അല്ല", "വേണ്ട", "ഇല്ലാ"]}
_NEG_END = {"ta": ["ாது"], "ur": ["نہیں"], "mr": ["नाही"]}
_NEG_END = {k: [_nfc(x) for x in v] for k, v in _NEG_END.items()}
_NEG_TOK = {k: frozenset(_nfc(t) for t in v.split() if "_" not in t) for k, v in _NEG_TOK.items()}
_NEG_SUB = {k: [_nfc(x) for x in v] for k, v in _NEG_SUB.items()}
_NEG_EN = frozenset("not no never none nothing nobody neither nor without cannot can't won't don't doesn't didn't isn't aren't wasn't weren't haven't hasn't hadn't couldn't wouldn't shouldn't".split())

# units / quantities that carry meaning (partial; time words and 'degree' are excluded: they have frequent legitimate synonyms/omissions). Symbols are checked in every language.
_UNIT_SYM = "%₹$€£°℃℉"
_UNITW = {
    "hi": "प्रतिशत डॉलर रुपये रुपए किलोमीटर किलो किलोग्राम मीटर सेंटीमीटर मील",
    "bn": "শতাংশ ডলার টাকা কিলোমিটার কিলো মিটার",
    "ta": "சதவீதம் டாலர் ரூபாய் கிலோமீட்டர் கிலோ மீட்டர்",
    "te": "శాతం డాలర్ రూపాయలు కిలోమీటర్ కిలో మీటర్",
    "mr": "टक्के डॉलर रुपये किलोमीटर किलो मीटर",
    "ur": "فیصد ڈالر روپے کلومیٹر کلو میٹر",
}
_UNITW = {k: frozenset(_nfc(w) for w in v.split()) for k, v in _UNITW.items()}

_COMMON_CAPS = frozenset('both being fit fixes only all not very really just more most some also then than here there what when where which while with from into over after before about because but and the for you your are was were has have had will would could should can may might must how why who this that these those yes no ok okay now new one two'.split())
_STOP_EN = frozenset("i i'm i've i'll i'd the a an and but or so if when then this that these those it he she they we you my his her our their".split())


def _norm_digits(tok):
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in tok).replace(",", "")


def _numbers(text):
    return Counter(_norm_digits(m) for m in _DIG.findall(text or ""))


def _has_digit_one(text):
    return _numbers(text).get("1", 0) > 0


def _toks(text):
    return [t.casefold() for t in _TOK.findall(_nfc(text or ""))]



# number words -> values (0-20, tens, hundred/thousand/lakh/crore) per language. Words are compared by VALUE together with digits.
def _mk(spec):
    out = {}
    for val, words in spec:
        for w in words.split():
            out[_nfc(w)] = val
    return out


_NUMV = {
    "hi": _mk([(0, "शून्य"), (1, "एक"), (2, "दो"), (3, "तीन"), (4, "चार"), (5, "पांच पाँच"), (6, "छह छः छे"), (7, "सात"), (8, "आठ"), (9, "नौ"), (10, "दस"),
               (11, "ग्यारह"), (12, "बारह"), (13, "तेरह"), (14, "चौदह"), (15, "पंद्रह पन्द्रह"), (16, "सोलह"), (17, "सत्रह"), (18, "अठारह"), (19, "उन्नीस"),
               (20, "बीस"), (30, "तीस"), (40, "चालीस"), (50, "पचास"), (60, "साठ"), (70, "सत्तर"), (80, "अस्सी"), (90, "नब्बे"), (100, "सौ"),
               (1000, "हज़ार हजार"), (100000, "लाख"), (10000000, "करोड़ करोड")]),
    "mr": _mk([(0, "शून्य"), (1, "एक"), (2, "दोन"), (3, "तीन"), (4, "चार"), (5, "पाच"), (6, "सहा"), (7, "सात"), (8, "आठ"), (9, "नऊ"), (10, "दहा"),
               (11, "अकरा"), (12, "बारा"), (13, "तेरा"), (14, "चौदा"), (15, "पंधरा"), (16, "सोळा"), (17, "सतरा"), (18, "अठरा"), (19, "एकोणीस"),
               (20, "वीस"), (30, "तीस"), (40, "चाळीस"), (50, "पन्नास"), (60, "साठ"), (70, "सत्तर"), (80, "ऐंशी"), (90, "नव्वद"), (100, "शंभर"),
               (1000, "हजार"), (100000, "लाख"), (10000000, "कोटी")]),
    "bn": _mk([(0, "শূন্য"), (1, "এক"), (2, "দুই দুটি দুটো"), (3, "তিন তিনটি"), (4, "চার"), (5, "পাঁচ"), (6, "ছয়"), (7, "সাত"), (2, "দু"), (8, "আট"), (10, "দশ"),
               (11, "এগারো"), (12, "বারো"), (13, "তেরো"), (14, "চৌদ্দ"), (15, "পনেরো"), (16, "ষোলো"), (17, "সতেরো"), (18, "আঠারো"), (19, "উনিশ"),
               (20, "বিশ"), (30, "ত্রিশ"), (40, "চল্লিশ"), (50, "পঞ্চাশ"), (60, "ষাট"), (70, "সত্তর"), (80, "আশি"), (90, "নব্বই"), (100, "একশো একশ শত"),
               (1000, "হাজার"), (100000, "লক্ষ লাখ"), (10000000, "কোটি")]),    # 9 'noy' is omitted: it is also 'is not'
    "ta": _mk([(0, "பூஜ்ஜியம்"), (1, "ஒன்று ஒரு"), (2, "இரண்டு"), (3, "மூன்று"), (4, "நான்கு"), (5, "ஐந்து"), (6, "ஆறு"), (7, "ஏழு"), (8, "எட்டு"), (9, "ஒன்பது"),
               (10, "பத்து"), (11, "பதினொன்று"), (12, "பன்னிரண்டு"), (13, "பதின்மூன்று"), (14, "பதினான்கு"), (15, "பதினைந்து"), (16, "பதினாறு"), (17, "பதினேழு"),
               (18, "பதினெட்டு"), (19, "பத்தொன்பது"), (20, "இருபது"), (30, "முப்பது"), (40, "நாற்பது"), (50, "ஐம்பது"), (60, "அறுபது"), (70, "எழுபது"),
               (80, "எண்பது"), (90, "தொண்ணூறு"), (100, "நூறு"), (1000, "ஆயிரம்"), (100000, "இலட்சம் லட்சம்"), (10000000, "கோடி")]),
    "te": _mk([(0, "సున్నా"), (1, "ఒకటి ఒక"), (2, "రెండు"), (3, "మూడు"), (4, "నాలుగు"), (5, "ఐదు"), (6, "ఆరు"), (7, "ఏడు"), (8, "ఎనిమిది"), (9, "తొమ్మిది"),
               (10, "పది"), (11, "పదకొండు"), (12, "పన్నెండు"), (13, "పదమూడు"), (14, "పద్నాలుగు"), (15, "పదిహేను"), (16, "పదహారు"), (17, "పదిహేడు"),
               (18, "పద్దెనిమిది"), (19, "పంతొమ్మిది"), (20, "ఇరవై"), (30, "ముప్పై"), (40, "నలభై"), (50, "యాభై"), (60, "అరవై"), (70, "డెబ్బై"),
               (80, "ఎనభై"), (90, "తొంభై"), (100, "వంద"), (1000, "వేయి వెయ్యి"), (100000, "లక్ష"), (10000000, "కోటి")]),
    "en": _mk([(0, "zero"), (1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five"), (6, "six"), (7, "seven"), (8, "eight"), (9, "nine"), (10, "ten"),
               (11, "eleven"), (12, "twelve"), (13, "thirteen"), (14, "fourteen"), (15, "fifteen"), (16, "sixteen"), (17, "seventeen"), (18, "eighteen"),
               (19, "nineteen"), (20, "twenty"), (30, "thirty"), (40, "forty"), (50, "fifty"), (60, "sixty"), (70, "seventy"), (80, "eighty"), (90, "ninety"),
               (100, "hundred"), (1000, "thousand thousands"), (100000, "lakh"), (1000000, "million"), (1000000000, "billion"), (10000000, "crore")]),
}
_NUMV["ne"] = _NUMV["mai"] = _NUMV["hi"]
_BN_COUNTERS = tuple(_nfc(x) for x in ("জন", "টি", "টা", "খানা", "টো"))
_BN_NINE = _nfc("নয়")
_BN_UNIT_NOUNS = frozenset(_nfc(x) for x in "বছর মিনিট ঘণ্টা ঘন্টা দিন মাস সপ্তাহ টাকা ডলার কিলোমিটার কিলো মিটার শতাংশ জন লোক সেকেন্ড".split())
_NUM_PREFIX = frozenset("ta te".split())      # case / emphatic suffixes are glued to the number word


def _num_values(text, lang):
    """Counter of numeric VALUES in `text`: digits (any script, normalised) and number words of `lang` (see _NUMV).
    Unmapped compound words (e.g. 45 written out in Hindi) are ignored; English tens+units are combined (twenty five = 25)."""
    c = Counter()
    for k, v in _numbers(text).items():
        try:
            c[int(k)] += v
        except ValueError:
            c[k] += v
    tab = _NUMV.get(lang)
    if not tab:
        return c
    prev = None
    toks = _toks(text)
    for ti, t in enumerate(toks):
        val = tab.get(t)
        if val is None and lang == "bn":
            for suf in _BN_COUNTERS:
                if t.endswith(suf) and len(t) > len(suf) and t[:-len(suf)] in tab:
                    val = tab[t[:-len(suf)]]
                    break
            if val is None:
                for suf in _BN_COUNTERS:
                    if t.endswith(suf) and t[:-len(suf)] == _BN_NINE:       # noy + any counter suffix (-khana, -ti, -ta, -jon, -to)
                        val = 9
                        break
            if val is None and t == _BN_NINE:       # bare 'noy' is nine only before a unit/noun or after another numeral; else 'is not'
                nxt = toks[ti + 1] if ti + 1 < len(toks) else ""
                if prev is not None or nxt in _BN_UNIT_NOUNS:
                    val = 9
        if val is None and lang in _NUM_PREFIX:
            for w in sorted((w for w in tab if len(w) >= 4 and t.startswith(w) and len(t) - len(w) <= 4), key=len, reverse=True):
                val = tab[w]
                break
        if val is None:
            prev = None
            continue
        if lang == "en" and prev and 20 <= prev <= 90 and 1 <= val <= 9:
            c[prev] -= 1
            if c[prev] <= 0:
                del c[prev]
            val = prev + val
        c[val] += 1
        prev = val if lang in ("en", "bn") else None
    return c



_HI100 = 'शून्य|एक|दो|तीन|चार|पाँच पांच|छह छः छे|सात|आठ|नौ|दस|ग्यारह|बारह|तेरह|चौदह|पंद्रह पन्द्रह|सोलह|सत्रह|अठारह|उन्नीस|बीस|इक्कीस|बाईस|तेईस|चौबीस|पच्चीस|छब्बीस|सत्ताईस|अट्ठाईस अठाईस|उनतीस|तीस|इकतीस|बत्तीस|तैंतीस|चौंतीस|पैंतीस|छत्तीस|सैंतीस|अड़तीस|उनतालीस|चालीस|इकतालीस|बयालीस|तैंतालीस|चौवालीस चौंतालीस|पैंतालीस|छियालीस|सैंतालीस|अड़तालीस|उनचास|पचास|इक्यावन|बावन|तिरपन तिरेपन|चौवन|पचपन|छप्पन|सत्तावन|अट्ठावन अठावन|उनसठ|साठ|इकसठ|बासठ|तिरसठ तिरेसठ|चौंसठ|पैंसठ|छियासठ|सड़सठ|अड़सठ|उनहत्तर|सत्तर|इकहत्तर|बहत्तर|तिहत्तर|चौहत्तर|पचहत्तर|छिहत्तर|सतहत्तर|अठहत्तर|उनासी|अस्सी|इक्यासी|बयासी|तिरासी|चौरासी|पचासी|छियासी|सत्तासी|अट्ठासी|नवासी|नब्बे|इक्यानवे|बानवे|तिरानवे|चौरानवे|पंचानवे|छियानवे|सत्तानवे|अट्ठानवे|निन्यानवे|सौ'
_MR100 = 'शून्य|एक|दोन|तीन|चार|पाच|सहा|सात|आठ|नऊ|दहा|अकरा|बारा|तेरा|चौदा|पंधरा|सोळा|सतरा|अठरा|एकोणीस|वीस|एकवीस|बावीस|तेवीस|चोवीस|पंचवीस|सव्वीस|सत्तावीस|अठ्ठावीस अठाविस|एकोणतीस|तीस|एकतीस|बत्तीस|तेहतीस|चौतीस|पस्तीस|छत्तीस|सदतीस|अडतीस|एकोणचाळीस|चाळीस|एक्केचाळीस|बेचाळीस|त्रेचाळीस|चव्वेचाळीस|पंचेचाळीस|शेहेचाळीस सेहेचाळीस|सत्तेचाळीस|अठ्ठेचाळीस|एकोणपन्नास|पन्नास|एक्कावन्न|बावन्न|त्रेपन्न|चोपन्न|पंचावन्न|छप्पन्न|सत्तावन्न|अठ्ठावन्न|एकोणसाठ|साठ|एकसष्ट|बासष्ट|त्रेसष्ट|चौसष्ट|पासष्ट|सहासष्ट|सदुसष्ट|अडुसष्ट|एकोणसत्तर|सत्तर|एकाहत्तर|बाहत्तर|त्र्याहत्तर|चौर्\u200dयाहत्तर|पंच्याहत्तर|शहात्तर|सत्याहत्तर|अठ्ठ्याहत्तर|एकोणऐंशी|ऐंशी|एक्क्याऐंशी|ब्याऐंशी|त्र्याऐंशी|चौऱ्याऐंशी|पंच्याऐंशी|शहाऐंशी|सत्त्याऐंशी|अठ्ठ्याऐंशी|एकोणनव्वद|नव्वद|एक्क्याण्णव|ब्याण्णव|त्र्याण्णव|चौऱ्याण्णव|पंच्याण्णव|शहाण्णव|सत्त्याण्णव|अठ्ठ्याण्णव|नव्व्याण्णव|शंभर'
for _l, _s in (("hi", _HI100), ("mr", _MR100)):
    for _v, _ws in enumerate(_s.split("|")):
        for _w in _ws.split():
            _NUMV[_l][_nfc(_w)] = _v
_NUMV["ne"] = _NUMV["mai"] = _NUMV["hi"]

# Fail-closed numeral lexicon for languages WITHOUT a full 1-100 table (bn, ta, te, ...): a token that starts with a numeral stem
# but is not an exactly mapped word is an "unverified numeral"; it must survive the rewrite verbatim.
_STEMS = {k: sorted({w for w in v if len(w) >= 3} - {w for w, x in _NUMV.get(k, {}).items() if x == 1}, key=len, reverse=True)
          for k, v in _NUMW.items() if k not in ("hi", "mr", "ne", "mai")}
_STEMS["bn"] = [w for w in _STEMS.get("bn", []) if len(w) >= 4] + [_nfc(x) for x in "এগার পনের আঠার চল্লিশ পঞ্চাশ ত্রিশ উনিশ ষোল সতের".split()]

_STEMS["bn"] += [_nfc(x) for x in "একুশ বাইশ তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আটাশ ঊনত্রিশ একত্রিশ বত্রিশ তেত্রিশ চৌত্রিশ পঁয়ত্রিশ ছত্রিশ সাঁইত্রিশ আটত্রিশ ঊনচল্লিশ একচল্লিশ বিয়াল্লিশ তেতাল্লিশ চুয়াল্লিশ পঁয়তাল্লিশ ছেচল্লিশ সাতচল্লিশ আটচল্লিশ ঊনপঞ্চাশ একান্ন বায়ান্ন তিপ্পান্ন চুয়ান্ন পঞ্চান্ন ছাপ্পান্ন সাতান্ন আটান্ন ঊনষাট একষট্টি বাষট্টি তেষট্টি চৌষট্টি পঁয়ষট্টি ছেষট্টি সাতষট্টি আটষট্টি ঊনসত্তর একাত্তর বাহাত্তর তিয়াত্তর চুয়াত্তর পঁচাত্তর ছিয়াত্তর সাতাত্তর আটাত্তর ঊনআশি একাশি বিরাশি তিরাশি চুরাশি পঁচাশি ছিয়াশি সাতাশি অষ্টাশি ঊননব্বই একানব্বই বিরানব্বই তিরানব্বই চুরানব্বই পঁচানব্বই ছিয়ানব্বই সাতানব্বই আটানব্বই নিরানব্বই".split()]
# sandhi forms: tens + unit are glued (ta) so only the tens/teen stem is stable
_STEMS["ta"] = _STEMS.get("ta", []) + [_nfc(x) for x in "இருபத் முப்பத் நாற்பத் ஐம்பத் அறுபத் எழுபத் எண்பத் தொண்ணூற் பதின் பத்தொன் நூற் ஆயிர".split()]
_STEMS["te"] = _STEMS.get("te", []) + [_nfc(x) for x in "ఇరవై ముప్పై నలభై యాభై అరవై డెబ్బై ఎనభై తొంభై పద".split()]


def _unverified_numerals(text, lang):
    st = _STEMS.get(lang)
    if not st:
        return Counter()
    tab = _NUMV.get(lang, {})
    def resolved(t):        # bn counter forms (baro+ta, noy+khana ...) are verified by VALUE in _num_values
        return lang == "bn" and any(t.endswith(s) and (t[:-len(s)] in tab or t[:-len(s)] == _BN_NINE) for s in _BN_COUNTERS)
    return Counter(t for t in _toks(text) if t not in tab and not resolved(t) and any(t.startswith(s) for s in st))


_AGGL = frozenset("ta te kn ml".split())


def _count_numw(toks, lang):
    w = _NUMW.get(lang) or frozenset()
    if lang in _AGGL:       # case/emphatic suffixes are glued to the number word (mūnru-thān)
        return sum(1 for t in toks if t in w or any(len(x) >= 3 and t.startswith(x) for x in w))
    return sum(1 for t in toks if t in w)


def _neg(text, lang):
    if lang == "en":
        return sum(1 for w in regex.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").casefold().replace("\u2019", "'")) if w in _NEG_EN or w.endswith("n't"))
    toks = _toks(text)
    st = _NEG_TOK.get(lang, frozenset())
    n = sum(1 for t in toks if t in st)
    for sub in _NEG_SUB.get(lang, ()):
        n += sum(1 for t in toks if sub in t)
    for suf in _NEG_END.get(lang, ()):
        n += sum(1 for t in toks if t.endswith(suf) and len(t) >= 4)
    return n


# --- script-agnostic consonant skeleton: every Indic block is mapped onto Devanagari by its ISCII offset, then onto a
# coarse Latin consonant class; vowels are dropped. Used only to find, in the MT/fit line, the transliteration of an
# English capitalised word (entity) and to compare it with the rewrite.
_BASES = [(0x0900, 0x0980), (0x0980, 0x0980), (0x0A00, 0x0980), (0x0A80, 0x0980), (0x0B00, 0x0980), (0x0B80, 0x0980),
          (0x0C00, 0x0980), (0x0C80, 0x0980), (0x0D00, 0x0980)]
_DEV = {"क": "k", "ख": "k", "ग": "g", "घ": "g", "ङ": "n", "च": "s", "छ": "s", "ज": "s", "झ": "s", "ञ": "n", "ट": "t", "ठ": "t",
        "ड": "d", "ढ": "d", "ण": "n", "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n", "प": "p", "फ": "f", "ब": "b", "भ": "b",
        "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v", "श": "s", "ष": "s", "स": "s", "ह": "h", "ज़": "z", "फ़": "f"}
_ARAB = {"ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "c", "ح": "h", "خ": "k", "د": "d", "ڈ": "d", "ذ": "z",
         "ر": "r", "ڑ": "r", "ز": "z", "ژ": "z", "س": "s", "ش": "s", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ق": "k", "ک": "k",
         "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "ہ": "h", "ھ": "h", "و": "v", "ی": "y", "ف": "f", "غ": "g", "ع": ""}
_LATCLS = {"c": "k", "q": "k", "x": "k", "z": "j", "w": "v", "f": "p", "g": "g", "d": "d", "b": "b", "k": "k", "j": "j", "y": "y"}


def _skeleton(tok):
    out = []
    for ch in unicodedata.normalize("NFC", tok.casefold()):
        o = ord(ch)
        if ch in _ARAB:
            c = _ARAB[ch]
        elif "a" <= ch <= "z":
            c = ch if ch not in "aeiou" else ""
        else:
            c = ""
            for base, _ in _BASES:
                if base <= o < base + 0x80:
                    c = _DEV.get(chr(0x0900 + (o - base)), "")
                    break
        # coarse classes: velar k/g, dental t/d, labial p/b/f/v/w, sibilant s/c(h)/j/z (Indic scripts mark far fewer contrasts)
        c = {"c": "k", "q": "k", "x": "ks", "g": "k", "d": "t", "b": "p", "f": "p", "v": "p", "w": "p", "j": "s", "z": "s", "h": ""}.get(c, c)
        for c1 in c:
            if not out or out[-1] != c1:
                out.append(c1)
    return "".join(out)


def _src_entities(src_text):
    """Capitalised source words that are not sentence-initial (and not 'I'): crude English NER."""
    ents = []
    words_all = regex.findall(r"[A-Za-z]{2,}", src_text or "")
    if words_all and sum(1 for w in words_all if w.isupper()) / len(words_all) > 0.5:
        return []           # ALL-CAPS ASR text: capitalisation carries no entity information
    for sent in regex.split(r"(?<=[.!?])\s+", src_text or ""):
        words = regex.findall(r"[\p{Script=Latin}][\p{Script=Latin}'’-]*", sent)
        for i, w in enumerate(words):
            if i == 0 or not w[0].isupper() or w.casefold() in _STOP_EN or len(w) < 3 or (w.isupper() and w.casefold() in _COMMON_CAPS):
                continue
            ents.append(w)
    return ents


def _src_neg(src_text, src_lang):
    """True/False: does the source line contain a negation; None when it cannot be told (unknown non-English language)."""
    if not src_text:
        return None
    lat = sum(1 for c in src_text if c.isalpha() and c.isascii())
    alpha = sum(1 for c in src_text if c.isalpha())
    if (src_lang == "en") or (not src_lang and alpha and lat / alpha > 0.8):
        return any((w.replace("’", "'") in _NEG_EN) or w.replace("’", "'").endswith("n't")
                   for w in regex.findall(r"[A-Za-z]+(?:['’][a-z]+)?", src_text.casefold()))
    if src_lang in _NEG_TOK:
        return _neg(src_text, src_lang) > 0
    return None


def _near(a, b, thr):
    return bool(a) and bool(b) and SequenceMatcher(None, a, b).ratio() >= thr


def meaning_issues(mt, new, lang, src_text=None, src_lang=None):
    """Return a list of short reason strings for facts present in the MT line but lost or changed in the rewrite `new`
    ([] = preserved). Checks: number values, Latin tokens, English-source named entities, negation, units."""
    issues = []
    mt_t, new_t = _toks(mt), _toks(new)
    # numbers: compare VALUES (digits and number words alike). A lost word-1 ('ek'/'oru') is ignored: it is mostly the article.
    vm, vn = _num_values(mt, lang), _num_values(new, lang)
    vs = (_num_values(src_text, src_lang or "en") | _num_values(src_text, "en")) if src_text else Counter()
    d_lost, d_extra = vm - vn, vn - vm
    # the ONLY exemption: exactly one spelled 'one' in the MT line (not a digit) and none (digit or word) in the rewrite
    if d_lost.get(1) and vm.get(1) == 1 and not vn.get(1) and not _has_digit_one(mt):
        del d_lost[1]
    lost = dict(d_lost)
    if lost:
        issues.append("number_lost:" + ",".join(sorted(str(v) for v in lost)))
    extra = {v: n for v, n in d_extra.items() if vn[v] > vs.get(v, 0)}      # multiplicity: occurrences beyond MT and source
    if extra:
        issues.append("number_added:" + ",".join(sorted(str(v) for v in extra)))
    ul = _unverified_numerals(mt, lang) - _unverified_numerals(new, lang)
    if ul:
        issues.append("numeral_unverified_lost:" + ",".join(sorted(ul)))
    # latin tokens (acronyms, brand names kept untransliterated by MT)
    latin_chk = lang in _SCRIPTS         # Latin-script targets (en, ...): every word is a Latin token, so Latin/entity checks do not apply
    ml = Counter(t.casefold() for t in _LAT_TOK.findall(mt or "")) if latin_chk else Counter()
    nl = Counter(t.casefold() for t in _LAT_TOK.findall(new or "")) if latin_chk else Counter()
    lost = ml - nl
    if lost:
        new_sk = [_skeleton(t) for t in new_t]
        new_cat = "".join(new_sk)
        for lt in sorted(lost):
            ls = _skeleton(lt)
            if len(ls) >= 3 and any(_near(k, ls, 0.7) for k in new_sk):
                continue            # transliterated into the target script
            if len(ls) < 3 and ls and ls in new_cat:
                continue
            issues.append("latin_lost:" + lt)
    # named entities from an English source: find the MT transliteration by skeleton, require it again in the rewrite
    for ent in (_src_entities(src_text) if latin_chk else []):
        es = _skeleton(ent)
        if len(es) < 3:
            continue
        best = max(mt_t, key=lambda t: SequenceMatcher(None, _skeleton(t), es).ratio(), default=None)
        if best is None or not _near(_skeleton(best), es, 0.8):
            continue
        hit, hs = best, _skeleton(best)
        if not any(t == hit or _near(_skeleton(t), hs, 0.75) for t in new_t) and ent.casefold() not in (x.casefold() for x in _LAT_TOK.findall(new or "")):
            issues.append("entity_lost:" + ent)
    # negation (presence)
    a, b = _neg(mt, lang), _neg(new, lang)
    if lang in _NEG_TOK or lang == "en":
        sn = _src_neg(src_text, src_lang)
        if sn is None:
            lost_, added_ = bool(a) and not b, bool(b) and not a
        else:           # judge against the source: an MT-only idiom ('jab tak ... nahin') is not a negation to preserve
            lost_, added_ = sn and a and not b, (not sn) and not a and bool(b)
        if lost_:
            issues.append("negation_lost")
        elif added_:
            issues.append("negation_added")
    # units
    sym_m = Counter(c for c in (mt or "") if c in _UNIT_SYM)
    sym_n = Counter(c for c in (new or "") if c in _UNIT_SYM)
    if sym_m - sym_n:
        issues.append("unit_symbol_lost:" + "".join(sorted(sym_m - sym_n)))
    uw = _UNITW.get(lang, frozenset())
    ulost = sorted({t for t in mt_t if t in uw} - set(new_t))
    if ulost:
        issues.append("unit_lost:" + ",".join(ulost))
    return issues


def _accept(mt, new, lang, src_text, check_words=True, src_lang=None):
    """Shared acceptance for one rewrite: returns (new_or_None, reason). Script, word-split, grapheme, meaning."""
    if not new:
        return None, "empty"
    if not script_ok(new, lang, strict=True):
        return None, "script"
    if check_words and len(new.split()) > len(mt.split()):
        return None, "word_split"
    if graphemes(new) >= graphemes(mt):
        return None, "not_shorter"
    if guard_enabled():
        iss = meaning_issues(mt, new, lang, src_text, src_lang)
        if iss:
            return None, "meaning:" + ";".join(iss)
    return new, "ok"


def must_keep_tokens(mt, lang, src_text=None, cap=20):
    """(d): the surface tokens of the MT line that the meaning guard counts (numbers by VALUE incl. number words, with multiplicity; Latin brand
    tokens; negation words; unit words/symbols), built from the guard's own extractors, as 'token' or 'token (xN)'. The rewriter must keep them verbatim."""
    out = []
    try:
        otoks = _TOK.findall(_nfc(mt or ""))            # ORIGINAL-case tokens; toks[i] is otoks[i].casefold() (same as _toks)
        toks = [t.casefold() for t in otoks]
        percat = {}
        order = []
        surface = {}
        def add(t, cat):
            k = t.casefold()
            if k not in surface:
                order.append(k)
                surface[k] = t          # keep the ORIGINAL spelling/case (iPhone, NASA) for the prompt; count case-insensitively like the guard
            percat.setdefault(k, Counter())[cat] += 1      # the same occurrence can be seen by several extractors (e.g. 'kg' = Latin token AND unit word)
        for m in _DIG.findall(mt or ""):
            add(m, "num")
        for i, t in enumerate(toks):
            if _num_values(t, lang) and not t.isdigit():
                add(otoks[i], "numw")                              # number word (ek / oru / three ...)
        if lang in _SCRIPTS:
            for t in _LAT_TOK.findall(mt or ""):
                add(t, "lat")
        if lang == "en":
            for w in regex.findall(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?", mt or ""):          # original case; casefold only for the lookup
                wl = w.casefold().replace("\u2019", "'")
                if wl in _NEG_EN or wl.endswith("n't"):
                    add(w, "neg")
        else:
            st = _NEG_TOK.get(lang, frozenset())
            for i, t in enumerate(toks):
                if t in st:
                    add(otoks[i], "neg")
        uw = _UNITW.get(lang, frozenset())
        for i, t in enumerate(toks):
            if t in uw:
                add(otoks[i], "unit")
        for c in (mt or ""):
            if c in _UNIT_SYM:
                add(c, "sym")
        for k in order[:cap]:
            n = max(percat[k].values())                    # overlapping categories describe the SAME occurrences: max, not sum
            out.append(surface[k] if n == 1 else f"{surface[k]} (x{n})")
    except Exception:
        return []
    return out


def guarded_rewrite(src_text, mt, budget, name, lang, key, src_name=None, tag="fit", call=None, src_lang=None):
    """One Gemini rewrite of `mt` to about `budget` graphemes with the full acceptance (script, word split, shorter,
    meaning guard) and ONE retry that names what the first answer lost. Returns the accepted line or None (caller keeps
    the MT line and lets tempo fit absorb it). Used by stage_fit and by the stage_tts refit path."""
    call = call or _llm_call
    reason = ""
    keep = must_keep_tokens(mt, lang, src_text) if os.environ.get("FIT_KEEP_LIST", "1").strip().lower() not in ("0", "false", "no", "off", "n", "f") else []
    kl = ("\nThese tokens from the current line must stay in your answer exactly as written (same spelling, same count): " + "; ".join(keep) + ".") if keep else ""
    prev = ""
    for attempt in (0, 1, 2) if keep else (0, 1):
        p = _prompt(src_text, mt, budget, name, src_name)
        if keep:
            p += kl + " Shorten ONLY by removing or replacing other words."
        if attempt and reason.startswith(("meaning:", "script")):
            p += ("\nYour previous attempt was rejected (" + reason + "). Redo it: keep every number with the same value, every name, "
                  f"every negation and every unit exactly, and write only in {name}.")
            if attempt == 2 and prev:
                _q = " ".join(str(prev).split())[:160].replace("<", " ").replace(">", " ")      # bounded + delimited: data, never instructions
                p += (f"\nLAST ATTEMPT. Your previous (rejected) answer, quoted as data between <<< and >>>: <<<{_q}>>>\nIt lost required tokens. Write it again with EVERY required token present the stated number of times; "
                      f"if you cannot reach the length without dropping one, keep the token and drop other words instead.")
        new = call(p, key)
        prev = new or prev
        ok, reason = _accept(mt, new, lang, src_text, src_lang=src_lang)
        if ok:
            return ok
        log.warning("%s rewrite rejected (%s)%s: %.60s", tag, reason, "; retrying" if attempt < (2 if keep else 1) else "", new or "")
        if reason not in ("empty",) and not reason.startswith(("meaning:", "script")):
            return None             # word split / not shorter: a retry will not help
    return None


_VOICE_RATES_CACHE = {}


def _valid_voice_rates(d):
    """{lang: {"ref": str, "rates": {voice: positive number}}} -- anything else is rejected (the last valid table stays in force)."""
    try:
        if not isinstance(d, dict) or not d:
            return False
        for lg, e in d.items():
            if not (isinstance(lg, str) and isinstance(e, dict) and isinstance(e.get("ref"), str) and isinstance(e.get("rates"), dict) and e["rates"]):
                return False
            if not all(isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for k, v in e["rates"].items()):
                return False
            if e["ref"] not in e["rates"]:
                return False
        return True
    except Exception:
        return False


def fit_voice_factor(lang, voice):
    """the fit budget (graphemes per slot-second) is calibrated per LANGUAGE on the old reference voice; a slower voice then overflows the slot.
    factor = measured g/s of this voice / measured g/s of the language's reference voice (voice_rates.json next to this file, measured on real chunks).
    Clamped to [0.7, 1.15]; 1.0 when unknown or FIT_VOICE_RATE=0."""
    if os.environ.get("FIT_VOICE_RATE", "1").strip().lower() in ("0", "false", "no", "off", "n", "f") or not voice:
        return 1.0
    try:
        p = DATA_DIR / "voice_rates.json"
        try:
            mt = p.stat().st_mtime
            if _VOICE_RATES_CACHE.get("m") != mt:
                _VOICE_RATES_CACHE["m"] = mt              # recorded BEFORE parsing: a broken file is parsed once, not on every segment
                cand = json.load(open(p, encoding="utf-8"))
                if _valid_voice_rates(cand):
                    _VOICE_RATES_CACHE["d"] = cand
                # structurally invalid: keep the last valid table
        except Exception:
            if "d" not in _VOICE_RATES_CACHE:       # never loaded: legacy behaviour; once loaded the last VALID table is kept (no cache-invalidation storm)
                return 1.0
        d = _VOICE_RATES_CACHE["d"].get(lang) or {}
        ref = (d.get("rates") or {}).get(d.get("ref"))
        v = (d.get("rates") or {}).get(voice)
        if not ref or not v:
            return 1.0
        return max(0.7, min(1.15, float(v) / float(ref)))
    except Exception:
        return 1.0


def stage_fit(args, out_dir: Path, segments):
    cache = out_dir / "fit.json"
    fitted = None
    if cache.exists():
        try:
            fitted = json.load(open(cache, encoding="utf-8"))
        except Exception as e:
            log.warning("fit.json unreadable (%s); recomputing", e)
    todo = segments
    _cs = confident_source_lang(out_dir)
    rate0 = VOICE_RATE.get(args.lang, DEFAULT_RATE)
    try:
        _t0 = next((x for x in json.load(open(DATA_DIR / "languages.json", encoding="utf-8")).get("targets", []) if x["code"] == args.lang), None)
        if _t0:
            rate0 = float(_t0.get("rate") or rate0)
    except Exception:
        pass
    _pv = getattr(args, "pick_voice", None)          # set by dub_pipeline: the SAME pure function (lang, gender, speaker; planned map fixed per process) the TTS stage calls

    def _fit_budget(s):
        try:
            vf = fit_voice_factor(args.lang, _pv(args.lang, s.get("gender"), s.get("speaker"))) if _pv else 1.0
        except Exception:
            vf = 1.0
        return max(4, int(rate0 * vf * (s["end"] - s["start"]) * SLACK))

    if fitted is not None:
        todo = []
        for s in segments:
            if s.get("edited"):
                continue            # never re-fit (rewrite) a user-edited line
            f = fitted.get(str(s["id"]))
            if f and s.get("tgt_text") == f["mt_text"] and f.get("budget") == _fit_budget(s) and not (
                    guard_enabled() and f["fit_text"] != f["mt_text"]
                    and meaning_issues(f["mt_text"], f["fit_text"], args.lang, s.get("src_text"), _cs)):
                s["mt_text"], s["tgt_text"] = f["mt_text"], f["fit_text"]
            else:           # a cached fit that fails the meaning guard is re-fitted through guarded_rewrite
                todo.append(s)      # new / changed MT text (e.g. a recovered MT fallback) must still be fitted
        if not todo:
            log.info("Stage fit: cached")
            return segments
        log.info("Stage fit: cached for %d segments, fitting %d new/changed", len(segments) - len(todo), len(todo))
    else:
        fitted = {}
        todo = [s for s in segments if not s.get("edited")]
    from sunwai.clients import llm as llm_client; key = llm_client.legacy_gemini_key()
    rate, name = VOICE_RATE.get(args.lang, DEFAULT_RATE), NAMES.get(args.lang, args.lang)
    try:
        cfg = json.load(open(DATA_DIR / "languages.json", encoding="utf-8"))
        t = next((x for x in cfg.get("targets", []) if x["code"] == args.lang), None)
        if t:
            rate, name = float(t.get("rate") or rate), t.get("name") or name
    except Exception as e:
        log.warning("languages.json not used: %s", e)
    n_rewritten = 0
    src_name = source_lang_name(out_dir)
    src_code = confident_source_lang(out_dir)
    for s in todo:
        mt = s.get("tgt_text") or ""
        s["mt_text"] = mt
        budget = _fit_budget(s)       # language rate x voice factor (was: language rate only)
        # skip failed/untranslated/fallback segments and lines that already fit
        if s.get("error") or s.get("fallback") or not mt or mt == s.get("src_text") or graphemes(mt) <= budget:
            fitted[str(s["id"])] = {"mt_text": mt, "fit_text": mt, "budget": budget}
            continue
        try:
            new = guarded_rewrite(s["src_text"], mt, budget, name, args.lang, key, src_name, src_lang=src_code, tag=f"fit seg {s['id']}")
            if new:
                s["tgt_text"] = new
                n_rewritten += 1
        except Exception as e:
            log.error("fit error seg %s: %s", s["id"], e)
        fitted[str(s["id"])] = {"mt_text": mt, "fit_text": s["tgt_text"], "budget": budget}
    _atomic_json(cache, fitted, indent=1)
    _atomic_json(out_dir / "segments.json", segments, indent=2)
    log.info("Stage fit: rewrote %d/%d segments (rate %.2f g/s)", n_rewritten, len(segments), rate)
    return segments
