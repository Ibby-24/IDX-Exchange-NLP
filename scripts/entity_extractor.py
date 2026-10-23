"""
Week 3: rule-based entity extraction from listing remarks.

Extracts bedrooms, bathrooms, square footage, list price and taxonomy amenities.

Design notes:
* The extractor works on text that went through the Week 2 ``TextCleaner``.
  ``extract_all(text)`` cleans for you by default (the cleaner is idempotent),
  so character offsets always refer to the *cleaned* text.
* Every numeric entity is found as a list of **mentions**. Each mention gets a
  *role* (main, partial, secondary_unit, market, records, conditional, range, ...)
  and only ``main`` mentions can become the answer. This is what keeps
  "two additional bedrooms", "a 1-bedroom ADU" or "$250 HOA" from being reported
  as the property's bedrooms or price.
* Price is cue-based (offered at, listed at, price improvement to ...). A bare
  "5+ digit number" rule would pick up zip codes, APNs, HUD case numbers and
  street numbers.
* Amenities are the Week 1 taxonomy terms, matched with word boundaries and
  longest-match, then filtered for negation, hypothetical use ("ideal for a home
  office") and community-only amenities ("community pool").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAXONOMY = ROOT / "data" / "processed" / "taxonomy.json"

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUMW = "|".join(NUMBER_WORDS)
_FRACTIONS = {"1/2": 0.5, "1/4": 0.25, "3/4": 0.75}

#context vocabularies (kept as data so they are easy to extend after error analysis)
PARTIAL_GAP_WORDS = {
    #words between the number and "bedroom" that make it a partial count
    "additional", "more", "other", "secondary", "extra", "remaining", "upstairs",
    "downstairs", "guest", "separate", "spare", "second", "half", "powder",
}
BAD_GAP_WORDS = {"of", "bath", "bathroom", "bathrooms", "baths", "bed", "beds", "bedroom", "bedrooms",
                 "car", "story", "stories", "floor", "floors", "acre", "acres", "unit", "units", "the",
                 "and", "plus", "with", "or"} | set(NUMBER_WORDS)
FLOOR_CUE_BEFORE = re.compile(
    r"(?:\b(?:main|first|ground|second|third|top|lower|upper)\s+(?:level|floor)\b|"
    r"\b(?:upstairs|downstairs)\b)[^.;!]{0,25}$")
FLOOR_CUE_AFTER = re.compile(
    r"^[\s,]*(?:are\s+|is\s+)?(?:located\s+)?(?:\w+\s+)?(?:on|in|at)\s+the\s+"
    r"(?:main|first|second|third|upper|lower|ground|top)\b|^[\s,]*(?:upstairs|downstairs)\b")
UNIT_CUE_BEFORE = re.compile(
    r"\b(?:adu|jadu|guest\s+(?:house|unit|home|quarters|suite)|casita|studio|cottage|"
    r"second\s+unit|back\s+unit|rear\s+unit|front\s+unit|each\s+unit|other\s+unit|one\s+unit|"
    r"separate|detached|apartment|in-law)\b[^.;]{0,25}$")
UNIT_CUE_AFTER = re.compile(
    r"^[^.;]{0,12}?\b(?:adu|jadu|guest\s+(?:house|unit|home|quarters)|casita|studio|cottage|"
    r"apartment|in-law)\b")
MAIN_CUE_BEFORE = re.compile(r"\bmain\s+(?:house|home|residence|dwelling|unit)\b[^.;]{0,15}$")
MARKET_CUE = re.compile(
    r"\b(?:average|comparable|comps?|similar|typical|compared|prices?\s+(?:for|of|currently)|"
    r"in the area|nearby)\b")
RECORDS_CUE_BEFORE = re.compile(
    r"\b(?:records?|assessor|permits?|permitted|per\s+(?:public|county|tax)|listed\s+as|"
    r"shows?\s+(?:only|as)|legal(?:ly)?)\b[^.;]{0,35}$")
CONDITIONAL_BEFORE = re.compile(
    r"\b(?:could|can\s+be|convert\w*|potential\s+(?:for|to)|possible|optional|easily|add|adding|"
    r"if|imagine|future)\b[^.;]{0,25}$")

LOT_IN_SENTENCE_BEFORE = re.compile(r"\b(?:lots?|parcels?)\b[^.;!?]{0,60}$")
OTHER_AREA_BEFORE = re.compile(
    r"\b(?:patio|deck|garage|loft|sunroom|porch|basement|bonus\s+room|workshop|addition)\b[^.;!?]{0,25}$")
DIRECT_LIVING_AFTER = re.compile(r"^\W{0,3}(?:of\s+)?(?:living|interior|home|house|residence|townhome|condo)\b")
RANGE_BEFORE = re.compile(
    r"(?:\d\s*(?:to|-|–|and|or)\s*|(?:square\s+f\w+|sq\.?\s?ft\.?)\s*(?:to|-|–|and|or)\s*"
    r"(?:approximately\s+|about\s+)?)$")
RANGE_AFTER = re.compile(
    r"^\s*(?:to|-|–|and|or)\s*\$?(?:approximately\s+)?\d[\d,]*\s*(?:square|sq)")
LOT_CUE_AFTER = re.compile(
    r"^\W{0,3}(?:[a-z]+\s+){0,2}?(?:lot|parcel|land|garage|loft|patio|deck|adu|jadu|barn|shop|"
    r"workshop|casita|sunroom|studio|yard|guest|bonus|primary|suite|unfinished|basement|"
    r"multipurpose|outbuilding|detached|storage|attic|office|building)\b")
LOT_CUE_BEFORE = re.compile(
    r"(?:\b(?:lot|parcel|land)(?:\s+size)?(?:\s+(?:of|is|measures|:))?|"
    r"\b(?:adu|jadu|guest\s+house|casita|studio|barn|workshop)\b[^.;]{0,30})\W*"
    r"(?:approximately\s+|about\s+|nearly\s+|over\s+)?$")
LIVING_CUE_AFTER = re.compile(
    r"^\W{0,3}(?:of\s+)?[^.;!]{0,45}?\b(?:living|interior|home|house|residence|floor\s*plan|space|"
    r"townhome|condo|cabin|property|built)\b")
PER_SQFT_BEFORE = re.compile(r"\b(?:per|price\s+per|cost\s+per)\s*$")

#price cues
_MONEY = r"\$\s?(?P<amt>\d[\d,]*(?:\.\d+)?)(?:\s?(?P<mult>million|[mMkK])\b)?"
PRICE_CUES = [
    r"offered(?:\s+for\s+sale)?\s+(?:at|for|under|below|from)",
    r"(?:now\s+|currently\s+|just\s+)?listed\s+(?:at|for|under|below|from)",
    r"(?:now\s+|newly\s+|recently\s+)?priced\s+(?:at|to|under|below|from)",
    r"list(?:ing)?\s+price\s*(?:is|of|:)?",
    r"asking(?:\s+price)?\s*(?:is|of|:)?",
    r"price\s+(?:improvement|reduction|drop|adjustment)\s+to",
    r"reduced\s+to",
    r"price\s+(?:is|of|:)",
    r"purchase\s+price\s+(?:of|is|:)",
]
PRICE_MODIFIER = r"(?:\s+(?:approximately|approx\.?|about|around|just|only|under|below|over|above|up\s+to|" \
                 r"starting\s+at|from|less\s+than|more\s+than))?"
PRICE_BOUND_WORDS = re.compile(r"\b(?:under|below|over|above|up\s+to|starting\s+at|from|less\s+than|more\s+than)\b")
PRICE_NOT_AFTER = re.compile(
    r"^\s*(?:/\s?(?:mo|month|yr|year)|per\s+(?:month|mo|year|yr|square|sq)|a\s+month|monthly|"
    r"\s*-\s*\$|\s+(?:and|to)\s+\$)", re.I)

#amenity context filters
#cues that only apply inside the current clause (a comma ends their reach) ...
AMENITY_CLAUSE_NEG = re.compile(
    r"\b(?:no|not|without|lacks?|lacking|potential|possible|possibly|could|can\s+be|may|might|"
    r"optional|eligible|obtain|if)\b[^,.;!?)]{0,60}$")
# ... and cues that govern a whole list ("suitable for guests, a den, or ...").
AMENITY_LIST_NEG = re.compile(
    r"\b(?:convert(?:ed|ing)?|add(?:ing)?|build(?:ing)?\s+(?:a|an|your)|putting\s+in|put\s+in|dream|"
    r"imagine|future|plans?\s+(?:for|to)|permits?|opportunity\s+to)\b[^.;!?)]{0,60}$")
#"ideal for / perfect for / used as" describe a suggested USE, so they only void room/type terms;
#"a deck ideal for enjoying the lake views" must keep "lake views".
AMENITY_USE_NEG = re.compile(
    r"\b(?:ideal\s+(?:for|as)|perfect\s+(?:for|as)|suited\s+(?:for|as)|great\s+for|use[ds]?\s+as)\b"
    r"[^.;!?)]{0,60}$")
USE_SENSITIVE_CATEGORIES = {"layout_rooms", "property_type", "financing_ownership"}
AMENITY_COMMUNITY_BEFORE = re.compile(
    r"\b(?:amenities|amenity|clubhouse|residents|community\s+(?:pool|offers|features|includes|"
    r"center)|hoa\s+(?:amenities|includes|covers|offers)|association|shared|common\s+area|"
    r"complex|enjoy\s+access)\b[^.;!?]{0,70}$")
COMMUNITY_SENSITIVE_CATEGORIES = {"outdoor_lot", "systems_parking_energy", "interior_features"}
FALSE_FRIEND_BEFORE = {"granite": re.compile(r"decomposed\s*$")}
CONDITION_TERMS_ABOUT_AREA = re.compile(r"^\s*(?:community|neighborhood|complex|development|area|tract)\b")
FALSE_FRIEND_AFTER = {
    "pool": re.compile(r"^\s*(?:table|hall|cue|stick)"),
    "spa": re.compile(r"^\s*(?:cious|ce)"),
}


@dataclass
class Mention:
    entity: str
    value: float
    start: int
    end: int
    text: str
    role: str = "main"
    notes: List[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _to_number(token: str) -> Optional[float]:
    """'3' -> 3, '2.5' -> 2.5, 'three' -> 3, 'single' -> 1 (used for baths)."""
    t = token.strip().lower()
    if t in NUMBER_WORDS:
        return float(NUMBER_WORDS[t])
    if t == "single":
        return 1.0
    try:
        return float(t)
    except ValueError:
        return None


def _clean_value(v: float):
    return int(v) if float(v).is_integer() else v


def _sentence_before(text: str, pos: int, width: int = 80) -> str:
    chunk = text[max(0, pos - width):pos]
    cut = max(chunk.rfind(". "), chunk.rfind("! "), chunk.rfind("? "), chunk.rfind("; "), chunk.rfind(") "))
    return chunk[cut + 2:] if cut >= 0 else chunk


class EntityExtractor:
    def __init__(self, taxonomy_path: Optional[Path] = None, cleaner=None):
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY
        self._cleaner = cleaner
        #loaded lazily so numeric extraction works without a taxonomy
        self._terms = None
        self._term_patterns = None

        #up to 3 descriptive words between the number and the unit ("2 spacious bedrooms");
        #words that start another count or a different noun are not allowed inside the gap
        bad = "|".join(sorted(BAD_GAP_WORDS, key=len, reverse=True))
        gap = rf"(?P<gap>(?:(?!(?:{bad})\b)[a-z']+(?:-|\s+)){{0,3}}?)"
        sep = r"(?:\s*-\s*|\s+)"
        lead = r"(?<![\w$#./])\*?"
        self._bed_re = re.compile(
            rf"{lead}(?P<num>\d{{1,2}}|{_NUMW})\*?{sep}{gap}(?P<unit>bed(?:room)?s?\b|bdrms?\b)", re.I)
        self._all_bed_re = re.compile(
            rf"\ball\s+(?P<num>\d{{1,2}}|{_NUMW})\s+(?:of\s+the\s+)?(?:\w+\s+){{0,2}}?(?P<unit>bed(?:room)?s\b)", re.I)
        self._all_bath_re = re.compile(
            rf"\ball\s+(?P<num>\d{{1,2}}|{_NUMW})\s+(?:of\s+the\s+)?(?:\w+\s+){{0,2}}?(?P<unit>bath(?:room)?s\b)", re.I)
        self._bed_label_re = re.compile(r"\bbed(?:room)?s?\s*[:=]\s*(?P<num>\d{1,2})\b", re.I)
        frac = r"(?:\s+(?:1/2|1/4|3/4)|\s*-?\s*and[- ]a[- ]half)"
        self._bath_re = re.compile(
            rf"{lead}(?P<num>\d{{1,2}}(?:\.\d{{1,2}})?|{_NUMW}|single)(?P<frac>{frac})?{sep}{gap}"
            rf"(?P<unit>bath(?:room)?s?\b|bth\b)", re.I)
        self._bath_and_half_re = re.compile(
            rf"{lead}(?P<num>\d{{1,2}}|{_NUMW})[-\s]+(?:and[-\s]+a[-\s]+half)[-\s]*(?:full[-\s]+)?"
            r"(?P<unit>bath(?:room)?s?\b)", re.I)
        self._full_half_re = re.compile(
            rf"{lead}(?P<full>\d{{1,2}}|{_NUMW})[-\s]*full(?:[-\s]*(?:bath(?:room)?s?))?"
            rf"\s*(?:,|and|\+|&|plus)\s*(?:(?:a|an|one|1|(?P<half>\d{{1,2}}|{_NUMW}))\s*)?"
            r"(?P<kind>half|1/2|three[- ]quarter|3/4|powder|quarter)[-\s]*(?:bath(?:room)?s?)?", re.I)
        self._half_after_re = re.compile(
            r"^\s*(?:,|;|and|\+|&|plus|with)?\s*(?:a|an|one|1)?\s*(?:additional\s+)?"
            r"(?P<kind>1/2|half|powder|three[- ]quarter|3/4|quarter)[-\s]*(?:bath(?:room)?s?|room)", re.I)
        self._sqft_re = re.compile(
            r"(?<![\w$#.,/])(?P<num>\d[\d,]*(?:\.\d+)?)(?P<plus>\s?\+(?:/-)?)?\s*-?\s*"
            r"(?P<unit>square[\s-]f(?:ee|oo)t(?:age)?|sq\.?\s?ft\.?|sqft)\b", re.I)
        self._sqft_label_re = re.compile(
            r"\bsquare\s+(?:footage|feet)\s*(?:of|is|:|=)?\s*(?:approximately\s+|about\s+)?"
            r"(?P<num>\d[\d,]{2,})\b(?!\s*(?:square|sq))", re.I)
        cue = "|".join(PRICE_CUES)
        self._price_re = re.compile(
            rf"(?P<cue>{cue})(?P<mod>{PRICE_MODIFIER})\s*(?:an?\s+\w+\s+price\s+of\s+)?{_MONEY}", re.I)

    #cleaning
    def clean(self, text: str) -> str:
        if self._cleaner is None:
            from text_cleaning import TextCleaner
            self._cleaner = TextCleaner()
        return self._cleaner.clean_text(text)

    #shared role logic for bedroom / bathroom mentions
    def _classify_count(self, text: str, start: int, end: int, gap_words: List[str]) -> (str, List[str]):
        notes: List[str] = []
        before = _sentence_before(text, start, 80).lower()
        after = text[end:end + 60].lower()
        near = text[max(0, start - 60):end + 60].lower()
        if any(w in PARTIAL_GAP_WORDS for w in gap_words):
            return "partial", ["gap word: " + ",".join(w for w in gap_words if w in PARTIAL_GAP_WORDS)]
        if re.search(r"\d\s*(?:-|to|or)\s*$|\d-\s+(?:and|or)\s+$", text[max(0, start - 10):start].lower()):
            return "range", ["range: '2- and 3-bedroom', '3-4 bedrooms'"]
        if RECORDS_CUE_BEFORE.search(before):
            return "records", ["public-records cue"]
        if MARKET_CUE.search(near):
            return "market", ["market comparison cue"]
        if CONDITIONAL_BEFORE.search(before):
            return "conditional", ["hypothetical cue"]
        if MAIN_CUE_BEFORE.search(before):
            return "main", ["main-house cue"]
        if UNIT_CUE_BEFORE.search(before) or UNIT_CUE_AFTER.search(after):
            return "secondary_unit", ["separate-unit cue"]
        if FLOOR_CUE_BEFORE.search(before) or FLOOR_CUE_AFTER.search(after):
            return "partial", ["floor/level cue"]
        return "main", notes

    @staticmethod
    def _pick(mentions: List[Mention]) -> Optional[Mention]:
        main = [m for m in mentions if m.role == "main"]
        if not main:
            return None
        counts: Dict[float, int] = {}
        for m in main:
            counts[m.value] = counts.get(m.value, 0) + 1
        best = max(counts.values())
        for m in main:
            #earliest among most frequent values
            if counts[m.value] == best:
                return m
        return None

    #bedrooms
    def find_bedroom_mentions(self, text: str) -> List[Mention]:
        out: List[Mention] = []
        for m in self._bed_re.finditer(text):
            gap_words = [w.lower() for w in re.split(r"[\s-]+", m.group("gap").strip()) if w]
            if any(w in BAD_GAP_WORDS for w in gap_words):
                continue
            value = _to_number(m.group("num"))
            if value is None or value > 20:
                continue
            role, notes = self._classify_count(text, m.start(), m.end(), gap_words)
            out.append(Mention("bedrooms", value, m.start(), m.end(), m.group(0), role, notes))
        for m in self._all_bed_re.finditer(text):
            if any(m.start() < o.end and m.end() > o.start for o in out):
                continue
            value = _to_number(m.group("num"))
            role, notes = self._classify_count(text, m.start(), m.end(), [])
            if value is not None and value <= 20:
                out.append(Mention("bedrooms", value, m.start(), m.end(), m.group(0), role, notes + ["'all N' form"]))
        for m in self._bed_label_re.finditer(text):
            value = _to_number(m.group("num"))
            if value is not None and value <= 20:
                out.append(Mention("bedrooms", value, m.start(), m.end(), m.group(0), "main", ["label form"]))
        return sorted(out, key=lambda x: x.start)

    def select_bedroom_mention(self, text: str) -> Optional[Mention]:
        return self._pick(self.find_bedroom_mentions(text))

    def extract_bedrooms(self, text: str) -> Optional[int]:
        m = self.select_bedroom_mention(text)
        return _clean_value(m.value) if m else None

    #bathrooms
    def find_bathroom_mentions(self, text: str) -> List[Mention]:
        out: List[Mention] = []
        taken = []

        def free(s, e):
            return not any(s < te and e > ts for ts, te in taken)

        #"2 full and 1 half bath", "3 full + powder"
        for m in self._full_half_re.finditer(text):
            full = _to_number(m.group("full"))
            half_n = _to_number(m.group("half")) if m.group("half") else 1.0
            kind = m.group("kind").lower()
            unit = 0.75 if ("quarter" in kind and "three" in kind) or kind == "3/4" else (
                0.25 if kind == "quarter" else 0.5)
            if full is None or half_n is None:
                continue
            role, notes = self._classify_count(text, m.start(), m.end(), [])
            out.append(Mention("bathrooms", full + half_n * unit, m.start(), m.end(), m.group(0), role,
                               notes + ["full + partial baths"]))
            taken.append((m.start(), m.end()))
        #"three-and-a-half bathrooms"
        for m in self._bath_and_half_re.finditer(text):
            if not free(m.start(), m.end()):
                continue
            n = _to_number(m.group("num"))
            role, notes = self._classify_count(text, m.start(), m.end(), [])
            out.append(Mention("bathrooms", n + 0.5, m.start(), m.end(), m.group(0), role, notes))
            taken.append((m.start(), m.end()))
        for m in self._bath_re.finditer(text):
            if not free(m.start(), m.end()):
                continue
            gap_words = [w.lower() for w in re.split(r"[\s-]+", m.group("gap").strip()) if w]
            if any(w in BAD_GAP_WORDS for w in gap_words):
                continue
            value = _to_number(m.group("num"))
            if value is None or value > 20:
                continue
            frac = m.group("frac")
            if frac:
                f = frac.strip()
                value += _FRACTIONS.get(f, 0.5) if f in _FRACTIONS else 0.5
            role, notes = self._classify_count(text, m.start(), m.end(), gap_words)
            #"2 full baths plus a half bath" -> 2.5 (only if the half bath follows closely)
            if "full" in gap_words and role == "main":
                h = self._half_after_re.search(text[m.end():m.end() + 30])
                if h:
                    kind = h.group("kind").lower()
                    value += 0.75 if ("three" in kind or kind == "3/4") else (0.25 if kind == "quarter" else 0.5)
                    notes.append("+ half bath")
            out.append(Mention("bathrooms", value, m.start(), m.end(), m.group(0), role, notes))
        for m in self._all_bath_re.finditer(text):
            if not free(m.start(), m.end()) or any(m.start() < o.end and m.end() > o.start for o in out):
                continue
            value = _to_number(m.group("num"))
            role, notes = self._classify_count(text, m.start(), m.end(), [])
            if value is not None and value <= 20:
                out.append(Mention("bathrooms", value, m.start(), m.end(), m.group(0), role, notes + ["'all N' form"]))
        return sorted(out, key=lambda x: x.start)

    def select_bathroom_mention(self, text: str) -> Optional[Mention]:
        return self._pick(self.find_bathroom_mentions(text))

    def extract_bathrooms(self, text: str) -> Optional[float]:
        m = self.select_bathroom_mention(text)
        return _clean_value(m.value) if m else None

    #square footage
    def find_sqft_mentions(self, text: str) -> List[Mention]:
        out: List[Mention] = []
        seen = set()
        candidates = [(m, "num") for m in self._sqft_re.finditer(text)] + \
                     [(m, "label") for m in self._sqft_label_re.finditer(text)]
        for m, kind in candidates:
            try:
                value = float(m.group("num").replace(",", ""))
            except ValueError:
                continue
            start, end = m.start(), m.end()
            before = text[max(0, start - 80):start].lower()
            after = text[end:end + 60]
            lower_after = after.lower()
            notes: List[str] = []
            role = "main"
            if PER_SQFT_BEFORE.search(before) or re.match(r"^\s*(?:per|/)", lower_after):
                role = "rate"
            elif RANGE_BEFORE.search(before) or RANGE_AFTER.match(lower_after):
                role = "range"
            elif not (250 <= value <= 30000):
                role = "implausible"
            elif not DIRECT_LIVING_AFTER.match(lower_after) and (
                    LOT_CUE_AFTER.search(lower_after) or LOT_CUE_BEFORE.search(before)
                    or LOT_IN_SENTENCE_BEFORE.search(_sentence_before(text, start, 80).lower())
                    or OTHER_AREA_BEFORE.search(_sentence_before(text, start, 80).lower())):
                if not MAIN_CUE_BEFORE.search(before):
                    role = "other_area"
            if role == "main" and LIVING_CUE_AFTER.search(lower_after):
                notes.append("living-space cue")
            key = (round(value), start // 20)
            if key in seen:
                continue
            seen.add(key)
            out.append(Mention("sqft", value, start, end, m.group(0), role, notes))
        return sorted(out, key=lambda x: x.start)

    def select_sqft_mention(self, text: str) -> Optional[Mention]:
        mentions = [m for m in self.find_sqft_mentions(text) if m.role == "main"]
        if not mentions:
            return None
        preferred = [m for m in mentions if "living-space cue" in m.notes]
        return (preferred or mentions)[0]

    def extract_sqft(self, text: str) -> Optional[int]:
        m = self.select_sqft_mention(text)
        return int(m.value) if m else None

    #price
    def find_price_mentions(self, text: str) -> List[Mention]:
        out: List[Mention] = []
        for m in self._price_re.finditer(text):
            amt = float(m.group("amt").replace(",", ""))
            mult = (m.group("mult") or "").lower()
            if mult in ("m", "million"):
                amt *= 1_000_000
            elif mult == "k":
                amt *= 1_000
            after = text[m.end():m.end() + 25]
            role = "main"
            notes: List[str] = []
            if PRICE_BOUND_WORDS.search(text[m.start("cue"):m.start("amt")].lower()):
                #"offered under $900000" is a ceiling, not the list price
                role = "bound"      
            elif PRICE_NOT_AFTER.search(after):
                role = "not_list_price"
            elif not (20_000 <= amt <= 100_000_000):
                role = "implausible"
            out.append(Mention("price", amt, m.start(), m.end(), m.group(0), role, notes))
        return out

    def select_price_mention(self, text: str) -> Optional[Mention]:
        for m in self.find_price_mentions(text):
            if m.role == "main":
                return m
        return None

    def extract_price(self, text: str) -> Optional[int]:
        m = self.select_price_mention(text)
        return int(m.value) if m else None

    #amenities
    @staticmethod
    def _normalize_term(term: str) -> str:
        return re.sub(r"\s+", " ", term.lower().replace("-", " ")).strip()

    def _load_taxonomy(self):
        if self._terms is not None:
            return
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(
                f"{self.taxonomy_path} not found. Run: python scripts/taxonomy_builder.py")
        with open(self.taxonomy_path) as f:
            self._terms = json.load(f)["terms"]
        pats = []
        for t in self._terms:
            words = [re.escape(w) for w in self._normalize_term(t["term"]).split(" ")]
            body = r"\s+".join(words)
            pats.append((t, re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")))
        self._term_patterns = pats

    def find_amenity_mentions(self, text: str, apply_filters: bool = True) -> List[dict]:
        """Taxonomy terms found in *text* (longest match wins; overlapping shorter terms dropped)."""
        self._load_taxonomy()
        norm = text.lower().replace("-", " ")
        raw = []
        for t, pat in self._term_patterns:
            for m in pat.finditer(norm):
                raw.append((m.start(), m.end(), t))
        #longest match first; on a tie the term whose own text is longer wins
        #("walk in closets" beats "walk in closet" + plural s)
        raw.sort(key=lambda r: (-(r[1] - r[0]), -len(self._normalize_term(r[2]["term"])), r[0]))
        kept = []
        for s, e, t in raw:
            if not any(s < ke and e > ks for ks, ke, _ in kept):
                kept.append((s, e, t))
        kept.sort(key=lambda r: r[0])
        out = []
        for s, e, t in kept:
            reason = None
            if apply_filters:
                term_norm = self._normalize_term(t["term"])
                before = _sentence_before(norm, s, 90)
                after = norm[e:e + 20]
                if term_norm in FALSE_FRIEND_BEFORE and FALSE_FRIEND_BEFORE[term_norm].search(before):
                    reason = "false friend"
                elif term_norm in FALSE_FRIEND_AFTER and FALSE_FRIEND_AFTER[term_norm].search(after):
                    reason = "false friend"
                elif t["category"] == "condition_updates" and CONDITION_TERMS_ABOUT_AREA.search(norm[e:e + 20]):
                    reason = "describes the community, not the home"
                elif AMENITY_CLAUSE_NEG.search(before) or AMENITY_LIST_NEG.search(before) or (
                        t["category"] in USE_SENSITIVE_CATEGORIES and AMENITY_USE_NEG.search(before)):
                    reason = "negated or hypothetical"
                elif (t["category"] in COMMUNITY_SENSITIVE_CATEGORIES
                      and "private" not in term_norm and AMENITY_COMMUNITY_BEFORE.search(before)):
                    reason = "community amenity"
            out.append({"id": t["id"], "term": t["term"], "category": t["category"],
                        "start": s, "end": e, "text": text[s:e], "dropped": reason})
        return out

    def extract_amenities(self, text: str) -> List[dict]:
        return [a for a in self.find_amenity_mentions(text) if a["dropped"] is None]

    #all entities
    def extract_all(self, text: str, clean: bool = True, with_amenities: bool = True) -> dict:
        """Return ``{'bedrooms','bathrooms','price','sqft','amenities'}`` for one remark."""
        if clean:
            text = self.clean(text)
        result = {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "price": self.extract_price(text),
            "sqft": self.extract_sqft(text),
        }
        if with_amenities:
            amen = self.extract_amenities(text)
            seen, names = set(), []
            for a in amen:
                if a["id"] not in seen:
                    seen.add(a["id"])
                    names.append(a["term"])
            result["amenities"] = names
        return result

    def extract_entities(self, text: str, clean: bool = True) -> dict:
        """Like extract_all, but returns every mention with span and role (for debugging / evaluation)."""
        if clean:
            text = self.clean(text)
        return {
            "text": text,
            "bedrooms": self.find_bedroom_mentions(text),
            "bathrooms": self.find_bathroom_mentions(text),
            "sqft": self.find_sqft_mentions(text),
            "price": self.find_price_mentions(text),
            "amenities": self.find_amenity_mentions(text),
        }
