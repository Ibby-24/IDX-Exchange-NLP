"""Week 4: QueryParser - turns a free-text home search into structured filters and safe SQL.

    from query_parser import QueryParser
    qp = QueryParser()                                   # needs data/processed/schema.json (build_schema.py)
    qp.parse("3 bed under 700k in Irvine")
    # {'city': 'Irvine', 'bedrooms': 3, 'price_max': 700000}
    sql, params = qp.to_sql(qp.parse("3 bed under 700k in Irvine"))
    # sql    -> "SELECT ... FROM rets_property WHERE L_City = %s AND L_SystemPrice <= %s AND L_Keyword2 = %s ..."
    # params -> ['Irvine', 700000, 3, 50]

Filters this parser can produce
    city / cities                       "in Irvine", "Corona or Riverside"
    price_min / price_max               "under 700k", "between 500k and 800k", "around 700k", "$400k-$600k"
    bedrooms / bedrooms_min / _max      "3 bed", "4+ bedrooms", "at least 3", "3 to 4", "more than 3"
    bathrooms / bathrooms_min / _max    same patterns; halves allowed ("2.5 bath")
    sqft_min / sqft_max                 "2000 sqft or more", "under 1200 square feet", "1500-2000 sq ft"
    amenities / exclude_amenities       Week 1 taxonomy terms: "with a pool", "no HOA", "without solar"

How it works (read this before changing a rule)
  1. The query goes through the Week 2 TextCleaner ("3br" -> "3 bedroom", "under 550k" -> "under 550000").
  2. Matchers run in a fixed order and each one *claims* the characters it used, so a number is
     never counted twice: cities -> square feet -> price -> bedrooms -> bathrooms -> "3/2" shorthand
     -> amenities. Inside one matcher, the more specific pattern runs first ("at least 3 bedrooms"
     before plain "3 bedrooms").
  3. Every accepted match records the *name of the pattern* that produced it (``PATTERN_NAMES``);
     tests check that every pattern is exercised.
  4. Nothing here touches the database. ``build_sql`` builds the statement from fixed text and passes
     every user-influenced value as a bound parameter (see its docstring).

Deliberate limits (also in docs/week4_report.md)
  * Amenities are matched in the free-text remarks (L_Remarks) as whole words/phrases. The
    PoolPrivateYN-style flag columns exist, but their value format has not been verified.
  * A city the parser cannot recognise is still returned (if it is capitalised after in/near/around)
    so the SchemaValidator can reject it with a suggestion. A lowercase unknown city is not detected.
  * A bare square-foot number ("a 2000 sqft home") is ignored and reported in ``ignored``.
  * Vague words ("cheap", "nice", "spacious") produce no filter.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "data" / "processed" / "schema.json"
DEFAULT_TAXONOMY = ROOT / "data" / "processed" / "taxonomy.json"

# --------------------------------------------------------------------------- shared regex pieces
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10}
_NUMW_RE = "|".join(NUMW)
NUM = rf"(?:\d+(?:\.\d+)?|{_NUMW_RE})"
BED_U = r"(?:bed(?:room)?s?|br|bd|bdrms?)\b"
BATH_U = r"(?:bath(?:room)?s?|ba|bth)\b"
SQ_U = r"(?:square\s+(?:feet|foot|footage)|sq\.?\s?ft\.?|sqft)(?![a-z])"

Q_MIN = r"(?:at\s+least|a\s+minimum\s+of|minimum(?:\s+of)?|min\.?|no\s+(?:fewer|less)\s+than|not\s+(?:fewer|less)\s+than)"
Q_MAX = r"(?:up\s+to|at\s+most|a\s+maximum\s+of|maximum(?:\s+of)?|max\.?|no\s+more\s+than|not\s+more\s+than)"
Q_GT = r"(?:more\s+than|greater\s+than|over|above|bigger\s+than|larger\s+than)"
Q_LT = r"(?:fewer\s+than|less\s+than|under|below|smaller\s+than)"
OR_MORE = r"(?:or\s+more|or\s+greater|and\s+up|and\s+above|minimum|min)\b"
OR_LESS = r"(?:or\s+fewer|or\s+less|and\s+under|and\s+below|maximum|max)\b"

MONEY = r"(?:\$\s?)?\d+(?:\.\d+)?\s?(?:k|m|mm|million|thousand|grand)?(?![\w$])(?!\s*(?:walk|drive|min|mile|mi\b|minute))"
MONEY_UNIT = {"k": 1000, "thousand": 1000, "grand": 1000, "m": 10**6, "mm": 10**6, "million": 10**6}

# The pattern catalogue. A name appears in ParsedQuery.matches when the pattern produced a filter.
PATTERN_NAMES = [
    # price
    "price_between", "price_dash_range", "price_max_word", "price_min_word", "price_plus",
    "price_or_less", "price_around", "price_budget", "price_bare",
    # bedrooms / bathrooms (the same nine shapes for each)
    "count_range", "count_or", "count_min_word", "count_max_word", "count_more_than",
    "count_fewer_than", "count_plus", "count_or_more", "count_or_fewer", "count_exact",
    "count_inherit", "beds_baths_slash",
    # square feet
    "sqft_range", "sqft_min_word", "sqft_min_post", "sqft_max_word", "sqft_max_post", "sqft_around",
    # city
    "city_known", "city_unknown",
    # amenities
    "amenity", "amenity_negated", "amenity_no_hoa",
]

# Taxonomy terms that appear in searches but are not features to filter on.
IGNORED_TERMS = {"hoa", "hoa fees", "hoa dues", "minutes from", "walking distance", "property taxes", "lender"}
HOA_TERMS = {"hoa", "hoa fees", "hoa dues"}                 # negated -> the taxonomy term "no hoa"

# City names that are also ordinary words: only accepted right after in/near/around/at.
AMBIGUOUS_CITIES = {"nice", "orange", "paradise", "vista", "industry", "commerce", "mission", "home",
                    "other", "bell", "rice", "winters", "live oak", "mountain view", "garden"}
LOCATIVE_BEFORE = re.compile(r"(?:\bin|\bnear|\baround|\bat|\boutside(?:\s+of)?|\bby)\s+$", re.I)

NEG_BEFORE = re.compile(r"(?:^|[\s,;(])(?:no|not|without|non|excluding|exclude|avoid|except|minus)\s+(?:an?\s+|the\s+|any\s+)?$")
NEG_LIST_GAP = re.compile(r"^\s*(?:,|/|or|nor|and|&|and/or)\s*$")     # "no pool, spa" / "without a pool or spa"

# "low HOA fees" / "no HOA" / "without HOA dues" are filters; a bare "HOA fees" is not. Handled before
# the generic longest-match so "low hoa" is not swallowed by the longer, ignored term "hoa fees".
HOA_PHRASE = re.compile(r"\b(?P<kind>low|no|without|zero|non)[\s-]+hoa(?:[\s-]+(?:fees?|dues?|payments?|costs?))?\b", re.I)
# singular forms people type for plural taxonomy terms
VIEW_SINGULAR = re.compile(r"\b(mountain|ocean|city|sunset|valley|canyon|lake|panoramic|golf\s+course)\s+view\b", re.I)

LOWER_TOKENS_NOT_CITIES = {"hoa", "adu", "rv", "ev", "usa", "mls", "fha", "va"}


# --------------------------------------------------------------------------- result object
@dataclass
class ParsedQuery:
    query: str
    cleaned: str
    filters: Dict = field(default_factory=dict)
    matches: List[dict] = field(default_factory=list)     # {"pattern", "text", "field"}
    ignored: List[str] = field(default_factory=list)      # things seen but deliberately not used

    def patterns_used(self) -> List[str]:
        return sorted({m["pattern"] for m in self.matches})


# --------------------------------------------------------------------------- helpers
def _num(token: str):
    token = token.lower()
    if token in NUMW:
        return NUMW[token]
    value = float(token)
    return int(value) if value.is_integer() else value


def _money(token: str) -> Optional[int]:
    """"$1.2m" -> 1200000, "450k" -> 450000, "450000" -> 450000, "$900" -> 900. None if not money."""
    m = re.fullmatch(r"(\$\s?)?(\d+(?:\.\d+)?)\s?(k|m|mm|million|thousand|grand)?", token.strip().lower())
    if not m:
        return None
    dollar, amount, unit = m.groups()
    mult = MONEY_UNIT.get(unit, 1)
    return int((Decimal(amount) * mult).to_integral_value())


def _looks_like_money(token: str, relaxed: bool) -> bool:
    """A number only counts as a price with "$" or k/m/million, or (when a price word came first) 5+ digits."""
    m = re.fullmatch(r"(\$\s?)?(\d+(?:\.\d+)?)\s?(k|m|mm|million|thousand|grand)?", token.strip().lower())
    if not m:
        return False
    dollar, amount, unit = m.groups()
    if dollar or unit:
        return True
    return relaxed and len(amount.split(".")[0]) >= 5


def _clean_filters(filters: dict) -> dict:
    """Whole floats become ints (3.0 -> 3) so results compare and serialise cleanly."""
    out = {}
    for k, v in filters.items():
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


# --------------------------------------------------------------------------- the parser
class QueryParser:
    def __init__(self, schema_path: Optional[Path] = None, schema: Optional[dict] = None,
                 taxonomy_path: Optional[Path] = None, cleaner=None, extractor=None):
        if schema is None:
            path = Path(schema_path) if schema_path else DEFAULT_SCHEMA
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Build it with: python scripts/build_schema.py "
                    "(add --from-db to read the cities from MySQL).")
            with open(path) as f:
                schema = json.load(f)
        self.schema = schema
        self.cities: List[str] = list(schema["cities"])
        self.columns: Dict[str, str] = dict(schema.get("columns", {}))
        self._cleaner = cleaner
        self._extractor = extractor
        self._taxonomy_path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY
        self._compile()

    # ------------------------------------------------------------------ setup
    def _get_cleaner(self):
        if self._cleaner is None:
            from text_cleaning import TextCleaner
            self._cleaner = TextCleaner()
        return self._cleaner

    def _get_extractor(self):
        if self._extractor is None:
            from entity_extractor import EntityExtractor
            self._extractor = EntityExtractor(taxonomy_path=self._taxonomy_path)
        return self._extractor

    def _compile(self):
        I = re.I
        # --- cities: longest name first so "Big Bear City" beats "Big Bear"
        names = sorted(self.cities, key=len, reverse=True)
        self._city_names = {" ".join(c.lower().split()): c for c in names}
        if names:
            alt = "|".join(r"\s+".join(re.escape(w) for w in c.split()) for c in names)
            self._city_re = re.compile(rf"(?<![A-Za-z0-9])({alt})(?![A-Za-z0-9])", I)
        else:
            self._city_re = None
        self._unknown_city_re = re.compile(
            r"\b(?i:in|near|around|outside\s+of|outside|at)\s+"
            r"(?P<c>[A-Z][A-Za-z.'\-]*(?:\s+(?:(?:de|del|la|las|los|el|of)\s+)?[A-Z][A-Za-z.'\-]*)*)")
        self._county_after = re.compile(r"\s+county\b", I)

        # --- square feet
        a = rf"(?P<a>\d+(?:\.\d+)?)"
        self._sqft = [
            ("sqft_range", re.compile(rf"(?<![\w.]){a}\s*(?:-|–|to)\s*(?P<b>\d+(?:\.\d+)?)\s*{SQ_U}", I), "range"),
            ("sqft_min_word", re.compile(rf"(?:{Q_MIN}|{Q_GT}|no\s+smaller\s+than)\s+{a}\s*\+?\s*{SQ_U}", I), "min"),
            ("sqft_min_post", re.compile(rf"(?<![\w.]){a}\s*(?:\+\s*{SQ_U}|{SQ_U}\s*(?:\+|(?:or\s+(?:more|larger|bigger|greater)|and\s+(?:up|above)|minimum|min)\b))", I), "min"),
            ("sqft_max_word", re.compile(rf"(?:{Q_MAX}|{Q_LT}|no\s+(?:larger|bigger)\s+than)\s+{a}\s*{SQ_U}", I), "max"),
            ("sqft_max_post", re.compile(rf"(?<![\w.]){a}\s*{SQ_U}\s*(?:or\s+(?:less|smaller|fewer)|and\s+(?:under|below)|max(?:imum)?\b)", I), "max"),
            ("sqft_around", re.compile(rf"(?:around|about|approximately|roughly)\s+{a}\s*{SQ_U}", I), "around"),
        ]
        self._sqft_bare = re.compile(rf"(?<![\w.]){a}\s*{SQ_U}", I)

        # --- price
        m = f"(?P<m>{MONEY})"
        m1, m2 = f"(?P<m1>{MONEY})", f"(?P<m2>{MONEY})"
        self._price = [
            ("price_between", re.compile(rf"\bbetween\s+{m1}\s*(?:and|to|-|–)\s*{m2}", I), "range"),
            ("price_dash_range", re.compile(rf"(?<![\w.]){m1}\s*(?:-|–|to)\s*{m2}", I), "range"),
            ("price_max_word", re.compile(
                rf"\b(?:under|below|less\s+than|up\s+to|at\s+most|max(?:imum)?(?:\s+of)?|no\s+more\s+than|"
                rf"not\s+more\s+than|not\s+(?:over|above|exceeding)|cheaper\s+than|budget\s+(?:of|is)|budget:)\s+{m}", I), "max"),
            ("price_min_word", re.compile(
                rf"\b(?:over|above|more\s+than|at\s+least|minimum(?:\s+of)?|min|starting\s+(?:at|from)|from|"
                rf"greater\s+than|exceeding|upwards\s+of)\s+{m}", I), "min"),
            ("price_plus", re.compile(rf"(?<![\w.]){m}\s*(?:\+|or\s+more|and\s+up|and\s+above|or\s+above)", I), "min"),
            ("price_or_less", re.compile(rf"(?<![\w.]){m}\s*(?:or\s+(?:less|under|below)|and\s+(?:under|below)|max(?:imum)?\b)", I), "max"),
            ("price_around", re.compile(rf"(?:\b(?:around|about|approximately|approx\.?|roughly|circa|close\s+to|near)\s*|~\s*){m}", I), "around"),
            ("price_budget", re.compile(rf"(?<![\w.]){m}\s+budget\b", I), "max"),
        ]
        self._price_bare = re.compile(rf"(?<![\w.]){m}", I)

        # --- bedrooms / bathrooms. phase 1 = patterns with a leading qualifier ("at least 3 bedrooms and
        # 2 bathrooms": the qualifier covers both counts), phase 2 = everything else.
        self._count = {}
        for fname, U in (("bedrooms", BED_U), ("bathrooms", BATH_U)):
            n = f"(?P<n>{NUM})"
            lead = r"(?<![\w.$/])"
            sep = r"[\s-]*"
            self._count[fname] = [
                ("count_min_word", re.compile(rf"\b{Q_MIN}\s+{n}{sep}{U}", I), "min", 1),
                ("count_max_word", re.compile(rf"\b{Q_MAX}\s+{n}{sep}{U}", I), "max", 1),
                ("count_more_than", re.compile(rf"\b{Q_GT}\s+{n}{sep}{U}", I), "gt", 1),
                ("count_fewer_than", re.compile(rf"\b{Q_LT}\s+{n}{sep}{U}", I), "lt", 1),
                ("count_range", re.compile(rf"\bbetween\s+{n}\s+(?:and|&|to)\s+(?P<m>{NUM}){sep}{U}", I), "range", 2),
                ("count_range", re.compile(rf"{lead}{n}\s*(?:-|–|to)\s*(?P<m>{NUM}){sep}{U}", I), "range", 2),
                ("count_plus", re.compile(rf"{lead}{n}\s*\+{sep}{U}", I), "min", 2),
                ("count_or", re.compile(rf"{lead}{n}\s+or\s+(?P<m>{NUM}){sep}{U}", I), "range", 2),
                ("count_or_more", re.compile(rf"{lead}{n}{sep}{U}\s*(?:\+|{OR_MORE})", I), "min", 2),
                ("count_or_more", re.compile(rf"{lead}{n}\s+(?:or\s+(?:more|greater)|and\s+(?:up|above)){sep}{U}", I), "min", 2),
                ("count_or_fewer", re.compile(rf"{lead}{n}{sep}{U}\s*{OR_LESS}", I), "max", 2),
                ("count_or_fewer", re.compile(rf"{lead}{n}\s+or\s+(?:fewer|less){sep}{U}", I), "max", 2),
                ("count_exact", re.compile(rf"{lead}{n}{sep}{U}", I), "exact", 2),
            ]
        self._inherit = {
            "bedrooms": re.compile(rf"^\s*(?:and|&)\s+(?P<n>{NUM})[\s-]*{BATH_U}", re.I),
            "bathrooms": re.compile(rf"^\s*(?:and|&)\s+(?P<n>{NUM})[\s-]*{BED_U}", re.I),
        }
        self._slash = re.compile(
            rf"(?<![\w.$/])(?P<b>\d{{1,2}})\s*/\s*(?P<a>\d{{1,2}}(?:\.\d)?)(?![\d/.]|\s*(?:bath|ba\b|bed|acre|mile|month|mo\b|%))", I)

    # ------------------------------------------------------------------ public API
    def parse(self, query: str) -> Dict:
        """Filters only: ``{'city': 'Irvine', 'bedrooms': 3, 'price_max': 700000}``."""
        return self.parse_detailed(query).filters

    def parse_detailed(self, query: str) -> ParsedQuery:
        if not isinstance(query, str):
            query = "" if query is None else str(query)
        query = query[:2000]                                     # hard cap; no real search is longer
        cleaned = self._get_cleaner().clean_text(query)
        pq = ParsedQuery(query=query, cleaned=cleaned)
        low = "".join(ch.lower() if len(ch.lower()) == 1 else ch for ch in cleaned)
        claimed = [False] * len(cleaned)

        self._find_cities(pq, cleaned, claimed)
        self._find_sqft(pq, low, claimed)
        self._find_price(pq, low, claimed)
        self._find_counts(pq, low, claimed)
        self._find_slash(pq, low, claimed)
        self._find_amenities(pq, cleaned, claimed)
        pq.filters = _clean_filters(pq.filters)
        return pq

    def search(self, query: str, validator=None, limit: int = 50) -> dict:
        """parse -> validate -> SQL in one call. SQL is built only when validation passes."""
        pq = self.parse_detailed(query)
        if validator is None:
            from schema_validator import SchemaValidator
            validator = SchemaValidator(schema=self.schema)
        ok, errors = validator.validate_query(pq.filters)
        out = {"query": query, "cleaned": pq.cleaned, "filters": pq.filters, "valid": ok,
               "errors": errors, "ignored": pq.ignored, "sql": None, "params": None}
        if ok:
            out["sql"], out["params"] = self.to_sql(pq.filters, limit=limit)
        return out

    def to_sql(self, filters: Dict, limit: int = 50) -> Tuple[str, list]:
        return build_sql(filters, limit=limit)

    # ------------------------------------------------------------------ claiming spans
    @staticmethod
    def _free(claimed, start, end):
        return not any(claimed[start:end])

    @staticmethod
    def _claim(claimed, start, end):
        for i in range(start, end):
            claimed[i] = True

    @staticmethod
    def _set(pq, key, value, pattern, text, fld):
        """Set a filter once. A second, different value for the same filter is reported, not merged."""
        if key in pq.filters:
            if pq.filters[key] != value:
                pq.ignored.append(f"second {key} value {value!r} ({text.strip()!r}) ignored; kept {pq.filters[key]!r}")
            return False
        pq.filters[key] = value
        pq.matches.append({"pattern": pattern, "text": text.strip(), "field": fld})
        return True

    # ------------------------------------------------------------------ cities
    def _find_cities(self, pq, text, claimed):
        found = []
        if self._city_re is not None:
            for m in self._city_re.finditer(text):
                canon = self._city_names[" ".join(m.group(1).lower().split())]
                if self._county_after.match(text, m.end()):
                    pq.ignored.append(f"'{m.group(1)} County' is a county, not a city; county search is not supported")
                    self._claim(claimed, m.start(), m.end())
                    continue
                if canon.lower() in AMBIGUOUS_CITIES and not LOCATIVE_BEFORE.search(text[:m.start()]):
                    continue
                if not self._free(claimed, m.start(), m.end()):
                    continue
                self._claim(claimed, m.start(), m.end())
                found.append((m.start(), canon, m.group(0)))
        # capitalised place after in/near/around that is NOT a known city: keep it so validation can reject it
        for m in self._unknown_city_re.finditer(text):
            c = m.group("c").strip(" .")
            s, e = m.start("c"), m.start("c") + len(m.group("c").rstrip(" ."))
            if not c or not self._free(claimed, s, e):
                continue
            words = c.split()
            if self._county_after.match(text, e) or c.lower().endswith(" county"):
                pq.ignored.append(f"'{c}' looks like a county; county search is not supported")
                self._claim(claimed, s, e)
                continue
            if len(words) == 1 and (c.upper() == c and len(c) <= 4 or c.lower() in LOWER_TOKENS_NOT_CITIES):
                continue
            if self._looks_like_taxonomy_term(c):
                continue
            self._claim(claimed, s, e)
            found.append((s, c, c))
            pq.matches.append({"pattern": "city_unknown", "text": c, "field": "city"})
        found.sort(key=lambda x: x[0])
        names, seen = [], set()
        for _, canon, raw in found:
            if canon.lower() not in seen:
                seen.add(canon.lower())
                names.append(canon)
                if canon in self.cities:
                    pq.matches.append({"pattern": "city_known", "text": raw, "field": "city"})
        if len(names) == 1:
            pq.filters["city"] = names[0]
        elif len(names) > 1:
            pq.filters["cities"] = names

    def _looks_like_taxonomy_term(self, phrase: str) -> bool:
        ex = self._get_extractor()
        return bool(ex.find_amenity_mentions(phrase.lower(), apply_filters=False))

    # ------------------------------------------------------------------ square feet
    def _find_sqft(self, pq, low, claimed):
        for name, rx, kind in self._sqft:
            for m in rx.finditer(low):
                if not self._free(claimed, m.start(), m.end()):
                    continue
                a = _num(m.group("a"))
                if kind == "range":
                    b = _num(m.group("b"))
                    lo, hi = sorted((a, b))
                    self._set(pq, "sqft_min", lo, name, m.group(0), "sqft")
                    self._set(pq, "sqft_max", hi, name, m.group(0), "sqft")
                elif kind == "min":
                    self._set(pq, "sqft_min", a, name, m.group(0), "sqft")
                elif kind == "max":
                    self._set(pq, "sqft_max", a, name, m.group(0), "sqft")
                else:  # around: +/- 10%
                    self._set(pq, "sqft_min", int(round(a * 0.9)), name, m.group(0), "sqft")
                    self._set(pq, "sqft_max", int(round(a * 1.1)), name, m.group(0), "sqft")
                self._claim(claimed, m.start(), m.end())
        for m in self._sqft_bare.finditer(low):
            if self._free(claimed, m.start(), m.end()):
                pq.ignored.append(f"'{m.group(0).strip()}' has no more/less/range wording, so no square-foot filter was set")
                self._claim(claimed, m.start(), m.end())

    # ------------------------------------------------------------------ price
    def _find_price(self, pq, low, claimed):
        for name, rx, kind in self._price:
            for m in rx.finditer(low):
                if not self._free(claimed, m.start(), m.end()):
                    continue
                if kind == "range":
                    t1, t2 = m.group("m1"), m.group("m2")
                    relaxed = name == "price_between"        # "between 450000 and 600000" needs no $ or k
                    if not _looks_like_money(t2, relaxed=relaxed):
                        continue
                    if not _looks_like_money(t1, relaxed=relaxed):
                        # "400-600k" / "between 600 and 800k": the unit on the second number also belongs to the first
                        u = re.search(r"(k|m|mm|million|thousand|grand)\s*$", t2.strip().lower())
                        if not (u and re.fullmatch(r"\d+(?:\.\d+)?", t1.strip())):
                            continue
                        t1 = t1.strip() + u.group(1)
                    v1, v2 = _money(t1), _money(t2)
                    if v1 is None or v2 is None:
                        continue
                    lo, hi = sorted((v1, v2))
                    self._set(pq, "price_min", lo, name, m.group(0), "price")
                    self._set(pq, "price_max", hi, name, m.group(0), "price")
                else:
                    tok = m.group("m")
                    if not _looks_like_money(tok, relaxed=True):
                        continue
                    v = _money(tok)
                    if v is None:
                        continue
                    if kind == "min":
                        self._set(pq, "price_min", v, name, m.group(0), "price")
                    elif kind == "max":
                        self._set(pq, "price_max", v, name, m.group(0), "price")
                    else:  # around: +/- 10%
                        self._set(pq, "price_min", int(round(v * 0.9)), name, m.group(0), "price")
                        self._set(pq, "price_max", int(round(v * 1.1)), name, m.group(0), "price")
                self._claim(claimed, m.start(), m.end())
        # a lone "$450k" / "450k" with no price word is read as the buyer's budget
        for m in self._price_bare.finditer(low):
            if not self._free(claimed, m.start(), m.end()):
                continue
            tok = m.group("m")
            if not _looks_like_money(tok, relaxed=False):
                continue
            # a bare "5m" (no $) is too easily "5 minutes"/"5 meters"; k and million are clear enough
            if "$" not in tok and re.search(r"(?:mm|m)\s*$", tok.strip().lower()) and "million" not in tok.lower():
                continue
            v = _money(tok)
            if v is not None and self._set(pq, "price_max", v, "price_bare", m.group(0), "price"):
                pq.ignored.append(f"'{tok.strip()}' had no under/over wording; treated as a maximum budget")
            self._claim(claimed, m.start(), m.end())

    # ------------------------------------------------------------------ bedrooms / bathrooms
    def _apply_count(self, pq, fname, kind, m, name):
        n = _num(m.group("n"))
        if fname == "bedrooms" and isinstance(n, float):
            pq.ignored.append(f"'{m.group(0).strip()}': bedrooms must be a whole number")
            return False
        text = m.group(0)
        if kind == "range":
            lo, hi = sorted((n, _num(m.group("m"))))
            ok1 = self._set(pq, f"{fname}_min", lo, name, text, fname)
            ok2 = self._set(pq, f"{fname}_max", hi, name, text, fname)
            return ok1 or ok2
        if kind == "min":
            return self._set(pq, f"{fname}_min", n, name, text, fname)
        if kind == "max":
            return self._set(pq, f"{fname}_max", n, name, text, fname)
        if kind == "gt":
            return self._set(pq, f"{fname}_min", math.floor(n) + 1, name, text, fname)
        if kind == "lt":
            return self._set(pq, f"{fname}_max", math.ceil(n) - 1, name, text, fname)
        return self._set(pq, fname, n, name, text, fname)

    def _find_counts(self, pq, low, claimed):
        for phase in (1, 2):
            for fname in ("bedrooms", "bathrooms"):
                for name, rx, kind, ph in self._count[fname]:
                    if ph != phase:
                        continue
                    for m in rx.finditer(low):
                        if not self._free(claimed, m.start(), m.end()):
                            continue
                        self._apply_count(pq, fname, kind, m, name)
                        self._claim(claimed, m.start(), m.end())
                        if phase == 1:
                            self._inherit_count(pq, low, claimed, fname, kind, m.end())

    def _inherit_count(self, pq, low, claimed, fname, kind, end):
        """"at least 3 bedrooms and 2 bathrooms": the qualifier of the first count covers the second."""
        m = self._inherit[fname].match(low[end:])
        if not m:
            return
        s, e = end + m.start("n"), end + m.end()
        if not self._free(claimed, s, e):
            return
        target = "bathrooms" if fname == "bedrooms" else "bedrooms"
        n = _num(m.group("n"))
        if target == "bedrooms" and isinstance(n, float):
            return
        if kind == "min":
            key, val = f"{target}_min", n
        elif kind == "max":
            key, val = f"{target}_max", n
        elif kind == "gt":
            key, val = f"{target}_min", math.floor(n) + 1
        else:
            key, val = f"{target}_max", math.ceil(n) - 1
        self._set(pq, key, val, "count_inherit", m.group(0), target)
        self._claim(claimed, s, e)

    def _find_slash(self, pq, low, claimed):
        """"3/2" -> 3 bedrooms, 2 bathrooms (only when neither is already set)."""
        for m in self._slash.finditer(low):
            if not self._free(claimed, m.start(), m.end()):
                continue
            b, a = _num(m.group("b")), _num(m.group("a"))
            if not (1 <= b <= 10 and 1 <= a <= 10):
                continue
            if "bedrooms" in pq.filters or "bathrooms" in pq.filters:
                continue
            pq.filters["bedrooms"], pq.filters["bathrooms"] = b, a
            pq.matches.append({"pattern": "beds_baths_slash", "text": m.group(0), "field": "bedrooms"})
            self._claim(claimed, m.start(), m.end())

    # ------------------------------------------------------------------ amenities
    def _find_amenities(self, pq, cleaned, claimed):
        masked = "".join(" " if claimed[i] else ch for i, ch in enumerate(cleaned))
        include, exclude = [], []
        # HOA phrases first: low/no/without HOA are filters, plain "HOA fees" is not
        def hoa_sub(m):
            kind = m.group("kind").lower()
            term = "low hoa" if kind == "low" else "no hoa"
            self._add_unique(include, term)
            pq.matches.append({"pattern": "amenity_no_hoa", "text": m.group(0), "field": "amenities"})
            return " " * len(m.group(0))
        masked = HOA_PHRASE.sub(hoa_sub, masked)
        # plural/singular tolerance, applied to the matching copy only: "communities" -> "community" (the taxonomy
        # matcher already accepts a trailing s/es, but not -ies),
        # "mountain view" -> "mountain views"
        masked = re.sub(r"\bcommunities\b", "community", masked, flags=re.I)
        masked = VIEW_SINGULAR.sub(lambda m: m.group(0) + "s", masked)
        ex = self._get_extractor()
        mentions = ex.find_amenity_mentions(masked, apply_filters=False)
        norm = masked.lower().replace("-", " ")
        terms = {t["term"].lower() for t in ex._terms}
        prev_end, prev_neg = None, False
        for a in mentions:
            term = a["term"].lower()
            tail = norm[:a["start"]]
            negated = bool(NEG_BEFORE.search(tail))
            if not negated and prev_neg and prev_end is not None:
                gap = norm[prev_end:a["start"]]
                if NEG_LIST_GAP.match(gap):
                    negated = True
            prev_end, prev_neg = a["end"], negated
            if term in IGNORED_TERMS and not (term in HOA_TERMS and negated):
                pq.ignored.append(f"'{a['text']}' is not a filterable feature")
                continue
            if term in HOA_TERMS and negated:
                self._add_unique(include, "no hoa")
                pq.matches.append({"pattern": "amenity_no_hoa", "text": a["text"], "field": "amenities"})
                continue
            # "pool and spa" is two features, but "shopping and dining" is one phrase
            parts = [term]
            if " and " in term:
                left, right = term.split(" and ", 1)
                if left in terms and right in terms:
                    parts = [left, right]
            for p in parts:
                canon = self._canonical(ex, p)
                if negated:
                    self._add_unique(exclude, canon)
                    pq.matches.append({"pattern": "amenity_negated", "text": a["text"], "field": "exclude_amenities"})
                else:
                    self._add_unique(include, canon)
                    pq.matches.append({"pattern": "amenity", "text": a["text"], "field": "amenities"})
        include = [t for t in include if t not in exclude]
        if include:
            pq.filters["amenities"] = include
        if exclude:
            pq.filters["exclude_amenities"] = exclude

    @staticmethod
    def _canonical(ex, term_lower: str) -> str:
        for t in ex._terms:
            if t["term"].lower() == term_lower:
                return t["term"].lower()
        return term_lower

    @staticmethod
    def _add_unique(lst, item):
        if item not in lst:
            lst.append(item)


# ===================================================================== SQL generation
TABLE = "rets_property"
COLUMN = {
    "city": "L_City", "price": "L_SystemPrice", "bedrooms": "L_Keyword2",
    "bathrooms": "LM_Dec_3", "sqft": "LM_Int2_3", "remarks": "L_Remarks",
}
SELECT_COLUMNS = ["L_ListingID", "L_Address", "L_City", "L_SystemPrice", "L_Keyword2", "LM_Dec_3", "LM_Int2_3"]

# filter name -> fixed SQL fragment. These strings are the ONLY SQL text ever added to a statement.
NUMERIC_SQL = {
    "price_min": f"{COLUMN['price']} >= %s",
    "price_max": f"{COLUMN['price']} <= %s",
    "bedrooms": f"{COLUMN['bedrooms']} = %s",
    "bedrooms_min": f"{COLUMN['bedrooms']} >= %s",
    "bedrooms_max": f"{COLUMN['bedrooms']} <= %s",
    "bathrooms": f"{COLUMN['bathrooms']} = %s",
    "bathrooms_min": f"{COLUMN['bathrooms']} >= %s",
    "bathrooms_max": f"{COLUMN['bathrooms']} <= %s",
    "sqft_min": f"{COLUMN['sqft']} >= %s",
    "sqft_max": f"{COLUMN['sqft']} <= %s",
}
NUMERIC_ORDER = ["price_min", "price_max", "bedrooms", "bedrooms_min", "bedrooms_max",
                 "bathrooms", "bathrooms_min", "bathrooms_max", "sqft_min", "sqft_max"]
ALLOWED_FILTERS = set(NUMERIC_SQL) | {"city", "cities", "amenities", "exclude_amenities"}
MAX_AMENITIES = 10
MAX_TERM_LENGTH = 60
MAX_CITIES = 10
MAX_LIMIT = 500


def amenity_regex(term: str) -> str:
    """Whole-word, case-insensitive regex for one amenity phrase, built character by character.

    Only letters/digits are copied through; spaces and hyphens become ``[ -]`` and every other
    character is backslash-escaped, so a term can never inject regex syntax ("." stays a dot).
    A trailing optional plural lets "condo" find "condos". Whole words matter: LIKE '%spa%' would
    match "spacious" and '%pool%' would match "carpool".
    """
    if not isinstance(term, str) or not term.strip():
        raise ValueError("amenity must be non-empty text")
    term = " ".join(term.lower().split())
    if len(term) > MAX_TERM_LENGTH:
        raise ValueError("amenity text too long")
    out = []
    for ch in term:
        if ch in " -":
            out.append("[ -]")
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append("\\" + ch)
    return "(^|[^a-z0-9])" + "".join(out) + "(s|es)?($|[^a-z0-9])"


def _checked_number(key, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    return int(value) if isinstance(value, float) and value.is_integer() else value


def build_sql(filters: Dict, limit: int = 50, columns: Optional[List[str]] = None) -> Tuple[str, list]:
    """Build ``(sql, params)`` for the %s style used by mysql.connector.

    Why this is injection-safe
      1. The statement text is assembled only from the constant fragments above (NUMERIC_SQL, the
         column whitelist and fixed keywords). No value from the user is ever put into the text, so two
         queries with the same filter *names* produce identical SQL no matter what the values are.
      2. Every value (city, numbers, amenity regex, limit) travels in ``params`` and is bound by the
         database driver.
      3. Filter names are checked against ``ALLOWED_FILTERS``; numbers must be real numbers (not
         strings, not bool); amenity text is turned into an escaped regex before binding.
      4. Anything unexpected raises ``ValueError`` instead of being passed along.
    """
    if not isinstance(filters, dict):
        raise ValueError("filters must be a dict")
    unknown = [k for k in filters if k not in ALLOWED_FILTERS]
    if unknown:
        raise ValueError(f"unknown filter(s): {sorted(map(str, unknown))}")
    active = {k: v for k, v in filters.items() if v not in (None, [], "")}
    if not active:
        raise ValueError("no filters: refusing to build an unrestricted query")

    cols = columns if columns is not None else SELECT_COLUMNS
    allowed_cols = set(SELECT_COLUMNS) | set(COLUMN.values())
    if not cols or any(c not in allowed_cols for c in cols):
        raise ValueError("unknown column requested")

    where, params = [], []

    # cities
    names = []
    if "city" in active:
        names.append(active["city"])
    if "cities" in active:
        if not isinstance(active["cities"], list):
            raise ValueError("cities must be a list")
        names.extend(active["cities"])
    if names:
        if len(names) > MAX_CITIES or not all(isinstance(c, str) and c.strip() for c in names):
            raise ValueError("city must be non-empty text (max %d)" % MAX_CITIES)
        names = list(dict.fromkeys(" ".join(c.split()) for c in names))
        if len(names) == 1:
            where.append(f"{COLUMN['city']} = %s")
        else:
            where.append(f"{COLUMN['city']} IN (" + ", ".join(["%s"] * len(names)) + ")")
        params.extend(names)

    # numbers
    for key in NUMERIC_ORDER:
        if key in active:
            where.append(NUMERIC_SQL[key])
            params.append(_checked_number(key, active[key]))

    # amenities (required)
    for key, template in (("amenities", f"REGEXP_LIKE({COLUMN['remarks']}, %s, 'i')"),
                          ("exclude_amenities", f"({COLUMN['remarks']} IS NULL OR NOT REGEXP_LIKE({COLUMN['remarks']}, %s, 'i'))")):
        if key in active:
            terms = active[key]
            if not isinstance(terms, list) or len(terms) > MAX_AMENITIES:
                raise ValueError(f"{key} must be a list of at most {MAX_AMENITIES} terms")
            for term in dict.fromkeys(terms):
                where.append(template)
                params.append(amenity_regex(term))

    if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= MAX_LIMIT):
        raise ValueError(f"limit must be an integer from 1 to {MAX_LIMIT}")
    sql = (f"SELECT {', '.join(cols)} FROM {TABLE} WHERE " + " AND ".join(where)
           + f" ORDER BY {COLUMN['price']} ASC LIMIT %s")
    params.append(limit)
    return sql, params
