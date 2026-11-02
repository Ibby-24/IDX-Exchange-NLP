"""Week 6: listing signal extraction.

Turns one listing record into a structured, search-ready JSON object:

    {
      "listing_id": "1190429395",
      "entities":  {"bedrooms": 3, "bathrooms": 2.5, "sqft": 1850, "price": null},
      "amenities": ["swimming pool", "fireplace", ...],            # feature amenities from the taxonomy
      "condition_keywords": ["renovated_updated", "new_components"],
      "financing_terms":    ["fha", "assumable_loan"],
      "location_features":  ["gated_community", "near_schools"],
      "fields":    {"hoa": "none", "mello_roos": null},
      "keywords":  ["cond:renovated_updated", "fin:fha", "loc:near_schools", "hoa:none", ...],
      "schema_version": 1
    }

Usage (from the repo root)::

    from signal_extractor import SignalExtractor
    ex = SignalExtractor()                                  # loads taxonomy + Week 3 EntityExtractor
    ex.extract_signals({"L_ListingID": "1", "L_Remarks": "Turnkey 3 bedroom home, no HOA, FHA ok."})

Design
------
* ``entities``  come straight from the Week 3 ``EntityExtractor`` (not retuned in Week 6).
* ``amenities`` are Week 3 taxonomy amenities from the *feature* categories (interior, outdoor, layout,
  systems/parking/energy, property type) after the Week 3 negation / hypothetical / community filters.
* The three signal families (condition, financing, location) use the canonical inventory in
  ``docs/signal_guidelines.md``. Each canonical signal is found two ways and the results are merged:
    1. taxonomy terms (Week 1) of the matching category, mapped to a canonical signal (TAXONOMY_TO_SIGNAL);
    2. regular-expression patterns (PATTERNS) for the many phrasings the taxonomy does not list
       ("seller willing to finance", "walk to the beach", "mountain and city views").
  Every candidate then goes through the same context filters: negation ("no", "without"), hypothetical /
  suggested use ("could be", "potential to", "ideal for") and, for location words that need a proximity cue
  ("near", "minutes from", "walking distance"), the cue check.
* HOA and Mello-Roos are fields, not yes/no signals: ``hoa`` in {none, low, present}, ``mello_roos`` in {none, present}.
* The output never contains remark text. ``extract_signals(..., evidence=True)`` adds the matched phrases
  (short snippets) for debugging; the batch processor leaves that off.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # works as ``scripts.signal_extractor`` and as a plain import from scripts/
    from entity_extractor import EntityExtractor
except ImportError:  # pragma: no cover
    from .entity_extractor import EntityExtractor

SCHEMA_VERSION = 1

CONDITION_SIGNALS = ["renovated_updated", "new_construction", "new_components", "move_in_ready",
                     "well_maintained", "needs_work"]
FINANCING_SIGNALS = ["seller_financing", "assumable_loan", "fha", "va", "cash_only", "exchange_1031", "short_sale",
                     "reo", "probate_estate", "tenant_occupied", "investment_income", "seller_credit",
                     "motivated_seller"]
LOCATION_SIGNALS = ["gated_community", "golf_course", "near_schools", "near_shopping_dining", "freeway_commute",
                    "public_transit", "walkable", "near_beach_water", "waterfront", "views", "quiet_street",
                    "parks_open_space", "senior_55plus", "near_downtown"]
FAMILIES = {"condition": CONDITION_SIGNALS, "financing": FINANCING_SIGNALS, "location": LOCATION_SIGNALS}
ALL_SIGNALS = CONDITION_SIGNALS + FINANCING_SIGNALS + LOCATION_SIGNALS
FAMILY_OF = {s: f for f, ss in FAMILIES.items() for s in ss}
FAMILY_PREFIX = {"condition": "cond", "financing": "fin", "location": "loc"}

FEATURE_CATEGORIES = {"interior_features", "outdoor_lot", "layout_rooms", "systems_parking_energy",
                      "property_type"}

# ---------------------------------------------------------------------------------------------
# Taxonomy terms -> canonical signal (term text as in taxonomy.json, lower case)
# ---------------------------------------------------------------------------------------------
# (well-maintained and bare "upgraded" are deliberately NOT mapped: the regex layer applies the extra context
#  rules they need, e.g. "beautifully maintained grounds" and "upgraded tile flooring" are not the home.)
TAXONOMY_TO_SIGNAL: Dict[str, str] = {
    # condition
    "turnkey": "move_in_ready", "move-in ready": "move_in_ready", "move-in condition": "move_in_ready",
    "beautifully updated": "renovated_updated", "beautifully remodeled": "renovated_updated",
    "recently renovated": "renovated_updated", "fully renovated": "renovated_updated",
    "renovated": "renovated_updated", "remodeled": "renovated_updated", "fully remodeled": "renovated_updated",
    "completely remodeled": "renovated_updated", "newly remodeled": "renovated_updated",
    "remodeled kitchen": "renovated_updated", "updated kitchen": "renovated_updated",
    "upgraded kitchen": "renovated_updated", "updated bathrooms": "renovated_updated",
    "updated throughout": "renovated_updated", "modern updates": "renovated_updated",
    "recent upgrades": "renovated_updated",
    "new roof": "new_components", "new flooring": "new_components", "new carpet": "new_components",
    "new paint": "new_components", "new interior paint": "new_components", "fresh paint": "new_components",
    "fresh interior paint": "new_components", "freshly painted": "new_components",
    "new appliances": "new_components", "new windows": "new_components", "new plumbing": "new_components",
    "new fixtures": "new_components",
    "newly built": "new_construction", "model home": "new_construction",
    "needs tlc": "needs_work", "fixer-upper": "needs_work", "investor special": "needs_work",
    "as-is": "needs_work", "needs work": "needs_work",
    # financing / ownership
    "seller financing": "seller_financing", "owner carry": "seller_financing",
    "assumable loan": "assumable_loan", "fha": "fha", "va loan": "va", "cash only": "cash_only",
    "1031 exchange": "exchange_1031", "short sale": "short_sale", "reo": "reo", "probate": "probate_estate",
    "investor": "investment_income",
    "investment property": "investment_income", "rental income": "investment_income",
    "income property": "investment_income", "income-producing": "investment_income",
    "rental potential": "investment_income", "short-term rental": "investment_income",
    "vacation rental": "investment_income", "airbnb": "investment_income",
    "motivated seller": "motivated_seller", "price improvement": "motivated_seller",
    "price reduced": "motivated_seller",
    # location
    "gated community": "gated_community", "guard-gated": "gated_community",
    "gated entry": "gated_community",     "walking trails": "parks_open_space", "hiking trails": "parks_open_space", "trails": "parks_open_space",
    "near parks": "parks_open_space",
    "near shopping": "near_shopping_dining", "close to shopping": "near_shopping_dining",
    "shopping and dining": "near_shopping_dining", "near restaurants": "near_shopping_dining",
    "near schools": "near_schools", "close to schools": "near_schools", "top-rated schools": "near_schools",
    "award-winning schools": "near_schools",
    "walking distance": "walkable", "walkable": "walkable",
    "freeway access": "freeway_commute", "major freeways": "freeway_commute", "commuter routes": "freeway_commute",
    "public transportation": "public_transit", "beach access": "near_beach_water", "lake community": "near_beach_water",
    "quiet street": "quiet_street", "quiet neighborhood": "quiet_street",
    "55+ community": "senior_55plus", "senior community": "senior_55plus",
}

# ---------------------------------------------------------------------------------------------
# Context vocabularies (data, so error analysis can extend them)
# ---------------------------------------------------------------------------------------------
NEG_BEFORE = re.compile(
    r"\b(?:no|not|without|never|nor|lacks?|lacking|isn't|aren't|wasn't|doesn't|don't|non)\b[^,.;!?)]{0,40}$", re.I)
# hypothetical / suggested-use cues: apply to condition and view-type claims
HYPO_BEFORE = re.compile(
    r"\b(?:could|can\s+be|may|might|would|potential\s+(?:to|for)|possibly|if|opportunity\s+to|imagine|"
    r"your\s+own|buyers?\s+(?:can|may|could|will|to)|ready\s+for\s+your|ready\s+to\s+(?:be\s+)?(?:remodel|renovat|update|upgrad)|"
    r"waiting\s+for\s+your|needs?\s+to\s+be|to\s+be|re-?imagine|future|plans?\s+(?:for|to)|easily|add(?:ing)?|"
    r"in\s+need\s+of|needing|needs?)\b[^,.;!?)]{0,40}$", re.I)

# proximity cue: words that say "this place is near X"
PROX = (r"(?:near(?:by|est)?|close\s+(?:to|by)|closeby|(?:north|south|east|west)\s+of|walk\s+score|minutes?\s+(?:from|to|away)|moments?\s+(?:from|to|away)|"
        r"steps?\s+(?:from|to)|blocks?\s+(?:from|to|away)|walk(?:ing)?\s+(?:distance\s+)?(?:to|from)|"
        r"walkable|short\s+(?:drive|walk|stroll)|proximity|conveniently|convenient(?:ly)?\s+(?:to|located)|"
        r"access\s+to|surrounded\s+by|adjacent|across\s+from|backs?\s+(?:up\s+)?to|within|mile|"
        r"easy\s+access|just\s+(?:minutes|steps|blocks)|short\s+distance|close\s+proximity|"
        r"near\s+all|nearby|walk\s+or\s+bike|bike\s+to)")
PROX_RE = re.compile(PROX, re.I)
# a proximity cue only counts when it is in the same sentence and not further than this many characters back
PROX_WINDOW = 140


def _sentence_before(text: str, pos: int, width: int) -> str:
    seg = text[max(0, pos - width):pos]
    cut = max(seg.rfind(". "), seg.rfind("! "), seg.rfind("? "), seg.rfind("\n"), seg.rfind(": "))
    return seg[cut + 1:] if cut >= 0 else seg


def _sentence_after(text: str, pos: int, width: int) -> str:
    seg = text[pos:pos + width]
    cut = min([i for i in (seg.find(". "), seg.find("! "), seg.find("? "), seg.find("\n")) if i >= 0] or [len(seg)])
    return seg[:cut]


# ---------------------------------------------------------------------------------------------
# Patterns. Each entry: signal -> list of (regex, options). Options: needs_prox, hypo (check hypothetical),
# neg (check negation, default True)
# ---------------------------------------------------------------------------------------------
_NAMED_VIEW = (r"(?:mountains?|ocean|sea|city|valley|canyons?|lakes?|hills?|hillsides?|golf(?:\s+course)?|"
               r"water|bay|coastal|sunsets?|treetops?|tree\s+tops?|desert|harbou?r|river|skyline|peaks?|forest|"
               r"vineyards?|pacific|island|catalina|snow|ski|slopes?|lights|foothills?|marina|canal|mountain-view|alpine|green\s?belt)")

P = {}
OPT = {}


def _add(signal: str, pattern: str, **opts):
    P.setdefault(signal, []).append((re.compile(pattern, re.I), opts))


# ---- condition
_add("renovated_updated",
     r"\b(?:remodel(?:ed|ing)?|renovat(?:ed|ion|ions)|rehabbed|refurbished|modernized|upgrades|updates)\b|"
     r"\b(?:upgraded|updated)\b(?!\s+(?:\w+\s+)?(?:flooring|floors?|tile|carpet|paint|lighting|light\s+fixtures|fixtures|"
     r"appliances|hardware|windows?|doors?|cabinets?|cabinetry|panel|electrical|plumbing|roof|hvac|a/c|water\s+heater|"
     r"countertops?|finishes)\b)", hypo=True)
_add("renovated_updated", r"\b(?:new|newer)\s+(?:kitchen|bath(?:room)?s?)\b", hypo=True)
_add("new_construction",
     r"\b(?:new\s+construction|newly\s+(?:built|constructed)|brand[- ]new\s+(?:home|construction|build|"
     r"townhome|condo|house)|never\s+lived\s+in|(?:estimated|est\.?|expected)\s+(?:completion|delivery)|"
     r"under\s+construction|model\s+home|to\s+be\s+built|now\s+building|move[- ]in\s+ready\s+(?:in|this)\s+\w+)\b",
     hypo=True)
_add("new_components",
     r"\b(?:new|newer|newly\s+(?:installed|replaced)|brand[- ]new|recently\s+(?:replaced|installed)|replaced)\s+"
     r"(?:[\w-]+\s+){0,2}?(?:roof|roofing|floor(?:ing|s)?|carpet(?:s|ing)?|paint|appliances?|windows?|hvac|a/c|ac|"
     r"air\s+condition\w*|furnace|water\s+heater|plumbing|electrical|panel|fence|fencing|fixtures?|cabinets?|"
     r"doors?|septic|stove|microwave|dishwasher|motor|siding|laminate|tile|hardware|lighting)\b", hypo=True)
_add("new_components", r"\b(?:fresh(?:ly)?\s+paint(?:ed)?|freshly\s+painted|fresh\s+interior\s+paint|new\s+paint)\b",
     hypo=True)
_add("new_components",
     r"\b(?:roof|water\s+heater|hvac|furnace|a/c|windows?|plumbing|electrical|panel|septic)\s+"
     r"(?:was\s+|were\s+|has\s+been\s+|have\s+been\s+)?(?:recently\s+|just\s+)?(?:replaced|new|updated)\b", hypo=True)
_add("move_in_ready", r"\b(?:turn[- ]?key|move[- ]in[- ]ready|move[- ]in\s+condition|ready\s+to\s+move\s+in|"
                       r"move\s+right\s+in|ready\s+for\s+(?:immediate\s+)?(?:move[- ]in|occupancy))\b")
_add("well_maintained",
     r"\b(?:(?:well|meticulously|beautifully|lovingly|immaculately|perfectly|nicely|thoughtfully|impeccably|"
     r"carefully)[- ](?:maintained|cared[- ]for|preserved|kept)|pristine|immaculate|pride\s+of\s+ownership)\b(?!\s+(?:\w+\s+)?(?:grounds|landscap\w*|common|building|complex|community|gardens|lawns?|streets|neighborhood|clubhouse))")
_add("needs_work",
     r"\b(?:fixer|needs?\s+(?:tlc|work|updating|repairs?|rehab|love|renovati\w+|remodel\w*|a\s+(?:little|lot\s+of)\s+(?:tlc|work))|"
     r"as[- ]is|handyman|investor\s+special|contractor\s+special|sweat\s+equity|original\s+condition|"
     r"bring\s+your\s+(?:vision|ideas|contractor)|blank\s+canvas|build\s+(?:instant\s+)?equity|"
     r"deferred\s+maintenance|tlc|cosmetic\s+(?:work|updates|fixer)|in\s+need\s+of)\b", neg=True)

# ---- financing / ownership
_add("seller_financing", r"\b(?:seller|owner)(?:'s)?[- ](?:\w+\s+){0,2}?(?:financ\w*|carry\w*)\b")
_add("seller_financing", r"\b(?:owner|seller)\s*carry|carry[- ]?back\b|owner[- ]financ\w*")
_add("assumable_loan", r"\bassum(?:able|ption)\b|\bassume\s+(?:the\s+|an?\s+)?(?:existing\s+)?(?:loan|mortgage|financing)\b")
_add("fha", r"\bfha\b")
_add("va", r"\bva\b(?!\s*(?:lley|nn))")
_add("cash_only", r"\bcash[- ](?:only|buyers?|offers?|sale|purchase|deal)\b|\ball[- ]cash\b|\bcash\s+or\s+hard\s+money\b")
_add("exchange_1031", r"\b1031\b")
_add("short_sale", r"\bshort[- ]sale\b")
_add("reo", r"\breo\b|\bbank[- ]owned\b|\bforeclos\w+|\blender[- ]owned\b")
_add("probate_estate", r"\bprobate\b|\bestate\s+sale\b|\btrust\s+sale\b|\bconservator\w*|\bcourt\s+confirm\w*|\btrustee\s+sale\b")
_add("tenant_occupied",
     r"\btenant[- ]occupied\b|\btenants?\s+in\s+place\b|\b(?:current|existing)\s+tenants?\b|"
     r"\bcurrently\s+(?:leased|rented|tenant)\b|\boccupied\s+by\s+(?:a\s+)?(?:tenants?|renters?)\b|"
     r"\bleased\s+(?:to|until|through)\b|\bsold\s+with\s+tenants?\b|\bmonth[- ]to[- ]month\s+tenan\w+")
_add("investment_income",
     r"\b(?:investors?|investment|income[- ](?:producing|property|potential)|rental\s+(?:income|property|potential)|"
     r"cap\s+rate|(?:short|long)[- ]term\s+rental|airbnb|vrbo|vacation\s+rental|income\s+(?:today|stream)|"
     r"gross\s+(?:rental\s+)?income|rent\s+the\s+other|positive\s+cash\s+flow|noi|str\s+permits?|"
     r"live\s+in\s+one)\b")
_add("seller_credit",
     r"\b(?:seller|builder|lender)\s+(?:credits?|concessions?|contribution|paid\s+closing)|"
     r"\b(?:buyer|seller)\s+concessions?|\bconcessions?\b|"
     r"\bclosing[- ]costs?\s+(?:credit|assistance|help|paid|contribution)|"
     r"\bcredits?\s+(?:toward|towards|for|at|to)\s+(?:the\s+)?(?:buyer|closing)|"
     r"\brate\s+buy[- ]?down\b|\bbuy[- ]?down\b|"
     r"\b(?:financing|builder|buyer|closing|special\s+financing)\s+incentives?\b|\bincentives?\s+(?:available|up\s+to)|"
     r"\b(?<!tax\s)(?<!energy\s)(?<!solar\s)(?:limited[- ]time\s+)?incentives?\b")
_add("motivated_seller",
     r"\b(?:motivated|priced\s+to\s+sell|price\s+(?:improvement|reduc\w*|adjust\w*|drop|lowered|deduction|cut)|"
     r"(?:just|recently|newly)\s+(?:reduced|lowered)|reduced\s+price|bring\s+(?:all\s+|your\s+)?offers|"
     r"must\s+sell|all\s+offers\s+considered|make\s+an\s+offer)\b")

# ---- location
_add("gated_community", r"\b(?:gated|guard[- ]gated|gate[- ]guarded)\b(?!\s+(?:underground|garage|parking|carport|rv|storage|yard|backyard|pool|patio|courtyard|driveway|side|front\s+door|lot))")
_add("golf_course",
     r"(?<!miniature\s)\bgolf(?:ing)?\s+(?:course|community|club|membership|resort|links|country|estates)s?\b|\bfairway\b|\bon\s+the\s+golf\b")
_add("golf_course", r"(?<!miniature\s)\bgolf(?:ing)?\b(?!\s+(?:cart|car\b))", needs_prox=True)
_add("near_schools", r"\bschools?\b", ctx="school")
_add("near_shopping_dining",
     r"\b(?:shopping|restaurants?|eateries|groceries|grocery|cafes?|boutiques|entertainment|retail|"
     r"dining(?!\s*(?:room|area|table|nook|space|set|bar|island|counter|outdoors?|and\s+entertaining|al\s+fresco|alfresco|chairs?))|supermarkets?|stores|shops|markets?)\b",
     needs_prox=True)
_add("freeway_commute",
     r"\b(?:freeways?|fwy|highways?|hwy|interstate|i-?\s?\d{1,3}|commuter|commute|commuting|"
     r"(?:major\s+)?transport(?:ation)?\s+corridors?|sr[- ]?\d{1,3}|state\s+route)\b")
_add("public_transit",
     r"\b(?:bart|metro(?:link)?|trolley|light\s+rail|subway(?!\s+tile)|transit|train(?:\s+station)?|amtrak|caltrain|sprinter|"
     r"coaster|bus\s+(?:stop|line|route|stops|lines)|public\s+transportation|railroad\s+station|"
     r"transportation\s+station|rail\s+station|metro\s+station)\b")
_add("walkable", r"\b(?:walkable|walkability|walk\s+score|walking\s+distance|walk(?:ing)?\s+to|short\s+walk|stroll\s+to|"
                  r"walk\s+or\s+bike|within\s+walking)\b")
_add("near_beach_water",
     r"\b(?:beach(?:es)?|ocean|bay(?!\s+(?:area|window|windows))|marina|harbou?r|coast(?:al)?|shore|sand|surf|boardwalk|"
     r"lakes?(?!\s+park)|waterways?)\b", needs_prox=True)
_add("near_beach_water", r"\bwater[- ]?front\b", needs_prox=True)
_add("near_beach_water", r"\blake\s+(?:access|rights|community|oriented|front)|\b(?:private|community|three|two|"
                          r"several|man-made|picturesque)\s+lakes?\b|\bbeach\s+(?:access|club|community)\b")
_add("waterfront",
     r"\b(?:water[- ]?front|ocean[- ]?front|beach[- ]?front|lake[- ]?front|canal[- ]?front|bay[- ]?front|"
     r"river[- ]?front|boat\s+dock|private\s+dock|own\s+dock|water\s+frontage|deep[- ]water)\b|"
     r"\bbacks?\s+(?:right\s+)?(?:up\s+)?(?:to|onto)\s+(?:the\s+)?(?:lake|ocean|water|bay|canal|river)\b|"
     r"\bon\s+the\s+(?:water|lake|bay|ocean|canal)\b")
_add("views",
     rf"\b{_NAMED_VIEW}\b[^.;!?]{{0,25}}?\b(?:views?|vistas?)\b|\b(?:views?|vistas?)\s+(?:of|over|across|to)\s+"
     rf"(?:the\s+|an?\s+|your\s+)?(?:\w+\s+){{0,2}}?{_NAMED_VIEW}\b|\bviewpoint\b|\bcity\s+lights\b", hypo=True)
_add("quiet_street",
     r"\b(?:quiet|peaceful|tranquil|serene|calm)[,\s]+(?:(?!(?:garden|backyard|yard|patio|outdoor|pool|balcony|deck|retreat|bedroom|spa)\b)[\w-]+[,\s]+){0,2}?(?:street|neighbou?rhood|community|cul|location|"
     r"setting|area|enclave|court|lane|road|tract|block|environment)\b|\bcul[- ]de[- ]sac\b|\bsecluded\b|\bseclusion\b|\bpeace\s+and\s+quiet\b|\bsense\s+of\s+quiet\b|\bquiet\s+residential\b|\bprivate\s+road\b|"
     r"\bdead[- ]end\b|\blittle\s+to\s+no\s+traffic\b|\bno\s+through\s+traffic\b")
_add("parks_open_space",
     r"\b(?:greenbelt|green\s+space|open\s+space|nature\s+preserve|preserve|trails?|hiking|wilderness|"
     r"parks?(?!ing)|park-like)\b", ctx="park")
_add("senior_55plus", r"\b55\s?\+|\b55\s+(?:and\s+(?:over|older|better)|plus)\b|\bover\s+55\b|\bactive\s+adult\b|"
                       r"\bage[- ]restricted\b|\bsenior\s+(?:communit(?:y|ies)|living|housing)\b|\bretirement\s+communit(?:y|ies)\b|"
                       r"\badult\s+communit(?:y|ies)\b")
_add("near_downtown", r"\b(?:downtown|city\s+center)\b", needs_prox=True)

# extra cues for "near_downtown" etc: heart of ...
HEART_RE = re.compile(r"\bheart\s+of\b")
HYPO_SAFE_FAMILIES = {"financing"}   # families where "could/may" cues are not applied


# HOA / Mello-Roos fields -------------------------------------------------------------------------------
HOA_NONE = re.compile(r"\b(?:no|without|not\s+in\s+an?|zero|non)[- ](?:\w+\s+){0,2}?(?:hoa|h\.o\.a\.?|association|homeowners?\s+association)\b|"
                      r"\bhoa\s*(?:fees?|dues)?\s*(?::|-)?\s*(?:none|\$?0\b|n/?a)", re.I)
HOA_LOW = re.compile(r"\b(?:low|lower|minimal|very\s+low|affordable|nominal|reasonable)\s+(?:monthly\s+)?(?:hoa|h\.o\.a\.?|"
                     r"association|dues|fees)\b|\b(?:hoa|association)\s+(?:dues\s+|fees\s+)?(?:is\s+|are\s+)?(?:very\s+)?low\b", re.I)
HOA_PRESENT = re.compile(r"\b(?:hoa|h\.o\.a\.?|homeowners?\s+association|association\s+(?:dues|fees)|monthly\s+dues|"
                         r"hoa\s+(?:dues|fees))\b", re.I)
MELLO_NONE = re.compile(r"\b(?:no|without|zero|non)[- ]mello[- ]?roos?\b|\bmello[- ]?roos?\s*(?::|-)?\s*(?:none|\$?0\b|n/?a)", re.I)
MELLO_PRESENT = re.compile(r"\bmello[- ]?roos?\b", re.I)


def hoa_field(text: str) -> Optional[str]:
    if HOA_NONE.search(text):
        return "none"
    if HOA_LOW.search(text):
        return "low"
    if HOA_PRESENT.search(text):
        return "present"
    return None


def mello_roos_field(text: str) -> Optional[str]:
    if MELLO_NONE.search(text):
        return "none"
    if MELLO_PRESENT.search(text):
        return "present"
    return None


# ---------------------------------------------------------------------------------------------
class SignalExtractor:
    """Extract condition / financing / location signals plus Week 3 entities from a listing record."""

    def __init__(self, taxonomy: Optional[dict] = None, entity_extractor: Optional[EntityExtractor] = None,
                 taxonomy_path: Optional[Path] = None):
        # ``taxonomy`` is the parsed taxonomy.json ({'terms': [...]}) or None to load the default file
        self.entity_extractor = entity_extractor or EntityExtractor(taxonomy_path=taxonomy_path)
        self.taxonomy = taxonomy
        self.entity_extractor._load_taxonomy()
        if self.taxonomy is None:
            self.taxonomy = {"terms": self.entity_extractor._terms}
        self._term_to_signal = {}
        for t in self.taxonomy["terms"]:
            key = EntityExtractor._normalize_term(t["term"])
            sig = TAXONOMY_TO_SIGNAL.get(t["term"].lower()) or TAXONOMY_TO_SIGNAL.get(key)
            if sig:
                self._term_to_signal[t["id"]] = sig

    # ------------------------------------------------------------------ context checks
    @staticmethod
    def _before(text: str, pos: int, width: int = 90) -> str:
        return _sentence_before(text, pos, width)

    def _blocked(self, signal: str, text: str, s: int, e: int, opts: dict) -> Optional[str]:
        before = self._before(text, s)
        if opts.get("neg", True) and NEG_BEFORE.search(before):
            # "no HOA" style negation of a signal; keep "not only", "not just"
            tail = before[NEG_BEFORE.search(before).start():]
            if not re.match(r"\b(?:not\s+(?:only|just)|no\s+(?:further|more)\b)", tail, re.I):
                return "negated"
        if opts.get("hypo") and FAMILY_OF[signal] != "financing" and HYPO_BEFORE.search(before):
            return "hypothetical"
        if signal == "tenant_occupied" and re.search(r"\b(?:previously|formerly|was|were|until)\b[^,.;!?)]{0,25}$", before, re.I):
            return "past tenancy"
        if signal == "waterfront" and re.search(
                r"\b(?:near|close\s+to|minutes?\s+(?:from|to)|walking\s+distance|walk(?:ing)?\s+to|steps?\s+(?:from|to)|"
                r"short\s+(?:drive|walk)|proximity|blocks?\s+(?:from|to)|access\s+to)\b[^.;!?]{0,60}$", before, re.I):
            return "near the waterfront, not on it"
        return None

    def _has_prox(self, text: str, s: int, e: int) -> bool:
        before = _sentence_before(text, s, PROX_WINDOW)
        if PROX_RE.search(before) or HEART_RE.search(before):
            return True
        after = _sentence_after(text, e, 40)
        return bool(re.match(r"[^.;]{0,30}?\b(?:nearby|close\s+by|within\s+(?:walking|\d|a\s+few)|are\s+(?:near|close)|just\s+minutes|minutes\s+away)\b", after, re.I))

    def _school_ok(self, text: str, s: int, e: int) -> bool:
        before = _sentence_before(text, s, 120)
        after = _sentence_after(text, e, 60)
        district = bool(re.match(r"\s*district", after, re.I))
        quality_pre = re.search(r"\b(?:award[- ]winning|top|highly|well|blue\s+ribbon|excellent|acclaimed|nationally|"
                                r"best)[^.;]{0,25}$", before, re.I)
        if district:                                  # a bare school-district name is not "near schools"
            return bool(quality_pre)
        if PROX_RE.search(before):
            return True
        quality = re.search(r"\b(?:top|highly|well|award|blue\s+ribbon|excellent|great|desirable|good|high[- ]scoring|"
                            r"acclaimed|sought[- ]after|best|nationally|rated|ranked|strong|quality|reputable)[^.;]{0,25}$", before, re.I)
        if quality:
            return True
        if re.match(r"[^.;]{0,25}\b(?:are|is)\s+(?:nearby|close|within|walkable|top|highly|excellent|great|rated)", after, re.I):
            return True
        return False

    def _park_ok(self, text: str, s: int, e: int) -> bool:
        word = text[s:e].lower()
        if word in {"greenbelt", "green space", "nature preserve", "wilderness", "hiking", "park-like"} or word.startswith("trail"):
            return True
        if word == "open space":
            after = _sentence_after(text, e, 25)
            before = _sentence_before(text, s, 25)
            if re.match(r"\s*(?:floor|concept|living|plan|layout|design|kitchen)", after, re.I):
                return False
            return not re.search(r"\b(?:an|the)\s*$", before, re.I)
        if word == "preserve":
            return bool(self._has_prox(text, s, e))
        if word.startswith("park"):
            before = _sentence_before(text, s, 25)
            if re.search(r"\b(?:business|industrial|mobile\s+home|rv|trailer|office|theme|amusement|water|vehicle|state\s+and|dog|skate|ball|car|trailer)\s+$", before, re.I):
                return False
            return self._has_prox(text, s, e)
        return False

    # ------------------------------------------------------------------ core
    def find_signal_mentions(self, text: str, amenity_mentions: Optional[List[dict]] = None) -> List[dict]:
        """Every candidate mention of a canonical signal, with the reason it was dropped (None = kept).

        ``text`` must be the cleaned text. Offsets refer to it.
        """
        mentions: List[dict] = []
        # 1. regex layer
        for signal, plist in P.items():
            for rx, opts in plist:
                for m in rx.finditer(text):
                    s, e = m.start(), m.end()
                    reason = self._blocked(signal, text, s, e, opts)
                    if reason is None and opts.get("needs_prox") and not self._has_prox(text, s, e):
                        reason = "no proximity cue"
                    if reason is None and opts.get("ctx") == "school" and not self._school_ok(text, s, e):
                        reason = "school without proximity or quality cue"
                    if reason is None and opts.get("ctx") == "park" and not self._park_ok(text, s, e):
                        reason = "park without proximity cue"
                    mentions.append({"signal": signal, "source": "pattern", "start": s, "end": e,
                                     "text": text[s:e], "dropped": reason})
        # 2. taxonomy layer (Week 3 filters already applied by EntityExtractor)
        if amenity_mentions is None:
            amenity_mentions = self.entity_extractor.find_amenity_mentions(text)
        for a in amenity_mentions:
            sig = self._term_to_signal.get(a["id"])
            if sig:
                reason = a["dropped"]
                if reason is None and FAMILY_OF[sig] != "financing":
                    before = self._before(text, a["start"])
                    if NEG_BEFORE.search(before) or HYPO_BEFORE.search(before):
                        reason = "negated or hypothetical"
                mentions.append({"signal": sig, "source": "taxonomy", "start": a["start"], "end": a["end"],
                                 "text": a["text"], "dropped": reason})
        return mentions

    def signals_from_text(self, text: str, clean: bool = True, amenity_mentions: Optional[List[dict]] = None) -> Tuple[Dict[str, List[str]], Dict[str, Optional[str]], List[dict]]:
        if clean:
            text = self.entity_extractor.clean(text)
        mentions = self.find_signal_mentions(text, amenity_mentions)
        kept = sorted({m["signal"] for m in mentions if m["dropped"] is None}, key=ALL_SIGNALS.index)
        # a "gated" mention inside "gated" + "no gated" etc. is fine; a few signals imply others
        by_family = {f: [s for s in kept if FAMILY_OF[s] == f] for f in FAMILIES}
        fields = {"hoa": hoa_field(text), "mello_roos": mello_roos_field(text)}
        return by_family, fields, mentions

    def extract_signals(self, listing_record: dict, evidence: bool = False) -> dict:
        """One listing record -> JSON-serialisable dict. Never contains the remark text (unless evidence=True,
        which adds short matched phrases for debugging)."""
        lid = str(listing_record.get("L_ListingID", listing_record.get("listing_id", "")))
        raw = listing_record.get("L_Remarks")
        if raw is None:
            raw = listing_record.get("remarks")
        raw = "" if raw is None or (isinstance(raw, float) and raw != raw) else str(raw)
        text = self.entity_extractor.clean(raw) if raw.strip() else ""

        ents = self.entity_extractor.extract_all(text, clean=False, with_amenities=False) if text else \
            {"bedrooms": None, "bathrooms": None, "price": None, "sqft": None}
        amen = []
        amen_mentions = self.entity_extractor.find_amenity_mentions(text) if text else []   # computed once, used twice
        seen = set()
        for a in amen_mentions:
            if a["dropped"] is None and a["category"] in FEATURE_CATEGORIES and a["id"] not in seen:
                seen.add(a["id"])
                amen.append(a["term"])
        by_family, fields, mentions = self.signals_from_text(text, clean=False, amenity_mentions=amen_mentions) if text else \
            ({f: [] for f in FAMILIES}, {"hoa": None, "mello_roos": None}, [])

        keywords = []
        for fam, sigs in by_family.items():
            keywords += [f"{FAMILY_PREFIX[fam]}:{s}" for s in sigs]
        if fields["hoa"]:
            keywords.append(f"hoa:{fields['hoa']}")
        if fields["mello_roos"]:
            keywords.append(f"mello_roos:{fields['mello_roos']}")

        out = {
            "listing_id": lid,
            "entities": {"bedrooms": ents["bedrooms"], "bathrooms": ents["bathrooms"],
                         "sqft": ents["sqft"], "price": ents["price"]},
            "amenities": amen,
            "condition_keywords": by_family["condition"],
            "financing_terms": by_family["financing"],
            "location_features": by_family["location"],
            "fields": fields,
            "keywords": keywords,
            "schema_version": SCHEMA_VERSION,
        }
        if evidence:
            out["evidence"] = [{"signal": m["signal"], "phrase": m["text"], "source": m["source"],
                                "kept": m["dropped"] is None, "dropped": m["dropped"]} for m in mentions]
        return out

    def predicted_signals(self, text: str) -> Tuple[set, Dict[str, Optional[str]]]:
        """(set of canonical signals, fields) for one remark. Used by the evaluator."""
        by_family, fields, _ = self.signals_from_text(text)
        return {s for ss in by_family.values() for s in ss}, fields
