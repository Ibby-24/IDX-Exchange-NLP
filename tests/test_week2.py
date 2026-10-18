"""Week 2 tests: these ARE the specification for scripts/text_cleaning.py.

Run from the repo root:
    pytest tests/test_week2.py -v                 # everything
    pytest tests/test_week2.py -v -k unicode      # one method at a time (use -k prices, -k measurements, ...)

Rules encoded here:
  * Only expand an abbreviation when it is unambiguous. HOA, ADU, HVAC, RV, EV, BBQ ... are NEVER changed.
  * "sf" means square feet only when it follows a number ("1,610 SF"), never on its own ("SF Bay Area").
  * k / m / million become numbers only with a "$" or right after a price word (under, over, around ...),
    so "FHA 203K loan", "(0.2m or ~4min walk)" and "3 million tourists" are left alone.
  * Expansions use singular forms ("3 bedroom"), lowercase, and never touch letters inside other words.
  * clean_text() must be idempotent: clean(clean(x)) == clean(x).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from text_cleaning import TextCleaner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONFIG_PATH = ROOT / "config" / "abbreviations.json"

def squash(text):
    """Collapse all whitespace so markup tests don't depend on exact spacing."""
    return " ".join(text.split())


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner()

UNICODE_CASES = [
    ("it\u2019s", "it's"),                                    #curly apostrophe
    ("\u201cWelcome\u201d", '"Welcome"'),                     #curly double quotes
    ("patio\u2014perfect", "patio - perfect"),                #em dash, no spaces
    ("location \u2014 an", "location - an"),                  #em dash with spaces
    ("2\u20133 bedrooms", "2 - 3 bedrooms"),                  #en dash
    ("today\u200b\u200c\u200b and", "today and"),             #zero-width characters
    ("walk\u00a0in", "walk in"),                              #non-breaking space
    ("\u00bd-bath", "1/2-bath"),                              #unicode fraction
    ("lot \u00b1 5 acres", "lot +/- 5 acres"),                #plus-minus
    ("Caltrain\u00ae station", "Caltrain station"),           #registered sign removed
    ("fa\u00c3\u00a7ade", "fa\u00e7ade"),                     #mojibake: faÃ§ade -> façade
    ("pied-\u00c3\u00a0-terre", "pied-\u00e0-terre"),         #mojibake with NBSP byte (order trap!)
    ("Plain text stays.", "Plain text stays."),               #untouched
    ("Caf\u00e9 stays", "Caf\u00e9 stays"),                   #legitimate accents survive
    ("Caltrain\u00c3\u00c2\u00ae station", "Caltrain station"),  #double-encoded leftovers
    ("\u2022 Pool \u2022 Spa", "- Pool - Spa"),                #bullets become dashes
]

@pytest.mark.parametrize("raw,expected", UNICODE_CASES)
def test_unicode(cleaner, raw, expected):
    assert cleaner.normalize_unicode(raw) == expected

WHITESPACE_CASES = [
    ("a  b", "a b"),
    ("line1\nline2", "line1 line2"),
    ("a\t\tb", "a b"),
    ("  padded  ", "padded"),
    ("a \n \n b", "a b"),
    ("", ""),
]


@pytest.mark.parametrize("raw,expected", WHITESPACE_CASES)
def test_whitespace(cleaner, raw, expected):
    assert cleaner.normalize_whitespace(raw) == expected

MARKUP_CASES = [
    ("<b>Call</b> today", "Call today"),
    ("Pool<br/>Spa", "Pool Spa"),
    ("<p>Hello</p>", "Hello"),
    ("Tom &amp; Jerry", "Tom & Jerry"),
    ("Pool&nbsp;Spa", "Pool Spa"),
    ("**Professional photos coming soon**", "Professional photos coming soon"),
    ("***PRICE IMPROVEMENT***Welcome", "PRICE IMPROVEMENT Welcome"), 
    ("Price under <$500k", "Price under <$500k"), 
    ("5 * 3", "5 * 3"),
]


@pytest.mark.parametrize("raw,expected", MARKUP_CASES)
def test_markup(cleaner, raw, expected):
    assert squash(cleaner.strip_markup(raw)) == expected


NUMBER_CASES = [
    ("$1,250,000", "$1250000"),
    ("2,000 sq ft", "2000 sq ft"),
    ("12,345,678", "12345678"),
    ("1,250.50", "1250.50"),
    ("built in 2015, remodeled", "built in 2015, remodeled"),
    ("rooms 1,2,3", "rooms 1,2,3"),
    ("call 555-1234, ext 12", "call 555-1234, ext 12"),
    ("3,2", "3,2"),
]


@pytest.mark.parametrize("raw,expected", NUMBER_CASES)
def test_numbers(cleaner, raw, expected):
    assert cleaner.normalize_numbers(raw) == expected


PRICE_CASES = [
    ("priced at 450k", "priced at 450000"),
    ("$1.2m home", "$1200000 home"),
    ("under 450k", "under 450000"),
    ("under 1.1m", "under 1100000"),
    ("over $1.2M", "over $1200000"),
    ("$1.3 million", "$1300000"),
    ("$950K - $980K", "$950000 - $980000"),
    ("$70k in upgrades", "$70000 in upgrades"),
    ("under $1M!!", "under $1000000!!"),
    ("$1.001m", "$1001000"),
    ("$1.15m", "$1150000"),
    ("around 500k", "around 500000"),
    ("up to 1.5 million", "up to 1500000"),
]


@pytest.mark.parametrize("raw,expected", PRICE_CASES)
def test_prices(cleaner, raw, expected):
    assert cleaner.normalize_prices(raw) == expected


PRICE_UNCHANGED = [
    "FHA 203K loan", 
    "conventional or FHA 203k renovation loan",
    "(0.2m or ~4min walk)",
    "3 million tourists visit",
    "a 4K TV",
    "10min drive to the beach",
    "approximately $80/month",
    "built in 2015",
]


@pytest.mark.parametrize("raw", PRICE_UNCHANGED)
def test_prices_left_alone(cleaner, raw):
    assert cleaner.normalize_prices(raw) == raw

MEASUREMENT_CASES = [
    ("2,000 sqft", "2000 square feet"), 
    ("1,610 SF residence", "1610 square feet residence"),
    ("750 sq. ft. lot", "750 square feet lot"),
    ("2,634 sq ft", "2634 square feet"),
    ("2000sf", "2000 square feet"),
    ("1,775 SF of modern living", "1775 square feet of modern living"),
    ("4,200 SF lot", "4200 square feet lot"),
    ("1,893 square feet", "1893 square feet"),
    ("900 sq.ft. condo", "900 square feet condo"),
    ("1,200 sq feet", "1200 square feet"),
    ("1,400+ sq ft", "1400+ square feet"), 
    ("+/-5944 sq. ft. corner lot", "+/-5944 square feet corner lot"),
    ("4076+/- Sq. Ft. lot", "4076+/- square feet lot"),
    ("4,400 sq/ft property", "4400 square feet property"),
]


@pytest.mark.parametrize("raw,expected", MEASUREMENT_CASES)
def test_measurements(cleaner, raw, expected):
    assert cleaner.normalize_measurements(raw) == expected


MEASUREMENT_UNCHANGED = [
    "SF Bay Area",
    "ADD SF to existing home",
    "36-ft covered patio", 
    "1/4 acre lot",
    "10 ft ceilings",
    "built in 2015",
    "BED / BATH: 2/2,0,0,0 square feet(src)",
]


@pytest.mark.parametrize("raw", MEASUREMENT_UNCHANGED)
def test_measurements_left_alone(cleaner, raw):
    assert cleaner.normalize_measurements(raw) == raw

ABBREVIATION_CASES = [
    ("3br 2ba", "3 bedroom 2 bathroom"),
    ("3 BR, 2 BA", "3 bedroom, 2 bathroom"),
    ("2BR/2BA layout", "2 bedroom 2 bathroom layout"),
    ("3BD/2BA", "3 bedroom 2 bathroom"),
    ("1br/1bth ADU", "1 bedroom 1 bathroom ADU"),
    ("1.5ba", "1.5 bathroom"),
    ("pool w/ spa", "pool with spa"),
    ("sold w/o garage", "sold without garage"),
    ("central A/C", "central air conditioning"),
    ("central AC", "central air conditioning"),
    ("LVP flooring", "luxury vinyl plank flooring"),
    ("approx. 4,200", "approximately 4,200"),
    ("approx 2 acres", "approximately 2 acres"),
    ("near hwy 101", "near highway 101"),
    ("fwy access", "freeway access"),
    ("mbr with balcony", "primary bedroom with balcony"),
    ("t&g ceilings", "tongue and groove ceilings"),
    ("W/D included", "washer and dryer included"),
    ("kitchen w/quartz island", "kitchen with quartz island"),
]


@pytest.mark.parametrize("raw,expected", ABBREVIATION_CASES)
def test_abbreviations(cleaner, raw, expected):
    assert cleaner.expand_abbreviations(raw) == expected


ABBREVIATION_UNCHANGED = [
    "HOA fees apply", 
    "ADU with RV parking",
    "HVAC and EV charger",
    "BBQ and TV",
    "FHA loan near BART",
    "UC Irvine in LA",
    "backyard brick patio", 
    "3 bank branches nearby", 
    "2 brick patios",
    "across the street",
    "approximately 2 acres", 
    "SF Bay Area",
]


@pytest.mark.parametrize("raw", ABBREVIATION_UNCHANGED)
def test_abbreviations_left_alone(cleaner, raw):
    assert cleaner.expand_abbreviations(raw) == raw

SLASH_CASES = [
    ("pool/spa", "pool spa"),
    ("indoor/outdoor living", "indoor outdoor living"),
    ("washer/dryer", "washer dryer"),
    ("den/office", "den office"),
]


@pytest.mark.parametrize("raw,expected", SLASH_CASES)
def test_slashes(cleaner, raw, expected):
    assert cleaner.normalize_slashes(raw) == expected


@pytest.mark.parametrize("raw", ["24/7 security", "1/2 bath", "$80/month", "N/A", "3/4 bath"])
def test_slashes_left_alone(cleaner, raw):
    assert cleaner.normalize_slashes(raw) == raw

INTEGRATION_CASES = [
    ("Gorgeous 3BR/2BA w/ A/C \u2014 approx. 1,850 SF; asking $1.2m",
     "Gorgeous 3 bedroom 2 bathroom with air conditioning - approximately 1850 square feet; asking $1200000"),
    ("3br 2ba under 450k w/ pool",
     "3 bedroom 2 bathroom under 450000 with pool"),
    ("2,000 sqft or larger in Oceanside under $1.2M move in ready",
     "2000 square feet or larger in Oceanside under $1200000 move in ready"),
    ("**Professional photos coming soon**  It\u2019s a must-see\u2014call today!",
     "Professional photos coming soon It's a must-see - call today!"),
    ("pool/spa, indoor/outdoor living & built-in BBQ",
     "pool spa, indoor outdoor living & built-in BBQ"),
    ("FHA 203K loan eligible, HOA $350/month",
     "FHA 203K loan eligible, HOA $350/month"),
    ("Close to hwy 101 & UC Irvine. HVAC, EV charger, RV parking, BBQ.",
     "Close to highway 101 & UC Irvine. HVAC, EV charger, RV parking, BBQ."),
    ("Walk in closet w/ built-ins; LVP floors",
     "Walk in closet with built-ins; luxury vinyl plank floors"),
    ("Spacious 1,200 sq  ft home", 
     "Spacious 1200 square feet home"),
    ("new HVAC/AC system", 
     "new HVAC air conditioning system"),
    ("\u2022 Pool/spa\n\u2022 4,400+ SF lot",
     "- Pool spa - 4400+ square feet lot"),
]


@pytest.mark.parametrize("raw,expected", INTEGRATION_CASES)
def test_clean_text_end_to_end(cleaner, raw, expected):
    assert cleaner.clean_text(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    (None, ""),
    (float("nan"), ""),
    ("", ""),
    ("   ", ""),
    ("\n\n", ""),
    (123, "123"),
])
def test_clean_text_edge_inputs(cleaner, raw, expected):
    assert cleaner.clean_text(raw) == expected


@pytest.mark.parametrize("raw", [case[0] for case in INTEGRATION_CASES])
def test_clean_text_is_idempotent(cleaner, raw):
    once = cleaner.clean_text(raw)
    assert cleaner.clean_text(once) == once


@pytest.mark.parametrize("raw", [
    "Beautiful single story home with a pool.",
    "Open floor plan, granite countertops, and a two-car garage.",
    "Located in Temecula near top-rated schools.",
])
def test_clean_text_leaves_clean_text_alone(cleaner, raw):
    assert cleaner.clean_text(raw) == raw

@pytest.fixture(scope="module")
def tiny_df():
    return pd.DataFrame({"remarks": [
        "Pool w/ spa. A/C included. 3br 2ba  home",
        "Call <b>today</b> for $450,000 deal\nnow",
        None,
        "Plain text only.",
        "caf\u00e9 \u2014 4,200 SF lot",
    ]})


def test_profile_has_required_keys(cleaner, tiny_df):
    profile = cleaner.profile_column(tiny_df, "remarks")
    for key in ["null_rate", "avg_length", "common_terms", "price_mentions", "has_html",
                "common_abbreviations", "non_ascii", "double_spaces", "newlines"]:
        assert key in profile


def test_profile_counts(cleaner, tiny_df):
    profile = cleaner.profile_column(tiny_df, "remarks")
    assert profile["null_rate"] == pytest.approx(0.2)
    assert profile["avg_length"] > 0
    assert profile["has_html"] == 1
    assert profile["price_mentions"] == 1
    assert profile["non_ascii"] == 1
    assert profile["double_spaces"] == 1
    assert profile["newlines"] == 1


def test_profile_detects_abbreviations(cleaner, tiny_df):
    abbreviations = dict(cleaner.profile_column(tiny_df, "remarks")["common_abbreviations"])
    for token in ["w/", "a/c", "br", "ba", "sf"]:
        assert token in abbreviations


def test_profile_does_not_modify_the_dataframe(cleaner, tiny_df):
    before = tiny_df.copy()
    cleaner.profile_column(tiny_df, "remarks")
    pd.testing.assert_frame_equal(tiny_df, before)

def test_abbreviation_config_has_30_plus_entries():
    cfg = json.load(open(CONFIG_PATH))
    assert len(cfg["standalone"]) + len(cfg["after_number"]) >= 30


def test_protected_acronyms_are_never_expanded():
    cfg = json.load(open(CONFIG_PATH)) 
    keys = set(cfg["standalone"]) | set(cfg["after_number"])
    assert not keys & set(cfg["protected"])


def test_abbreviation_keys_are_lowercase():
    cfg = json.load(open(CONFIG_PATH))
    for key in list(cfg["standalone"]) + list(cfg["after_number"]):
        assert key == key.lower()
