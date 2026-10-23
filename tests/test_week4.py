"""Week 4 tests: QueryParser, SchemaValidator, safe SQL generation and SQL-injection protection.

Run from the repo root:
    pytest tests/test_week4.py -v

* The parser cases use an inline city list (MINI_SCHEMA) and invented queries, so they need no data files
  except taxonomy.json (Week 1).
* The accuracy gate on the 60 Week 1 queries needs data/processed/schema.json (python scripts/build_schema.py)
  and skips, instead of failing, when it is missing.
* The held-out queries in query_gold.json are deliberately NOT used here. Score them once with:
    python scripts/evaluate_queries.py --heldout --final
* Injection tests run the generated SQL for real on an in-memory SQLite table that has the same column
  names as rets_property (REGEXP_LIKE is registered to behave like MySQL 8's).
"""
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from query_parser import (ALLOWED_FILTERS, PATTERN_NAMES, QueryParser, amenity_regex,  # noqa: E402
                          build_sql)
from run_query import execute_sqlite, sqlite_connection, to_qmark  # noqa: E402
from schema_validator import DEFAULT_LIMITS, SchemaValidator       # noqa: E402

SCHEMA_FILE = ROOT / "data" / "processed" / "schema.json"
GOLD = ROOT / "data" / "processed" / "query_gold.json"
TAXONOMY = ROOT / "data" / "processed" / "taxonomy.json"

MINI_SCHEMA = {
    "cities": ["Irvine", "Corona", "Riverside", "Temecula", "Murrieta", "Pasadena", "Oakland", "Hemet",
               "Escondido", "Lancaster", "San Jose", "Santee", "Fontana", "Indio", "Menifee",
               "Palm Desert", "San Diego", "Long Beach", "Los Angeles", "Walnut Creek", "Oceanside",
               "Big Bear City", "Big Bear Lake", "Nice", "Orange", "Simi Valley"],
    "columns": {"city": "L_City", "price": "L_SystemPrice", "bedrooms": "L_Keyword2",
                "bathrooms": "LM_Dec_3", "sqft": "LM_Int2_3", "remarks": "L_Remarks"},
}


@pytest.fixture(scope="module")
def parser():
    return QueryParser(schema=MINI_SCHEMA)


@pytest.fixture(scope="module")
def validator():
    return SchemaValidator(schema=MINI_SCHEMA)


# ======================================================================================
# 1. Parser: 70 invented queries -> expected filters
# ======================================================================================
CASES = [
    # --- the spec example and basic shapes
    ("3 bed under 700k in Irvine", {"city": "Irvine", "bedrooms": 3, "price_max": 700000}),
    ("2 bedroom condo in Pasadena under $650,000", {"city": "Pasadena", "bedrooms": 2, "price_max": 650000, "amenities": ["condo"]}),
    ("four bedroom house in Temecula", {"city": "Temecula", "bedrooms": 4}),
    ("3br/2ba in Corona", {"city": "Corona", "bedrooms": 3, "bathrooms": 2}),
    ("4/3 in Murrieta", {"city": "Murrieta", "bedrooms": 4, "bathrooms": 3}),
    ("3-bedroom home in Hemet", {"city": "Hemet", "bedrooms": 3}),
    # --- bedroom / bathroom ranges
    ("at least 4 bedrooms in Corona", {"city": "Corona", "bedrooms_min": 4}),
    ("minimum 3 bathrooms in Irvine", {"city": "Irvine", "bathrooms_min": 3}),
    ("5+ bedroom homes in Riverside", {"city": "Riverside", "bedrooms_min": 5}),
    ("2 bedrooms or more in Oakland", {"city": "Oakland", "bedrooms_min": 2}),
    ("3 or more bedrooms in Oakland", {"city": "Oakland", "bedrooms_min": 3}),
    ("3 bedrooms max in Hemet", {"city": "Hemet", "bedrooms_max": 3}),
    ("up to 3 bedrooms in Escondido", {"city": "Escondido", "bedrooms_max": 3}),
    ("fewer than 3 bedrooms in Lancaster", {"city": "Lancaster", "bedrooms_max": 2}),
    ("more than 2 bathrooms in San Jose", {"city": "San Jose", "bathrooms_min": 3}),
    ("2 to 3 bathrooms in Murrieta", {"city": "Murrieta", "bathrooms_min": 2, "bathrooms_max": 3}),
    ("2-3 bedroom in Santee", {"city": "Santee", "bedrooms_min": 2, "bedrooms_max": 3}),
    ("3 or 4 bedrooms in Fontana", {"city": "Fontana", "bedrooms_min": 3, "bedrooms_max": 4}),
    ("1.5 bath in Hemet", {"city": "Hemet", "bathrooms": 1.5}),
    ("2.5 bathrooms or more in Walnut Creek", {"city": "Walnut Creek", "bathrooms_min": 2.5}),
    ("at least 3 bedrooms and 2 bathrooms in Indio", {"city": "Indio", "bedrooms_min": 3, "bathrooms_min": 2}),
    ("at least 2 bathrooms and 4 bedrooms in Indio", {"city": "Indio", "bedrooms_min": 4, "bathrooms_min": 2}),
    ("more than 2 bedrooms and 3 bathrooms in Indio", {"city": "Indio", "bedrooms_min": 3, "bathrooms_min": 4}),
    ("fewer than 4 bedrooms and 3 bathrooms in Indio", {"city": "Indio", "bedrooms_max": 3, "bathrooms_max": 2}),
    ("up to 4 bedrooms and 3 bathrooms in Indio", {"city": "Indio", "bedrooms_max": 4, "bathrooms_max": 3}),
    ("3 bed 2 bath with a pool in Corona", {"city": "Corona", "bedrooms": 3, "bathrooms": 2, "amenities": ["pool"]}),
    ("under 4 bedrooms in Menifee", {"city": "Menifee", "bedrooms_max": 3}),
    # --- prices
    ("between $300,000 and $450,000 in Lancaster", {"city": "Lancaster", "price_min": 300000, "price_max": 450000}),
    ("between 1m and 2m in Pasadena", {"city": "Pasadena", "price_min": 1000000, "price_max": 2000000}),
    ("400-600k in Hemet", {"city": "Hemet", "price_min": 400000, "price_max": 600000}),
    ("houses in Hemet between 600 and 800k", {"city": "Hemet", "price_min": 600000, "price_max": 800000}),
    ("between 450000 and 600000 in Hemet", {"city": "Hemet", "price_min": 450000, "price_max": 600000}),
    ("between 2 and 3 bedrooms in Hemet", {"city": "Hemet", "bedrooms_min": 2, "bedrooms_max": 3}),
    ("$400k-$600k in Hemet", {"city": "Hemet", "price_min": 400000, "price_max": 600000}),
    ("$1.2M max in Irvine", {"city": "Irvine", "price_max": 1200000}),
    ("over $2 million in Los Angeles", {"city": "Los Angeles", "price_min": 2000000}),
    ("above 750k in Oakland", {"city": "Oakland", "price_min": 750000}),
    ("starting at $500k in Corona", {"city": "Corona", "price_min": 500000}),
    ("at least 500k in Corona", {"city": "Corona", "price_min": 500000}),
    ("around 600k in Riverside", {"city": "Riverside", "price_min": 540000, "price_max": 660000}),
    ("about 1.5 million in Irvine", {"city": "Irvine", "price_min": 1350000, "price_max": 1650000}),
    ("600k or less in Hemet", {"city": "Hemet", "price_max": 600000}),
    ("$500k+ in San Diego", {"city": "San Diego", "price_min": 500000}),
    ("$800k budget in Menifee", {"city": "Menifee", "price_max": 800000}),
    ("no more than $900,000 in Murrieta", {"city": "Murrieta", "price_max": 900000}),
    ("under 450000 in Indio", {"city": "Indio", "price_max": 450000}),
    ("homes in Corona 650k", {"city": "Corona", "price_max": 650000}),
    # --- square feet
    ("less than 1500 square feet in Indio", {"city": "Indio", "sqft_max": 1500}),
    ("at least 2500 sqft in Corona", {"city": "Corona", "sqft_min": 2500}),
    ("1800+ sq ft in Temecula", {"city": "Temecula", "sqft_min": 1800}),
    ("2,000 to 3,000 sqft in San Diego", {"city": "San Diego", "sqft_min": 2000, "sqft_max": 3000}),
    ("under 1000 sq ft condo in Long Beach", {"city": "Long Beach", "sqft_max": 1000, "amenities": ["condo"]}),
    ("2000 sqft or larger in Santee", {"city": "Santee", "sqft_min": 2000}),
    ("1200 sqft or less in Hemet", {"city": "Hemet", "sqft_max": 1200}),
    ("around 1800 sq ft in Indio", {"city": "Indio", "sqft_min": 1620, "sqft_max": 1980}),
    ("a 2400 sqft home in Corona", {"city": "Corona"}),
    ("4 bed 3 bath 2,500 sqft or more under 900k in Irvine",
     {"city": "Irvine", "bedrooms": 4, "bathrooms": 3, "sqft_min": 2500, "price_max": 900000}),
    # --- amenities and negation
    ("no pool in Corona", {"city": "Corona", "exclude_amenities": ["pool"]}),
    ("without a pool or spa in Indio", {"city": "Indio", "exclude_amenities": ["pool", "spa"]}),
    ("homes with solar and a pool in Riverside", {"city": "Riverside", "amenities": ["solar", "pool"]}),
    ("pool and spa in Palm Desert", {"city": "Palm Desert", "amenities": ["pool", "spa"]}),
    ("no HOA in Menifee", {"city": "Menifee", "amenities": ["no hoa"]}),
    ("w/o HOA in Hemet", {"city": "Hemet", "amenities": ["no hoa"]}),
    ("low HOA fees townhome in Irvine", {"city": "Irvine", "amenities": ["low hoa", "townhome"]}),
    ("not a fixer upper in Corona", {"city": "Corona", "exclude_amenities": ["fixer-upper"]}),
    ("gated communities in Indio", {"city": "Indio", "amenities": ["gated community"]}),
    ("pool but no HOA in Menifee", {"city": "Menifee", "amenities": ["pool", "no hoa"]}),
    ("3 car garage in Corona", {"city": "Corona", "amenities": ["3 car garage"]}),
    ("2 story home in Murrieta", {"city": "Murrieta", "amenities": ["2 story"]}),
    ("55+ community in Hemet", {"city": "Hemet", "amenities": ["55+ community"]}),
    ("move-in ready 3 bed in Santee", {"city": "Santee", "bedrooms": 3, "amenities": ["move-in ready"]}),
    # --- cities
    ("homes in Corona or Riverside under 500k", {"cities": ["Corona", "Riverside"], "price_max": 500000}),
    ("3 bed in irvine", {"city": "Irvine", "bedrooms": 3}),
    ("IRVINE 3 bed", {"city": "Irvine", "bedrooms": 3}),
    ("Orange County homes", {}),
    ("homes in Nice", {"city": "Nice"}),
    ("nice home with a pool", {"amenities": ["pool"]}),
    ("houses in Gotham under 450k", {"city": "Gotham", "price_max": 450000}),
    ("homes near Big Bear City", {"city": "Big Bear City"}),
    ("homes in Big Bear", {"city": "Big Bear"}),
    ("condos in Corona'; DROP TABLE rets_property;--", {"city": "Corona", "amenities": ["condo"]}),
    # --- things that must NOT become filters
    ("homes in 92612", {}),
    ("built in 1995 in Irvine", {"city": "Irvine"}),
    ("within 5m of the freeway in Corona", {"city": "Corona"}),
    ("walking distance to schools in Irvine", {"city": "Irvine"}),
    ("good morning", {}),
    ("affordable houses please", {}),
    ("", {}),
    ("   ", {}),
]


def _norm(f):
    out = dict(f)
    for k in ("amenities", "exclude_amenities"):
        if k in out:
            out[k] = frozenset(out[k])
    return out


@pytest.mark.parametrize("query,expected", CASES, ids=[c[0][:50] or "(empty)" for c in CASES])
def test_parse_case(parser, query, expected):
    assert _norm(parser.parse(query)) == _norm(expected)


def test_at_least_70_cases():
    assert len(CASES) >= 70


def test_none_and_non_text_input_do_not_crash(parser):
    assert parser.parse(None) == {}
    assert parser.parse(12345) == {}
    assert parser.parse("x" * 100000) == {}


def test_every_pattern_is_exercised(parser):
    used = set()
    for q, _ in CASES:
        used.update(parser.parse_detailed(q).patterns_used())
    never = [p for p in PATTERN_NAMES if p not in used]
    assert not never, f"patterns never exercised by CASES: {never}"


def test_at_least_20_distinct_patterns():
    assert len(PATTERN_NAMES) >= 20 and len(set(PATTERN_NAMES)) == len(PATTERN_NAMES)


def test_bare_sqft_and_bare_price_are_reported(parser):
    pq = parser.parse_detailed("a 2400 sqft home in Corona")
    assert any("square-foot" in n for n in pq.ignored)
    pq = parser.parse_detailed("homes in Corona 650k")
    assert any("maximum budget" in n for n in pq.ignored)


def test_conflicting_values_keep_first_and_report(parser):
    pq = parser.parse_detailed("3 bedroom or 4 bedroom? 2 bed in Corona")
    assert "bedrooms" in pq.filters or "bedrooms_min" in pq.filters
    assert pq.ignored


def test_parse_is_deterministic(parser):
    for q, _ in CASES[:20]:
        assert parser.parse(q) == parser.parse(q)


# ======================================================================================
# 2. Accuracy on the Week 1 sample queries (the tuning set) - needs schema.json
# ======================================================================================
@pytest.mark.skipif(not (SCHEMA_FILE.exists() and GOLD.exists()), reason="run scripts/build_schema.py first")
def test_week1_query_accuracy_at_least_90_percent():
    real = QueryParser(schema_path=SCHEMA_FILE)
    rows = json.load(open(GOLD))["week1"]
    assert len(rows) == 60
    ok = sum(_norm(real.parse(r["query"])) == _norm(r["expected"]) for r in rows)
    assert ok / len(rows) >= 0.90, f"{ok}/{len(rows)}"


@pytest.mark.skipif(not (SCHEMA_FILE.exists() and GOLD.exists()), reason="run scripts/build_schema.py first")
def test_week1_cities_are_in_the_schema():
    real = SchemaValidator(schema_path=SCHEMA_FILE)
    for r in json.load(open(GOLD))["week1"]:
        for c in ([r["expected"]["city"]] if "city" in r["expected"] else r["expected"].get("cities", [])):
            assert real.canonical_city(c), c


@pytest.mark.skipif(not GOLD.exists(), reason="query_gold.json missing")
def test_gold_file_integrity():
    g = json.load(open(GOLD))
    assert len(g["week1"]) == 60 and len(g["heldout"]) >= 40
    seen = [r["query"] for r in g["week1"]] + [r["query"] for r in g["heldout"]]
    assert len(seen) == len(set(seen)), "duplicate query between/within sets"
    inline = {q for q, _ in CASES}
    assert not inline & set(r["query"] for r in g["heldout"]), "held-out query also used in the dev cases"
    tax = {t["term"].lower() for t in json.load(open(TAXONOMY))["terms"]}
    for r in g["week1"] + g["heldout"]:
        assert set(r["expected"]) <= ALLOWED_FILTERS
        for k in ("amenities", "exclude_amenities"):
            assert set(r["expected"].get(k, [])) <= tax


# ======================================================================================
# 3. SchemaValidator
# ======================================================================================
class TestValidator:
    def test_valid_query_passes(self, validator):
        assert validator.validate_query({"city": "Irvine", "bedrooms": 3, "price_max": 700000}) == (True, [])

    def test_unknown_city_with_suggestion(self, validator):
        ok, errs = validator.validate_query({"city": "Irvin"})
        assert not ok and "Unknown city" in errs[0] and "'Irvine'" in errs[0]

    def test_unknown_city_without_suggestion(self, validator):
        ok, errs = validator.validate_query({"city": "Springfield"})
        assert not ok and "Unknown city" in errs[0] and "Did you mean" not in errs[0]

    def test_partial_city_gets_prefix_suggestion(self, validator):
        ok, errs = validator.validate_query({"city": "Big Bear"})
        assert not ok and "Big Bear City" in errs[0] and "Big Bear Lake" in errs[0]

    def test_city_match_is_case_and_space_insensitive(self, validator):
        assert validator.validate_query({"city": "  palm   desert "})[0]

    def test_every_city_in_a_list_is_checked(self, validator):
        ok, errs = validator.validate_query({"cities": ["Irvine", "Atlantis", "Gotham"]})
        assert not ok and len(errs) == 2

    @pytest.mark.parametrize("filters,fragment", [
        ({"city": "Irvine", "price_max": 50000}, "below the minimum allowed price"),
        ({"city": "Irvine", "price_min": 20_000_000}, "above the maximum allowed price"),
        ({"city": "Irvine", "bedrooms": 0}, "below the minimum allowed bedrooms"),
        ({"city": "Irvine", "bedrooms": 11}, "above the maximum allowed bedrooms"),
        ({"city": "Irvine", "bedrooms_min": 12}, "above the maximum allowed bedrooms"),
        ({"city": "Irvine", "bathrooms": 15}, "above the maximum allowed bathrooms"),
        ({"city": "Irvine", "sqft_min": 10}, "below the minimum allowed square feet"),
        ({"city": "Irvine", "sqft_max": 10_000_000}, "above the maximum allowed square feet"),
    ])
    def test_out_of_range_values(self, validator, filters, fragment):
        ok, errs = validator.validate_query(filters)
        assert not ok and any(fragment in e for e in errs), errs

    @pytest.mark.parametrize("filters", [
        {"price_max": 100000}, {"price_max": 10_000_000}, {"bedrooms": 1}, {"bedrooms": 10},
    ])
    def test_boundaries_are_inclusive(self, validator, filters):
        assert validator.validate_query(filters)[0]

    def test_min_greater_than_max(self, validator):
        ok, errs = validator.validate_query({"price_min": 800000, "price_max": 500000})
        assert not ok and "greater than" in errs[0]
        ok, errs = validator.validate_query({"bedrooms_min": 5, "bedrooms_max": 3})
        assert not ok and "greater than" in errs[0]

    def test_exact_bedrooms_conflicting_with_range(self, validator):
        assert not validator.validate_query({"bedrooms": 2, "bedrooms_min": 3})[0]

    @pytest.mark.parametrize("bad", ["3", True, False, None.__class__, float("nan"), float("inf"), [3], {"a": 1}])
    def test_wrong_types(self, validator, bad):
        ok, errs = validator.validate_query({"bedrooms": bad})
        assert not ok

    def test_unknown_filter_name(self, validator):
        ok, errs = validator.validate_query({"city": "Irvine", "L_City; DROP TABLE x": 1})
        assert not ok and "Unknown filter" in errs[0]

    def test_empty_and_nonsense(self, validator):
        assert not validator.validate_query({})[0]
        assert "No search criteria" in validator.validate_query({})[1][0]
        assert not validator.validate_query(None)[0]
        assert not validator.validate_query("city=Irvine")[0]

    def test_amenity_required_and_excluded(self, validator):
        ok, errs = validator.validate_query({"amenities": ["pool"], "exclude_amenities": ["pool"]})
        assert not ok and "both required and excluded" in errs[0]

    def test_amenity_lists_must_be_text(self, validator):
        assert not validator.validate_query({"amenities": [1, 2]})[0]
        assert not validator.validate_query({"amenities": "pool"})[0]

    def test_limits_come_from_schema(self):
        v = SchemaValidator(schema={"cities": ["Irvine"], "limits": {"price": [50000, 60_000_000]}})
        assert v.validate_query({"price_max": 55_000_000})[0]
        assert not v.validate_query({"price_max": 40000})[0]
        assert v.limits["bedrooms"] == DEFAULT_LIMITS["bedrooms"]

    def test_missing_schema_file_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_schema.py"):
            SchemaValidator(schema_path=tmp_path / "nope.json")

    def test_parser_output_goes_through_validator(self, parser, validator):
        ok, errs = validator.validate_query(parser.parse("3 bed under 50k in Hemet"))
        assert not ok and "below the minimum allowed price" in errs[0]
        ok, errs = validator.validate_query(parser.parse("homes in Springfield under 500k"))
        assert not ok and "Unknown city" in errs[0]
        assert validator.validate_query(parser.parse("3 bed under 700k in Irvine"))[0]


# ======================================================================================
# 4. SQL generation
# ======================================================================================
class TestSqlBuilder:
    def test_spec_example(self, parser):
        sql, params = parser.to_sql(parser.parse("3 bed under 700k in Irvine"))
        assert sql == ("SELECT L_ListingID, L_Address, L_City, L_SystemPrice, L_Keyword2, LM_Dec_3, LM_Int2_3 "
                       "FROM rets_property WHERE L_City = %s AND L_SystemPrice <= %s AND L_Keyword2 = %s "
                       "ORDER BY L_SystemPrice ASC LIMIT %s")
        assert params == ["Irvine", 700000, 3, 50]

    def test_all_placeholders_have_a_parameter(self):
        f = {"cities": ["Irvine", "Corona"], "price_min": 1, "price_max": 2, "bedrooms_min": 3,
             "bathrooms": 2, "sqft_min": 1000, "amenities": ["pool", "spa"], "exclude_amenities": ["hoa"]}
        sql, params = build_sql(f)
        assert sql.count("%s") == len(params)
        assert "IN (%s, %s)" in sql

    def test_no_values_in_sql_text(self):
        sql, params = build_sql({"city": "Irvine", "price_max": 700000, "amenities": ["pool"]})
        assert "Irvine" not in sql and "700000" not in sql and "pool" not in sql

    def test_uses_no_select_star_and_whitelisted_columns(self):
        sql, _ = build_sql({"city": "Irvine"})
        assert "*" not in sql
        used = set(re.findall(r"\b(?:L_|LM_)\w+", sql))
        assert used <= {"L_ListingID", "L_Address", "L_City", "L_SystemPrice", "L_Keyword2", "LM_Dec_3", "LM_Int2_3"}

    def test_exclusion_keeps_rows_with_no_remarks(self):
        sql, _ = build_sql({"exclude_amenities": ["pool"]})
        assert "L_Remarks IS NULL OR NOT REGEXP_LIKE" in sql

    def test_empty_filters_refuse_to_dump_the_table(self):
        with pytest.raises(ValueError, match="no filters"):
            build_sql({})
        with pytest.raises(ValueError, match="no filters"):
            build_sql({"city": None, "amenities": []})

    @pytest.mark.parametrize("limit", [0, -1, 501, "50", 5.5, True, None])
    def test_bad_limit(self, limit):
        with pytest.raises(ValueError):
            build_sql({"city": "Irvine"}, limit=limit)

    def test_amenity_regex_whole_words_and_escaping(self):
        rx = amenity_regex("spa")
        assert re.search(rx, "Has a spa.", re.I) and re.search(rx, "private spas", re.I)
        assert not re.search(rx, "spacious rooms", re.I)
        assert re.search(amenity_regex("move-in ready"), "Move in ready!", re.I)
        assert re.search(amenity_regex("55+ community"), "a 55+ community", re.I)
        assert not re.search(amenity_regex("55+ community"), "555 community", re.I)
        assert not re.search(amenity_regex(".*"), "anything at all", re.I)
        assert re.search(amenity_regex(".*"), "literal .* here", re.I)

    def test_amenity_limits(self):
        with pytest.raises(ValueError):
            build_sql({"amenities": ["a"] * 11})
        with pytest.raises(ValueError):
            amenity_regex("x" * 61)
        with pytest.raises(ValueError):
            amenity_regex("   ")

    def test_search_helper_builds_sql_only_when_valid(self, parser):
        ok = parser.search("3 bed under 700k in Irvine")
        assert ok["valid"] and ok["sql"] and ok["params"][0] == "Irvine"
        bad = parser.search("3 bed under 700k in Springfield")
        assert not bad["valid"] and bad["sql"] is None and bad["errors"]


# ======================================================================================
# 5. SQL injection protection - the generated SQL is executed on SQLite
# ======================================================================================
ROWS = [
    # id, address, city, price, beds, baths, sqft, remarks
    ("A1", "1 Test St", "Irvine", 650000, 3, 2, 1500, "Light and bright with a private pool and spa."),
    ("A2", "2 Test St", "Irvine", 720000, 3, 2, 1700, "Spacious home near the carpool lane."),
    ("A3", "3 Test St", "Irvine", 480000, 2, 2, 1100, "Cozy condos nearby; no HOA."),
    ("A4", "4 Test St", "Corona", 560000, 4, 3, 2100, None),
    ("A5", "5 Test St", "Corona", 910000, 5, 4, 3200, "Gated community, solar owned. Chef's kitchen."),
    ("A6", "6 Test St", "Hemet", 310000, 3, 1, 1000, "Fixer-upper with potential. Mello-Roos applies."),
]


@pytest.fixture()
def db():
    conn = sqlite_connection()
    conn.executemany("INSERT INTO rets_property VALUES (?,?,?,?,?,?,?,?)", ROWS)
    yield conn
    conn.close()


def ids(rows):
    return sorted(r[0] for r in rows)


def run(db, filters, **kw):
    sql, params = build_sql(filters, **kw)
    return execute_sqlite(db, sql, params)


PAYLOADS = [
    "'; DROP TABLE rets_property; --",
    "Irvine'; DROP TABLE rets_property; --",
    "' OR '1'='1",
    "' OR 1=1 --",
    '" OR ""="',
    "Robert'); DROP TABLE rets_property;--",
    "1; DELETE FROM rets_property",
    "' UNION SELECT L_Remarks, L_Remarks, L_Remarks, 1, 1, 1, 1 FROM rets_property --",
    "%", "_", "\\", "\\'", "\x00", "ünïcode'; --", "/*", "*/", "`", ")" * 50,
]


class TestInjection:
    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_sql_text_is_independent_of_the_value(self, payload):
        base_city = build_sql({"city": "Irvine"})[0]
        assert build_sql({"city": payload})[0] == base_city
        assert build_sql({"cities": [payload, "Corona"]})[0] == build_sql({"cities": ["Irvine", "Corona"]})[0]
        assert build_sql({"amenities": [payload]})[0] == build_sql({"amenities": ["pool"]})[0] \
            if len(payload) <= 60 else True

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_hostile_city_is_just_a_value(self, db, payload):
        assert run(db, {"city": payload}) == []
        assert db.execute("SELECT COUNT(*) FROM rets_property").fetchone()[0] == len(ROWS)

    @pytest.mark.parametrize("payload", [p for p in PAYLOADS if len(p) <= 60])
    def test_hostile_amenity_is_just_a_value(self, db, payload):
        assert run(db, {"amenities": [payload]}) == []
        # excluding something that appears nowhere removes nothing
        assert len(run(db, {"exclude_amenities": [payload], "city": "Irvine"})) == 3
        assert db.execute("SELECT COUNT(*) FROM rets_property").fetchone()[0] == len(ROWS)

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_hostile_free_text_end_to_end(self, db, parser, payload):
        for q in (payload, payload + " 3 bed in Irvine", "3 bed in Irvine " + payload):
            res = parser.search(q)
            assert res["sql"] is None or res["sql"].count("%s") == len(res["params"])
            if res["valid"]:
                execute_sqlite(db, res["sql"], res["params"])
            assert db.execute("SELECT COUNT(*) FROM rets_property").fetchone()[0] == len(ROWS)

    def test_injection_text_in_a_query_does_not_widen_the_result(self, db, parser):
        normal = parser.search("3 bed in Irvine")
        evil = parser.search("3 bed in Irvine' OR '1'='1")
        assert evil["valid"] and evil["sql"] == normal["sql"] and evil["params"] == normal["params"]
        assert ids(execute_sqlite(db, evil["sql"], evil["params"])) == ["A1", "A2"]

    @pytest.mark.parametrize("bad", ["1 OR 1=1", "700000; DROP TABLE rets_property", True, float("nan"),
                                     float("inf"), [1], None.__class__, b"1"])
    def test_numeric_filters_reject_non_numbers(self, bad):
        for key in ("price_max", "price_min", "bedrooms", "bedrooms_min", "bathrooms_max", "sqft_min"):
            with pytest.raises(ValueError):
                build_sql({key: bad})

    @pytest.mark.parametrize("key", ["L_City", "city; DROP TABLE x", "price_max OR 1=1", "", "bedrooms)--", "limit"])
    def test_unknown_filter_names_are_refused(self, key):
        with pytest.raises(ValueError, match="unknown filter"):
            build_sql({"city": "Irvine", key: 1})

    def test_requested_columns_are_whitelisted(self):
        with pytest.raises(ValueError, match="unknown column"):
            build_sql({"city": "Irvine"}, columns=["L_City", "password FROM users --"])
        with pytest.raises(ValueError):
            build_sql({"city": "Irvine"}, columns=[])

    def test_like_and_regex_wildcards_are_not_wildcards(self, db):
        for wild in ("%", "_", ".*", ".", "^", "$", "(a|b)", "[a-z]", "p.ol"):
            assert run(db, {"amenities": [wild]}) == [], wild

    def test_stacked_statements_cannot_run(self, db):
        with pytest.raises(sqlite3.Error):
            db.execute("SELECT 1; DROP TABLE rets_property")
        assert db.execute("SELECT COUNT(*) FROM rets_property").fetchone()[0] == len(ROWS)

    def test_fuzz_random_strings_never_change_the_sql(self):
        import random
        rnd = random.Random(4)
        alphabet = "abc'\";-- /*)(\\%_\x00 \n\tÅ"
        template = build_sql({"city": "x", "amenities": ["x"], "exclude_amenities": ["x"]})[0]
        for _ in range(300):
            s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 40)))
            try:
                sql, params = build_sql({"city": s, "amenities": [s], "exclude_amenities": [s]})
            except ValueError:
                continue          # e.g. blank after trimming: refused, not passed on
            assert sql == template and sql.count("%s") == len(params)


# ======================================================================================
# 6. End to end on a tiny table: query text -> rows
# ======================================================================================
class TestEndToEnd:
    @pytest.mark.parametrize("query,expected_ids", [
        ("3 bed under 700k in Irvine", ["A1"]),
        ("3 bed in Irvine", ["A1", "A2"]),
        ("2 bed condo in Irvine", ["A3"]),
        ("pool in Irvine", ["A1"]),                 # "carpool" must not match
        ("spa in Irvine", ["A1"]),                  # "spacious" must not match
        ("no HOA in Irvine", ["A3"]),
        ("without a pool in Irvine", ["A2", "A3"]),
        ("without a pool in Corona", ["A4", "A5"]),  # A4 has no remarks at all: still returned
        ("4+ bedrooms in Corona", ["A4", "A5"]),
        ("gated community in Corona", ["A5"]),
        ("at least 2000 sqft in Corona", ["A4", "A5"]),
        ("between 300k and 500k", ["A3", "A6"]),
        ("no mello-roos under 400k", []),
        ("fixer-upper in Hemet", ["A6"]),
    ])
    def test_query_to_rows(self, db, parser, query, expected_ids):
        res = parser.search(query)
        assert res["valid"], res["errors"]
        assert ids(execute_sqlite(db, res["sql"], res["params"])) == expected_ids

    def test_invalid_query_never_reaches_the_database(self, parser):
        res = parser.search("3 bed under 50k in Hemet")
        assert not res["valid"] and res["sql"] is None


def test_to_qmark_only_touches_placeholders():
    assert to_qmark("a = %s AND b = %s") == "a = ? AND b = ?"
