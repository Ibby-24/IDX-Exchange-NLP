"""Week 6 tests: SignalExtractor, full-table processing helpers, signal gold file, evaluator helpers.

Run from the repo root:
    pytest tests/test_week6.py -v

* No database is needed. The keyset-pagination test uses an in-memory sqlite table that has the same column names.
* Tests that need data/processed/listing_sample.csv skip, instead of failing, when it is missing.
"""
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_signals as es                                   # noqa: E402
import process_listings as pl                                   # noqa: E402
from signal_extractor import (ALL_SIGNALS, CONDITION_SIGNALS, FAMILIES, FINANCING_SIGNALS,   # noqa: E402
                              LOCATION_SIGNALS, SignalExtractor, hoa_field, mello_roos_field)

GOLD = ROOT / "data" / "processed" / "signal_gold.json"
GUIDE = ROOT / "docs" / "signal_guidelines.md"
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"


@pytest.fixture(scope="module")
def ex():
    return SignalExtractor()


def sig(ex, text):
    rec = ex.extract_signals({"L_ListingID": "1", "L_Remarks": text})
    return set(rec["condition_keywords"]) | set(rec["financing_terms"]) | set(rec["location_features"]), rec


# ---------------------------------------------------------------- output schema
def test_record_schema_and_types(ex):
    s, rec = sig(ex, "Turnkey 3 bedroom, 2 bath home with 1,850 square feet, pool and fireplace. FHA ok. Near schools.")
    assert set(rec) == {"listing_id", "entities", "amenities", "condition_keywords", "financing_terms",
                        "location_features", "fields", "keywords", "schema_version"}
    assert rec["listing_id"] == "1"
    assert rec["entities"] == {"bedrooms": 3, "bathrooms": 2, "sqft": 1850, "price": None}
    assert "move_in_ready" in rec["condition_keywords"] and "fha" in rec["financing_terms"]
    assert "near_schools" in rec["location_features"]
    assert "cond:move_in_ready" in rec["keywords"] and "fin:fha" in rec["keywords"] and "loc:near_schools" in rec["keywords"]
    json.dumps(rec)           # serialisable


def test_output_never_contains_remark_text(ex):
    text = "Quiet cul-de-sac. Fully remodeled kitchen with granite countertops. Seller financing available."
    rec = ex.extract_signals({"L_ListingID": "9", "L_Remarks": text})
    blob = json.dumps(rec).lower()
    for chunk in ("fully remodeled kitchen", "quiet cul-de-sac", "seller financing available"):
        assert chunk not in blob
    assert "evidence" not in rec
    assert "evidence" in ex.extract_signals({"L_ListingID": "9", "L_Remarks": text}, evidence=True)


def test_empty_and_missing_remarks(ex):
    for rec in ({"L_ListingID": "5"}, {"L_ListingID": "5", "L_Remarks": None}, {"L_ListingID": "5", "L_Remarks": "  "},
                {"L_ListingID": "5", "L_Remarks": float("nan")}):
        out = ex.extract_signals(rec)
        assert out["listing_id"] == "5" and out["keywords"] == [] and out["amenities"] == []
        assert out["entities"] == {"bedrooms": None, "bathrooms": None, "sqft": None, "price": None}


def test_accepts_remarks_key_and_numeric_id(ex):
    out = ex.extract_signals({"L_ListingID": 123, "remarks": "Gated community."})
    assert out["listing_id"] == "123" and "gated_community" in out["location_features"]


def test_signal_inventory_matches_guidelines():
    text = GUIDE.read_text()
    for s in ALL_SIGNALS:
        assert f"`{s}`" in text, f"{s} missing from docs/signal_guidelines.md"
    assert len(CONDITION_SIGNALS) == 6 and len(FINANCING_SIGNALS) == 13 and len(LOCATION_SIGNALS) == 14
    assert len(set(ALL_SIGNALS)) == 33


# ---------------------------------------------------------------- positives, one per signal
POSITIVE = [
    ("Fully remodeled kitchen and baths.", "renovated_updated"),
    ("Brand new construction in a new community.", "new_construction"),
    ("New roof and new HVAC installed.", "new_components"),
    ("Freshly painted throughout.", "new_components"),
    ("Turnkey and ready to go.", "move_in_ready"),
    ("Meticulously maintained by the original owner.", "well_maintained"),
    ("This fixer needs TLC and is sold as-is.", "needs_work"),
    ("Seller will carry a note; owner financing available.", "seller_financing"),
    ("Assumable loan at 2.5%.", "assumable_loan"),
    ("FHA buyers welcome.", "fha"),
    ("VA loan approved.", "va"),
    ("Cash only please.", "cash_only"),
    ("Perfect for a 1031 exchange.", "exchange_1031"),
    ("This is a short sale subject to lender approval.", "short_sale"),
    ("Bank owned REO property.", "reo"),
    ("Probate sale, no court confirmation.", "probate_estate"),
    ("Sold with tenants in place.", "tenant_occupied"),
    ("Great investment property with rental income.", "investment_income"),
    ("Seller offering closing cost credit and a rate buy-down.", "seller_credit"),
    ("Motivated seller, priced to sell.", "motivated_seller"),
    ("Located in a guard-gated community.", "gated_community"),
    ("Backs to the golf course.", "golf_course"),
    ("Close to top-rated schools.", "near_schools"),
    ("Minutes from shopping and dining.", "near_shopping_dining"),
    ("Easy freeway access for commuters.", "freeway_commute"),
    ("Steps from the Metrolink station.", "public_transit"),
    ("Walking distance to the village.", "walkable"),
    ("Minutes from the beach.", "near_beach_water"),
    ("Stunning oceanfront estate.", "waterfront"),
    ("Panoramic mountain and city views.", "views"),
    ("Tucked at the end of a quiet cul-de-sac.", "quiet_street"),
    ("Backs to open space with hiking trails.", "parks_open_space"),
    ("Active adult 55+ community.", "senior_55plus"),
    ("Close to downtown Redlands.", "near_downtown"),
]


@pytest.mark.parametrize("text,expected", POSITIVE)
def test_each_signal_fires(ex, text, expected):
    s, _ = sig(ex, text)
    assert expected in s, (text, s)


def test_every_signal_has_a_positive_test():
    assert {e for _, e in POSITIVE} == set(ALL_SIGNALS)


# ---------------------------------------------------------------- traps (should NOT fire)
NEGATIVE = [
    ("No HOA and no gated entry.", "gated_community"),
    ("Could be remodeled to your taste.", "renovated_updated"),
    ("Ready for your updates.", "renovated_updated"),
    ("Potential to add a new roof.", "new_components"),
    ("Not a short sale.", "short_sale"),
    ("Normal sale!! (Not probate).", "probate_estate"),               # upper-case "Not" must also negate
    ("Previously tenant occupied, now vacant.", "tenant_occupied"),
    ("Carrara marble subway tile backsplash.", "public_transit"),
    ("Parking in a gated underground garage.", "gated_community"),
    ("Surrounded by the greater Bay Area.", "near_beach_water"),
    ("Includes a miniature golf course for residents' kids.", "golf_course"),
    ("Located within the Glendora Unified School District.", "near_schools"),
    ("A walk-in closet and walking path.", "walkable"),
    ("Open space floor plan with high ceilings.", "parks_open_space"),
    ("Perfect for outdoor dining and entertaining.", "near_shopping_dining"),
    ("Beautifully maintained grounds and common areas.", "well_maintained"),
    ("Upgraded tile flooring.", "renovated_updated"),
    ("A 3 car garage and a golf cart garage.", "golf_course"),
    ("No cash buyers needed, financing welcome.", "cash_only"),
]


@pytest.mark.parametrize("text,absent", NEGATIVE)
def test_traps_do_not_fire(ex, text, absent):
    s, _ = sig(ex, text)
    assert absent not in s, (text, s)


def test_near_downtown_needs_proximity(ex):
    assert "near_downtown" not in sig(ex, "She worked in downtown Los Angeles for years.")[0]
    assert "near_downtown" in sig(ex, "One block from historic downtown.")[0]


def test_waterfront_vs_near_water(ex):
    s, _ = sig(ex, "Walking distance to the waterfront and local shops.")
    assert "waterfront" not in s and "near_beach_water" in s
    s, _ = sig(ex, "Waterfront home with a private dock.")
    assert "waterfront" in s


def test_signals_are_yes_no_not_counts(ex):
    _, rec = sig(ex, "Remodeled kitchen. Remodeled baths. Fully remodeled throughout.")
    assert rec["condition_keywords"].count("renovated_updated") == 1


# ---------------------------------------------------------------- HOA / Mello-Roos fields
@pytest.mark.parametrize("text,expected", [
    ("No HOA!", "none"), ("This home has no association fees.", "none"), ("NO HOA fees", "none"),
    ("Low HOA of $120", "low"), ("The HOA is very low.", "low"), ("Very low association dues.", "low"),
    ("HOA dues are $350 monthly.", "present"), ("Homeowners association applies.", "present"),
    ("Nice home.", None),
])
def test_hoa_field(text, expected):
    assert hoa_field(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("No Mello-Roos.", "none"), ("Low Mello Roos tax.", "present"), ("Mello-Roos applies", "present"), ("Nothing.", None),
])
def test_mello_roos_field(text, expected):
    assert mello_roos_field(text) == expected


def test_fields_in_record(ex):
    _, rec = sig(ex, "No HOA and no Mello-Roos.")
    assert rec["fields"] == {"hoa": "none", "mello_roos": "none"}
    assert "hoa:none" in rec["keywords"] and "mello_roos:none" in rec["keywords"]


# ---------------------------------------------------------------- relation to Week 3
def test_entities_equal_week3_extractor(ex):
    text = "Charming 4 bedroom, 3 bath home, 2,200 sq ft, offered at $899,000."
    week3 = ex.entity_extractor.extract_all(text, with_amenities=False)
    out = ex.extract_signals({"L_ListingID": "x", "L_Remarks": text})["entities"]
    assert out == {k: week3[k] for k in ("bedrooms", "bathrooms", "sqft", "price")}


def test_amenities_exclude_signal_categories(ex):
    _, rec = sig(ex, "Pool, fireplace and a gated community with a remodeled kitchen.")
    assert any("pool" in a for a in rec["amenities"]) and "fireplace" in rec["amenities"]
    assert not any("gated" in a or "remodel" in a for a in rec["amenities"])


# ---------------------------------------------------------------- processing helpers
def _db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rets_property (L_ListingID TEXT, L_Remarks TEXT, L_Keyword2 INT, LM_Dec_3 REAL, "
                 "LM_Int2_3 INT, L_SystemPrice INT)")
    conn.executemany("INSERT INTO rets_property VALUES (?,?,?,?,?,?)", rows)
    return conn


ROWS = [(f"{i:04d}", f"Gated community, {i % 5 + 1} bedroom home.", i % 5 + 1, 2.0, 1500, 500000) for i in range(1, 8)]


def test_keyset_pagination_visits_every_row_once_in_order():
    conn = _db(ROWS)
    got = [r["L_ListingID"] for r in pl.iter_db_rows(conn, False, batch=3, qmark=True)]
    assert got == [r[0] for r in ROWS]


def test_pagination_limit_and_start_after():
    conn = _db(ROWS)
    assert len(list(pl.iter_db_rows(conn, False, batch=3, limit=5, qmark=True))) == 5
    got = [r["L_ListingID"] for r in pl.iter_db_rows(conn, False, batch=3, start_after="0004", qmark=True)]
    assert got == ["0005", "0006", "0007"]


def test_sql_binds_parameters():
    sql = pl.build_batch_sql(True, 10)
    assert sql.count("%s") == 2 and "ORDER BY L_ListingID" in sql
    assert "L_Keyword2" in sql and "L_Keyword2" not in pl.build_batch_sql(False, 10)


def test_run_writes_jsonl_and_summary(tmp_path, ex):
    conn = _db(ROWS)
    out = tmp_path / "o.jsonl"
    summ = pl.run(pl.iter_db_rows(conn, True, batch=4, qmark=True), out, ex, compare=True, progress_every=0)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 7 and all("gated_community" in r["location_features"] for r in lines)
    d = summ.as_dict(1.0)
    assert d["listings"] == 7 and d["signal_listing_counts"]["gated_community"] == 7
    assert d["agreement_with_db_columns"]["bedrooms"]["agreement"] == 1.0
    assert pl.last_id(out) == "0007"


def test_last_id_of_missing_or_empty_file(tmp_path):
    assert pl.last_id(tmp_path / "none.jsonl") == ""
    (tmp_path / "e.jsonl").write_text("")
    assert pl.last_id(tmp_path / "e.jsonl") == ""


# ---------------------------------------------------------------- gold file and evaluator
def test_gold_file_is_clean_and_consistent():
    g = json.load(open(GOLD))["listings"]
    assert len(g) == 120
    assert {v["split"] for v in g.values()} == {"dev", "test"}
    for lid, v in g.items():
        assert set(v["signals"]) <= set(ALL_SIGNALS), lid
        assert v["hoa"] in (None, "none", "low", "present") and v["mello_roos"] in (None, "none", "present")
        assert re.fullmatch(r"[0-9a-f]{40}", v["text_sha1"])
        assert set(v) == {"split", "signals", "hoa", "mello_roos", "text_sha1"}      # ids and labels only: no text
    assert len(set(g)) == 120


def test_gold_overlaps_nothing_it_should_not_leak_text():
    raw = GOLD.read_text()
    assert len(raw) < 200_000 and "bedroom" not in raw.lower().replace("bedrooms", "")


def test_prf():
    assert es.prf(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert es.prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_score_signals_on_tiny_gold(ex):
    texts = {"a": "Gated community near top-rated schools.", "b": "Cozy home."}
    gold = {"a": {"split": "dev", "signals": ["gated_community", "near_schools", "views"], "hoa": None, "mello_roos": None,
                  "text_sha1": None},
            "b": {"split": "dev", "signals": [], "hoa": None, "mello_roos": None, "text_sha1": None}}
    per_sig, flds, per_listing, errors, used, skipped = es.score_signals("dev", gold, texts, ex)
    assert used == 2 and skipped == 0
    assert per_sig["gated_community"]["tp"] == 1 and per_sig["near_schools"]["tp"] == 1
    assert per_sig["views"]["fn"] == 1
    summ = es.summarize_signals(per_sig)
    assert summ["overall"]["tp"] == 2 and summ["overall"]["fn"] == 1 and summ["overall"]["fp"] == 0
    assert ("a", "views", "missed") in errors


def test_changed_text_is_skipped_not_scored(ex):
    gold = {"a": {"split": "dev", "signals": ["gated_community"], "hoa": None, "mello_roos": None, "text_sha1": "0" * 40}}
    _, _, _, _, used, skipped = es.score_signals("dev", gold, {"a": "Gated community."}, ex)
    assert used == 0 and skipped == 1


def test_coverage_guard_stops():
    with pytest.raises(SystemExit):
        es.check_coverage(50, 100, "x", allow_low=False)
    es.check_coverage(95, 100, "x", allow_low=False)
    es.check_coverage(50, 100, "x", allow_low=True)


def test_load_texts_reads_csvs_without_db(tmp_path):
    p = tmp_path / "s.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L_ListingID", "remarks"])
        w.writerow(["1", "hello"])
    got = es.load_texts(["1", "2"], use_db=False, sources=[p])
    assert got == {"1": "hello"}


def test_test_split_requires_final_flag():
    with pytest.raises(SystemExit):
        es.main(["--split", "test"])


# ---------------------------------------------------------------- end to end on the sample, if present
@pytest.mark.skipif(not SAMPLE.exists(), reason="listing_sample.csv not present")
def test_sample_runs_and_signals_are_not_degenerate(ex):
    rows = list(csv.DictReader(open(SAMPLE)))[:150]
    recs = [ex.extract_signals({"L_ListingID": r["L_ListingID"], "L_Remarks": r["remarks"]}) for r in rows]
    assert len(recs) == 150
    assert sum(bool(r["location_features"]) for r in recs) > 60       # most listings mention location
    assert sum(bool(r["condition_keywords"]) for r in recs) > 30
    for r in recs:
        assert all(k.split(":")[0] in {"cond", "fin", "loc", "hoa", "mello_roos"} for k in r["keywords"])
