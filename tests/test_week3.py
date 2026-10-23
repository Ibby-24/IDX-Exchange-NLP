"""
Week 3 tests: EntityExtractor rules, gold-label integrity, dev-split quality gate.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from entity_extractor import EntityExtractor 
from text_cleaning import TextCleaner

SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
GOLD = ROOT / "data" / "processed" / "gold_labels.json"
TAXONOMY = ROOT / "data" / "processed" / "taxonomy.json"


@pytest.fixture(scope="module")
def ex():
    return EntityExtractor()


#Bedrooms
@pytest.mark.parametrize("text,expected", [
    ("Charming 3-bedroom, 2-bath home.", 3),
    ("This 4 bedroom 3 bathroom house sits on a corner lot.", 4),
    ("Offering five bedrooms and four bathrooms.", 5),
    ("A spacious 4 generously sized bedrooms layout.", 4),                 
    ("Beautiful 2 bed, 1 bath condo.", 2),
    ("Bedrooms: 3", 3),
    ("Welcome to this *3* bedroom condo.", 3),                          
    ("The home features 3 bedrooms plus an office.", 3),
    ("Lovely home. Upstairs, two additional bedrooms share a bath.", None), 
    ("Plus three additional bedrooms for guests.", None),                   
    ("The main level has 3 bedrooms and the loft adds more space.", None),  
    ("A separate 1-bedroom ADU sits behind the main home.", None),          
    ("Main house: 4 bedrooms, 3 baths. Studio ADU with one bedroom.", 4),
    ("Prices for a 4-bedroom home in the area keep rising.", None),         
    ("Tax records show only 2 bedrooms, but it lives larger.", None),       
    ("Choose from 2- and 3-bedroom floor plans.", None),                    
    ("This 3-4 bedroom home is great.", None),                              
    ("One of the bedrooms is a bonus room.", None),                         
    ("Located at 1234 Oak Street with 3 bedrooms.", 3),                     
    ("It could be a 4 bedroom with a small addition.", None),       
    ("No bedroom count is given here.", None),
])
def test_bedrooms(ex, text, expected):
    assert ex.extract_bedrooms(text) == expected


def test_bedrooms_all_n_form(ex):
    assert ex.extract_bedrooms("All four bedrooms are en-suite.") == 4


def test_bedrooms_majority_vote_beats_first_mention(ex):
    text = "A 3-bedroom home. Truly a 4 bedroom layout. Four bedrooms for everyone."
    assert ex.extract_bedrooms(text) == 4


#bathrooms
@pytest.mark.parametrize("text,expected", [
    ("3-bedroom, 2-bath home", 2),
    ("with 2.5 bathrooms", 2.5),
    ("a 1.75-bathroom plan", 1.75),
    ("offers 3 1/2 bathrooms", 3.5),
    ("featuring three-and-a-half bathrooms", 3.5),
    ("a two and a half-bath plan", 2.5),
    ("2 full baths", 2),
    ("2 full baths plus a half bath", 2.5),
    ("2 full and 1 half bath", 2.5),
    ("3 full + powder room", 3.5),
    ("Quaint 2-bedroom, single-bath home", 1),
    ("two bedrooms and one bath", 1),          
    ("All three of the bathrooms were remodeled.", 3),
    ("Upstairs, two additional bathrooms are shared.", None),
    ("A 1/2 bath on the main level.", None),    
    ("The back unit has a 3/4 bathroom.", None),
    ("Beautiful bathroom with soaking tub.", None),
])
def test_bathrooms(ex, text, expected):
    assert ex.extract_bathrooms(text) == expected


#Square footage
@pytest.mark.parametrize("text,expected", [
    ("This home offers 1853 square feet of living space.", 1853),
    ("Approximately 2,400 square feet.", 2400),
    ("A 2400-square-foot home", 2400),
    ("Offering 1400+ square feet", 1400),
    ("Square footage of 1040 square feet", 1040),
    ("Set on a 6662 square feet lot. The home offers 1853 square feet.", 1853),
    ("The 1200 square feet home sits on a 7000 square feet lot.", 1200),
    ("Set on a generous lot, spanning approximately 5242 square feet, the home offers 2696 square feet of living space.", 2696),
    ("An enclosed patio approximately 400 square feet on top of the 1600 square feet home.", 1600),
    ("Villas range from approximately 1400 square feet to 2000 square feet.", None),
    ("The ADU adds 450 square feet.", None),
    ("A detached 960 square feet garage.", None),
    ("Price is $581 per square foot.", None),
    ("Offering 2389 square feet and 2.5 baths.", 2389),
    ("A 120 square feet closet.", None),
])
def test_sqft(ex, text, expected):
    assert ex.extract_sqft(text) == expected

#price
@pytest.mark.parametrize("text,expected", [
    ("Offered at $695000, this home is tenant occupied.", 695000),
    ("PRICE IMPROVEMENT TO $472000! Beautiful condo.", 472000),
    ("Recently appraised at $1300000 and currently listed at $1225000.", 1225000),
    ("LIST PRICE: $439500 Enter from Main St.", 439500),
    ("Listed at $1.2 million for a quick sale.", 1200000),
    ("HOA dues are $784/month.", None),
    ("Currently rented for $4500/month.", None),
    ("Seller offering a $10000 flooring credit.", None),
    ("Over $70000 in upgrades!", None),
    ("Now offered under $900000!", None),
    ("Priced from $500000 for the smallest plan.", None),
    ("Offered at approximately $695000.", 695000),
    ("Offers between $950000 and $980000 considered.", None),
    ("$150000 PRICE REDUCTION!", None),
    ("Reduced $2100000 from the previous list price.", None),
    ("HUD case number 198-194713. Zip 92220. APN 123456789.", None),
    ("Offered at $581 per square foot.", None),
    ("1666 Roosevelt Ave and 92501 area.", None),
])
def test_price(ex, text, expected):
    assert ex.extract_price(text) == expected

# extract_all / cleaning
def test_extract_all_shape_and_cleaning(ex):
    raw = "Stunning 3BR/2BA home w/ 1,850 sq. ft. and a pool.  Offered at $1.2M."
    out = ex.extract_all(raw)
    assert set(out) == {"bedrooms", "bathrooms", "price", "sqft", "amenities"}
    assert out["bedrooms"] == 3
    assert out["bathrooms"] == 2
    assert out["sqft"] == 1850
    assert out["price"] == 1200000
    assert "pool" in out["amenities"]


def test_extract_all_clean_flag_is_equivalent_on_cleaned_text(ex):
    cleaned = TextCleaner().clean_text("Lovely 4BR/3BA home with 2,000 sqft and a pool")
    assert ex.extract_all(cleaned, clean=False) == ex.extract_all(cleaned, clean=True)


def test_extract_all_empty_text(ex):
    out = ex.extract_all("")
    assert out["bedrooms"] is None and out["bathrooms"] is None
    assert out["sqft"] is None and out["price"] is None and out["amenities"] == []


def test_mentions_carry_spans_and_roles(ex):
    text = "A 3-bedroom home. Upstairs, two additional bedrooms."
    ments = ex.find_bedroom_mentions(text)
    assert [m.role for m in ments] == ["main", "partial"]
    first = ments[0]
    assert text[first.start:first.end] == first.text


#amenities
@pytest.fixture(scope="module")
def amen(ex):
    if not TAXONOMY.exists():
        pytest.skip("taxonomy.json missing - run: python scripts/taxonomy_builder.py")
    return lambda text: {a["term"] for a in ex.extract_amenities(text)}


def test_amenity_basic_hits(amen):
    found = amen("Updated kitchen with granite countertops, a fireplace and a covered patio.")
    assert {"granite countertops", "fireplace", "covered patio"} <= found


def test_amenity_word_boundaries(amen):
    found = amen("A spacious residence with cabinetry and resident parking.")
    assert "spa" not in found and "cabin" not in found and "den" not in found


def test_amenity_longest_match_wins(amen):
    found = amen("Enjoy the swimming pool all summer.")
    assert "swimming pool" in found and "pool" not in found


def test_amenity_exact_term_beats_plural_of_shorter(amen):
    assert "walk-in closets" in amen("Two bedrooms with walk-in closets.")


def test_amenity_hyphen_and_case_insensitive(amen):
    assert "walk-in closet" in amen("Huge WALK IN CLOSET in the primary suite.")


@pytest.mark.parametrize("text,term", [
    ("There is no pool, but the yard is large.", "pool"),
    ("Perfect for a home office or hobby room.", "home office"),
    ("Lot is ready for putting in a pool.", "pool"),
    ("Potential to add an ADU in the backyard.", "adu"),
    ("Dream of building a garden? Start here.", "garden"),
    ("The pool table stays with the house.", "pool"),
    ("Decomposed granite paths wind through the yard.", "granite"),
])
def test_amenity_negated_or_hypothetical_not_reported(amen, text, term):
    assert term not in amen(text)


def test_amenity_use_cue_does_not_void_views(amen):
    assert "lake views" in amen("The deck is ideal for lounging with lake views.")


def test_amenity_clause_cue_stops_at_comma(amen):
    assert "brand new" in amen("Although it could use fresh paint, the house includes a brand new roof.")


def test_community_amenity_not_counted_as_private(amen):
    found = amen("Community amenities include a pool, spa and fitness center.")
    assert "pool" not in found and "spa" not in found
    assert "fitness center" in found


def test_private_pool_kept_inside_a_community(amen):
    assert "private pool" in amen("Located in a gated community, with a private pool.")

#gold labels
@pytest.fixture(scope="module")
def gold():
    assert GOLD.exists(), "Missing data/processed/gold_labels.json"
    with open(GOLD) as f:
        return json.load(f)


def test_gold_size_and_splits(gold):
    listings = gold["listings"]
    random_ones = [g for g in listings.values() if g["source"] == "random"]
    assert 200 <= len(random_ones) <= 300
    assert {g["split"] for g in listings.values()} == {"dev", "test"}
    n_test = sum(g["split"] == "test" for g in random_ones)
    assert 0.3 <= n_test / len(random_ones) <= 0.5


def test_gold_contains_no_remark_text(gold):
    """Only ids, numbers, offsets and short categorical strings may appear in the listing entries."""
    for lid, g in gold["listings"].items():
        assert lid.isdigit()
        for key, val in g.items():
            if isinstance(val, str):
                assert len(val) <= 40, f"{lid}.{key} looks like free text"
        for key in ("bedrooms", "bathrooms", "sqft", "price"):
            v = g.get(key)
            if isinstance(v, dict) and "spans" in v:
                assert all(isinstance(x, int) for span in v["spans"] for x in span)


def test_gold_entries_are_well_formed(gold):
    taxonomy_ids = {t["id"] for t in json.load(open(TAXONOMY))["terms"]} if TAXONOMY.exists() else None
    for lid, g in gold["listings"].items():
        for key in ("bedrooms", "bathrooms", "sqft", "price"):
            if key not in g:
                assert g["scope"] == "price_only" and key != "price"
                continue
            v = g[key]
            assert v is None or v.get("ambiguous") is True or (isinstance(v.get("value"), (int, float)) and v["spans"])
        if "amenities" in g and taxonomy_ids is not None:
            assert set(g["amenities"]) <= taxonomy_ids


def test_gold_spans_match_cleaned_text(gold):
    if not SAMPLE.exists():
        pytest.skip("listing_sample.csv not found - run: python scripts/data_loading.py")
    df = pd.read_csv(SAMPLE)
    df["L_ListingID"] = df["L_ListingID"].astype(str)
    df = df.set_index("L_ListingID")
    cleaner = TextCleaner()
    checked = 0
    for lid, g in gold["listings"].items():
        text = cleaner.clean_text(df.loc[lid, "remarks"])
        assert hashlib.sha1(text.encode()).hexdigest() == g["text_sha1"], f"{lid}: cleaned text changed"
        for key in ("bedrooms", "bathrooms", "sqft", "price"):
            v = g.get(key)
            if isinstance(v, dict) and "spans" in v:
                for s, e in v["spans"]:
                    assert 0 <= s < e <= len(text)
                    checked += 1
    assert checked > 300


def test_dev_and_test_splits_do_not_share_listings(gold):
    ids = list(gold["listings"])
    assert len(ids) == len(set(ids))

#quality gate on the DEV split
@pytest.fixture(scope="module")
def dev_scores():
    if not SAMPLE.exists():
        pytest.skip("listing_sample.csv not found - run: python scripts/data_loading.py")
    from evaluate_extractor import evaluate, load_data, prf
    df, gold_ = load_data()
    extractor = EntityExtractor()
    extractor._load_taxonomy()
    counts, _, amen_counts, _, _, _, skipped = evaluate("dev", df, gold_, extractor, TextCleaner())
    assert skipped == 0
    return counts, amen_counts, prf


@pytest.mark.parametrize("entity", ["bedrooms", "bathrooms", "sqft"])
def test_dev_f1_meets_target_for_core_entities(dev_scores, entity):
    counts, _, prf = dev_scores
    c = counts[entity]
    assert prf(c["tp"], c["fp"], c["fn"])[2] >= 0.85


def test_dev_numeric_micro_f1_meets_target(dev_scores):
    counts, _, prf = dev_scores
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values())
    fn = sum(c["fn"] for c in counts.values())
    assert prf(tp, fp, fn)[2] >= 0.85


def test_dev_amenity_f1_meets_target(dev_scores):
    _, a, prf = dev_scores
    assert prf(a["tp"], a["fp"], a["fn"])[2] >= 0.85


def test_extractor_beats_naive_baseline_on_price_precision(dev_scores):
    """The starter's \\$?(\\d{5,}) regex is wrong for this data (zips, APNs, HUD numbers)."""
    from baseline_extractor import NaiveBaseline
    from evaluate_extractor import evaluate, load_data, prf
    counts, _, _ = dev_scores
    df, gold_ = load_data()
    base_counts, *_ = evaluate("dev", df, gold_, NaiveBaseline(), TextCleaner())
    new_p = prf(counts["price"]["tp"], counts["price"]["fp"], counts["price"]["fn"])[0]
    old_p = prf(base_counts["price"]["tp"], base_counts["price"]["fp"], base_counts["price"]["fn"])[0]
    assert new_p > old_p + 0.5


def test_every_sample_remark_runs_without_error_and_in_range(ex):
    if not SAMPLE.exists():
        pytest.skip("listing_sample.csv not found - run: python scripts/data_loading.py")
    df = pd.read_csv(SAMPLE)
    for text in df["remarks"].head(300):
        out = ex.extract_all(text)
        assert out["bedrooms"] is None or 0 < out["bedrooms"] <= 20
        assert out["bathrooms"] is None or 0 < out["bathrooms"] <= 20
        assert out["sqft"] is None or 250 <= out["sqft"] <= 30000
        assert out["price"] is None or 20_000 <= out["price"] <= 100_000_000
        assert isinstance(out["amenities"], list)
