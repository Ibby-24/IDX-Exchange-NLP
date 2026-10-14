"""Week 1 validation tests.

Run from the repo root:
    pytest tests/test_week1.py -v

Prerequisites (in this order):
    python scripts/data_loading.py        # creates data/processed/listing_sample.csv
    python scripts/taxonomy_builder.py    # creates data/processed/taxonomy.json
    data/processed/sample_queries.json    # hand-labeled queries (committed to git)

listing_sample.csv is MLS data and is git-ignored, so the tests that need it
skip (instead of failing) on a fresh clone until you run data_loading.py.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "data" / "processed" / "taxonomy.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "listing_sample.csv"
QUERIES_PATH = ROOT / "data" / "processed" / "sample_queries.json"

EXPECTED_CATEGORIES = {
    "property_type",
    "interior_features",
    "layout_rooms",
    "outdoor_lot",
    "systems_parking_energy",
    "condition_updates",
    "location_community",
    "financing_ownership",
}
VALID_INTENTS = {"browsing", "researching", "high_intent_inquiry"}

MIN_TERMS_TOTAL = 200
MIN_TERMS_PER_CATEGORY = 15
MIN_SAMPLE_ROWS = 500
MIN_COVERAGE = 0.30           # >=30% of remarks must contain at least one term
MIN_CATEGORY_PREVALENCE = 0.05
MIN_QUERIES = 50
MIN_QUERIES_PER_INTENT = 10


def normalize(text):
    """Lowercase and treat hyphens as spaces so 'walk in closet' == 'walk-in closet'."""
    return re.sub(r"\s+", " ", str(text).lower().replace("-", " ")).strip()


def term_regex(term):
    # \b = word boundary, so 'spa' does NOT match 'spacious'
    return re.compile(r"\b" + re.escape(normalize(term)) + r"\b")


@pytest.fixture(scope="module")
def taxonomy():
    assert TAXONOMY_PATH.exists(), "Run: python scripts/taxonomy_builder.py"
    with open(TAXONOMY_PATH) as f:
        return json.load(f)["terms"]


@pytest.fixture(scope="module")
def sample_df():
    if not SAMPLE_PATH.exists():
        pytest.skip("listing_sample.csv not found - run: python scripts/data_loading.py")
    return pd.read_csv(SAMPLE_PATH)


@pytest.fixture(scope="module")
def remarks(sample_df):
    return sample_df["remarks"].dropna().map(normalize).tolist()


@pytest.fixture(scope="module")
def queries():
    assert QUERIES_PATH.exists(), "Missing data/processed/sample_queries.json"
    with open(QUERIES_PATH) as f:
        return json.load(f)



#Taxonomy structure

def test_taxonomy_loaded(taxonomy):
    assert len(taxonomy) >= MIN_TERMS_TOTAL
    assert all("id" in t and "term" in t for t in taxonomy)


def test_taxonomy_has_exactly_8_expected_categories(taxonomy):
    assert {t["category"] for t in taxonomy} == EXPECTED_CATEGORIES


def test_each_category_has_enough_terms(taxonomy):
    counts = {}
    for t in taxonomy:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    too_small = {c: n for c, n in counts.items() if n < MIN_TERMS_PER_CATEGORY}
    assert not too_small, f"Categories under {MIN_TERMS_PER_CATEGORY} terms: {too_small}"


def test_taxonomy_ids_are_unique(taxonomy):
    ids = [t["id"] for t in taxonomy]
    assert len(ids) == len(set(ids))


def test_taxonomy_terms_are_unique_across_categories(taxonomy):
    terms = [normalize(t["term"]) for t in taxonomy]
    dupes = {x for x in terms if terms.count(x) > 1}
    assert not dupes, f"Duplicate terms: {sorted(dupes)}"


def test_taxonomy_terms_are_clean(taxonomy):
    for t in taxonomy:
        term = t["term"]
        assert term == term.strip().lower(), f"Term not lowercase/trimmed: {term!r}"
        assert term, "Empty term"



#Sample dataset

def test_sample_data_quality(sample_df):
    assert len(sample_df) >= MIN_SAMPLE_ROWS
    assert sample_df["remarks"].str.len().min() > 50


def test_sample_has_expected_columns(sample_df):
    expected = {"L_ListingID", "L_Address", "L_City", "beds", "baths", "price", "remarks"}
    assert expected.issubset(sample_df.columns)


def test_sample_listing_ids_are_unique(sample_df):
    assert sample_df["L_ListingID"].is_unique



#Taxonomy coverage
def test_taxonomy_coverage_meets_threshold(taxonomy, remarks):
    patterns = [term_regex(t["term"]) for t in taxonomy]
    covered = sum(1 for text in remarks if any(p.search(text) for p in patterns))
    coverage = covered / len(remarks)
    assert coverage >= MIN_COVERAGE, f"Coverage {coverage:.1%} is below {MIN_COVERAGE:.0%}"


def test_every_category_appears_in_sample(taxonomy, remarks):
    by_category = {}
    for t in taxonomy:
        by_category.setdefault(t["category"], []).append(term_regex(t["term"]))
    weak = {}
    for category, patterns in by_category.items():
        prevalence = sum(1 for text in remarks if any(p.search(text) for p in patterns)) / len(remarks)
        if prevalence < MIN_CATEGORY_PREVALENCE:
            weak[category] = round(prevalence, 3)
    assert not weak, f"Categories rarely seen in the sample: {weak}"


#Labeled queries
def test_queries_loaded_and_well_formed(queries):
    assert len(queries) >= MIN_QUERIES
    for item in queries:
        assert set(item) == {"query", "intent"}
        assert isinstance(item["query"], str) and item["query"].strip()
        assert item["intent"] in VALID_INTENTS


def test_queries_balanced_across_intents(queries):
    counts = {intent: 0 for intent in VALID_INTENTS}
    for item in queries:
        counts[item["intent"]] += 1
    short = {i: n for i, n in counts.items() if n < MIN_QUERIES_PER_INTENT}
    assert not short, f"Intents under {MIN_QUERIES_PER_INTENT} examples: {short}"


def test_queries_are_unique(queries):
    texts = [normalize(item["query"]) for item in queries]
    assert len(texts) == len(set(texts))


def test_queries_use_taxonomy_vocabulary(queries, taxonomy):
    patterns = [term_regex(t["term"]) for t in taxonomy]
    hits = sum(1 for item in queries if any(p.search(normalize(item["query"])) for p in patterns))
    assert hits / len(queries) >= 0.40
