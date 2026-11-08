"""Week 7 tests: QueryIntentClassifier, the labeled query file, the evaluator helpers, Week 4 integration.

Run from the repo root:
    pytest tests/test_week7.py -v

No database and no downloads are needed. The held-out test split is never scored here (only dev).
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_intent as ev                                    # noqa: E402
import query_intent as qi                                       # noqa: E402
from query_intent import LABELS, QueryIntentClassifier, normalize_query, understand   # noqa: E402

DATA = ROOT / "data" / "processed" / "intent_queries.json"


@pytest.fixture(scope="module")
def rows():
    return json.load(open(DATA))["queries"]


@pytest.fixture(scope="module")
def clf(rows):
    r = [x for x in rows if x["split"] in ("train", "dev")]
    return QueryIntentClassifier().train([x["query"] for x in r], [x["intent"] for x in r])


# ---------------------------------------------------------------- the labeled dataset
def test_dataset_size_and_labels(rows):
    assert len(rows) >= 200                                    # spec: 200+ labeled queries
    assert {r["intent"] for r in rows} == set(LABELS)
    assert Counter(r["split"] for r in rows).keys() == {"train", "dev", "test"}
    for lab in LABELS:                                         # roughly balanced
        assert sum(r["intent"] == lab for r in rows) >= 80


def test_no_duplicate_queries(rows):
    seen = Counter(r["query"].strip().lower() for r in rows)
    assert [q for q, c in seen.items() if c > 1] == []


def test_near_duplicate_groups_do_not_straddle_splits(rows):
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert [g for g, s in by_group.items() if len(s) > 1] == []


def test_required_fields_and_flags(rows):
    for r in rows:
        assert set(r) >= {"id", "query", "intent", "style", "hard", "source", "split", "group"}
        assert r["query"].strip() and isinstance(r["hard"], bool)
    assert sum(r["hard"] for r in rows) >= 10
    assert len({r["id"] for r in rows}) == len(rows)


def test_week1_queries_kept_their_labels(rows):
    wk1 = json.load(open(ROOT / "data" / "processed" / "sample_queries.json"))
    ours = {r["query"]: r["intent"] for r in rows if r["source"] == "week1"}
    assert len(ours) == 60 and all(ours[x["query"]] == x["intent"] for x in wk1)


def test_guidelines_exist_and_name_all_labels():
    text = (ROOT / "docs" / "intent_guidelines.md").read_text()
    for lab in LABELS:
        assert f"`{lab}`" in text


# ---------------------------------------------------------------- label order bug in the spec's starter code
def test_predict_uses_model_classes_not_a_fixed_order():
    """scikit-learn sorts classes alphabetically; a fixed ['browsing','researching','high_intent_inquiry'] list would swap two labels."""
    q = ["show me homes in A", "show me homes in B", "show me houses in C",
         "what is the average hoa fee", "how much are property taxes", "which areas have low crime",
         "open house this weekend", "schedule a showing tomorrow", "ready to make an offer today"]
    y = ["browsing"] * 3 + ["researching"] * 3 + ["high_intent_inquiry"] * 3
    c = QueryIntentClassifier(parser_features=False).train(q, y)
    assert c.classes_ == sorted(LABELS)
    assert c.classes_ != LABELS                                # the fixed order really would be wrong
    assert c.predict("schedule a showing this weekend")[0] == "high_intent_inquiry"
    assert c.predict("what is the average property tax")[0] == "researching"
    assert c.predict("show me houses in D")[0] == "browsing"


# ---------------------------------------------------------------- interface
def test_predict_shape_and_ranges(clf):
    intent, conf = clf.predict("homes with open houses this weekend")
    assert intent in LABELS and isinstance(intent, str) and 1 / 3 - 1e-9 <= conf <= 1.0


def test_detailed_probabilities_sum_to_one_and_are_consistent(clf):
    d = clf.predict_detailed("condos near UC Irvine with low HOA")
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert d["intent"] == max(d["probabilities"], key=d["probabilities"].get)
    assert d["confidence"] == pytest.approx(max(d["probabilities"].values()), abs=1e-3)
    assert d["margin"] >= 0 and d["second_choice"] != d["intent"]


def test_uncertain_flag_follows_threshold(clf):
    d = clf.predict_detailed("homes")
    assert d["uncertain"] == (d["confidence"] < clf.uncertain_below)
    strict = QueryIntentClassifier(uncertain_below=0.999)
    strict.pipeline, strict.trained = clf.pipeline, True
    assert strict.predict_detailed("show me homes in Irvine")["uncertain"] is True
    lax = QueryIntentClassifier(uncertain_below=0.0)
    lax.pipeline, lax.trained = clf.pipeline, True
    assert lax.predict_detailed("homes")["uncertain"] is False


def test_out_of_scope_text_is_flagged_uncertain(clf):
    """There is no 'other' class, so non-housing text must be caught by the in-scope check instead."""
    for q in ["what is the weather in Paris", "pizza near me", "asdf qwerty", "how do I bake bread"]:
        d = clf.predict_detailed(q)
        assert d["in_scope"] is False and d["uncertain"] is True, q
    for q in ["show me homes in Irvine", "3br condo Fresno", "average HOA dues", "Irvine"]:
        assert clf.predict_detailed(q)["in_scope"] is True, q


def test_clear_query_is_confident(clf):
    d = clf.predict_detailed("3 bed 2 bath in Escondido under 650k, available now, schedule a tour")
    assert d["intent"] == "high_intent_inquiry" and d["confidence"] > 0.8 and not d["uncertain"]


def test_batch_matches_single(clf):
    qs = ["show me homes in Corona", "average HOA dues in Palm Desert", "open houses this weekend in Irvine"]
    assert [d["intent"] for d in clf.predict_detailed_many(qs)] == [clf.predict(q)[0] for q in qs]


def test_untrained_and_bad_labels_raise():
    c = QueryIntentClassifier()
    with pytest.raises(RuntimeError):
        c.predict("homes in Irvine")
    with pytest.raises(ValueError):
        c.train(["a", "b"], ["browsing", "spam"])
    with pytest.raises(ValueError):
        c.train(["a", "b"], ["browsing", "browsing"])


@pytest.mark.parametrize("q", ["", "   ", "???", "a" * 5000, "ÅÄÖ 房子 $$$", None])
def test_odd_input_does_not_crash(clf, q):
    d = clf.predict_detailed("" if q is None else q)
    assert d["intent"] in LABELS


def test_save_and_load_roundtrip(clf, tmp_path):
    path = clf.save(tmp_path / "m.joblib")
    again = QueryIntentClassifier.load(path)
    for q in ["show me homes in Corona", "average HOA dues", "tour 3 bed tomorrow under 700k"]:
        assert again.predict_detailed(q)["probabilities"] == clf.predict_detailed(q)["probabilities"]
    assert again.uncertain_below == clf.uncertain_below


def test_normalize_query_collapses_prices():
    assert normalize_query("under $1.2M and 900k") == "under pricetoken and pricetoken"
    assert normalize_query(None) == ""


# ---------------------------------------------------------------- behaviour on the spec's examples (model fit on train+dev)
SPEC_EXAMPLES = [
    ("show me homes in San Diego", "browsing"),
    ("luxury homes in Beverly Hills", "browsing"),
    ("areas in San Diego with low property taxes", "researching"),
    ("move-in ready homes in San Diego under 1.2m", "high_intent_inquiry"),
    ("new listings in Irvine under 900k with seller financing", "high_intent_inquiry"),
]


@pytest.mark.parametrize("q,expected", SPEC_EXAMPLES)
def test_spec_examples(clf, q, expected):
    assert clf.predict(q)[0] == expected


# ---------------------------------------------------------------- parser features
def test_parser_features_columns_and_values():
    f = qi.ParserFeatures()
    row = dict(zip(f.NAMES, f.features("3 bed 2 bath under 700k in Irvine with a pool?")))
    assert row["has_city"] == 1 and row["has_price"] == 1 and row["has_beds"] == 1 and row["has_baths"] == 1
    assert row["has_question_mark"] == 1 and row["n_filters"] >= 4
    empty = dict(zip(f.NAMES, f.features("homes")))
    assert empty["n_filters"] == 0 and empty["has_price"] == 0


def test_parser_features_degrade_without_parser():
    f = qi.ParserFeatures(use_parser=False)
    assert f.features("3 bed under 700k")[:7] == [0.0] * 7


# ---------------------------------------------------------------- Week 4 integration
def test_understand_combines_parser_and_intent(clf):
    res = understand("open houses this weekend in Irvine under 900k", clf)
    assert res["filters"]["city"] == "Irvine" and res["filters"]["price_max"] == 900000
    assert res["valid"] and res["sql"] and "%s" in res["sql"]
    assert res["intent"] == "high_intent_inquiry" and res["suggested_handling"]["mode"] == "act"


def test_understand_invalid_query_has_no_sql(clf):
    res = understand("homes in Atlantis", clf)
    assert res["valid"] is False and res["sql"] is None and res["intent"] in LABELS


def test_uncertain_queries_get_the_cautious_handling(clf):
    low = QueryIntentClassifier(uncertain_below=0.9999)
    low.pipeline, low.trained = clf.pipeline, True
    res = understand("show me homes in Irvine", low)
    assert res["uncertain"] and res["suggested_handling"]["mode"] == "explore"


# ---------------------------------------------------------------- evaluator helpers
def test_metrics_perfect_and_wrong():
    gold = ["browsing", "researching", "high_intent_inquiry"] * 2
    assert ev.metrics(gold, gold)["accuracy"] == 1.0
    m = ev.metrics(gold, ["browsing"] * 6)
    assert m["accuracy"] == pytest.approx(2 / 6, abs=1e-3) and m["per_class"]["browsing"]["recall"] == 1.0
    assert m["per_class"]["researching"]["recall"] == 0.0 and sum(m["confusion"]["researching"].values()) == 2


def test_bootstrap_interval_contains_accuracy():
    correct = [1] * 80 + [0] * 20
    lo, hi = ev.bootstrap_accuracy(correct)
    assert lo < 0.8 < hi and 0.65 < lo and hi < 0.92


def test_calibration_perfect_case():
    conf = np.array([0.9] * 10)
    correct = np.array([1] * 9 + [0])
    assert ev.calibration(conf, correct)["ece"] == pytest.approx(0.0, abs=1e-6)


def test_coverage_table_is_monotone():
    conf = np.array([0.4, 0.5, 0.6, 0.7, 0.9, 0.95])
    correct = np.array([0, 0, 1, 1, 1, 1])
    t = ev.coverage_table(conf, correct)
    cov = [r["coverage"] for r in t]
    assert cov == sorted(cov, reverse=True) and t[0]["coverage"] == 1.0


def test_perturbations_change_text_but_not_length_much():
    rng = random.Random(1)
    q = "show me beautiful homes in Pleasanton"
    assert ev.PERTURBATIONS["UPPER CASE"](q, rng) == q.upper()
    assert "," not in ev.PERTURBATIONS["no punctuation"]("a, b! c", rng)
    t = ev.typo(q, rng)
    assert t != q and sorted(t) == sorted(q)


def test_rule_baseline_follows_guidelines():
    rb = ev.RuleBaseline()
    assert rb.predict(["are there open houses in Irvine this weekend?"]) == ["high_intent_inquiry"]   # timing beats question
    assert rb.predict(["what is the average price of a 3 bedroom in Irvine"]) == ["researching"]
    assert rb.predict(["3 bed under 700k in Corona"]) == ["high_intent_inquiry"]
    assert rb.predict(["homes in Irvine under 900k"]) == ["browsing"]
    assert rb.predict(["show me homes in Irvine"]) == ["browsing"]


def test_test_split_requires_final_flag():
    with pytest.raises(SystemExit):
        ev.main(["--split", "test"])


def test_read_extra_csv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("query,label\nshow me condos in Fresno,browsing\nhow much is HOA,researching\n")
    got = ev.read_extra(p)
    assert [r["intent"] for r in got] == ["browsing", "researching"]
    bad = tmp_path / "bad.csv"
    bad.write_text("query,label\nfoo,spam\n")
    with pytest.raises(SystemExit):
        ev.read_extra(bad)


# ---------------------------------------------------------------- dev quality gate (the test split is not touched here)
def test_dev_quality_gate(rows):
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    m, errs, _ = ev.evaluate(train, dev, qi.DEFAULT_UNCERTAIN_BELOW, "dev")
    assert m["accuracy"] >= 0.80 and m["macro_f1"] >= 0.80
    assert all(v["recall"] >= 0.7 for v in m["per_class"].values())
    assert m["calibration"]["accuracy"] > 0.8
