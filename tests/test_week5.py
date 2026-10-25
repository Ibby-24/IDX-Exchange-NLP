"""Week 5 tests: chunking, SemanticSearcher, BM25, ranking metrics, judgments file, latency helpers.

Run from the repo root:
    pytest tests/test_week5.py -v

* No model download is needed. Searcher tests use HashEncoder (a deterministic bag-of-words hash encoder
  defined below) or the LSA stand-in. The real SentenceTransformer path is covered by a test that is
  skipped when sentence-transformers is not installed.
* Tests that need data/processed/listing_sample.csv skip, instead of failing, when it is missing.
"""
import csv
import json
import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_latency as bl                                    # noqa: E402
import build_eval_corpus as bec                                   # noqa: E402
import evaluate_search as ev                                      # noqa: E402
import label_judgments as lj                                      # noqa: E402
from bm25_search import BM25Searcher, STOPWORDS, stem, tokenize   # noqa: E402
from semantic_search import (LSAEncoder, SearchHit, SemanticSearcher, chunk_text,  # noqa: E402
                             default_corpus_path, l2_normalize, load_corpus, make_encoder)

SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
JUDGMENTS = ROOT / "data" / "processed" / "relevance_judgments.json"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="listing_sample.csv not present")
needs_judgments = pytest.mark.skipif(not JUDGMENTS.exists(), reason="relevance_judgments.json not present")


class HashEncoder:
    """Deterministic test encoder: word counts hashed into `dim` buckets. Texts sharing words are similar."""
    needs_fit = False
    kind = "hash"
    name = "hash-test"

    def __init__(self, dim=64):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        return out

    def config(self):
        return {"kind": self.kind, "dim": self.dim}


CORPUS = [
    "Quiet cul-de-sac with a big backyard and a swimming pool",
    "Downtown loft near restaurants coffee shops and transit",
    "Horse property with barn corrals and acres of pasture",
    "Fixer upper needs work great for investors",
    "Mountain cabin among tall pine trees with a wood stove",
]
IDS = ["a1", "b2", "c3", "d4", "e5"]


def searcher(**kw):
    kw.setdefault("clean", False)
    kw.setdefault("encoder", HashEncoder())
    return SemanticSearcher(**kw).build_index(CORPUS, IDS)


# ============================================================================ l2_normalize
class TestNormalize:
    def test_rows_have_unit_length(self):
        a = l2_normalize(np.random.default_rng(0).standard_normal((20, 7)))
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)

    def test_zero_row_stays_zero_not_nan(self):
        a = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
        assert np.all(np.isfinite(a)) and np.allclose(a[0], 0) and np.allclose(a[1], [0.6, 0.8])

    def test_one_dimensional_input_becomes_one_row(self):
        assert l2_normalize(np.array([1.0, 0.0])).shape == (1, 2)

    def test_dtype_and_layout_for_faiss(self):
        a = l2_normalize(np.ones((3, 4), dtype=np.float64))
        assert a.dtype == np.float32 and a.flags["C_CONTIGUOUS"]


# ============================================================================ chunking
class TestChunking:
    LONG = " ".join(f"Sentence number {i} has exactly seven words." for i in range(30))   # 30 sentences x 7 = 210 words

    def test_short_text_is_one_passage(self):
        assert chunk_text("A few words only.", 100) == ["A few words only."]

    def test_empty_and_none(self):
        assert chunk_text("", 10) == [] and chunk_text("   ", 10) == [] and chunk_text(None, 10) == []

    def test_no_passage_longer_than_limit(self):
        assert all(len(c.split()) <= 50 for c in chunk_text(self.LONG, 50, 10))

    def test_every_word_is_kept_without_overlap(self):
        chunks = chunk_text(self.LONG, 50, 0)
        assert " ".join(chunks).split() == self.LONG.split()

    def test_overlap_repeats_text_and_loses_nothing(self):
        plain, over = chunk_text(self.LONG, 50, 0), chunk_text(self.LONG, 50, 14)
        assert len(over) >= len(plain)
        assert set(" ".join(over).split()) == set(self.LONG.split())
        assert over[1].split()[:3] == over[0].split()[-7:][:3] or set(over[1].split()) & set(over[0].split())

    def test_limit_is_exact_when_sentences_do_not_divide_it(self):
        text = " ".join(f"This sentence has six words {i}." for i in range(40))      # 7 words each incl. the number
        six = " ".join("Short sentence with six words." for _ in range(40))           # 5 words each
        for t in (text, six):
            sizes = [len(c.split()) for c in chunk_text(t, 48, 0)]
            assert max(sizes) <= 48 and max(sizes) >= 40

    def test_sentences_are_not_cut_when_they_fit(self):
        for c in chunk_text(self.LONG, 50, 0):
            assert c.rstrip().endswith(".")

    def test_run_on_sentence_is_hard_split(self):
        text = " ".join(f"w{i}" for i in range(250))          # no punctuation at all
        chunks = chunk_text(text, 100, 0)
        assert [len(c.split()) for c in chunks] == [100, 100, 50]

    @pytest.mark.parametrize("max_words,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
    def test_bad_parameters(self, max_words, overlap):
        with pytest.raises(ValueError):
            chunk_text("some text here", max_words, overlap)

    def test_100_word_default_on_typical_remark(self):
        text = " ".join(["This home has a lovely kitchen."] * 60)       # 360 words
        assert len(chunk_text(text, 100, 20)) >= 4


# ============================================================================ SemanticSearcher
class TestSemanticSearcher:
    def test_finds_the_listing_that_shares_the_words(self):
        s = searcher()
        assert s.search("horse barn pasture", 1)[0].listing_id == "c3"
        assert s.search("restaurants coffee downtown", 1)[0].listing_id == "b2"

    def test_exact_text_scores_one(self):
        s = searcher()
        hit = s.search(CORPUS[3], 1)[0]
        assert hit.listing_id == "d4" and hit.score == pytest.approx(1.0, abs=1e-5)

    def test_results_are_sorted_best_first_with_ranks(self):
        hits = searcher().search("backyard pool cabin trees", 5)
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
        assert all(hits[i].score >= hits[i + 1].score for i in range(4))

    def test_hit_fields(self):
        h = searcher().search("barn", 1)[0]
        assert isinstance(h, SearchHit) and h.index == IDS.index(h.listing_id)

    def test_top_k_larger_than_corpus_returns_everything_once(self):
        hits = searcher().search("home", 50)
        assert sorted(h.listing_id for h in hits) == sorted(IDS)

    def test_scores_stay_in_cosine_range(self):
        assert all(-1.0001 <= h.score <= 1.0001 for h in searcher().search("pool", 5))

    @pytest.mark.parametrize("bad", [0, -3, 2.5, "5", None, True])
    def test_bad_top_k(self, bad):
        with pytest.raises(ValueError):
            searcher().search("pool", bad)

    @pytest.mark.parametrize("q", ["", "   ", None])
    def test_empty_query_returns_nothing(self, q):
        assert searcher().search(q, 3) == []

    def test_search_before_build_is_an_error(self):
        with pytest.raises(RuntimeError):
            SemanticSearcher(HashEncoder(), clean=False).search("x", 1)

    def test_ids_and_texts_must_match(self):
        with pytest.raises(ValueError):
            SemanticSearcher(HashEncoder(), clean=False).build_index(["a", "b"], ["x"])

    def test_all_empty_texts_is_an_error(self):
        with pytest.raises(ValueError):
            SemanticSearcher(HashEncoder(), clean=False).build_index(["", "  "])

    def test_default_ids_are_positions(self):
        s = SemanticSearcher(HashEncoder(), clean=False).build_index(CORPUS)
        assert s.search("horse barn", 1)[0].listing_id == 2

    def test_empty_remark_is_skipped_but_keeps_its_position(self):
        s = SemanticSearcher(HashEncoder(), clean=False).build_index(["horse barn", "", "pool deck"], ["x", "y", "z"])
        assert s.stats()["listings"] == 3 and s.stats()["passages"] == 2
        assert [h.listing_id for h in s.search("pool", 3)][0] == "z"

    def test_cleaning_is_applied_to_text_and_query(self):
        s = SemanticSearcher(HashEncoder(), clean=True).build_index(["Spacious home w/ pool", "Mountain cabin"], ["p", "m"])
        assert s.search("pool", 1)[0].listing_id == "p"

    # ---- chunking inside the searcher
    def test_late_mention_is_found_only_with_chunking(self):
        filler = " ".join(["Lovely home with nice rooms and bright windows."] * 30)
        texts = [filler + " Also there is a wet bar.", filler + " Also there is a garden."]
        whole = SemanticSearcher(HashEncoder(256), clean=False).build_index(texts, ["bar", "garden"])
        chunked = SemanticSearcher(HashEncoder(256), clean=False, chunk_words=20, chunk_overlap=5).build_index(texts, ["bar", "garden"])
        gap_whole = dict((h.listing_id, h.score) for h in whole.search("wet bar", 2))
        gap_chunk = dict((h.listing_id, h.score) for h in chunked.search("wet bar", 2))
        assert (gap_chunk["bar"] - gap_chunk["garden"]) > (gap_whole["bar"] - gap_whole["garden"])

    def test_listing_score_is_best_passage_and_listing_appears_once(self):
        text = "alpha alpha alpha. " * 10 + "beta beta beta. " * 10
        s = SemanticSearcher(HashEncoder(), clean=False, chunk_words=10, chunk_overlap=0).build_index([text, "gamma"], ["L", "G"])
        assert s.stats()["passages"] > 2
        hits = s.search("alpha alpha alpha", 5)
        assert [h.listing_id for h in hits].count("L") == 1
        assert hits[0].listing_id == "L" and hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_many_passages_of_one_listing_do_not_hide_other_listings(self):
        big = "pool " * 400                       # 40 passages that all match "pool"
        s = SemanticSearcher(HashEncoder(), clean=False, chunk_words=10).build_index([big, "pool deck", "pool house"], ["big", "m", "h"])
        assert len(s.search("pool", 3)) == 3

    # ---- backends
    def test_faiss_and_numpy_return_the_same_ranking(self):
        pytest.importorskip("faiss")
        a, b = searcher(use_faiss=True), searcher(use_faiss=False)
        assert a.backend == "faiss-flat-ip" and b.backend == "numpy"
        for q in ["pool backyard", "horse barn", "investors work", "coffee downtown"]:
            ha, hb = a.search(q, 5), b.search(q, 5)
            assert [h.listing_id for h in ha] == [h.listing_id for h in hb]
            assert np.allclose([h.score for h in ha], [h.score for h in hb], atol=1e-5)

    def test_numpy_fallback_when_faiss_missing(self, monkeypatch):
        import semantic_search
        monkeypatch.setattr(semantic_search, "_faiss", lambda: None)
        s = searcher()
        assert s.backend == "numpy" and s.search("horse", 1)[0].listing_id == "c3"

    # ---- save / load
    def test_save_load_round_trip(self, tmp_path):
        s = searcher(chunk_words=8, chunk_overlap=2)
        s.save(tmp_path)
        t = SemanticSearcher.load(tmp_path, encoder=HashEncoder())
        for q in ["horse barn", "pool backyard", "cabin pine"]:
            assert [(h.listing_id, round(h.score, 5)) for h in s.search(q, 5)] == \
                   [(h.listing_id, round(h.score, 5)) for h in t.search(q, 5)]
        assert t.chunk_words == 8 and t.chunk_overlap == 2

    def test_saved_files_contain_no_remark_text(self, tmp_path):
        marker = "UNIQUEZEBRAMARKER"
        SemanticSearcher(HashEncoder(), clean=False).build_index([f"{marker} horse barn", "pool deck"], ["x", "y"]).save(tmp_path)
        for p in tmp_path.iterdir():
            assert marker.encode() not in p.read_bytes() and marker.lower().encode() not in p.read_bytes()

    def test_load_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SemanticSearcher.load(tmp_path / "nothing")

    def test_load_rejects_mismatched_embeddings(self, tmp_path):
        searcher().save(tmp_path)
        np.save(tmp_path / "embeddings.npy", np.zeros((5, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            SemanticSearcher.load(tmp_path, encoder=HashEncoder())

    def test_stats(self):
        st = searcher().stats()
        assert st["listings"] == 5 and st["passages"] == 5 and st["dim"] == 64


# ============================================================================ encoders
class TestEncoders:
    def test_make_encoder_kinds(self):
        assert make_encoder("lsa").kind == "lsa"
        e = make_encoder("minilm")
        assert e.kind == "sentence-transformers" and e.model_name == "all-MiniLM-L6-v2"
        assert make_encoder("mpnet").model_name == "all-mpnet-base-v2"

    def test_sentence_transformer_import_error_is_helpful(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="pip install"):
            make_encoder("minilm").encode(["x"])

    def test_sentence_transformer_real_model(self):
        pytest.importorskip("sentence_transformers")
        enc = make_encoder("minilm")
        v = l2_normalize(enc.encode(["quiet street", "peaceful road", "tankless water heater"]))
        assert v.shape == (3, 384) and enc.dim == 384
        assert v[0] @ v[1] > v[0] @ v[2]          # the point of embeddings: synonyms are closer than strangers

    def test_lsa_needs_fit(self):
        with pytest.raises(RuntimeError):
            LSAEncoder().encode(["x"])

    def test_lsa_groups_cooccurring_words(self):
        docs = ["horse barn pasture corral", "horse stable pasture arena", "pool spa patio deck", "pool spa deck yard"] * 6
        enc = LSAEncoder(n_components=3).fit(docs)
        v = l2_normalize(enc.encode(["horse corral", "stable arena", "spa deck"]))
        assert v[0] @ v[1] > v[0] @ v[2]

    def test_lsa_save_load(self, tmp_path):
        docs = ["horse barn pasture corral", "pool spa patio deck", "kitchen island granite quartz"] * 5
        enc = LSAEncoder(n_components=3).fit(docs)
        enc.save(tmp_path / "e.joblib")
        again = LSAEncoder.load(tmp_path / "e.joblib", 3)
        assert np.allclose(enc.encode(["horse pool"]), again.encode(["horse pool"]))

    def test_searcher_fits_lsa_itself(self):
        s = SemanticSearcher(LSAEncoder(n_components=3), clean=False).build_index(CORPUS * 3, IDS * 3)
        assert len(s.search("horse barn pasture", 3)) == 3


# ============================================================================ BM25
class TestBM25:
    def test_stem_plurals_but_not_false_plurals(self):
        assert stem("pools") == "pool" and stem("communities") == "community"
        for w in ("glass", "bonus", "this", "gas", "pool"):
            assert stem(w) == w

    def test_tokenize_drops_stopwords_and_single_letters(self):
        assert tokenize("The pool and a spa in the yard") == ["pool", "spa", "yard"]
        assert "no" in STOPWORDS and "not" in STOPWORDS

    def test_tokenize_keeps_numbers(self):
        assert tokenize("3 bed 2 bath 55+ community") == ["3", "bed", "2", "bath", "55", "community"]

    def test_score_matches_the_formula_by_hand(self):
        docs = ["alpha beta", "alpha gamma gamma", "delta"]
        bm = BM25Searcher(clean=False, use_stem=False).build_index(docs, ["d0", "d1", "d2"])
        n, avg = 3, (2 + 3 + 1) / 3
        idf = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))           # "alpha" is in 2 of 3 documents
        expected = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / avg))
        assert bm.scores("alpha")[0] == pytest.approx(expected)

    def test_rarer_word_counts_more(self):
        docs = ["pool pool deck", "pool deck", "yard deck", "yard deck", "yard deck", "wetbar yard"]
        bm = BM25Searcher(clean=False).build_index(docs)
        assert bm.search("wetbar", 1)[0].index == 5
        assert bm.idf["wetbar"] > bm.idf["deck"]

    def test_repeated_word_helps_but_with_diminishing_returns(self):
        docs = ["pool", "pool pool", "pool pool pool pool pool pool", "yard"] + ["other words here"] * 5
        bm = BM25Searcher(clean=False, b=0).build_index(docs)
        s = bm.scores("pool")
        assert s[0] < s[1] < s[2] and (s[1] - s[0]) > (s[2] - s[1]) / 5

    def test_long_document_is_penalised(self):
        docs = ["pool " + "filler " * 80, "pool filler"] + ["other"] * 3
        bm = BM25Searcher(clean=False).build_index(docs)
        assert bm.search("pool", 2)[0].index == 1

    def test_no_match_returns_empty(self):
        assert BM25Searcher(clean=False).build_index(CORPUS, IDS).search("xylophone", 5) == []

    def test_query_of_only_stopwords_returns_empty(self):
        assert BM25Searcher(clean=False).build_index(CORPUS, IDS).search("the and of", 5) == []

    def test_plural_query_matches_singular_text(self):
        bm = BM25Searcher(clean=False).build_index(["a pool and spa", "a garden"], ["p", "g"])
        assert bm.search("pools", 1)[0].listing_id == "p"

    def test_no_hoa_cannot_be_told_from_hoa(self):
        # The weakness the study documents: "no" is a stopword, so the query "no HOA" is just "HOA",
        # and a listing that says "HOA fees $400 monthly" matches it as well as one that says "No HOA".
        assert tokenize("no hoa") == tokenize("hoa") == ["hoa"]
        bm = BM25Searcher(clean=False).build_index(["No HOA fees here", "HOA fees $400 monthly"], ["none", "has"])
        s = bm.scores("no hoa")
        assert s[0] > 0 and s[1] > 0

    def test_ties_broken_by_position(self):
        bm = BM25Searcher(clean=False).build_index(["pool", "pool", "pool"], ["a", "b", "c"])
        assert [h.listing_id for h in bm.search("pool", 3)] == ["a", "b", "c"]

    @pytest.mark.parametrize("kw", [{"k1": -1}, {"b": 1.5}, {"b": -0.1}])
    def test_bad_parameters(self, kw):
        with pytest.raises(ValueError):
            BM25Searcher(**kw)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, None, True])
    def test_bad_top_k(self, bad):
        with pytest.raises(ValueError):
            BM25Searcher(clean=False).build_index(CORPUS).search("pool", bad)

    def test_search_before_build(self):
        with pytest.raises(RuntimeError):
            BM25Searcher().search("pool")

    def test_build_input_errors(self):
        with pytest.raises(ValueError):
            BM25Searcher().build_index([])
        with pytest.raises(ValueError):
            BM25Searcher().build_index(["a"], ["x", "y"])

    def test_none_and_empty_documents_are_tolerated(self):
        bm = BM25Searcher(clean=False).build_index([None, "", "pool deck"], ["n", "e", "p"])
        assert [h.listing_id for h in bm.search("pool", 3)] == ["p"]


# ============================================================================ metrics
J = {"a": 2, "b": 1, "c": 0, "d": 2, "e": 0}


class TestMetrics:
    def test_precision(self):
        assert ev.precision_at_k(["a", "c", "b", "e", "d"], J, 5) == pytest.approx(3 / 5)
        assert ev.precision_at_k(["a", "c", "b", "e", "d"], J, 5, rel_min=2) == pytest.approx(2 / 5)

    def test_precision_divides_by_k_even_with_few_results(self):
        assert ev.precision_at_k(["a"], J, 5) == pytest.approx(1 / 5)

    def test_unjudged_counts_as_not_relevant(self):
        assert ev.precision_at_k(["zzz", "a"], J, 2) == pytest.approx(0.5)

    def test_precision_bad_k(self):
        with pytest.raises(ValueError):
            ev.precision_at_k(["a"], J, 0)

    def test_reciprocal_rank(self):
        assert ev.reciprocal_rank(["c", "e", "b"], J) == pytest.approx(1 / 3)
        assert ev.reciprocal_rank(["c", "e"], J) == 0.0
        assert ev.reciprocal_rank(["c", "a"], J, rel_min=2) == pytest.approx(1 / 2)

    def test_ndcg_perfect_and_zero(self):
        assert ev.ndcg_at_k(["a", "d", "b", "c", "e"], J) == pytest.approx(1.0)
        assert ev.ndcg_at_k(["c", "e"], J) == 0.0

    def test_ndcg_known_value(self):
        # ranking [b, a]: DCG = 1/log2(2) + 2/log2(3); ideal [a, d, b]: 2 + 2/log2(3) + 1/2
        got = ev.ndcg_at_k(["b", "a"], J, 3)
        dcg = 1 / 1 + 2 / math.log2(3)
        ideal = 2 / 1 + 2 / math.log2(3) + 1 / 2
        assert got == pytest.approx(dcg / ideal)

    def test_ndcg_order_matters(self):
        assert ev.ndcg_at_k(["a", "b", "c"], J) > ev.ndcg_at_k(["c", "b", "a"], J)

    def test_ndcg_with_no_relevant_judgments(self):
        assert ev.ndcg_at_k(["a"], {"a": 0, "b": 0}) == 0.0

    def test_unjudged_share(self):
        assert ev.unjudged_at_k(["a", "x", "y", "c"], J) == pytest.approx(0.5)
        assert ev.unjudged_at_k([], J) == 0.0

    def test_ids_are_compared_as_text(self):
        assert ev.precision_at_k([1158147085], {"1158147085": 2}, 1) == 1.0

    def test_rrf_prefers_items_high_in_both_lists(self):
        assert ev.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["a", "b"]
        assert ev.reciprocal_rank_fusion([["a", "b"], ["c", "b"]])[0] == "b"

    def test_rrf_includes_items_from_either_list(self):
        assert set(ev.reciprocal_rank_fusion([["a"], ["b"], ["c"]])) == {"a", "b", "c"}
        assert ev.reciprocal_rank_fusion([]) == []

    def test_rrf_score_formula(self):
        # item "a" is rank 1 in one list and absent in the other; "b" is rank 2 in both
        assert ev.reciprocal_rank_fusion([["a", "b"], ["x", "b"]], k=60)[0] == "b"

    def test_bootstrap_identical_scores_give_zero(self):
        m, lo, hi = ev.paired_bootstrap([0.5, 0.7, 0.9], [0.5, 0.7, 0.9])
        assert m == lo == hi == 0.0

    def test_bootstrap_consistent_gain_excludes_zero(self):
        a = [0.9] * 20
        b = [0.5, 0.6, 0.4, 0.5] * 5
        m, lo, hi = ev.paired_bootstrap(a, b)
        assert m > 0 and lo > 0

    def test_bootstrap_mixed_differences_include_zero(self):
        a = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        b = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        _, lo, hi = ev.paired_bootstrap(a, b)
        assert lo < 0 < hi

    def test_bootstrap_interval_is_a_95_percent_interval(self):
        a, b = [1.0, 0.0] * 10, [0.0, 1.0] * 10                  # differences +1, -1 alternating: mean 0, sd 1
        _, lo, hi = ev.paired_bootstrap(a, b, n_boot=20000, seed=1)
        assert 0.75 < hi - lo < 0.95                              # about 2 * 1.96 / sqrt(20) = 0.877

    def test_bootstrap_is_reproducible_and_validates(self):
        a, b = [0.2, 0.4, 0.9], [0.1, 0.5, 0.3]
        assert ev.paired_bootstrap(a, b, 500, seed=3) == ev.paired_bootstrap(a, b, 500, seed=3)
        with pytest.raises(ValueError):
            ev.paired_bootstrap([1], [1, 2])
        with pytest.raises(ValueError):
            ev.paired_bootstrap([], [])

    def test_p10_upper_bound(self):
        j = {"a": 2, "b": 0, "c": 1}
        assert ev.p10_upper_bound(["a", "b", "x", "y"], j) == pytest.approx(3 / 10)        # a, x, y could count
        assert ev.p10_upper_bound(["a", "c"], j, rel_min=2) == pytest.approx(1 / 10)
        assert ev.p10_upper_bound([f"u{i}" for i in range(10)], j) == 1.0

    def test_win_tie_loss(self):
        assert ev.win_tie_loss([0.9, 0.5, 0.2], [0.5, 0.5, 0.6]) == (1, 1, 1)
        assert ev.win_tie_loss([0.505], [0.5]) == (0, 1, 0)       # inside the tolerance

    def test_evaluate_system_perfect_ranker(self):
        queries = [{"id": "q", "query": "x", "type": "t", "judgments": {"a": 2, "b": 0, "c": 1}}]
        out, rk = ev.evaluate_system(lambda q: ["a", "c", "b"], queries)
        assert out["ndcg10"] == [pytest.approx(1.0)] and out["mrr"] == [1.0] and rk["q"] == ["a", "c", "b"]
        assert out["unjudged10"] == [0.0]


# ============================================================================ judgments file
@pytest.fixture(scope="module")
def data():
    return json.load(open(JUDGMENTS))


@needs_judgments
class TestJudgmentsFile:

    def test_at_least_50_judged_pairs_and_20_queries(self, data):
        assert len(data["queries"]) >= 20
        assert sum(len(q["judgments"]) for q in data["queries"]) >= 50

    def test_structure_has_no_text_fields(self, data):
        for q in data["queries"]:
            assert set(q) == {"id", "query", "type", "judgments"}
            assert all(re.fullmatch(r"\d+", k) for k in q["judgments"])        # listing ids only, no remark text

    def test_grades_valid(self, data):
        assert all(v in (0, 1, 2) for q in data["queries"] for v in q["judgments"].values())

    def test_query_ids_unique_and_types_known(self, data):
        ids = [q["id"] for q in data["queries"]]
        assert len(ids) == len(set(ids))
        assert {q["type"] for q in data["queries"]} <= {"concept", "mixed", "lexical"}
        assert {"concept", "lexical"} <= {q["type"] for q in data["queries"]}

    def test_every_query_has_relevant_and_irrelevant_examples(self, data):
        for q in data["queries"]:
            grades = set(q["judgments"].values())
            assert 2 in grades, q["id"]
        assert any(0 in {v for v in q["judgments"].values()} for q in data["queries"])

    @needs_sample
    def test_judged_listings_exist_in_the_corpus_that_is_searched(self, data):
        """Fails when listing_sample.csv is a different random draw than the one that was graded.
        Fix: python scripts/build_eval_corpus.py (creates listing_sample_eval.csv, which is then used)."""
        ids = {r["L_ListingID"] for r in csv.DictReader(open(default_corpus_path(), newline="", encoding="utf-8"))}
        missing = [(q["id"], k) for q in data["queries"] for k in q["judgments"] if k not in ids]
        assert not missing, f"{len(missing)} judged listings are not in {default_corpus_path().name}: run scripts/build_eval_corpus.py"

    def test_queries_are_not_listing_text_copies(self, data):
        assert all(len(q["query"].split()) <= 15 for q in data["queries"])


# ============================================================================ end to end on the sample
@pytest.fixture(scope="module")
def world():
    ids, texts = load_corpus()
    queries = ev.load_judgments()
    found, total, _ = ev.judged_coverage(queries, ids)
    if found / total < ev.MIN_COVERAGE:
        pytest.skip("judged listings are not in the corpus: run scripts/build_eval_corpus.py")
    bm = BM25Searcher().build_index(texts, ids)
    sem = SemanticSearcher(LSAEncoder(60), chunk_words=100, chunk_overlap=20).build_index(texts, ids)
    return ids, texts, queries, bm, sem


@needs_sample
@needs_judgments
class TestEndToEnd:

    def test_bm25_beats_a_shuffled_ranking(self, world):
        ids, _, queries, bm, _ = world
        rng = np.random.default_rng(0)
        real, _ = ev.evaluate_system(lambda q: [h.listing_id for h in bm.search(q, 50)], queries)
        fake, _ = ev.evaluate_system(lambda q: list(rng.permutation(ids)[:50]), queries)
        assert ev.mean(real["ndcg10"]) > ev.mean(fake["ndcg10"]) + 0.2

    def test_semantic_beats_a_shuffled_ranking(self, world):
        ids, _, queries, _, sem = world
        rng = np.random.default_rng(0)
        real, _ = ev.evaluate_system(lambda q: [h.listing_id for h in sem.search(q, 50)], queries)
        fake, _ = ev.evaluate_system(lambda q: list(rng.permutation(ids)[:50]), queries)
        assert ev.mean(real["ndcg10"]) > ev.mean(fake["ndcg10"]) + 0.2

    def test_wet_bar_query_finds_wet_bar_listings(self, world):
        ids, texts, _, bm, _ = world
        hits = bm.search("wet bar", 5)
        remarks = dict(zip(ids, texts))
        assert hits and all("wet bar" in remarks[h.listing_id].lower() for h in hits)

    def test_semantic_hits_exist_in_the_sample(self, world):
        ids, _, _, _, sem = world
        assert {h.listing_id for h in sem.search("quiet neighborhood for a family", 10)} <= set(ids)

    def test_search_is_deterministic(self, world):
        sem = world[4]
        a, b = sem.search("room for horses", 10), sem.search("room for horses", 10)
        assert [(h.listing_id, h.score) for h in a] == [(h.listing_id, h.score) for h in b]

    def test_evaluate_cli_runs(self, world, tmp_path, capsys):
        out = tmp_path / "r.json"
        assert ev.main(["--encoder", "lsa", "--out", str(out), "--no-whole"]) == 0
        res = json.load(open(out))
        assert {"bm25", "semantic", "hybrid-rrf"} <= set(res["summary"])
        assert 0 <= res["summary"]["bm25"]["ndcg10"] <= 1

    def test_build_and_search_cli(self, world, tmp_path, capsys):
        import semantic_search as ss
        assert ss.main(["build", "--encoder", "lsa", "--out", str(tmp_path)]) == 0
        assert (tmp_path / "embeddings.npy").exists() and (tmp_path / "meta.json").exists()
        assert ss.main(["search", "room for horses", "--index-dir", str(tmp_path), "--top", "3"]) == 0
        assert "result(s)" in capsys.readouterr().out
        meta = json.load(open(tmp_path / "meta.json"))
        assert set(meta) >= {"ids", "owner", "encoder", "dim"} and "remarks" not in meta


# ============================================================================ labeling helpers
def _queries():
    return [{"id": "q1", "query": "wet bar", "type": "lexical", "judgments": {"a": 2, "b": 0}},
            {"id": "q2", "query": "pool", "type": "lexical", "judgments": {"c": 1}}]


class TestLabelHelpers:
    def test_find_unjudged_dedupes_across_systems(self):
        rankings = {"s1": {"q1": ["a", "x", "y"], "q2": ["c", "z"]}, "s2": {"q1": ["x", "w"], "q2": ["z"]}}
        pairs = lj.find_unjudged(rankings, _queries(), top=3)
        assert sorted(pairs) == [("q1", "w"), ("q1", "x"), ("q1", "y"), ("q2", "z")]

    def test_find_unjudged_respects_top(self):
        pairs = lj.find_unjudged({"s": {"q1": ["a", "b", "x"]}}, _queries(), top=2)
        assert pairs == []

    def test_find_unjudged_is_reproducible(self):
        r = {"s": {"q1": [f"u{i}" for i in range(30)]}}
        assert lj.find_unjudged(r, _queries(), 30) == lj.find_unjudged(r, _queries(), 30)

    @pytest.mark.parametrize("v,expected", [("2", 2), (" 1 ", 1), ("0", 0), ("2.0", 2), (1, 1)])
    def test_parse_grade_ok(self, v, expected):
        assert lj.parse_grade(v) == expected

    @pytest.mark.parametrize("v", ["", "3", "-1", "1.5", "yes", None])
    def test_parse_grade_bad(self, v):
        with pytest.raises((ValueError, TypeError)):
            lj.parse_grade(v)

    def test_merge_adds_and_keeps_existing(self):
        qs = _queries()
        rep = lj.merge_grades(qs, [("q1", "n", 1), ("q1", "a", 2), ("q1", "b", 2), ("q9", "z", 1)])
        assert rep["added"] == 1 and rep["unchanged"] == 1 and rep["conflicts"] == [("q1", "b", 0, 2)]
        assert rep["unknown_query"] == ["q9"] and qs[0]["judgments"]["b"] == 0

    def test_merge_overwrite(self):
        qs = _queries()
        assert lj.merge_grades(qs, [("q1", "b", 2)], overwrite=True)["changed"] == 1
        assert qs[0]["judgments"]["b"] == 2

    def test_read_graded_csv_skips_blank_and_reports_bad(self, tmp_path):
        p = tmp_path / "g.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "query", "listing_id", "remarks", "grade"])
            w.writerows([["q1", "x", "n1", "text", "2"], ["q1", "x", "n2", "text", ""], ["q1", "x", "n3", "text", "7"]])
        triples, problems = lj.read_graded_csv(p)
        assert triples == [("q1", "n1", 2)] and len(problems) == 1 and "line 4" in problems[0]

    def test_save_judgments_keeps_other_fields(self, tmp_path):
        p = tmp_path / "j.json"
        json.dump({"version": 1, "note": "keep me", "queries": []}, open(p, "w"))
        qs = _queries()
        lj.save_judgments(qs, p)
        data = json.load(open(p))
        assert data["note"] == "keep me" and len(data["queries"]) == 2

    def test_import_command_end_to_end(self, tmp_path):
        j = tmp_path / "j.json"
        json.dump({"queries": _queries()}, open(j, "w"))
        g = tmp_path / "g.csv"
        with open(g, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "query", "listing_id", "remarks", "grade"])
            w.writerow(["q2", "pool", "newid", "some text", "2"])
        lj.main(["--judgments", str(j), "import", str(g)])
        saved = json.load(open(j))
        assert saved["queries"][1]["judgments"]["newid"] == 2
        assert "some text" not in j.read_text()                       # remark text never reaches the judgments file


# ============================================================================ latency helpers and the 100 ms spec
class TestLatency:
    def test_percentile(self):
        assert bl.percentile([1, 2, 3, 4, 5], 50) == 3
        assert bl.percentile([10], 99) == 10
        assert bl.percentile([0, 10], 25) == pytest.approx(2.5)
        assert bl.percentile([5, 1, 3], 100) == 5
        with pytest.raises(ValueError):
            bl.percentile([], 50)

    def test_time_calls_discards_warmup(self):
        calls = []
        r = bl.time_calls(lambda i: calls.append(i), runs=10, warmup=3)
        assert len(calls) == 13 and r["runs"] == 10 and r["p50_ms"] >= 0

    def test_random_vectors_are_unit_and_reproducible(self):
        a = bl.random_unit_vectors(50, 16, seed=1)
        assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
        assert np.array_equal(a, bl.random_unit_vectors(50, 16, seed=1))

    def test_bench_vector_search_reports_sizes(self):
        r = bl.bench_vector_search(500, 3, 32, 10, runs=10, use_faiss=False)
        assert r["passages"] == 1500 and r["backend"] == "numpy" and r["p95_ms"] >= r["p50_ms"]

    def test_10k_listings_vector_search_is_well_under_100ms(self):
        r = bl.bench_vector_search(10000, 4, 384, 10, runs=30, use_faiss=False)
        assert r["p95_ms"] < ev_limit()

    def test_10k_listings_with_faiss_is_well_under_100ms(self):
        pytest.importorskip("faiss")
        r = bl.bench_vector_search(10000, 4, 384, 10, runs=30, use_faiss=True)
        assert r["backend"] == "faiss-flat-ip" and r["p95_ms"] < ev_limit()

    def test_full_path_on_a_small_index(self):
        s = searcher(chunk_words=6)
        r = bl.bench_full_path(s, ["horse barn", "pool"], runs=10)
        assert r["listings"] == 5 and r["p50_ms"] >= 0


def ev_limit():
    return bl.SPEC_LIMIT_MS


# ============================================================================ the judged listings must be in the corpus
class _MySQLShapedConnection:
    """sqlite3 connection that accepts %s placeholders, so build_eval_corpus.main can be run without MySQL."""
    def __init__(self, conn):
        self._c = conn

    def cursor(self):
        outer = self._c.cursor()

        class Cur:
            def execute(self, sql, params=()):
                outer.execute(sql.replace("%s", "?"), params)

            def fetchall(self):
                return outer.fetchall()
        return Cur()

    def close(self):
        pass


def _write_sample(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["L_ListingID", "remarks"])
        w.writerows(rows)


def _queries_for(ids_by_query):
    return [{"id": q, "query": q, "type": "t", "judgments": {i: 2 for i in ids}} for q, ids in ids_by_query.items()]


class TestCorpusCoverage:
    def test_coverage_counts_judged_pairs_in_corpus(self):
        qs = _queries_for({"q1": ["a", "b"], "q2": ["b", "c"]})
        assert ev.judged_coverage(qs, ["a", "b", "x"])[:2] == (3, 4)      # (q1,a) (q1,b) (q2,b) found; (q2,c) not
        found, total, missing = ev.judged_coverage(qs, [])
        assert (found, total) == (0, 4) and ("q2", "c") in missing

    def test_coverage_compares_ids_as_text(self):
        assert ev.judged_coverage(_queries_for({"q": ["1158147085"]}), [1158147085])[:2] == (1, 1)

    def test_default_corpus_prefers_the_eval_file(self, tmp_path, monkeypatch):
        import semantic_search as ss
        eval_file = tmp_path / "listing_sample_eval.csv"
        monkeypatch.setattr(ss, "EVAL_CORPUS", eval_file)
        assert ss.default_corpus_path() == ss.SAMPLE
        eval_file.write_text("L_ListingID,remarks\n1,hello\n")
        assert ss.default_corpus_path() == eval_file
        assert ss.load_corpus() == (["1"], ["hello"])

    def test_evaluation_refuses_to_run_when_judged_listings_are_missing(self, tmp_path):
        """The situation behind the first MiniLM run: all-zero scores because the judged listings were not searched."""
        corpus = tmp_path / "other.csv"
        _write_sample(corpus, [(f"9{i}", f"unrelated listing number {i} with a pool") for i in range(30)])
        with pytest.raises(SystemExit, match="build_eval_corpus"):
            ev.main(["--encoder", "lsa", "--corpus", str(corpus), "--out", str(tmp_path / "r.json")])

    def test_heavily_unjudged_results_trigger_a_lower_bound_warning(self, tmp_path, capsys):
        corpus = tmp_path / "other.csv"
        _write_sample(corpus, [(f"9{i}", f"unrelated listing number {i} with a pool and a deck") for i in range(40)])
        ev.main(["--encoder", "lsa", "--corpus", str(corpus), "--allow-low-coverage", "--no-whole",
                 "--out", str(tmp_path / "r.json")])
        out = capsys.readouterr().out
        assert "LOWER BOUNDS" in out and "P@10 max" in out

    def test_override_flag_allows_the_run(self, tmp_path):
        corpus = tmp_path / "other.csv"
        _write_sample(corpus, [(f"9{i}", f"unrelated listing number {i} with a pool and a deck") for i in range(40)])
        assert ev.main(["--encoder", "lsa", "--corpus", str(corpus), "--allow-low-coverage", "--no-whole",
                        "--out", str(tmp_path / "r.json")]) == 0

    def test_fetch_sql_has_one_placeholder_per_id_and_no_values(self):
        sql = bec.build_fetch_sql(3)
        assert sql.count("%s") == 3 and "1158" not in sql
        with pytest.raises(ValueError):
            bec.build_fetch_sql(0)

    def test_fetch_remarks_batches_and_skips_empty(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE rets_property (L_ListingID TEXT, L_Remarks TEXT)")
        db.executemany("INSERT INTO rets_property VALUES (?,?)",
                       [("1", "one"), ("2", "  two  "), ("3", ""), ("4", None), ("5", "five")])
        got = bec.fetch_remarks(db, ["1", "2", "3", "4", "5", "6"], qmark=True, batch=2)
        assert got == {"1": "one", "2": "two", "5": "five"}

    def test_injection_text_in_an_id_is_just_a_value(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE rets_property (L_ListingID TEXT, L_Remarks TEXT)")
        db.execute("INSERT INTO rets_property VALUES ('1','one')")
        assert bec.fetch_remarks(db, ["1' OR '1'='1", "1; DROP TABLE rets_property"], qmark=True) == {}
        assert db.execute("SELECT COUNT(*) FROM rets_property").fetchone()[0] == 1

    def test_merge_puts_judged_first_and_does_not_duplicate(self):
        merged = bec.merge_corpus({"j1": "a", "s1": "judged version"}, {"s1": "sample version", "s2": "b"})
        assert list(merged) == ["j1", "s1", "s2"] and merged["s1"] == "judged version"

    @needs_sample
    @needs_judgments
    def test_foreign_sample_is_repaired_by_build_eval_corpus(self, tmp_path, monkeypatch):
        """Simulate the laptop: a sample with none of the judged listings, a database that has everything."""
        import sqlite3
        ids, texts = load_corpus()                      # the corpus that holds every judged listing
        judged = set(bec.judged_ids())
        if not judged <= set(ids):
            pytest.skip("this machine's corpus lacks the judged listings: run scripts/build_eval_corpus.py first")
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE rets_property (L_ListingID TEXT, L_Remarks TEXT)")
        db.executemany("INSERT INTO rets_property VALUES (?,?)", list(zip(ids, texts)))
        foreign = tmp_path / "listing_sample.csv"
        _write_sample(foreign, [(i, t) for i, t in zip(ids, texts) if i not in judged][:600])
        queries = ev.load_judgments()
        assert ev.judged_coverage(queries, load_corpus(foreign)[0])[0] == 0          # reproduces the problem

        import build_schema
        monkeypatch.setattr(build_schema, "connect_or_exit", lambda args: _MySQLShapedConnection(db))
        out = tmp_path / "listing_sample_eval.csv"
        assert bec.main(["--sample", str(foreign), "--out", str(out)]) == 0
        fixed_ids, fixed_texts = load_corpus(out)
        found, total, _ = ev.judged_coverage(queries, fixed_ids)
        assert found == total
        assert len(fixed_ids) == len(set(fixed_ids)) == len(judged) + 600          # judged + the 600 background rows

        res = tmp_path / "r.json"
        assert ev.main(["--encoder", "lsa", "--corpus", str(out), "--no-whole", "--out", str(res)]) == 0
        summary = json.load(open(res))["summary"]
        assert summary["bm25"]["ndcg10"] > 0.3 and summary["semantic"]["ndcg10"] > 0.3     # not the all-zero table

    def test_build_eval_corpus_exit_code_flags_missing_ids(self, tmp_path, monkeypatch):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE rets_property (L_ListingID TEXT, L_Remarks TEXT)")
        db.execute("INSERT INTO rets_property VALUES ('1','one')")
        j = tmp_path / "j.json"
        json.dump({"queries": [{"id": "q", "query": "x", "type": "t", "judgments": {"1": 2, "2": 2, "3": 2}}]}, open(j, "w"))
        import build_schema
        monkeypatch.setattr(build_schema, "connect_or_exit", lambda args: _MySQLShapedConnection(db))
        rc = bec.main(["--judgments", str(j), "--sample", str(tmp_path / "none.csv"), "--out", str(tmp_path / "o.csv")])
        assert rc == 1
        assert load_corpus(tmp_path / "o.csv") == (["1"], ["one"])
