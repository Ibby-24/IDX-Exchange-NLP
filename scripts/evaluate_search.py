"""Week 5: compare semantic search with BM25 keyword search on hand-judged queries.

    python scripts/evaluate_search.py --encoder lsa      # offline stand-in encoder (no downloads)
    python scripts/evaluate_search.py --encoder minilm   # the real thing: needs sentence-transformers (see README)
    python scripts/evaluate_search.py --encoder minilm --chunk-words 0   # embed each remark whole (no chunking)

Data: data/processed/relevance_judgments.json (20 queries, grades 0/1/2 per judged listing) and
data/processed/listing_sample.csv (the 1,000-listing sample; its remark text is read locally, never copied).

Systems compared (all return the same SearchHit objects, all see the same TextCleaner output)
  bm25          keyword search, Okapi BM25                         scripts/bm25_search.py
  semantic      embeddings + FAISS, remarks split into passages    scripts/semantic_search.py
  semantic-whole  the same encoder, each remark embedded as one text (only when chunking is on)
  hybrid-rrf    BM25 and semantic rankings merged by reciprocal rank fusion

Metrics (per query, then averaged over queries)
  P@5, P@10   share of the top 5/10 that are relevant (grade >= --relevant-min, default 1)
  nDCG@10     graded: gain = grade (0/1/2), discounted by log2(rank + 1), divided by the best order
              of the judged listings for that query
  MRR         1 / rank of the first relevant result (0 if none in the top 10)
  unjudged@10 share of the top 10 that has no judgment. Those count as 0, so a high number means the
              score is a lower bound (see label_judgments.py to judge them)
  P@10 max    P@10 if every unjudged result were relevant: the ceiling for that system

The "difference" rows are paired bootstrap 95% intervals over queries (10,000 resamples): if the interval
contains 0, the data do not show that one system is better than the other.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "data" / "processed" / "relevance_judgments.json"
DOCS = ROOT / "docs"
TIE_TOLERANCE = 0.01


# ----------------------------------------------------------------------------- metrics (pure functions)
def precision_at_k(ranked: Sequence, judg: Dict[str, int], k: int, rel_min: int = 1) -> float:
    """Share of the first k results that are relevant. Fewer than k results still divide by k."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return sum(1 for d in list(ranked)[:k] if judg.get(str(d), 0) >= rel_min) / k


def reciprocal_rank(ranked: Sequence, judg: Dict[str, int], k: int = 10, rel_min: int = 1) -> float:
    for r, d in enumerate(list(ranked)[:k], 1):
        if judg.get(str(d), 0) >= rel_min:
            return 1.0 / r
    return 0.0


def p10_upper_bound(ranked: Sequence, judg: Dict[str, int], rel_min: int = 1) -> float:
    """P@10 if every unjudged result in the top 10 turned out to be relevant (the most the score could be)."""
    return sum(1 for d in list(ranked)[:10] if str(d) not in judg or judg[str(d)] >= rel_min) / 10


def dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: Sequence, judg: Dict[str, int], k: int = 10) -> float:
    """Graded nDCG. The ideal order is the best k grades among ALL judged listings of the query."""
    ideal = sorted(judg.values(), reverse=True)[:k]
    best = dcg(ideal)
    if best == 0:
        return 0.0
    return dcg([judg.get(str(d), 0) for d in list(ranked)[:k]]) / best


def unjudged_at_k(ranked: Sequence, judg: Dict[str, int], k: int = 10) -> float:
    top = list(ranked)[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if str(d) not in judg) / len(top)


def reciprocal_rank_fusion(rankings: Sequence[Sequence], k: int = 60) -> List:
    """Merge several best-first id lists. score(d) = sum over lists of 1 / (k + rank). Ties break by first list."""
    score: Dict = {}
    first_seen: Dict = {}
    for lst in rankings:
        for r, d in enumerate(lst, 1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
            first_seen.setdefault(d, len(first_seen))
    return sorted(score, key=lambda d: (-score[d], first_seen[d]))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0):
    """Mean of (a - b) over queries with a 95% percentile interval from resampling queries."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need two equally long, non-empty score lists")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def win_tie_loss(a: Sequence[float], b: Sequence[float], tol: float = TIE_TOLERANCE):
    w = sum(1 for x, y in zip(a, b) if x - y > tol)
    l = sum(1 for x, y in zip(a, b) if y - x > tol)
    return w, len(a) - w - l, l


# ----------------------------------------------------------------------------- data
def load_judgments(path: Path = JUDGMENTS) -> List[dict]:
    if not Path(path).exists():
        sys.exit(f"{path} not found.")
    with open(path) as f:
        data = json.load(f)
    queries = data["queries"]
    for q in queries:
        q["judgments"] = {str(k): int(v) for k, v in q["judgments"].items()}
    return queries


def evaluate_system(search_fn, queries: List[dict], top_k: int = 10, rel_min: int = 1):
    """search_fn(query_text) -> best-first list of listing ids. Returns per-query metric lists + rankings."""
    out = {"p5": [], "p10": [], "ndcg10": [], "mrr": [], "unjudged10": [], "strict_p10": [], "p10_max": []}
    rankings = {}
    for q in queries:
        ranked = [str(d) for d in search_fn(q["query"])][:top_k]
        rankings[q["id"]] = ranked
        j = q["judgments"]
        out["p5"].append(precision_at_k(ranked, j, 5, rel_min))
        out["p10"].append(precision_at_k(ranked, j, 10, rel_min))
        out["strict_p10"].append(precision_at_k(ranked, j, 10, 2))
        out["ndcg10"].append(ndcg_at_k(ranked, j, 10))
        out["mrr"].append(reciprocal_rank(ranked, j, 10, rel_min))
        out["unjudged10"].append(unjudged_at_k(ranked, j, 10))
        out["p10_max"].append(p10_upper_bound(ranked, j, rel_min))
    return out, rankings


def judged_coverage(queries: List[dict], corpus_ids: Sequence) -> tuple:
    """(judged listings found in the corpus, judged listings in total, ids missing) over all queries."""
    have = {str(i) for i in corpus_ids}
    wanted = {(q["id"], str(k)) for q in queries for k in q["judgments"]}
    missing = sorted(w for w in wanted if w[1] not in have)
    return len(wanted) - len(missing), len(wanted), missing


MIN_COVERAGE = 0.90
UNJUDGED_WARN = 0.25


def mean(xs):
    return float(sum(xs) / len(xs)) if len(xs) else 0.0


# ----------------------------------------------------------------------------- main
def check_coverage(args, ids):
    """Stop, instead of printing meaningless zeros, when the judged listings are not in the searched corpus."""
    queries = load_judgments(getattr(args, "judgments", JUDGMENTS))
    found, total, _ = judged_coverage(queries, ids)
    if total and found / total < MIN_COVERAGE:
        sys.exit(
            f"STOP: only {found} of {total} judged listings ({found / total:.0%}) are in the corpus being searched.\n"
            "The relevance judgments refer to specific listings; if they are not in the corpus every score is 0 and\n"
            "means nothing. Build a corpus that contains them (needs the MySQL container running):\n"
            "    python scripts/build_eval_corpus.py\n"
            "then run this command again. (--allow-low-coverage overrides this check.)")


def build_systems(args):
    from bm25_search import BM25Searcher
    from semantic_search import SemanticSearcher, load_corpus, make_encoder

    ids, texts = load_corpus(getattr(args, "corpus", None))
    if not getattr(args, "allow_low_coverage", False):
        check_coverage(args, ids)
    t0 = time.time()
    bm25 = BM25Searcher().build_index(texts, ids)
    systems = {"bm25": lambda q: [h.listing_id for h in bm25.search(q, 50)]}
    found, total, _ = judged_coverage(load_judgments(getattr(args, "judgments", JUDGMENTS)), ids)
    info = {"listings": len(ids), "judged_in_corpus": [found, total], "bm25_build_s": round(time.time() - t0, 2)}

    chunk = args.chunk_words or None
    t0 = time.time()
    sem = SemanticSearcher(make_encoder(args.encoder), chunk_words=chunk, chunk_overlap=args.overlap if chunk else 0)
    sem.build_index(texts, ids)
    info["semantic_build_s"] = round(time.time() - t0, 2)
    info["semantic"] = sem.stats()
    systems["semantic"] = lambda q: [h.listing_id for h in sem.search(q, 50)]

    if chunk and not args.no_whole:
        whole = SemanticSearcher(make_encoder(args.encoder), chunk_words=None)
        whole.build_index(texts, ids)
        info["semantic_whole"] = whole.stats()
        systems["semantic-whole"] = lambda q: [h.listing_id for h in whole.search(q, 50)]

    systems["hybrid-rrf"] = lambda q: reciprocal_rank_fusion(
        [[h.listing_id for h in bm25.search(q, 50)], [h.listing_id for h in sem.search(q, 50)]])
    return systems, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="lsa", help="lsa | minilm | mpnet | any sentence-transformers model")
    ap.add_argument("--chunk-words", type=int, default=100, help="passage size in words; 0 = whole remarks")
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--no-whole", action="store_true", help="skip the unchunked comparison system")
    ap.add_argument("--relevant-min", type=int, default=1, choices=[1, 2],
                    help="grade that counts as relevant for P@k and MRR (1 = partly or fully, 2 = fully)")
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="CSV with L_ListingID and remarks; default listing_sample_eval.csv if present, else listing_sample.csv")
    ap.add_argument("--allow-low-coverage", action="store_true", help="skip the judged-listings-present check")
    ap.add_argument("--out", type=Path, default=None, help="default: docs/week5_results_<encoder>.json")
    ap.add_argument("--errors", action="store_true", help="print, per query, the top-3 ids of every system")
    args = ap.parse_args(argv)

    queries = load_judgments(args.judgments)
    systems, info = build_systems(args)

    results, rankings = {}, {}
    for name, fn in systems.items():
        results[name], rankings[name] = evaluate_system(fn, queries, 10, args.relevant_min)

    enc_label = info["semantic"]["encoder"]
    found, total = info["judged_in_corpus"]
    print(f"\njudged listings present in the corpus: {found}/{total}")
    print(f"encoder: {enc_label}   backend: {info['semantic']['backend']}   chunking: "
          f"{info['semantic']['chunk_words'] or 'off'}   queries: {len(queries)}   listings: {info['listings']}")
    print(f"relevant = grade >= {args.relevant_min}\n")
    head = f"{'system':<16}{'P@5':>7}{'P@10':>7}{'nDCG@10':>9}{'MRR':>7}{'unjudged@10':>13}{'P@10 max':>10}"
    print(head)
    print("-" * len(head))
    for name, m in results.items():
        print(f"{name:<16}{mean(m['p5']):>7.3f}{mean(m['p10']):>7.3f}{mean(m['ndcg10']):>9.3f}"
              f"{mean(m['mrr']):>7.3f}{mean(m['unjudged10']):>13.3f}{mean(m['p10_max']):>10.3f}")
    heavy = [n for n, m in results.items() if mean(m["unjudged10"]) > UNJUDGED_WARN]
    if heavy:
        print(f"\nWARNING: more than {UNJUDGED_WARN:.0%} of the top 10 has no judgment for: {', '.join(heavy)}.")
        print("  Unjudged results count as 0, so these scores are LOWER BOUNDS ('P@10 max' is the ceiling if every")
        print("  unjudged result were relevant). The differences below are not a fair comparison until those results")
        print("  are graded: python scripts/label_judgments.py export / import.")

    comparisons = {}
    print("\nnDCG@10 difference (paired bootstrap over queries, 95% interval; wins/ties/losses with tolerance "
          f"{TIE_TOLERANCE})")
    pairs = [("semantic", "bm25"), ("hybrid-rrf", "bm25"), ("hybrid-rrf", "semantic")]
    if "semantic-whole" in results:
        pairs.append(("semantic", "semantic-whole"))
    for a, b in pairs:
        d, lo, hi = paired_bootstrap(results[a]["ndcg10"], results[b]["ndcg10"])
        w, t, l = win_tie_loss(results[a]["ndcg10"], results[b]["ndcg10"])
        comparisons[f"{a} - {b}"] = {"mean_diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                     "wins": w, "ties": t, "losses": l,
                                     "significant": bool(lo > 0 or hi < 0)}
        flag = "interval excludes 0" if (lo > 0 or hi < 0) else "interval contains 0"
        print(f"  {a:>14} - {b:<14} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  W/T/L {w}/{t}/{l}   {flag}")

    types = sorted({q["type"] for q in queries})
    by_type = {}
    print("\nmean nDCG@10 by query type")
    print(f"{'type':<10}{'n':>3}" + "".join(f"{n:>16}" for n in results))
    for ty in types:
        idx = [i for i, q in enumerate(queries) if q["type"] == ty]
        row = {n: round(mean([results[n]["ndcg10"][i] for i in idx]), 3) for n in results}
        by_type[ty] = {"n": len(idx), **row}
        print(f"{ty:<10}{len(idx):>3}" + "".join(f"{row[n]:>16.3f}" for n in results))

    per_query = []
    for i, q in enumerate(queries):
        per_query.append({"id": q["id"], "type": q["type"], "query": q["query"],
                          **{f"ndcg10_{n}": round(results[n]["ndcg10"][i], 3) for n in results}})
    if args.errors:
        print("\nper query nDCG@10")
        for row in per_query:
            print(f"  {row['id']} [{row['type']:<7}] " + "  ".join(f"{n}={row[f'ndcg10_{n}']:.2f}" for n in results)
                  + f"   {row['query']}")

    summary = {n: {k: round(mean(v), 4) for k, v in m.items()} for n, m in results.items()}
    out = args.out or (DOCS / f"week5_results_{args.encoder.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"encoder": enc_label, "relevant_min": args.relevant_min, "queries": len(queries),
                   "info": info, "summary": summary, "comparisons": comparisons, "by_type": by_type,
                   "per_query": per_query}, f, indent=1)
    print(f"\nwrote {out}")
    print("Reminder: results are only as good as the judgments (single annotator, pooled). "
          "Unjudged results count as 0; see label_judgments.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
