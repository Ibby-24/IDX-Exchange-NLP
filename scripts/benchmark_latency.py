"""Week 5: how long does one search take? The spec asks for < 100 ms on 10,000 listings.

    python scripts/benchmark_latency.py                       # vector search only, synthetic vectors, 1k..50k listings
    python scripts/benchmark_latency.py --encoder lsa         # + the full query path on the 1,000-listing sample
    python scripts/benchmark_latency.py --encoder minilm      # + the full query path with the real model
    python scripts/benchmark_latency.py --encoder minilm --real-listings 10000 --source db   # real remarks from MySQL

One search has two parts, timed separately:
  1. encode the query  - run the model on ~10 words. Depends on the encoder and the CPU, NOT on how many listings exist.
  2. vector search     - compare the query vector with every stored passage vector (FAISS IndexFlatIP, exact).
                         Grows with the number of passages: listings x passages per listing.
The total is what a user waits. Times are the median (p50), 95th and 99th percentile over many runs;
the first runs are discarded as warm-up (model loading, caches).

Synthetic vectors are random unit vectors. Exact search time depends only on how many vectors there are and how
long they are, not on what the numbers are, so they give the right timing for the search step. They say nothing
about result quality - that is evaluate_search.py's job.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SPEC_LIMIT_MS = 100.0


def percentile(values: List[float], p: float) -> float:
    """Linear-interpolated percentile of a non-empty list (p in 0..100)."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def time_calls(fn: Callable[[int], object], runs: int, warmup: int = 5) -> Dict[str, float]:
    """Call fn(i) warmup + runs times; report milliseconds for the timed ones."""
    for i in range(warmup):
        fn(i)
    ms = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(warmup + i)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return {"p50_ms": round(percentile(ms, 50), 3), "p95_ms": round(percentile(ms, 95), 3),
            "p99_ms": round(percentile(ms, 99), 3), "mean_ms": round(statistics.fmean(ms), 3), "runs": runs}


def random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    from semantic_search import l2_normalize
    return l2_normalize(np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32))


def bench_vector_search(listings: int, passages_per_listing: float, dim: int, top_k: int, runs: int,
                        use_faiss: bool) -> dict:
    """Time SemanticSearcher's search step alone, on synthetic unit vectors."""
    from semantic_search import SemanticSearcher
    n_pass = int(listings * passages_per_listing)
    emb = random_unit_vectors(n_pass, dim)
    owner = np.minimum((np.arange(n_pass) / passages_per_listing).astype(np.int64), listings - 1)
    s = SemanticSearcher(encoder=object(), use_faiss=use_faiss)
    s.ids = list(range(listings))
    s._set_vectors(emb, owner)
    queries = random_unit_vectors(64, dim, seed=1)

    def one(i):
        q = queries[i % len(queries)][None, :]
        scores, idx = s._knn(q, top_k * 4)          # the vector-search step
        best = {}
        for sc, p in zip(scores, idx):               # and the passage -> listing roll-up
            best.setdefault(int(s.owner[p]), float(sc))
        return best

    out = time_calls(one, runs)
    out.update({"listings": listings, "passages": n_pass, "dim": dim, "backend": s.backend})
    return out


def bench_full_path(searcher, queries: List[str], runs: int) -> dict:
    """Time searcher.search(text) end to end (clean + encode + search + roll-up)."""
    def one(i):
        return searcher.search(queries[i % len(queries)], 10)
    out = time_calls(one, runs)
    out.update(searcher.stats())
    return out


def bench_query_encoding(searcher, queries: List[str], runs: int) -> dict:
    def one(i):
        return searcher.encoder.encode([queries[i % len(queries)]])
    return time_calls(one, runs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default=None, help="also time the full path with this encoder: lsa | minilm | mpnet")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000], help="listing counts for the synthetic test")
    ap.add_argument("--passages-per-listing", type=float, default=4.0,
                    help="average passages per listing when chunking is on (the sample gives about 4)")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--chunk-words", type=int, default=100)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--real-listings", type=int, default=0,
                    help="encode this many REAL remarks for the full-path test (needs --encoder; slow on CPU)")
    ap.add_argument("--source", choices=["csv", "db"], default="csv")
    ap.add_argument("--corpus", type=Path, default=None, help="CSV to encode for the full-path test (default: sample)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = {"spec_limit_ms": SPEC_LIMIT_MS, "vector_search": [], "full_path": None}
    print(f"vector search (synthetic unit vectors, dim {args.dim}, {args.passages_per_listing:g} passages per listing, "
          f"top {args.top}, {args.runs} runs)")
    head = f"{'listings':>9}{'passages':>10}{'backend':>16}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}"
    print(head)
    print("-" * len(head))
    for use_faiss in (True, False):
        for n in args.sizes:
            r = bench_vector_search(n, args.passages_per_listing, args.dim, args.top, args.runs, use_faiss)
            report["vector_search"].append(r)
            print(f"{r['listings']:>9}{r['passages']:>10}{r['backend']:>16}{r['p50_ms']:>9.3f}{r['p95_ms']:>9.3f}{r['p99_ms']:>9.3f}")

    if args.encoder:
        from semantic_search import SemanticSearcher, load_corpus, make_encoder
        if args.real_listings and args.source == "db":
            from semantic_search import _rows_from_db
            import os
            ns = argparse.Namespace(host=os.environ.get("DB_HOST", "localhost"), user=os.environ.get("DB_USER", "root"),
                                    password=os.environ.get("DB_PASSWORD", "root"),
                                    database=os.environ.get("DB_NAME", "real_estate"))
            ids, texts = _rows_from_db(args.real_listings, ns)
        else:
            ids, texts = load_corpus(args.corpus)
            if args.real_listings:
                ids, texts = ids[:args.real_listings], texts[:args.real_listings]
        print(f"\nfull query path: encoder {args.encoder}, {len(ids)} real listings, chunking {args.chunk_words or 'off'}")
        s = SemanticSearcher(make_encoder(args.encoder), chunk_words=args.chunk_words or None,
                             chunk_overlap=args.overlap if args.chunk_words else 0)
        t0 = time.time()
        s.build_index(texts, ids)
        build_s = time.time() - t0
        qs = ["quiet street good for a young family", "room for horses", "wet bar", "walk to coffee shops",
              "energy efficient home with solar", "views of the city lights", "no HOA", "separate guest house"]
        enc = bench_query_encoding(s, qs, args.runs)
        full = bench_full_path(s, qs, args.runs)
        full["index_build_s"] = round(build_s, 1)
        full["query_encoding"] = enc
        report["full_path"] = full
        print(f"  index build: {build_s:.1f} s for {full['passages']} passages ({full['backend']})")
        print(f"  encode query only : p50 {enc['p50_ms']:.2f} ms   p95 {enc['p95_ms']:.2f}   p99 {enc['p99_ms']:.2f}")
        print(f"  whole search      : p50 {full['p50_ms']:.2f} ms   p95 {full['p95_ms']:.2f}   p99 {full['p99_ms']:.2f}")
        verdict = "PASS" if full["p95_ms"] < SPEC_LIMIT_MS else "FAIL"
        print(f"  spec (< {SPEC_LIMIT_MS:g} ms, judged on the 95th percentile): {verdict}")

    ten_k = [r for r in report["vector_search"] if r["listings"] == 10000 and r["backend"] != "numpy"]
    if ten_k:
        print(f"\n10,000 listings, vector search only: p95 {ten_k[0]['p95_ms']:.3f} ms "
              f"(limit {SPEC_LIMIT_MS:g} ms). Add the query-encoding time from the full-path line above for the total.")
    out = args.out or (DOCS / f"week5_latency_{args.encoder or 'vectors'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
