# Week 5 Report: Semantic Search with Embeddings

**Code:** `scripts/semantic_search.py` (SemanticSearcher, encoders, chunking, FAISS), `scripts/bm25_search.py` (keyword baseline), `scripts/evaluate_search.py` (the comparison), `scripts/benchmark_latency.py` (the 100 ms check), `scripts/label_judgments.py` (judge unjudged results).
**Data:** `data/processed/relevance_judgments.json` (20 queries, 633 hand-graded query/listing pairs: ids and grades only). Raw numbers: `docs/week5_results_minilm.json`, `docs/week5_results_lsa.json`, `docs/week5_latency_*.json`.
**Privacy:** no remark text appears in this report, in the judgments file or in a saved index (the index holds vectors and listing ids only).

> **Status.** The MiniLM and LSA comparisons below were run on the same corpus (1,260 listings) against the same 633 judgments, so the rows are directly comparable. The earlier sandbox-only LSA table (1,000 listings, 305 judgments) is superseded and was removed.

## 1. What was built

| Piece | What it does |
|---|---|
| `SemanticSearcher` | Cleans text with the Week 2 `TextCleaner`, splits long remarks into passages, encodes every passage, L2-normalizes the vectors, stores them in a FAISS `IndexFlatIP`, and returns the top listings (a listing's score is its best passage's score). Falls back to a numpy matrix product when FAISS is not installed. Saves and loads vectors + ids + settings, never remark text. |
| Encoders | `SentenceTransformerEncoder` (`all-MiniLM-L6-v2`, 384 dims, or `all-mpnet-base-v2`, 768 dims) and `LSAEncoder` (offline stand-in). Same interface, chosen with `--encoder`. |
| Chunking | Passages of about 100 words (20 overlap) on sentence boundaries. MiniLM reads at most 256 word pieces and silently drops the rest. The sample's remarks have a median of about 1,260 characters (max 4,000), so without chunking the end of many remarks is never seen. |
| `BM25Searcher` | Okapi BM25 written from scratch (k1 1.5, b 0.75), same cleaning, stopwords and a tiny plural stemmer. The formula is tested against a hand calculation. |
| `evaluate_search.py` | P@5, P@10, nDCG@10, MRR, unjudged@10, paired bootstrap confidence intervals over queries, win/tie/loss, results by query type. Also scores a hybrid (BM25 + semantic merged by reciprocal rank fusion) and the unchunked variant. |
| `benchmark_latency.py` | Times the query encoding and the vector search separately, at 1k, 10k and 50k listings, with FAISS and with numpy. |

## 2. How relevance was judged

- 20 queries written **before** any search was run: 10 *concept* (meaning, few keywords: "quiet peaceful neighborhood good for raising a young family"), 5 *mixed* ("bright open kitchen great for cooking"), 5 *lexical* (exact feature: "wet bar", "RV parking", "no HOA").
- For each query a pool of 10 to 19 candidate listings was shuffled and read in full; each got a grade: **2** clearly satisfies, **1** partly, **0** no. 305 judgments in the first round (204 / 65 / 36 for grades 2 / 1 / 0).
- Second round: the top-10 results of every system that still had no grade were exported (`label_judgments.py export`, no system names shown) and graded against the query text only: 328 more pairs (209 / 66 / 53). Total 633 (413 / 131 / 89).
- Metrics treat grade >= 1 as relevant (`--relevant-min 2` for the strict version).

**Limits that matter**
1. One annotator (the AI assistant), no second opinion. Grades are consistent, not independently verified. A human spot-check of a sample of pairs is the obvious next step.
2. After the second round every top-10 result of BM25, semantic (chunked), hybrid is graded (unjudged@10 = 0 for LSA and MiniLM). The MiniLM *whole-remark* variant still has 50% unjudged results, so its row is a lower bound and is marked as such.
3. Only the top 10 of these systems is judged. A relevant listing that none of them ranks highly stays unknown, so the numbers say nothing about recall.
4. P@10 is high for every system (0.80 to 0.94): on a 1,260-listing corpus, with queries that mostly name concrete features, ranking is easy. Compare systems with each other, not with an absolute target.
5. 20 queries is small; the confidence intervals below show how small. Differences whose interval touches 0 are not findings.

## 3. Results (1,260 listings, 20 queries, 633 judgments, relevant = grade >= 1)

| System | P@5 | P@10 | nDCG@10 | MRR | unjudged@10 |
|---|---|---|---|---|---|
| BM25 | **0.950** | 0.920 | 0.852 | **0.975** | 0.000 |
| Semantic, MiniLM (chunked) | 0.860 | 0.795 | 0.728 | 0.950 | 0.000 |
| Semantic, MiniLM (whole remark) * | 0.510 | 0.420 | 0.422 | 0.746 | 0.505 |
| Hybrid MiniLM + BM25 (RRF) | 0.930 | 0.910 | **0.853** | 0.967 | 0.000 |
| Semantic, LSA (chunked) | 0.890 | 0.895 | 0.812 | 0.880 | 0.000 |
| Semantic, LSA (whole remark) | 0.860 | 0.800 | 0.770 | 0.938 | 0.130 |
| Hybrid LSA + BM25 (RRF) | 0.940 | **0.935** | **0.880** | 0.960 | 0.000 |

\* lower bound: half of its top 10 has no judgment and counts as 0.

Paired bootstrap on nDCG@10 (10,000 resamples of queries; wins / ties / losses with tolerance 0.01):

| Comparison | MiniLM | LSA |
|---|---|---|
| semantic - BM25 | -0.124 [-0.241, -0.008], 5/2/13 | -0.040 [-0.131, +0.053], 8/4/8 |
| hybrid - BM25 | +0.001 [-0.050, +0.048], 10/3/7 | +0.028 [-0.013, +0.074], 8/8/4 |
| hybrid - semantic | +0.125 [+0.045, +0.209], 15/1/4 | +0.068 [+0.012, +0.128], 8/7/5 |
| chunked - whole remark | +0.306 [+0.217, +0.402], 19/0/1 (lower bound, see \*) | +0.042 [-0.053, +0.148], 11/1/8 |

**By query type (mean nDCG@10)**

| Type | n | BM25 | MiniLM | MiniLM hybrid | LSA | LSA hybrid |
|---|---|---|---|---|---|---|
| concept | 10 | 0.871 | 0.798 | **0.910** | 0.833 | 0.896 |
| mixed | 5 | 0.789 | 0.809 | **0.839** | **0.924** | 0.902 |
| lexical | 5 | **0.877** | 0.506 | 0.754 | 0.659 | 0.828 |

**What this does and does not show**
- **Semantic search alone did not beat BM25 on this set.** MiniLM is 0.12 nDCG below BM25 (the interval only just excludes 0, and with 20 queries I would call it suggestive, not settled). LSA is within noise of BM25.
- **The hybrid is the safe choice.** With MiniLM it ties BM25 overall (+0.001) and is clearly better than MiniLM alone; it has the best concept and mixed scores in the MiniLM run. With LSA it has the best overall nDCG (0.880), but its gap to BM25 is inside the noise. So: "hybrid never loses to either side", not "hybrid wins".
- **MiniLM is weakest on exact feature names** (lexical 0.506 vs 0.877 for BM25): queries like "wet bar", "RV parking", "tankless water heater" want the literal phrase, and a dense model returns listings about the same topic without it. This is the expected weakness of dense retrieval.
- **MiniLM did not win the paraphrase queries clearly either** (mixed 0.809 vs BM25 0.789; concept 0.798 vs 0.871). One reason may be that this corpus is a few hundred long, descriptive remarks where keyword overlap is already strong; I did not test that.
- **The LSA stand-in scored higher than MiniLM** (0.812 vs 0.728). LSA is fitted on this corpus, so it knows its vocabulary; MiniLM is general purpose. Do not read this as "LSA is better than transformers".
- **Chunking:** with LSA no measurable difference. With MiniLM the +0.306 is inflated by the unjudged half of the whole-remark row; the real gap is smaller. It is consistent with the 256-word-piece argument, not proof.
- **Negation fails for both** ("no HOA", "ready to move in no work needed"): neither ranker understands "no". A structured filter (Week 4 parser) is the right tool for negation.

## 4. Latency (spec: < 100 ms for 10,000 listings)

Sandbox with 2 CPU cores, 384 dimensions, about 4 passages per listing, top 10, 200 runs. Times in milliseconds.

| Listings | Passages | FAISS flat p50 / p95 | numpy p50 / p95 |
|---|---|---|---|
| 1,000 | 4,000 | 0.29 / 0.34 | 0.16 / 0.24 |
| **10,000** | 40,000 | **4.2 / 5.3** | 1.4 / 1.7 |
| 50,000 | 200,000 | 29.9 / 42.9 | 14.2 / 19.5 |

- The vector search for 10,000 listings is about 5 ms, 20 times under the limit. The other half of a search is encoding the query: with the LSA encoder the whole search on the 1,000-listing sample takes 23.5 ms median, 28 ms at the 95th percentile (almost all of it query encoding). **The MiniLM encode time is measured on the laptop, below.**
- On the 2-core sandbox the plain numpy matrix product was faster than FAISS for one query at a time (FAISS's multi-threading costs more than it saves on 40,000 vectors); on the laptop the two are about equal. FAISS is the spec's tool and is the default; the point of `IndexFlatIP` is the same exact answer, and its advantages (batches, approximate indexes at around a million vectors) do not apply at this size. Both backends are tested to return identical rankings.
- Index build for the sample: 7 s with LSA.

**Measured with MiniLM on the laptop** (CPU only, 1,000 listings, 2,725 passages; this does not depend on the judgments, so it stands):

| Step | p50 | p95 |
|---|---|---|
| Encode one query (MiniLM) | 11.3 ms | 42.1 ms |
| Whole search (clean + encode + FAISS + roll-up) | 12.1 ms | 28.0 ms |
| Vector search alone, 10,000 listings (40,000 passages) | 7.4 ms | 10.0 ms |
| Index build, 2,725 passages | 73 s | |

The spec (< 100 ms for 10,000 listings) is met: about 10 ms of search plus about 28 ms of encoding at the 95th percentile. On this laptop FAISS and numpy are about equal at 10,000 listings (7.4 vs 6.8 ms median), so the sandbox's "numpy is faster" result was specific to that machine. Building the index for the full database will take proportionally longer (about 27 ms per passage, so 400,000 passages is about 3 hours); start with `--limit 10000`.

## 5. Tests

`tests/test_week5.py`: 152 tests passed, 1 skipped ( the skipped one is the real-MiniLM test that needs `sentence-transformers`).
- Chunking: size limit exact, no words lost, overlap, run-on sentences, bad parameters.
- SemanticSearcher with a deterministic hash encoder (no download): self-retrieval scores 1.0, ranking order, duplicates collapsed to one hit per listing, empty/bad queries, `top_k` validation, FAISS and numpy give the same ranking, save/load round trip, **saved files contain no remark text**.
- BM25: score equals the formula by hand, rarer words count more, diminishing returns, length penalty, stemming limits, stopwords (including the `no` limitation).
- Metrics: P@k, MRR, nDCG against hand-computed values, RRF, bootstrap (reproducible; mixed differences include 0; interval width matches theory), win/tie/loss.
- Judgments file: at least 50 pairs, grades in {0,1,2}, only ids and grades stored, every judged listing exists in the sample, every query has a relevant example.
- End to end on the sample: both systems beat a shuffled ranking by a wide margin, "wet bar" results all contain "wet bar", CLI build/search/evaluate run.
- Mutation check: 16 deliberate bugs (wrong discount, `>` for `>=`, no normalization, wrong idf, stopwords off, 80% instead of 95% interval, ...). Two survived the first test version (chunk size off by 5 words, interval too narrow); tests were added and all 16 are now caught.

## 6. Running it with MiniLM (laptop)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu     # small CPU build, no GPU needed
pip install sentence-transformers faiss-cpu scikit-learn numpy
docker compose up -d                                                    # MySQL must be running for the next line
python scripts/build_eval_corpus.py                                     # makes listing_sample_eval.csv (see below)
python scripts/evaluate_search.py --encoder minilm                      # writes docs/week5_results_minilm.json
python scripts/benchmark_latency.py --encoder minilm                    # the 100 ms check including query encoding
```

**Why `build_eval_corpus.py` is required.** The first MiniLM run printed all zeros (99-100% "unjudged"). Cause: `listing_sample.csv` is a random draw, so the laptop's 1,000 listings were not the 1,000 that were graded, and the judgments pointed at listings that were not being searched. The tell was the passage count (2,725 on the laptop vs 2,730 where the judgments were made). The fix is a corpus that contains every judged listing (fetched from MySQL by id) plus your sample as background. `evaluate_search.py` now stops with an explanation when fewer than 90% of the judged listings are in the corpus, so this cannot silently happen again. Because the background listings differ from the ones used for the LSA numbers in Section 3, re-run the LSA comparison on the same corpus (`--encoder lsa`) so the BM25, LSA and MiniLM rows are directly comparable.

**Closing the unjudged gap.** After `build_eval_corpus.py`, 77% of MiniLM's top-10 results had no grade, which made the first table meaningless (every score a lower bound). `label_judgments.py export` listed the ungraded pairs, they were graded against the query text, and `label_judgments.py import` added them. The result is Section 3. If you change an encoder or chunk size, run `export` again: a new system returns new unjudged listings.

## 7. Decisions and limits

1. **Cosine on unit vectors.** Vectors are L2-normalized, so FAISS inner product equals cosine similarity.
2. **Exact search.** `IndexFlatIP` is exact; no approximate index is justified below about a million vectors.
3. **Embedding all remarks.** `semantic_search.py build --source db --limit N` encodes remarks from MySQL. Memory: vectors are 384 x 4 bytes per passage, about 1.5 KB; at 4 passages per listing, 100,000 listings is about 600 MB. CPU encoding speed is the limit (measure it on the 1,000-listing sample first and extrapolate), so start with `--limit 10000`.
4. **Vectors are derived from MLS text.** `data/models/` must stay git-ignored. The index files are not committed.
5. **`relevance_judgments.json` contains MLS listing ids** (no text), like `gold_labels.json`. Whether to commit it is your decision under the "only code" rule; the evaluation needs it, so without it the results cannot be reproduced by teammates.
6. **The unjudged export contains remark text.** It is named `listing_sample_unjudged.csv` so the existing `listing_sample*.csv` ignore rule covers it. Do not commit it.
7. **Query text is cleaned the same way as remarks.** The Week 2 cleaner expands some abbreviations; "w/" in a query becomes "with" on both sides.
8. **No structured filters here.** City, price and bedroom limits come from the Week 4 parser; combining them with semantic ranking is the natural Week 6 step.

## 8. Spec checklist

| Requirement | Status |
|---|---|
| Embeddings for listing remarks, 384 or 768 dims | Implemented (MiniLM 384, MPNet 768); MiniLM index built and timed on the laptop |
| FAISS index for efficient similarity search | `IndexFlatIP` on normalized vectors, numpy fallback, tested identical |
| SemanticSearcher with query embedding and retrieval | Yes, plus chunking, save/load |
| Comparison study: semantic vs BM25 | Done with MiniLM and the LSA stand-in on the same corpus and judgments (Section 3) |
| Latency < 100 ms for 10k listings | Met with MiniLM on the laptop: about 10 ms search + 28 ms encoding at p95 (Section 4) |
| Relevance evaluation on 50 query-result pairs | 633 judged pairs over 20 queries |

## 9. Next

- Have a person re-grade a random sample of about 40 pairs and compare with the AI grades (agreement rate); that is the weakest link in the study.
- Add more queries, especially more paraphrase-style ones, so the intervals narrow.
- Week 6: combine the parser's structured filters (city, price, bedrooms, negated amenities) with a hybrid ranking, so "no HOA under 800k in Irvine, quiet street for a family" uses SQL for the filters and BM25 + embeddings for "quiet street for a family".
