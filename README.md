# IDX-Exchange-NLP

NLP system for California real estate listings, built as part of the IDX Exchange NLP Engineer internship (12-week curriculum).
The system turns messy MLS listing remarks and free-text search queries into structured, searchable signals.

## Status

| Week | Topic | Status |
|------|-------|--------|
| 0 | Environment setup (Docker, MySQL, Python) | Done |
| 1 | Domain understanding + taxonomy | Done |
| 2 | Text cleaning and normalization | Done |
| 3 | Named entity extraction | Done |
| 4 | Query parser and safe SQL | Done |
| 5 | Semantic search with embeddings | Done (see `docs/week5_report.md`) |
| 6 | Listing signal extraction | Done (see `docs/week6_report.md`) |
| 7 | Query intent classification | Done (see `docs/week7_report.md`) |
| 8 | See curriculum | Next |

## Repository layout

```
config/
  abbreviations.json       abbreviation map (standalone, after-number) + protected acronyms
scripts/
  data_loading.py          pulls a 1,000-listing sample of remarks from MySQL
  taxonomy_builder.py      builds data/processed/taxonomy.json (8 categories)
  check_taxonomy_hits.py   reports how often each taxonomy term appears in the sample
  text_cleaning.py         TextCleaner: unicode, whitespace, markup, numbers, prices, measurements,
                           slashes, abbreviations, plus column profiling
  generate_clean_sample.py cleans the sample and prints a before/after summary
  entity_extractor.py      EntityExtractor: bedrooms, bathrooms, square footage, price, amenities
  baseline_extractor.py    naive starter-style baseline (used only for comparison)
  evaluate_extractor.py    precision/recall/F1 against the gold labels; dev and held-out test
  query_parser.py          QueryParser (free text -> filters) and build_sql (parameterized SQL)
  schema_validator.py      SchemaValidator: unknown cities, out-of-range values, wrong types
  build_schema.py          builds data/processed/schema.json (cities + limits); --from-db for MySQL
  run_query.py             try a search: filters, validation, SQL; --sample (SQLite) or --mysql to run it
  evaluate_queries.py      exact-match accuracy against data/processed/query_gold.json
  semantic_search.py       SemanticSearcher: chunking, sentence embeddings (MiniLM/MPNet or offline LSA), FAISS
  bm25_search.py           BM25Searcher: the keyword baseline
  evaluate_search.py       semantic vs BM25 vs hybrid on the judged queries (P@k, nDCG, MRR, bootstrap)
  benchmark_latency.py     query-encoding and vector-search timing (spec: < 100 ms at 10k listings)
  label_judgments.py       judge results nobody has graded yet (export / import / interactive)
  build_eval_corpus.py     fetches the judged listings from MySQL into listing_sample_eval.csv (git-ignored)
  signal_extractor.py      SignalExtractor: Week 3 entities + taxonomy/pattern signals (condition, financing, location) -> JSON
  process_listings.py      runs SignalExtractor over the whole rets_property table (or the sample) -> listing_signals.jsonl
  evaluate_signals.py      scores the signals (free text) and the structured fields; dev and held-out test
  query_intent.py          QueryIntentClassifier (browsing / researching / high_intent_inquiry + confidence) and understand()
  evaluate_intent.py       accuracy, per-class, calibration, baselines, ablations, CV, robustness; dev and held-out test
notebooks/
  week1_data_exploration.ipynb
  week2_before_after.ipynb
docs/
  week2_profiling_report.md
  week3_error_analysis.md   results, failure patterns, planned fixes
  annotation_guidelines.md  how the gold set was labeled and its limits
  week3_results.json        scores (counts only)
  week4_report.md           results, injection checks, decisions and limits
  week4_results.json        scores (counts only)
  week5_report.md           comparison study, latency, limits, what to run with MiniLM
  week5_results_lsa.json    scores with the offline LSA encoder; week5_results_minilm.json with MiniLM (counts and averages only)
  week5_latency_lsa.json    timings
  signal_guidelines.md      the 33 signals + 2 fields and the labeling rules (written before labeling)
  week6_report.md           results, error patterns, limits, how to run on the full table
  week6_results.json        scores (counts and rates only)
  week6_sample_summary.json signal rates on the 1,000-listing sample (counts only)
  intent_guidelines.md      the three intent labels and tie-break rules (written before labeling)
  week7_report.md           results, baselines, limits of language-based intent inference
  week7_results.json        scores (counts and rates only)
  week7_second_pass.json    blind relabel of 80 queries (same annotator; consistency check only)
  intent_extra_template.csv empty query,label sheet for queries you label yourself
tests/
  test_setup.py            environment smoke test (Week 0)
  test_week1.py            taxonomy, sample data, coverage and labeled-query tests
  test_week2.py            159 cases that specify the TextCleaner rules
  test_week3.py            extractor rules, gold-file integrity, dev quality gate
  test_week4.py            parser cases, validator, SQL builder, injection tests run on SQLite
  test_week5.py            chunking, searchers, BM25 formula, metrics, judgments file, latency helpers
  test_week6.py            every signal fires and every trap does not, HOA/Mello-Roos fields, output schema, pagination, gold file
  test_week7.py            dataset integrity, label-order bug, probabilities, uncertain flag, save/load, parser integration, test-split guard
data/
  raw/                     SQL dumps, NOT committed
  processed/
    taxonomy.json          committed
    sample_queries.json    committed (60 hand-labeled queries)
    gold_labels.json       committed (301 hand-labeled listings: ids, values, offsets; no remark text)
    query_gold.json        committed (60 Week 1 + 43 held-out invented queries with expected filters)
    schema.json            generated by build_schema.py (city names + limits); commit is a team decision
    signal_gold.json       committed (120 hand-labeled listings: ids, split, signal labels; no remark text)
    intent_queries.json    committed (380 invented search queries with intent labels, style, hard flag, group, split)
    intent_stress.json     committed (45 indirect queries written after the classifier was frozen)
    relevance_judgments.json  20 queries x graded listing ids (no text); commit is a team decision (MLS ids)
    listing_sample*.csv    NOT committed (MLS data; includes listing_sample_unjudged.csv from label_judgments.py
                           and listing_sample_signals.csv, the cache evaluate_signals.py fills from MySQL)
    listing_signals.jsonl  NOT committed: Week 6 output for every listing (derived from MLS data); listing_signals_summary.json too
  models/                  embeddings, indexes and intent_model.joblib, NOT committed
docker-compose.yml         MySQL 8.0 container, auto-imports data/raw/*.sql
requirements.txt
```

## Data policy

Only code and small hand-authored files are committed. MLS data (the SQL dumps and any CSV derived from them) stays local and is git-ignored. Notebooks never print raw remark text in committed outputs.

## Setup

Requires Python 3.11+ and Docker (with the Compose plugin).

```bash
git clone https://github.com/<your-username>/IDX-Exchange-NLP.git
cd IDX-Exchange-NLP
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw data/processed data/models
# place california_sold.sql, rets_openhouse.sql and rets_property.sql in data/raw/
docker compose up -d
docker compose logs -f mysql      # wait for the final "ready for connections"
python tests/test_setup.py
```

## Run order

```bash
python scripts/data_loading.py          # Week 1: creates data/processed/listing_sample.csv
python scripts/taxonomy_builder.py      # Week 1: creates data/processed/taxonomy.json
python scripts/generate_clean_sample.py # Week 2: creates data/processed/listing_sample_clean.csv
python scripts/evaluate_extractor.py    # Week 3: scores the extractor on the dev split
python scripts/build_schema.py --from-db  # Week 4: creates data/processed/schema.json from MySQL (valid cities)
python scripts/evaluate_queries.py      # Week 4: scores the query parser on the 60 Week 1 queries
python scripts/run_query.py "3 bed under 700k in Irvine" --mysql   # Week 4: parse, validate, build SQL, run it
python scripts/build_eval_corpus.py                        # Week 5: corpus containing the judged listings (needs MySQL)
python scripts/semantic_search.py build --encoder minilm   # Week 5: embeds the sample, saves data/models/semantic_index/
python scripts/semantic_search.py search "room for horses" --top 5
python scripts/evaluate_search.py --encoder minilm         # Week 5: semantic vs BM25 on the judged queries
python scripts/benchmark_latency.py --encoder minilm       # Week 5: latency check
python scripts/evaluate_signals.py                         # Week 6: signal + structured-field accuracy on the dev split
python scripts/process_listings.py --sample                # Week 6: signals for the 1,000-listing sample (no database)
python scripts/process_listings.py --compare-columns       # Week 6: the ENTIRE rets_property table (MySQL), checked against the DB columns
python scripts/evaluate_intent.py                          # Week 7: intent accuracy on the dev split
python scripts/evaluate_intent.py --compare                # Week 7: baselines and ablations
python scripts/query_intent.py understand "open house in Irvine this weekend"   # Week 7: filters + intent + confidence
pytest tests/ -v
```

Week 5 needs `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install sentence-transformers faiss-cpu scikit-learn numpy`. Use `--encoder lsa` to run everything without downloading a model.

Tests that need `listing_sample.csv` or `schema.json` are skipped on a fresh clone until `data_loading.py` has been run.

## Results so far

**Week 1**
- Taxonomy: 379 terms in 8 categories; about 99% of sampled remarks contain at least one term.
- 60 labeled queries (`browsing`, `researching`, `high_intent_inquiry`), 20 each.

**Week 2**
- `TextCleaner` with 8 normalizers and a 37-entry abbreviation map; protected acronyms (HOA, ADU, HVAC, RV, ...) are never expanded.
- 83% of sampled remarks change; every profiled noise pattern drops to zero; cleaning is idempotent.
- Taxonomy matches rise about 1% after cleaning, with none lost.
- Findings: `docs/week2_profiling_report.md` and `notebooks/week2_before_after.ipynb`.

**Week 3**
- `EntityExtractor` finds bedrooms, bathrooms, square footage, list price and Week 1 amenities. Every numeric mention gets a role (main, partial, secondary unit, other area, bound, ...), and only main mentions are returned.
- Gold set: 250 random listings (150 dev / 100 test) plus 51 listings found by price cue; 48 listings with amenity labels.
- Held-out test F1: bedrooms 0.95, bathrooms 0.95, square footage 0.94, numeric micro 0.95, amenities 0.99 (target 0.85). The naive starter-style baseline scores 0.66 on the numeric entities.
- Caveats: single annotator, price has only 8 positive examples, dev was used for tuning. See `docs/week3_error_analysis.md`.
- The held-out split is scored only with `python scripts/evaluate_extractor.py --split test --final`.

**Week 4**
- `QueryParser` turns a search into filters (city, price range, bed/bath min/max, square feet, amenities, negation) through 32 named patterns; `build_sql` makes a parameterized statement from fixed text plus bound values; `SchemaValidator` rejects unknown cities and out-of-range numbers before any SQL is built.
- 60/60 on the Week 1 queries (90.0% on the first run, before tuning) and 43/43 on 43 fresh queries scored once. Both sets were labeled by the same person who wrote the parser, so treat them as optimistic. See `docs/week4_report.md`.
- Injection checks run the generated SQL on SQLite with hostile city, amenity and free-text input; the SQL text never contains a user value. Not yet run on the MySQL container.
- The held-out queries are scored only with `python scripts/evaluate_queries.py --heldout --final`.

**Week 5**
- `SemanticSearcher` (sentence embeddings, passage chunking, FAISS `IndexFlatIP`, numpy fallback) and a from-scratch `BM25Searcher` return the same hit objects, so they are compared on equal terms. Judgments: 20 queries (10 concept, 5 mixed, 5 lexical), 633 graded query/listing pairs.
- nDCG@10 on 1,260 listings (MiniLM / LSA run on the same judgments): BM25 0.852; semantic 0.728 (MiniLM) and 0.812 (LSA); hybrid 0.853 (MiniLM) and 0.880 (LSA). Semantic alone did not beat BM25 (MiniLM -0.12, interval just excludes 0; LSA within noise); the hybrid ties or slightly beats BM25 but not significantly. MiniLM is weakest on exact-feature queries ("wet bar", "RV parking": 0.51 vs 0.88). Negation ("no HOA") fails for both rankers.
- Latency with MiniLM on CPU: query encoding 11 ms median, whole search 12 ms median / 28 ms at p95, vector search over 10,000 listings about 10 ms at p95 (spec: < 100 ms). The relevance table needs `build_eval_corpus.py` first: `docs/week5_report.md` section 6.
- Caveats: single annotator (an AI assistant, not independently verified), 20 queries, only the top 10 of each system is judged. See `docs/week5_report.md`.

**Week 6**
- `SignalExtractor` turns a listing record into one JSON object: Week 3 entities (bedrooms, bathrooms, sqft, price), feature amenities, and 33 yes/no signals in three families (6 condition, 13 financing/ownership, 14 location) plus two fields (`hoa`: none/low/present, `mello_roos`: none/present) and a flat `keywords` list for indexing (`cond:renovated_updated`, `fin:fha`, `loc:near_schools`, `hoa:none`). Signals combine mapped taxonomy terms with regex patterns, then pass negation, hypothetical and proximity-cue filters. Output has no remark text. `process_listings.py` streams the whole table in id order, in batches, resumable.
- Held-out test (54 listings x 33 signals, scored once): free-text micro F1 0.88 (precision 0.90, recall 0.86; target 0.75); structured fields on the Week 3 gold, 108 test listings: F1 0.95, exact-match accuracy 96% (target 0.90). On dev (tuned on): free text 0.98, structured 0.99.
- Weak spots on test: financing signals (only 9 positives, F1 0.78), views (recall 0.71), near_downtown (0.67), waterfront (0.57). Six financing signals (cash only, 1031, short sale, REO, probate, tenant occupied) have no positives in the 120 labeled listings, so they were audited by hand on the wider sample instead. See `docs/week6_report.md`.
- Caveats: single annotator (an AI assistant) who also saw the texts while writing rules, 120 listings, 4 dev labels corrected after reviewing dev disagreements (listed in the report; test labels untouched). The held-out split is scored only with `python scripts/evaluate_signals.py --split test --final`.

**Week 7**
- `QueryIntentClassifier` (logistic regression on word and character TF-IDF plus nine Week 4 `QueryParser` features) labels a query `browsing`, `researching` or `high_intent_inquiry` with a confidence and an `uncertain` flag (confidence below 0.65 or not a housing query). `understand(query)` returns parser filters, SQL, intent and a suggested handling note; `run_query.py --intent` prints it.
- Dataset: 380 invented queries (228 train / 68 dev / 84 test, near-duplicates kept in one split), guidelines written before labeling. Held-out test (scored once): accuracy 91.7% (95% interval 84.5% to 96.4%; target 80%), macro-F1 91.6%. Dev 92.7%, 5-fold grouped CV 87.9%. Flagged-uncertain queries were right 68.8% of the time vs 97.1% for the rest.
- A rule baseline written from the same guidelines scores 95.2% on test, higher than the model, so the model mostly reproduces the labeling rules. A 45-query stress set written after freezing: 91.1%; all four misses are tour/availability questions phrased without cue words.
- The spec's starter skeleton maps probabilities to labels in a fixed order, but scikit-learn sorts classes alphabetically; this code uses `model.classes_` and has a test for it.
- Caveats: single annotator (an AI assistant) who also built the classifier, invented rather than real queries, 84 test queries. The held-out split is scored only with `python scripts/evaluate_intent.py --split test --final`. See `docs/week7_report.md`.
