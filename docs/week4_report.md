# Week 4 Report: Query Parser and Safe SQL

**Code:** `scripts/query_parser.py` (parser + SQL builder), `scripts/schema_validator.py`, `scripts/build_schema.py`, `scripts/run_query.py` (try it), `scripts/evaluate_queries.py` (scoring).
**Data:** `data/processed/query_gold.json` (expected filters, written before the parser existed), `data/processed/schema.json` (generated; city list and limits). Raw numbers: `docs/week4_results.json`.
**Privacy:** every query in this repo is invented or from the Week 1 sample queries. No listing text appears in this report.

## 1. What it does

`"3 bed under 700k in Irvine"` becomes

```
filters : {city: Irvine, price_max: 700000, bedrooms: 3}
sql     : SELECT ... FROM rets_property WHERE L_City = %s AND L_SystemPrice <= %s AND L_Keyword2 = %s ORDER BY L_SystemPrice ASC LIMIT %s
params  : ['Irvine', 700000, 3, 50]
```

Pipeline: Week 2 `TextCleaner` -> parser (32 named patterns) -> `SchemaValidator` -> `build_sql`. SQL is only built when validation passes.

| Filter | Examples it understands |
|---|---|
| city / cities | `in Irvine`, `Corona or Riverside`, `irvine` (any case), unknown capitalised places are kept so the validator can reject them |
| price_min / price_max | `under 700k`, `$1.2M max`, `between 600 and 800k`, `400-600k`, `over $2 million`, `around 600k` (+/-10%), `$800k budget` |
| bedrooms (+ `_min`, `_max`) | `3 bed`, `3br`, `four bedroom`, `4+`, `at least 3`, `3 or more`, `2 to 3`, `more than 2`, `fewer than 4`, `3/2` |
| bathrooms (+ `_min`, `_max`) | same shapes, halves allowed (`2.5 bath`) |
| sqft_min / sqft_max | `2000 sqft or more`, `under 1200 square feet`, `1500-2000 sq ft` |
| amenities / exclude_amenities | Week 1 taxonomy terms: `with a pool`, `no HOA`, `low HOA fees`, `without a pool or spa`, `not a fixer upper` |

## 2. Results

| Set | Queries | Exactly right | Note |
|---|---|---|---|
| Week 1 sample queries (tuning set) | 60 | **60 (100%)** | First run, before any tuning: 54/60 (90.0%). Three fixes (below) took it to 60/60 |
| Held-out queries (scored once) | 43 | **43 (100%)** | Scored after the parser was frozen; includes numbers, negation, ranges, hostile text, nonsense |

"Exactly right" means every filter matches the hand-written expectation (amenity lists compare as sets). Spec target: 90%+ on the Week 1 queries: met.

The three fixes after the first Week 1 run, all on amenity wording:
- "communities" did not match the term "community" (the matcher only adds a plain `s`).
- "mountain view" (singular) did not match "mountain views".
- "low HOA fees" was swallowed by the longer, ignored term "hoa fees", so `low hoa` was lost.

**How much to trust these numbers**
- Only 7 of the 32 patterns occur in the Week 1 queries (mostly city, one price shape, a few counts, amenities). That score says little about price, bedroom and square-foot handling. The 89 dev cases in `tests/test_week4.py` and the held-out set are what cover those.
- The same author (the AI assistant) wrote the expected filters and the parser. The expectations were written first, but the held-out queries are invented, not real user text. Real searches will be messier; expect a lower score.
- Four held-out queries were also tried informally during development (they passed without changes); the dev cases were then rewritten so the sets do not overlap (a test enforces this).
- After the held-out score, two small gaps were fixed: `between 600 and 800k` (the `k` belongs to both numbers) and `between 2 and 3 bedrooms`. Neither touches a held-out query; re-running confirmed 43/43.
- An informal probe of 20 messier queries found these gaps, listed in section 5: spelled-out prices ("six hundred thousand"), vague words ("cheap", "big yard"), "studio".

## 3. Tests

`tests/test_week4.py`: 256 tests (the whole repo: 535, all passing).
- 89 parse cases across every pattern (a test fails if any of the 32 patterns is never exercised), plus empty, `None`, 100,000-character and non-text input.
- 25+ validator tests: unknown city with suggestion, out-of-range price/bedrooms/bathrooms/sqft, inclusive boundaries, min > max, wrong types (text, `True`, NaN, infinity), unknown filter names, empty query, limits read from `schema.json`.
- SQL builder tests: exact statement for the spec example, placeholder count equals parameter count, no values in the SQL text, no `SELECT *`, whitelisted columns.
- Injection tests (section 4), and 14 end-to-end queries run on a small table.
- Mutation check: about 20 deliberate bugs (wrong comparison, regex without word boundaries, negation off, value pasted into SQL, validator accepting `True`, ...). One survived (the "more than N ... and M" rule); a test now covers it, and all are caught.

## 4. SQL-injection protection

Verified by executing the generated SQL, not by reading it.
- The statement text is built only from fixed fragments. No user-influenced value is ever put in the text. Tests assert that `build_sql({"city": payload})` produces byte-identical SQL for 18 hostile payloads (`'; DROP TABLE rets_property; --`, `' OR '1'='1`, `UNION SELECT ...`, backslashes, NUL bytes, comment markers, unicode) and for 300 random strings.
- Values travel as bound parameters. On a SQLite table with `rets_property`'s column names, hostile city and amenity values return zero rows, the table keeps all its rows, and stacked statements cannot run.
- Numbers must be real numbers (not text, not `True`, not NaN); filter names, columns and the row limit are checked against whitelists; at most 10 amenities and 60 characters per amenity.
- Amenity text is converted to a regex character by character (letters and digits copied, spaces/hyphens become `[ -]`, everything else escaped), so `.*`, `%`, `_`, `[a-z]` match nothing special.
- Free-text queries containing injection attempts (`... in Irvine' OR '1'='1`) parse to the same SQL and parameters as the clean query.

Limit of this check: the engine used in tests is SQLite with `REGEXP_LIKE` registered to mimic MySQL 8. The design (bound parameters, fixed text) is the same on MySQL, but the SQL has not been run against the real MySQL container from here. Run `python scripts/run_query.py "..." --mysql` once to confirm.

## 5. Decisions and limits (read before trusting results)

1. **Amenities search the free-text remarks**, as whole words (`REGEXP_LIKE`). `LIKE '%spa%'` would match "spacious" and `'%pool%'` "carpool". The flag columns (`PoolPrivateYN`, ...) exist, but their value format is unverified, so they are not used yet.
2. **Bare numbers.** `650k` with no under/over is read as a budget (`price_max`) and reported in `ignored`. A bare `2400 sqft` sets no filter (also reported): it could mean "about" or "at least". `around` means +/-10% for price and square feet.
3. **Bedrooms are exact** (`3 bed` -> `= 3`) as in the spec example; `3+`/`at least 3` give a minimum. Bathrooms in the sample are all whole numbers, so `1.5 bath` (exact) will match nothing; `2.5 bathrooms or more` works.
4. **Validator limits** are the spec's (price 100,000 to 10,000,000; bedrooms 1 to 10), plus bathrooms 1 to 10 and square feet 200 to 50,000 (my additions). In the 1,000-listing sample 13 listings are above $10M, 6 have 0 bedrooms and 1 has more than 10, so searches for those are rejected. The limits live in `schema.json` and can be changed without touching code.
5. **Cities.** The valid list is built from the data (`build_schema.py`). The sample has 353 cities, but the database has more: build it with `--from-db`. City names that are ordinary words (Nice, Orange, Vista, ...) only count after `in`/`near`/`around`, and "Orange County" is not the city. A lowercase unknown city is not detected ("homes in springfield" returns no city filter, no error). The MLS data names Big Bear three ways ("Big Bear", "Big Bear City", "Big Bear Lake"; 2, 9 and 4 sample listings), so a search for one spelling misses the others.
6. **Not understood (silently ignored):** spelled-out prices, vague words ("cheap", "spacious", "big yard"), "studio", counties, "near X" distances. A query with no usable filter is rejected by the validator rather than returning the whole table.
7. **Ordering:** results are sorted by price ascending and capped at 50 rows (max 500).

## 6. Spec checklist

| Requirement | Status |
|---|---|
| QueryParser, 20+ query patterns | 32 named patterns, all exercised by tests |
| Safe SQL generator, parameterized, no string concatenation | `build_sql`; fixed text + bound parameters |
| Price ranges, bed/bath min/max, amenities, negation | yes (section 1) |
| Test suite with 50+ query examples | 89 parse cases + 103 gold queries |
| 90%+ on the Week 1 sample queries | 100% (54/60 = 90.0% on the first run) |
| SQL-injection protection verified | section 4 |
| Schema validation: invalid cities, out-of-range values | `SchemaValidator.validate_query(filters) -> (bool, errors)` |

## 7. Next

- Week 5 can reuse `QueryParser.search()` as the front door; ranking/semantic matching would replace the plain remark-text match for amenities.
- Collect 30 to 50 real (or realistic, independently written) queries and label them: that is the only way to get an honest accuracy number.
