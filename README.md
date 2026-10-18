# IDX-Exchange-NLP

NLP system for California real estate listings, built as part of the 12-week IDX Exchange NLP Engineer internship.
The system turns messy MLS listing remarks and free-text search queries into structured, searchable signals.

## Status

| Week | Topic | Status |
|------|-------|--------|
| 0 | Environment setup (Docker, MySQL, Python) | Done |
| 1 | Domain understanding + taxonomy | Done |
| 2 | Text cleaning and normalization | Done |
| 3 | Named entity extraction | Next |

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
notebooks/
  week1_data_exploration.ipynb
  week2_before_after.ipynb
docs/
  week2_profiling_report.md
tests/
  test_setup.py            environment smoke test (Week 0)
  test_week1.py            taxonomy, sample data, coverage and labeled-query tests
  test_week2.py            159 cases that specify the TextCleaner rules
data/
  raw/                     SQL dumps, NOT committed
  processed/
    taxonomy.json          committed
    sample_queries.json    committed (60 hand-labeled queries)
    listing_sample*.csv    NOT committed (MLS data)
docker-compose.yml         MySQL 8.0 container, auto-imports data/raw/*.sql
requirements.txt
```

## Data policy

Only code and small hand-authored files are committed. MLS data (the SQL dumps and any CSV derived from them) stays local and is git-ignored. Notebooks never print raw remark text in committed outputs.

## Setup

Requires Python 3.11+ and Docker (with the Compose plugin).

```bash
git clone https://github.com/ibby-24/IDX-Exchange-NLP.git
cd IDX-Exchange-NLP
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw data/processed data/models
#place california_sold.sql, rets_openhouse.sql and rets_property.sql in data/raw/
docker compose up -d
docker compose logs -f mysql
python tests/test_setup.py
```

## Run order

```bash
python scripts/data_loading.py          # Week 1: creates data/processed/listing_sample.csv
python scripts/taxonomy_builder.py      # Week 1: creates data/processed/taxonomy.json
python scripts/generate_clean_sample.py # Week 2: creates data/processed/listing_sample_clean.csv
pytest tests/ -v
```

Tests that need `listing_sample.csv` are skipped on a fresh clone until `data_loading.py` has been run.

## Results so far

**Week 1**
- Taxonomy: 379 terms in 8 categories; about 99% of sampled remarks contain at least one term.
- 60 labeled queries (`browsing`, `researching`, `high_intent_inquiry`), 20 each.

**Week 2**
- `TextCleaner` with 8 normalizers and a 37-entry abbreviation map; protected acronyms (HOA, ADU, HVAC, RV, ...) are never expanded.
- 83% of sampled remarks change; every profiled noise pattern drops to zero; cleaning is idempotent.
- Taxonomy matches rise about 1% after cleaning, with none lost.
- Findings: `docs/week2_profiling_report.md` and `notebooks/week2_before_after.ipynb`.
