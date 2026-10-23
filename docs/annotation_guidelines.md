# Week 3 Annotation Guidelines and Gold Dataset

**File:** `data/processed/gold_labels.json` (committed; labels, offsets and ids only, no remark text).
**Evaluation code:** `scripts/evaluate_extractor.py`. **Extractor:** `scripts/entity_extractor.py`.

## 1. What is in the gold set

| Part | Listings | Labeled fields | How chosen |
|---|---|---|---|
| Random sample | 250 (150 dev, 100 test) | bedrooms, bathrooms, sqft, price | random draw (fixed seed) from the 1,000-listing sample |
| Price-cue search | 51 (about 40 dev, 11 test) | price only | every other sample listing that contains a `$` amount or a price cue; added because only about 1% of listings state a list price, so the random set holds too few examples |
| Amenity subset | 48 (first 24 dev, first 24 test of the random sample) | taxonomy term ids | manual review, see section 4 |

Each entry stores: split, the SHA-1 of the cleaned text (detects drift if the Week 2 cleaner changes), and for every labeled field either
`null` (no explicit value stated), `{"value": ..., "spans": [[start, end], ...]}` (character offsets into the **cleaned** text; every mention of the same fact is listed), or `{"ambiguous": true}` (excluded from scoring for that field).

Counts after labeling (dev / test): bedrooms 92 / 71 positive, bathrooms 87 / 67, sqft 45 / 40, price 5 / 3. Ambiguous: bedrooms 5 / 0, bathrooms 4 / 0, sqft 2 / 1, price 4 / 3.

## 2. Labeling rules

**The target is the main residence.** A listing may describe a main house plus an ADU, a guest house, per-floor counts, or "additional" rooms. The gold value is the count the remarks state for the whole main residence. If only partial or per-unit counts appear, the gold is `null`.

| Situation | Label |
|---|---|
| "3-bedroom, 2-bath home" | 3 and 2 |
| "3 bedrooms plus an office" | 3 |
| "4.5 baths", "3 1/2 baths", "three-and-a-half baths", "2 full + 1 half" | 4.5, 3.5, 3.5, 2.5 |
| "single-bath" | 1 |
| A count repeated in several places | one value, all mentions listed as spans |
| "two additional bedrooms", "3 bedrooms upstairs", "the main level has 3 bedrooms" | partial: not the total, so `null` unless a total is also stated |
| Main house and ADU or guest house both counted | main house only |
| "Each unit has 2 bedrooms" (multi-unit) | `null` |
| A range ("2- and 3-bedroom plans", "3-4 bedrooms") | `null` |
| Public-record count that the text then contradicts | `ambiguous` |
| Two conflicting totals for the same house | `ambiguous` |
| Another building's or the market's count ("prices for a 4-bedroom in the area") | not labeled |
| Square footage of living space, "nearly 2800", "1400+" | the number |
| Lot, garage, loft, patio, ADU, barn, bonus-room or "of additional outdoor space" areas | not labeled as sqft |
| A square-footage range | `null` |
| Price: "Offered at $X", "price improvement to $X", "currently listed at $X", "LIST PRICE: $X" | X |
| Price ceilings and ranges ("under $X", "offers between $X and $Y") | `ambiguous` |
| HOA, rent, credits, grants, upgrades, appraisals, price per square foot, reduction amounts | not a price |

When the remarks and the structured MLS fields disagree, **the text wins**: the task is extraction from text, and the structured bedroom count is a property-wide total that can include units the main-residence text does not describe.

## 3. Scoring

Per listing and field: gold value `g`, predicted value `p`.
`TP` = `g` exists and `p == g`. `FP` = `p` exists and `p != g` (spurious or wrong value). `FN` = `g` exists and `p != g` (missing or wrong value). A wrong value costs one FP and one FN. "Span hit" additionally checks that the chosen mention overlaps a gold span. All 4 numeric entities are also reported as one micro-average.
Amenities are scored as **sets of taxonomy term ids per listing**; the 95% confidence intervals are bootstrapped over listings.

## 4. How the amenity labels were made

1. For each of the 48 listings, every taxonomy term was located by plain substring search (no word boundaries, no context rules), longest match first.
2. Matches inside longer words ("spa" in "spacious", "den" in "residence", "cabin" in "cabinet") were rejected automatically.
3. Every remaining match was read in context and accepted or rejected. Rejected: negated or hypothetical uses ("putting in a pool", "future ADU", "ideal as a home office"), community-only amenities listed as private features (the pool in "community amenities include a pool"), and false friends ("pool table", "decomposed granite", "describes the community, not the home").
4. Four substring variants were accepted by hand ("decking", "carpeting", "backsplashes", a term glued to the next word).

Scope: this measures the extractor against the **Week 1 vocabulary**. It does not measure how much of the real amenity language the taxonomy misses.

## 5. Known limits of this gold set (read before trusting the scores)

- **One annotator.** All labels were made by one annotator (the AI assistant that also wrote the extractor), and no second annotator checked them, so there is no inter-annotator agreement number. Same-author labels favor the extractor's own rules. Dev scores are the most inflated, because the rules were tuned on dev errors.
- **Mitigations:** candidates were produced by broader, independent patterns than the extractor uses; labels were written before the extractor existed; the test split was scored once after the extractor was frozen; and an independent cross-check against the structured MLS fields is reported.
- **Corrections after the first test scoring** (both are clear annotation errors, found while reading test errors; the extractor was not changed): one square-footage statement I had skipped (`1190103900`), and one amenity (`renovated`, id `cond_260`) that my tool had hidden behind a substring match. First-scored test F1 for numeric entities was 0.945; after the corrections it is 0.948.
- **Price has very few examples** (3 test positives, 8 positives in total). Its F1 is not a reliable estimate.
- **Amenities:** 24 test listings, one vocabulary, one annotator.
- **Recommended next step:** have a second person label 30 test listings blind and compare.
