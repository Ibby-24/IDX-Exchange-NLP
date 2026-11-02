# Week 6: Listing signal extraction

## 1. What was built

`SignalExtractor` (`scripts/signal_extractor.py`) takes one listing record and returns one JSON object that a search index or a filter can use directly. It reuses the Week 3 `EntityExtractor` for the numbers and the Week 1 taxonomy for amenities, and adds three families of yes/no **signals** (condition, financing and ownership, location) that the taxonomy alone does not capture.

`process_listings.py` runs it over the whole `rets_property` table (id order, batches of 2,000, resumable) and writes `data/processed/listing_signals.jsonl`. `evaluate_signals.py` scores it against hand labels.

Spec checklist

| Spec item | Where |
|---|---|
| `SignalExtractor.__init__(taxonomy, entity_extractor)` and `extract_signals(listing_record)` | `scripts/signal_extractor.py` |
| JSON schema: entities + amenities + keywords | section 2 |
| Process the entire `rets_property` table, save to `processed/` | `scripts/process_listings.py` |
| 90%+ on structured fields | section 5 |
| 75%+ on free text | section 5 |
| Output suitable for search indexing / filtering | `keywords` list and one field per signal family (section 2) |

## 2. Output schema

One line of `listing_signals.jsonl` (invented example, not a real listing):

```json
{
  "listing_id": "1000000001",
  "entities": {"bedrooms": 3, "bathrooms": 2, "sqft": 1850, "price": null},
  "amenities": ["swimming pool", "fireplace"],
  "condition_keywords": ["renovated_updated", "move_in_ready"],
  "financing_terms": ["fha"],
  "location_features": ["gated_community", "near_schools"],
  "fields": {"hoa": "low", "mello_roos": null},
  "keywords": ["cond:renovated_updated", "cond:move_in_ready", "fin:fha",
               "loc:gated_community", "loc:near_schools", "hoa:low"],
  "schema_version": 1
}
```

* `listing_id` is `L_ListingID` as text. `entities` are exactly the Week 3 values (`null` = not stated in the remarks).
* `amenities` are taxonomy terms from the feature categories (interior, outdoor, layout, systems/parking/energy, property type) that survived the Week 3 negation / hypothetical / community filters. The condition, financing and location categories feed the signals instead.
* The three lists hold canonical signal names (section 3), each at most once: a signal is yes/no per listing, not a count.
* `keywords` is the same information flattened to `family:signal` strings, so one inverted index (or a `LIKE` / `JSON_CONTAINS` filter) covers everything: `loc:near_schools AND fin:fha AND hoa:none`.
* No remark text is ever written to the output. `extract_signals(record, evidence=True)` adds the matched phrases for debugging; the batch processor never turns it on.

## 3. Signal inventory (defined before labeling)

Full definitions and the "not this" cases are in `docs/signal_guidelines.md`, written before any listing was labeled and before the extractor existed.

* **Condition (6):** renovated_updated, new_construction, new_components, move_in_ready, well_maintained, needs_work
* **Financing and ownership (13):** seller_financing, assumable_loan, fha, va, cash_only, exchange_1031, short_sale, reo, probate_estate, tenant_occupied, investment_income, seller_credit, motivated_seller
* **Location (14):** gated_community, golf_course, near_schools, near_shopping_dining, freeway_commute, public_transit, walkable, near_beach_water, waterfront, views, quiet_street, parks_open_space, senior_55plus, near_downtown
* **Fields:** `hoa` in {none, low, present}, `mello_roos` in {none, present}; null when the remarks do not say.

Rules shared by every signal: negated, hypothetical or "could be" statements do not count ("no HOA", "could be remodeled", "ideal for a home office"); "vacant" is not a signal; a bare school-district name is not "near schools".

## 4. How extraction works

1. Clean the remark with the Week 2 `TextCleaner` (same as Weeks 3 to 5).
2. **Taxonomy layer:** the Week 3 amenity matcher finds taxonomy terms. A hand-written table (`TAXONOMY_TO_SIGNAL`) maps the condition, financing and location terms to canonical signals ("owner carry" and "seller financing" both become `seller_financing`). The Week 3 filters already drop negated, hypothetical and community-only mentions.
3. **Pattern layer:** about 40 regular expressions (several cover many phrasings each) cover the phrasings the taxonomy lacks ("seller willing to finance", "walk to the beach", "mountain and city views", "bring all offers").
4. **Context filters** on every candidate: negation in the same clause ("no", "not", "without", "never"); hypothetical cues ("could", "potential to", "ready for your updates"); a **proximity cue** for words that only mean something when the place is nearby (shopping, dining, beach, lake, downtown, schools, parks, bare "golf"); and false friends (`subway tile`, `gated garage`, `Bay Area`, `miniature golf`, `open space floor plan`, `school district`).
5. HOA and Mello-Roos are read by separate rules: "no HOA" beats "low HOA" beats a plain HOA mention.

Where the two layers would disagree, the pattern layer wins: terms such as "upgraded", "gated", "waterfront", "open space" and "tenant occupied" are deliberately not mapped from the taxonomy, because their context rules live in the pattern layer.

## 5. Results

### Gold sets

* **Signals:** 120 listings (seed-606 draw from the 1,000-listing sample; 66 dev, 54 test), hand-labeled against the guidelines before the extractor existed: 33 signals each, 3,960 yes/no decisions. Stored as ids, labels and a SHA-1 of the cleaned text (`data/processed/signal_gold.json`; no remark text).
* **Structured fields:** the Week 3 gold (301 listings; 186 dev, 108 test usable here), read through the new pipeline. The Week 3 extractor was not retuned.

### Held-out test (scored once)

| | Target | Result |
|---|---|---|
| Free-text signals, micro F1 (54 listings, 237 gold positives) | 75%+ | **87.7%** (precision 89.8%, recall 85.7%; bootstrap 95% interval for F1 83.2% to 91.6%) |
| Structured fields, F1 (108 listings) | 90%+ | **94.8%** (exact-match accuracy 96.1%) |

By family on test (F1): condition 91.4%, financing 77.8% (only 9 positives), location 86.4%.

Structured fields on test: bedrooms F1 95.2% (exact-match 93.0%), bathrooms 94.7% (95.0%), square footage 93.7% (96.0%), price 100% (only 3 positives). `hoa`: 5 of 5 correct; `mello_roos`: no listing in the gold set mentions it.

Weaker signals on test (F1): waterfront 57%, investment_income 57%, near_downtown 67%, motivated_seller 67%, needs_work 50% (3 positives). The rest of the 33 signals are 83% or higher where they have positives. Per-signal tables are in `docs/week6_results.json`.

### Dev (the half the rules were tuned on)

Free text 97.6% (precision 99.6%, recall 95.5%), structured 99.1% F1. **Do not quote the dev number as accuracy**; it measures how well the rules fit the listings they were written against. The 10-point drop to test is the honest size of that fit.

### What changed while tuning (dev only)

* First run on dev, before any tuning: free-text F1 88.3%, structured 99.1%.
* Two real bugs found by reading dev errors: the proximity and negation regexes were case-sensitive, so "Close to schools" and "(Not probate)" were missed. Fixing them changed more results than any single new pattern.
* Other fixes were general rules: false friends (subway tile, gated garage, Bay Area, miniature golf, open space floor plan), "waterfront" near vs on the water, "upgraded" + single item is not a renovation, "district" alone is not schools, plural forms ("retirement communities"), hyphenated modifiers ("newer whole-house HVAC").
* **Label corrections.** Reviewing the remaining dev disagreements against the written guidelines showed four slips in my own labels on three dev listings (two signals the text clearly states were missing on one listing, one named view missing on another, one generic "gorgeous views" labeled as a named view on a third). I corrected those four labels and nothing else. Disagreements where the guideline was ambiguous were left as they were (the extractor counts as wrong). Test labels were not reviewed or changed, so the test number is, if anything, slightly pessimistic relative to dev.
* The test split was run once, after the extractor was frozen.

### Rare financing signals

Six signals (cash_only, exchange_1031, short_sale, reo, probate_estate, tenant_occupied) have **no positives** in the 120 gold listings, and seller_financing and assumable_loan have 3 and 2. Their recall is therefore unmeasured. To check precision instead, I read the hits on the other 880 sample listings: 47 distinct listings were flagged for the rare financing signals (cash only 2, 1031 1, short sale 2, REO 1, probate 3, tenant occupied 8, seller financing 3, assumable 2, FHA 7, VA 8, seller credit 10). I read about 45 of them; one was wrong ("Normal sale (Not probate)"), caused by the case-sensitivity bug above and now covered by a test. Recall for these signals is not known.

### Full-table run

On the 1,000-listing sample (`docs/week6_sample_summary.json`): 74 listings/s, so 13.5 s. At that rate the full table takes (rows / 74) seconds; for example 100,000 rows is about 23 minutes. Signal rates on the sample: any location signal 84%, any condition signal 69%, any financing signal 26%. Most common: near_shopping_dining 49%, renovated_updated 47%, parks_open_space 37%, new_components 29%, freeway_commute 29%. Rarest: reo 0.1%, exchange_1031 0.1%, cash_only 0.2%, short_sale 0.2%.

Cross-check with the database columns (`process_listings.py --compare-columns`), sample only: when both the remarks and the column give a value, bedrooms agree 89.4% (n=688) and bathrooms 91.0% (n=608, counting 2.5 baths = 3 as agreement, as noted in Week 3). This is a consistency check, not accuracy: the column and the remarks can both be right (the remark may describe the main house only) or the column may be stale. The extractor finds no bedroom count in 31% of remarks that have one in the column, because the remarks simply do not state it. The square-footage and price columns need the full-table run (the sample CSV has no sqft column).

## 6. Limits

* **One annotator, and not independent.** The 120 listings were labeled by the AI assistant that also wrote the extractor rules. Labels were written first, blind to the extractor, but the same person saw the listing texts while writing patterns, so test is not truly unseen. A human spot-check of 20 to 30 labeled listings would be the best use of an hour.
* **Small.** 120 listings, 54 on test. The interval on test F1 is about plus or minus 4 points. Financing signals have 9 positives on test, so their F1 swings with one listing.
* **Rare signals are unmeasured** (above).
* **Judgment calls.** `views` (named view vs generic "views"), `quiet_street`, `near_downtown`, `investment_income` ("great for an investor") and `near_beach_water` for lake communities are the ones where a second labeler would most likely disagree. They are also among the weakest on test.
* **Precision from proximity cues is heuristic.** A cue ("near", "minutes from") counts if it is in the same sentence within 140 characters; lists of places can be attached to the wrong cue.
* **Signals describe what the remark says**, not the property. A listing that never mentions a freeway has no `freeway_commute`, even if one is next door.
* The DB comparison has only been run on the sample here. Run it on the full table before relying on the structured fields at scale.

## 7. How to reproduce

```bash
python scripts/evaluate_signals.py                      # dev
python scripts/evaluate_signals.py --split test --final # held-out, once
python scripts/process_listings.py --sample             # sample, no database
python scripts/process_listings.py --compare-columns    # the whole table (MySQL running)
pytest tests/test_week6.py -v
```

`evaluate_signals.py` loads listing texts by id from `listing_sample.csv`, `listing_sample_eval.csv` or `listing_sample_signals.csv`, and fetches any missing ids from MySQL (cached in `listing_sample_signals.csv`, git-ignored). If fewer than 90% of the gold listings can be loaded it stops with instructions instead of printing meaningless scores. The sample CSV is a random draw, so on a machine that did not make the gold set, the first run needs the MySQL container running.

## 8. Next

* Week 7 will want to use these signals as filters on top of the Week 4 parser and the Week 5 ranker (for example `fin:fha`, `hoa:none`), which also addresses the Week 5 negation failure ("no HOA") with a field instead of a text match.
* Add 20 to 30 listings that mention rare financing terms to the gold set so their recall can be measured.
