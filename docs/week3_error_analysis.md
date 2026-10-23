# Week 3 Error Analysis: Entity Extraction

**Code:** `scripts/entity_extractor.py` (rules), `scripts/evaluate_extractor.py` (scoring), `scripts/baseline_extractor.py` (naive baseline).
**Data:** hand-labeled gold set, see `docs/annotation_guidelines.md`. Raw numbers: `docs/week3_results.json`.
**Privacy:** this report contains counts and invented example phrases only. No listing text is reproduced.

## 1. Results

Target: 85%+ F1 on the test set.

| Entity | Test P | Test R | **Test F1** (95% CI) | Dev F1 | Naive baseline, test F1 |
|---|---|---|---|---|---|
| Bedrooms | 0.932 | 0.972 | **0.952** [0.91, 0.98] | 0.984 | 0.604 |
| Bathrooms | 0.955 | 0.940 | **0.947** [0.89, 0.99] | 1.000 | 0.634 |
| Square footage | 0.949 | 0.925 | **0.937** [0.85, 0.99] | 1.000 | 0.905 |
| Price | 1.000 | 1.000 | 1.000 (3 positives; CI is [0, 1]) | 0.889 | 0.231 |
| **Numeric, micro-average** | 0.945 | 0.950 | **0.948** | 0.991 | 0.662 |
| Amenities (24 listings, term-id sets) | 0.987 | 0.994 | **0.990** [0.98, 1.00] | 0.992 | 0.870 |
| Numeric + amenities, micro | 0.972 | 0.978 | **0.975** | 0.992 | 0.805 |

The 85% target is met on the held-out split for every entity.

**How much to trust these numbers**
- Dev was used to tune the rules, so dev scores are optimistic. The first dev run, before any tuning, scored 0.967 micro on the numeric entities; after fixes 0.991. The test split was scored once, after the extractor was frozen (numeric micro F1 0.945 at first scoring; 0.948 after two label corrections described in `annotation_guidelines.md`).
- The same annotator wrote the gold labels and the extractor (see the limits section of the guidelines). Treat a drop of a few points with an independent second annotator as likely.
- Price has 8 positives in total, so its F1 means little. What does matter for price is precision: the starter-style regex fires on 140 of the 1,000 sample remarks, while only 7 state a list price.
- Amenity scores are against the Week 1 vocabulary only.
- Confidence intervals are bootstrap intervals over listings and are wide for small entities.

**Independent cross-check against the structured MLS fields** (all 1,000 sample rows, no labels involved):

| Field | Rows with both values | Exact agreement |
|---|---|---|
| Bedrooms | 688 | 89.4% |
| Bathrooms | 608 | 68.1% exact, **91.0% after rounding the extracted value up** |

Finding: the structured bathroom column is a whole number (2.5 baths is stored as 3). Of the 194 exact mismatches for bathrooms, 135 differ by exactly 0.5. For bedrooms the extracted count is usually smaller than the structured one (most of the 73 disagreements are by 1 or 2), which fits the structured field being a property-wide total that includes ADUs, dens and bonus rooms the main-residence text does not count.

## 2. What the data looks like (1,000-listing sample)

| Entity | Remarks with any mention | Remarks where a main value is returned |
|---|---|---|
| Bedrooms | 756 (76%) | 688 (69%) |
| Bathrooms | 627 (63%) | 608 (61%) |
| Square footage | 383 (38%) | 333 (33%) |
| List price | 11 (1%) | 7 (0.7%) |

Mention roles assigned by the extractor:

| Entity | main | partial | secondary unit | other area / not the property | other |
|---|---|---|---|---|---|
| Bedrooms (1,001 mentions) | 798 | 164 | 26 | 2 market | 8 public-records, 3 range |
| Bathrooms (702) | 655 | 18 | 18 | 1 market | 5 hypothetical, 5 public-records |
| Square footage (532) | 373 | - | - | 147 other area (lot, garage, loft, ADU, ...) | 10 implausible, 2 range |
| Price (11) | 7 | - | - | - | 3 bound ("under $X"), 1 implausible |

Two consequences drove the design:
1. **Counting mentions is not extracting.** About 1 in 6 bedroom mentions is a partial count ("two additional bedrooms", "3 bedrooms upstairs"), and 28% of square-footage mentions are not living area. A role is assigned to every mention and only `main` mentions can become the answer. The naive baseline has no roles; it misses about half of the bedroom and bathroom values (it cannot read "3 spacious bedrooms", number words or "two-bath" forms) and adds 6 wrong square-footage values on test (lot sizes, garage and loft areas).
2. **Price must be cue-based.** Most dollar amounts in remarks are HOA dues, rent, upgrade costs, credits, grants or appraisals, and the digit-run rule picks up zip codes, APNs, HUD case numbers and street numbers. Baseline price precision is 0.06 (dev) and 0.13 (test); the cue-based extractor has no false positives on either split.

Amenity filters removed 830 term matches across the 1,000 sample remarks: 480 negated or hypothetical, 328 community-only amenities, 12 terms describing the community instead of the home ("well maintained community"), and 10 false friends. The plain substring baseline, which has no boundaries and no filters, reaches only 0.73-0.77 amenity precision (dev, test), with outdoor and property-type terms hurt most (precision 0.57 and 0.74 on test) versus 0.99 for the extractor.

## 3. Remaining failure patterns (test split)

Every item below is an error the frozen extractor makes on the test split. Example phrases are invented. Numeric errors: 10 false positives and 9 false negatives (a wrong value counts as one of each). Amenity errors: 4 false positives and 2 false negatives.

| # | Pattern | Entity | Errors | Cause | Planned fix |
|---|---|---|---|---|---|
| 1 | Partial counts promoted to the property total | bedrooms (4 FP), bathrooms (1 FP) | 5 | "an additional 3 bedrooms" (the word sits before the number, outside the gap that is checked); "two bedrooms share a bath"; a sentence that starts with "Upstairs, a hallway connects four bedrooms" (floor cue further than 25 characters away); a list of rooms ("...two bedrooms, and a bathroom") | check the words before the number; treat "share" and enumerated-room lists as partial; widen the floor-cue window to the whole clause |
| 2 | Main-house cue overrides a guest-house cue | bedrooms | 1 | "a main residence plus a one-bedroom guesthouse": the main-house cue wins because it is checked first | test the unit cue after the number before the main-house cue |
| 3 | A hypothetical cue leaks onto the next item | bathrooms | 3 (2 missed, 1 wrong value) | "(a den that converts to a 4th bedroom), 2.5 bathrooms" and "an optional 5th bedroom plus 3 updated bathrooms": the cue is meant for the bedroom, but the bathroom count is within its reach | end the cue's reach at the first closing parenthesis, comma or "and" |
| 4 | Half bath too far from the full-bath count | bathrooms | 1 wrong value | "2 full baths and a bonus room with its own 1/2 bath" is 2.5; the half bath is 40 characters away and the window is 30 | widen to the clause and require "bath" to follow |
| 5 | Unit cue also fires on property types | bedrooms | 1 FN | "cottage cabin" is the house itself, but "cottage" is on the secondary-unit list | use "cottage" only when it follows "guest", "back" or "rear", or sits next to "main house" |
| 6 | Unusual phrasing | bedrooms | 1 FN | "2 plus an optional bedroom" (the gap cannot contain "plus") | add a "N plus (an) optional bedroom" form |
| 7 | Area rules too eager | sqft | 2 FN | a lot or loft is mentioned earlier in the same sentence ("a corner lot with a fenced yard ... this 1500 square feet property"; "a townhome with a versatile loft offers 1600 square feet") | trust a directly following "home/house/property/townhome" and drop sentence-wide lot and loft cues when the area is the grammatical object of offers/features/boasts |
| 8 | Area attributed to the wrong unit | sqft | 1 wrong value + 1 FP | "Main house: ..., 2200 square feet PLUS studio ADU ...": the area after "PLUS" is read as an ADU cue for the main-house area, while the ADU's own area is accepted; "1400 square feet of additional outdoor living space" is accepted | stop the after-window at "plus/and/;"; add "outdoor living / entertaining" and "additional ... space" to the non-living list |
| 9 | Amenity suggested without a cue word | amenities | 1 FP | "extra space for a game room, office or ...": a suggested use without ideal/perfect/potential | add "space for" and "room for" to the use cues for room terms |
| 10 | Community amenities with no cue word nearby | amenities | 2 FP | "five swimming pools, seven clubhouses" and "a smart pool, splash park, playground" appear in lists with no "amenities" or "community" in the sentence | detect lists of facility nouns (pools, clubhouses, courts, parks, playground) and treat sentences with 2 or more of them as community |
| 11 | False friend | amenities | 1 FP | "gated yard" (means fenced) matches the community term "gated" | add "gated yard/area/patio" to false friends |
| 12 | Words glued together in the source | amenities | 1 FN | "butler's pantrycreate" has no word boundary after the term | allow a lowercase run-on only for terms longer than 8 characters, after checking precision |
| 13 | Cue too broad | amenities | 1 FN | "a chance to buy an income producing property" is read as hypothetical because of "opportunity to" | limit "opportunity to" to the verbs add, build, create, convert |

Dev errors that remain after tuning (4 bedroom/price, 4 amenity) are in the same families: a multi-building property ("the second 5-bedroom building"), a list of villa types, a two-unit property where the second unit's count sits earlier in the text, a price that is a program cap ("the purchase price cannot exceed $X") caught by the bound rule, vocabulary variants ("decking", "carpeting"), "dream for ... rental income" read as hypothetical, and guest parking in a building-amenity list.

## 4. What the dev errors changed in the code (generalizable fixes)

1. The conditional cue "potential" now requires "for" or "to" ("offers lot of potential with 5 bed" is a real count).
2. The gap between number and unit may not contain other count words (and, plus, number words, bed/bath words); the old gap read "two bedrooms and one bath" as two baths.
3. A range is recognized after a unit ("1400 square feet to 2000 square feet"), but "and 2.5 baths" is no longer a range.
4. A lot, patio, deck, garage or loft earlier in the sentence marks the area as non-living, unless "home/house/residence" follows directly (this rule is also the source of test pattern 7).
5. "All N bedrooms/bathrooms" and "All N of the bathrooms" are totals.
6. Amenity ties: the term whose own text is longer wins ("walk-in closets" over "walk-in closet" + plural).
7. Amenity cues are split by reach: "no/could/may/potential" stop at a comma, "dream/adding/building a" run across a list, and "ideal for/perfect for" apply only to room, property-type and financing terms (so "a deck ideal for enjoying the lake views" keeps "lake views").
8. Terms that describe the community ("well maintained community") are dropped for the home.
9. Price modifiers (under, below, over, from, up to) between a cue and a dollar amount now mark a bound instead of a price.

## 5. Recommended Week 4+ follow-ups

- Apply the fixes in section 3 as a second version, re-score on a **fresh** labeled batch (not the current test split) so the test number stays honest.
- Have a second person label 30 test listings blind.
- Use the structured bedroom and bathroom fields as a feature in Week 4+ rather than a replacement: the text gives the main-residence count, the field gives the property total, and a difference is a signal for ADUs and extra units.
- Optional spaCy NER (not done): the gaps above are semantic role problems (main vs additional unit), which a trained span model would learn from the 301 labeled listings only with many more examples; the rules are the better first step.
