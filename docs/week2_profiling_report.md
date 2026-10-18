# Week 2 Data Profiling Report: Listing Remarks (`L_Remarks`)

**Purpose:** document what is actually in the listing remarks *before* cleaning, so every rule in `TextCleaner` is justified by evidence and every risky rule has a test.

**Data:** `data/processed/listing_sample.csv`, a random sample of 1,000 listings from `rets_property` (remarks longer than 50 characters). Produced by `scripts/data_loading.py`.
**Tool:** `TextCleaner.profile_column()` plus the pattern counts below. All numbers refer to this sample; a new random draw will differ slightly.
**Privacy note:** this report contains counts and short tokens only. No listing text is reproduced, because the source is MLS data.

---

## 1. Summary

| Finding | Value | Consequence |
|---|---|---|
| Completeness | 0% null (the SQL filter requires non-null remarks) | no imputation needed |
| Typical length | median 1,262 characters, mean 1,321, range 51 to 4,000 | full paragraphs; plenty of text to extract from |
| HTML | 0 remarks | no heavy markup, but the cleaner still handles tags and entities |
| Non-ASCII characters | 384 remarks (38%) | Unicode normalization is required |
| Double spaces | 323 remarks (32%) | whitespace must be normalized **before** other regexes |
| Embedded newlines | 232 remarks (23%) | newlines become spaces |
| Mostly full words, little shorthand | "bedrooms" appears in 825 remarks; `3br`-style shorthand in about 20 | the abbreviation list matters more for **search queries** than for remarks |
| Price shorthand | 10 remarks with `$`+k/m; a naive k/m rule alters 19 remarks and disagrees with the context-aware rule on 13 | price conversion needs context rules |

## 2. Length

| Statistic | Characters |
|---|---|
| Minimum | 51 |
| 10th percentile | 611 |
| Median | 1,262 |
| 90th percentile | 2,201 |
| Maximum | 4,000 |

Only 1 remark reaches 4,000 characters, which matches the 4,000-character limit on `PublicRemarks` in the Trestle metadata, so truncation is rare.

## 3. Character-level problems

| Issue | Remarks | Handling |
|---|---|---|
| Curly quotes and apostrophes | 257 (26%) | convert to straight quotes (this also unhides taxonomy terms such as "chef's kitchen") |
| Em or en dashes | 204 (20%) | replace with " - ", because they often join words with no spaces |
| Zero-width characters | 2 (80 invisible characters in total) | remove; invisible, but they break word matching |
| Non-breaking spaces | 4 | convert to normal spaces |
| Bullets (•) | 7 | convert to "-" |
| Unicode fractions (½, ¼, ¾) | 7 | convert to "1/2", "1/4", "3/4" (NFKC alone would produce a different "fraction slash" character) |
| Plus-minus sign (±) | 7 | convert to "+/-" |
| ® and ™ | 6 | remove |
| Mojibake (text decoded with the wrong encoding) | 3 | repair per fragment; remove unrepairable leftovers |

**Ordering insight:** the mojibake repair must run before the non-breaking-space replacement, because one common broken sequence contains a NBSP byte.

## 4. Whitespace and markup

| Issue | Remarks | Handling |
|---|---|---|
| Two or more spaces in a row | 323 (32%) | collapse; also run early, since patterns like "sq  ft" would otherwise be missed |
| Single newlines inside the text | 232 (23%) | replace with a space (no double newlines in this sample) |
| Markdown bold markers (`**`) | 28 | replace with a space (removing them outright merged two words in one remark) |
| HTML tags | 0 | handled anyway; a lone `<` is not treated as a tag |
| `!!` runs | 58 | left alone (harmless for extraction) |
| ALL-CAPS words (4+ letters) | 282 (28%) | no case folding in the cleaner; matching code lowercases |

## 5. Numbers, prices and measurements

**Square footage** appears in many forms:

| Form | Remarks |
|---|---|
| "square feet" (already canonical) | 167 |
| "sq. ft." | 87 |
| "sq ft" | 56 |
| number + "SF" | 26 |
| "sqft" | 26 |
| "sq/ft" | 1 |

Some are written with a `+` or `+/-` between the number and the unit. A bare "SF" can also mean San Francisco, so it is expanded **only directly after a number**.

**Prices:** 69 remarks contain a dollar amount. Only 10 use `$` with k/m shorthand, and 8 mention "million". Some look-alikes are *not* prices: an FHA renovation-loan name that ends in "K", a walking distance written with an "m", a duration written like "4min", and a count of people in millions. A naive "digits followed by k or m" rule alters 19 remarks in this sample, and its output differs from the context-aware rule in 13 of them. The cleaner therefore converts shorthand **only after a `$` or a price word** (under, over, around, ...).

**Thousands separators** appear in many numbers, and some junk values such as comma lists with single digits exist, so only valid three-digit grouping is treated as a thousands separator.

## 6. Abbreviations and shorthand

Counts from `_detect_abbreviations` (remarks, approximate):

| Token | Count | Decision |
|---|---|---|
| HOA, ADU, RV, BBQ, HVAC, EV, LED, TV | 24 to 133 | **protected**: never expanded |
| `A/C` (35) and `AC` (36) | 71 | expand to "air conditioning" (matches taxonomy term) |
| `sq.` / `ft.` / `sqft` / `SF` | 30 to 116 | handled by the measurement rule |
| `w/` | 31 (12 remarks use it as a word) | "with", including when glued to the next word |
| `LVP` | 14 | "luxury vinyl plank" |
| `approx.` | 9 | "approximately" |
| `hwy` / `fwy` | 14 | "highway" / "freeway" |
| `3br`, `2ba`, `bd`, `bth` style | about 20 remarks | "3 bedroom", "2 bathroom" |
| `LA`, `CA`, `UC`, `BART`, `FHA`, `VA`, `LAX` | 7 to 27 | protected (places and programs) |

Detection is heuristic, so a few false positives appear, such as a sentence ending in a short word followed by a lowercase letter.

**Slash compounds:** 172 remarks contain word/word forms ("pool/spa", "indoor/outdoor"). Splitting them lets taxonomy terms such as "indoor-outdoor living" match.

**Addresses inside remarks:** about 79 remarks (8%) contain a street address. The cleaner leaves them alone; Week 3 entity extraction must not mistake an address number for a bedroom count.

## 7. Vocabulary (top bigrams after stopword removal)

| Bigram | Count |
|---|---|
| primary suite | 346 |
| floor plan | 317 |
| car garage | 295 |
| living room | 292 |
| home offers | 290 |
| living space | 281 |
| natural light | 260 |
| bedroom bath | 212 |
| shopping dining | 208 |
| walk closet | 201 |

Some are artifacts of stopword removal rather than real phrases: "walk closet" is *walk-in closet*, "car garage" is *two-car garage*, and "shopping dining" comes from "shopping and dining". Several top bigrams ("home offers", "living space", "everyday living") are marketing filler.

## 8. From findings to rules

| Finding | Rule | Method |
|---|---|---|
| Curly quotes, dashes, invisible characters, mojibake | normalize characters | `normalize_unicode` |
| Double spaces, newlines | collapse whitespace (early and late) | `normalize_whitespace` |
| `**`, HTML tags and entities | strip markup | `strip_markup` |
| Thousands separators | remove valid groupings | `normalize_numbers` |
| k/m/million shorthand | convert only with `$` or a price word | `normalize_prices` |
| sq ft variants | "N square feet", number-anchored | `normalize_measurements` |
| word/word compounds | split on slash | `normalize_slashes` |
| shorthand, bed/bath abbreviations | expand from config, protect acronyms | `expand_abbreviations` |

Pipeline order is pinned by tests (whitespace before measurements, slashes before abbreviations, mojibake before NBSP, markers replaced with a space).

## 9. Assumptions and limitations

- One 1,000-row random sample from a single MLS table; rare patterns may be missing, and counts shift with each draw.
- Abbreviation detection is heuristic and was reviewed by hand.
- Expansions are singular and lowercase by design ("3 bedroom"), and `mbr` maps to "primary bedroom" because the data says "primary".
- The price-word rule is a heuristic: a count written right after a price word (for example "over 3 million visitors") would still be converted. This is rare in listing text.
- Unrepairable source typos (for example a malformed number) are left unchanged rather than guessed.
- The cleaner is not meant to fix grammar or spelling.

## 10. Hand-off to Week 3

Cleaned text keeps all information and standardizes its spelling: bedroom and bathroom shorthand becomes "N bedroom" / "N bathroom", prices become plain numbers, and areas become "N square feet". Entity extraction can therefore rely on a small set of canonical patterns.
