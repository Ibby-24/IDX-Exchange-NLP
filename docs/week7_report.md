# Week 7: Search query intent classification

## 1. What was built

`QueryIntentClassifier` (`scripts/query_intent.py`) reads one search query and returns an intent (`browsing`, `researching` or `high_intent_inquiry`) with a confidence score and an `uncertain` flag. It is a multinomial logistic regression on three kinds of features: word 1 to 2-grams (TF-IDF), character 2 to 5-grams (TF-IDF, inside word boundaries), and nine small numeric features taken from the Week 4 `QueryParser` (how many filters it found, whether it found a city, a price, beds, baths, square feet, amenities, plus question mark and log length).

Intent is inferred **only from the words of the query**. No click history, no CRM data, no knowledge of the person.

Spec checklist

| Spec item | Where | Result |
|---|---|---|
| Labeled dataset, 200+ queries | `data/processed/intent_queries.json` | 380 queries (228 train, 68 dev, 84 test), about equal across the three labels |
| `QueryIntentClassifier` | `scripts/query_intent.py` | logistic regression |
| 80%+ accuracy on a held-out test set | section 4 | **91.7%** (n=84, scored once; 95% interval 84.5% to 96.4%) |
| Confidence scores for uncertain classifications | `predict`, `predict_detailed`, `uncertain` flag | section 5 |
| Integration with the Week 4 query parser | `understand()`, `run_query.py --intent`, parser features | section 6 |
| Limitations of language-based intent inference | section 8 | |

## 2. The labels and how they were assigned

Full rules are in `docs/intent_guidelines.md`, written **before** any new query was labeled and before the classifier existed.

* **browsing:** open-ended looking: a place and at most a type or one descriptive attribute ("condos Santa Monica", "homes with a pool in Riverside").
* **researching:** wants information or a comparison, not a list to act on: questions, "vs", costs, rules, schools, commute, "is it a good time" ("average HOA fees in Irvine", "which areas have low property taxes").
* **high_intent_inquiry:** readiness to act on specific homes: timing or availability ("this weekend", "open house", "just listed"), contact or tour words, pre-approval, a hard budget plus another concrete spec, or a financing term used as a filter ("FHA", "seller financing").

Four ordered tie-break rules decide overlaps (readiness words beat question form; question form beats a price; budget plus another spec is high intent; everything else is browsing). The consequence that surprises people: "homes in Irvine under 900k" is **browsing** (one constraint besides the place), flagged `hard`.

## 3. The dataset

* 380 queries: 60 from Week 1 (kept with their original labels), 277 written for Week 7, 43 added to the training split only after the first dev run (they were checked so none is a near-duplicate of a dev or test query).
* Styles: natural 234, keyword 58, question 45, noisy 25 (typos, shorthand), long 18. 24 queries are flagged `hard`.
* **Near-duplicates stay together.** Queries that differ only in a city or number share a `group`, and a group never straddles train/dev/test. Without that, a random split would score memorized templates.
* The queries are **invented**, not real search traffic, so the file contains no MLS data and is safe to commit.
* A **stress set** of 45 queries (`data/processed/intent_stress.json`, 15 per label) was written after the classifier was frozen, with indirect phrasing, and is used only as a robustness check.
* **Second pass:** 80 queries were relabeled blind (`docs/week7_second_pass.json`) and agreed 80 of 80. That is the same annotator checking consistency, **not** independent agreement, so it should not be quoted as inter-annotator agreement.

## 4. Results

### Held-out test (84 queries, scored once, model fit on train + dev)

| | Target | Result |
|---|---|---|
| Accuracy | 80%+ | **91.7%** (95% bootstrap interval 84.5% to 96.4%) |
| Macro-F1 | | 91.6% |

Per label (F1): browsing 91.2%, researching 93.1%, high_intent_inquiry 90.6% (precision 96%, recall 85.7%).

Confusion (rows gold, columns predicted: browsing / researching / high intent):

```
browsing              26   1   1
researching            1  27   0
high_intent_inquiry    2   2  24
```

Where it is weaker on test: keyword-style queries 81.8% (n=11), long queries 66.7% (n=3), hard-flagged queries 83.3% (n=6), compared with 94.8% on natural-sentence queries (n=58). These subgroups are tiny; treat the direction, not the number.

### Baselines

| | Test accuracy | Dev accuracy |
|---|---|---|
| Majority class | 33.3% | 33.8% |
| **Rule baseline** (the guidelines turned into regexes) | **95.2%** | 89.7% |
| LR words only | | 86.8% |
| LR words + chars | | 86.8% |
| LR words + chars + parser features (final) | 91.7% | 92.6% |

**The rule baseline beats the model on test (95.2% vs 91.7%).** The honest reading: the labels were assigned by written rules, so rules that restate those rules do very well, and the ML model is mostly learning the same rules from examples. The model's advantage, if any, shows on wording the rules did not anticipate (stress set below), and it supplies a calibrated-ish confidence, which the regexes do not. The parser features added about 6 points on dev; the character n-grams added nothing on dev.

### Dev (what I tuned on)

92.7% accuracy (n=68). 5-fold grouped cross-validation on train + dev: mean 87.9% (folds 83.6% to 91.7%), which is the better estimate of typical performance than any single split.

### Stress set (45 indirect queries, written after freezing)

91.1% accuracy (macro-F1 90.9%); the rule baseline gets 84.4%. All four errors are high-intent queries predicted as researching because they are phrased as questions without a stock cue word:

* "what time can we see the house on Maple Street in Pasadena"
* "my agent says the Sunnyvale listing is still available, can we tour it"
* "who do I contact to see the open house in Alameda on Sunday"
* "3 bedroom Concord home I saw online, is it still on the market and can I tour it"

Browsing and researching were 15 of 15 each. This is the clearest weakness: **intent expressed through context rather than keywords**.

### Robustness (test / stress)

| Change to the query | Test accuracy | Stress accuracy |
|---|---|---|
| none | 91.7% | 91.1% |
| UPPER CASE | 91.7% | 88.9% |
| punctuation removed | 88.1% | 91.1% |
| one typo | 88.1% | 86.2% |

## 5. Confidence and the `uncertain` flag

`confidence` is the model's probability for the top label. A query is marked `uncertain` when confidence is below **0.65** (chosen from out-of-fold predictions on train + dev, not from test) **or** the query looks outside real-estate search.

On test, 16 of 84 queries were flagged: they were right 68.8% of the time, versus 97.1% for the 68 confident ones. So the flag does separate the shaky answers from the solid ones, at the cost of sending about 19% of queries to a fallback. The coverage table in `docs/week7_results.json` shows the trade: answering only above 0.7 covers 76% of queries at 98.4% accuracy; above 0.8, 64% at 100%.

Calibration is imperfect and **under**confident on test (mean confidence 0.82 vs accuracy 0.92; expected calibration error 0.093). Confidence reflects similarity to training queries, not how hard the query is: "houses" gets 0.997 for browsing because it looks exactly like many training queries, even though it carries almost no information.

**Out-of-scope input.** There is no "other" label, so without a guard the classifier labels anything ("what is the weather in Paris" came out researching at 0.79 during development). An `in_scope` check (housing vocabulary, or a non-city parser filter, or a known city) flags these as uncertain: 15 of 16 non-housing probes were flagged in the final evaluation ("restaurants in Pasadena" got through, because it parses as a known city). The vocabulary was widened once after I had seen some labeled test queries wrongly flagged (no labeled query is wrongly flagged now); accuracy was not affected, but this means the in-scope word list is not independent of test either.

## 6. Integration with the Week 4 parser

* **As features:** `ParserFeatures` runs `QueryParser` on the query and feeds the counts above into the classifier. This is what lifted dev accuracy from 86.8% to 92.6%.
* **As a pipeline:** `understand(query)` returns the parser's filters, the validator result and SQL, plus intent, confidence, `uncertain` and a suggested handling note (what a product could do: show results, show an answer or comparison, or route toward a contact/tour flow; or ask a clarifying question when uncertain).
* **On the command line:** `python scripts/run_query.py "open house in Irvine this weekend" --intent`.
* The intent never changes the SQL; it is advisory.

## 7. Bug in the starter skeleton

The spec's skeleton picks the label with `self.labels[probas.argmax()]` using a fixed order (browsing, researching, high_intent_inquiry). scikit-learn orders `model.classes_` alphabetically (browsing, **high_intent_inquiry**, researching), so the skeleton would return the wrong label for two of the three classes. This code uses `model.classes_`; `tests/test_week7.py` has a test that fails if the order is ever assumed again.

## 8. Limitations of language-based intent inference

1. **Words are a weak, noisy signal.** The same words come from different people: "open house this weekend" can be a buyer, a neighbor or an agent. A label is "what the wording most probably signals", not a measure of the person or a prediction that they will buy.
2. **Intent without cue words is missed.** The stress set shows it: tour/availability questions phrased as questions are read as research. Short keyword queries ("3 bed Irvine 800k") are ambiguous even for a human; the guidelines call that browsing, a real user might not mean it.
3. **Labels are a convention.** The boundary between browsing and high intent (is a price plus a city enough?) is a rule I wrote, not a fact. 24 of 380 queries are flagged hard for that reason.
4. **Invented queries, one annotator.** The AI assistant wrote the guidelines, the queries, the labels and the classifier. The 80/80 blind relabel shows self-consistency only. Real users' typos, abbreviations, other languages and multi-intent queries ("what are taxes in Irvine and can I see 12 Oak Lane Saturday") are under-represented or absent. Real accuracy on live traffic could easily be well below 91.7%.
5. **The model mostly reproduces the labeling rules.** A regex baseline written from the same guidelines scores higher on test. Treat the classifier as a way to generalize those rules to unseen wording and to produce a confidence, not as evidence that intent can be learned this accurately from real data.
6. **Three labels only, single query only.** No "other", no mixed intent, no session context. Intent often shifts across a session; that needs behavior data, which is explicitly out of scope here.
7. **Small test set.** 84 queries; the interval is about plus or minus 6 points, and subgroup numbers (keyword 11, long 3, hard 6) are anecdotes.
8. **Confidence is not difficulty.** It is similarity to training data. A short, familiar query is scored very confident; an unusual but easy one is scored low.
9. **Fairness and use.** Do not use the label to treat people differently in price, availability or contact priority without a human review; it was built to choose a result layout or a follow-up prompt.

## 9. How the numbers were protected

* Guidelines before labeling; stress set written after freezing; test split run once with `python scripts/evaluate_intent.py --split test --final` (the script refuses `--split test` without `--final`).
* Dev was used for all tuning (C, character n-grams on/off, parser feature scale, the 0.65 threshold from cross-validation). 43 training queries were added after the first dev run; this is disclosed above and does not touch test.
* Group-aware splits and cross-validation prevent near-duplicate leakage.

## 10. How to reproduce

```bash
python scripts/evaluate_intent.py                          # dev
python scripts/evaluate_intent.py --errors                 # dev errors
python scripts/evaluate_intent.py --compare                # baselines and ablations
python scripts/evaluate_intent.py --cv                     # 5-fold grouped CV
python scripts/evaluate_intent.py --extra data/processed/intent_stress.json   # stress set
python scripts/evaluate_intent.py --split test --final     # held-out, ONCE
python scripts/query_intent.py train                       # saves data/models/intent_model.joblib (not committed)
python scripts/query_intent.py understand "open house in Irvine this weekend"
python scripts/run_query.py "3 bed under 700k in Irvine" --intent
pytest tests/test_week7.py -v
```

To check the model on queries you wrote yourself, fill `docs/intent_extra_template.csv` (columns `query,label`, label one of the three) and run `python scripts/evaluate_intent.py --extra docs/intent_extra_template.csv`. Thirty queries from someone other than the person who built the classifier is the most useful next test.

## 11. Next

* Collect real (anonymized) search queries and have a second person label 100 of them; compare against this set.
* Add training examples of tour/availability questions without stock cue words (the stress-set failures), then re-measure on a fresh stress set.
* Decide with the team whether an "other / not a search" label is wanted, and what each intent should change in the product.
