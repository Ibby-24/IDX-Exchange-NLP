# Week 7 query intent labeling guidelines

Written **before** any new query was labeled and before the classifier existed. Intent is inferred **only from the words of the query**. Nothing is known about the person, their history or whether they ever buy; a label is "what the wording most probably signals", not a prediction of behavior.

## The three labels

**`browsing`**: open-ended looking around. The query asks to see homes by place and, at most, a type, a style or one or two descriptive attributes (a pool, a view, a vibe, "luxury", "single story"). No budget plus specs, no timing, no readiness, no question.

**`researching`**: the query wants *information or a comparison*, not a list to act on. Typical wording: question forms (what / which / how / is / are / does / can), compare, vs, pros and cons, average, typical, cost, trend, best, "areas / neighborhoods / cities / communities that ...". Typical topics: ownership costs and rules (HOA, property tax, Mello-Roos, insurance, flood zone, rent vs buy, ADU rules, short-term rental rules), schools, crime, commute, "is it a good time". A cost-of-ownership attribute as the main point of the query ("condos near UC Irvine with low HOA", "areas with low property taxes") counts as researching.

**`high_intent_inquiry`**: wording that signals readiness to act on specific homes. Any of: timing or availability ("this weekend", "open house", "available now", "can close in 30 days", "just listed", "new listings", "price reduced"); a readiness or contact verb ("schedule a showing", "tour", "make an offer", "contact agent", "pre-approved", "cash buyer"); a hard budget **plus** at least one more concrete spec (beds, baths, sqft, a feature, a condition such as move-in ready); an actual financing term used as a search filter (seller financing, FHA, VA, assumable, cash offer).

## Tie-break rules (apply in this order)

1. **Readiness, timing or contact words win.** "Are there open houses in Irvine this weekend?" is `high_intent_inquiry` even though it is a question.
2. **Information-seeking form or topic next.** If it is a question, a comparison, or about costs / rules / schools / commute / market, it is `researching` even when a city or a price appears ("average price of a 3 bedroom in Irvine", "can I afford a 900k house on 120k income").
3. **Hard budget plus another concrete spec** (beds, baths, sqft, feature, condition) or **three or more concrete constraints** is `high_intent_inquiry`.
4. **Everything else is `browsing`.**

Consequences worth stating:
* A budget with only a place and a property type ("homes in Irvine under 900k") has one concrete constraint besides the place, so it is `browsing` by rule 4. It is flagged **hard** (a second labeler could reasonably say high intent).
* "fixer upper houses in Los Angeles", "4 bedroom houses in Riverside", "55+ communities in Menifee" are `browsing`: a descriptor, no cue.
* A bare financing word with no search ("what is an FHA loan") is `researching`, not high intent.
* Out-of-scope or non-search text (a greeting, a listing address pasted alone) is not in the dataset. The classifier will still return one of the three labels for it; that is a stated limitation.

## Flags

* `hard`: the query is a plausible disagreement between two labelers, or a cue and the label point different ways (a question that is high intent, a price with no other spec). Hard queries are scored separately.
* `style`: `keyword` (terse), `natural` (a sentence), `question`, `long`, `noisy` (typos, no punctuation, shorthand such as 3br/2ba). Used to check the classifier is not only learning one writing style.

## What this is not

Not buyer analytics, not conversion prediction, not a measure of the person. The same words from two different people can mean different things. See the limitations section of `docs/week7_report.md`.
