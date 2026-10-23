"""Week 4: score QueryParser against the hand-labeled expected filters.

    python scripts/evaluate_queries.py                 # the 60 Week 1 queries (the set used while building the parser)
    python scripts/evaluate_queries.py --errors        # list every query the parser got wrong
    python scripts/evaluate_queries.py --heldout --final   # the fresh queries: run ONCE, after the parser is frozen

Data: data/processed/query_gold.json (committed; invented queries and expected filters, no MLS text).

Scoring
  * query accuracy  = the parsed filters equal the expected filters exactly
                      (amenity lists compare as sets; numbers compare exactly)
  * field accuracy  = for each filter name (city, price_max, ...) over the queries that expect it
                      or got it: recall (expected and right) and precision (produced and right)
  * valid-filter rate = how many parsed outputs pass SchemaValidator (informational)

The Week 1 queries were available while the rules were written, so that score is optimistic.
The held-out score is the honest estimate, and it only means something if it is run once.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_parser import QueryParser          # noqa: E402
from schema_validator import SchemaValidator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "processed" / "query_gold.json"
RESULTS = ROOT / "docs" / "week4_results.json"
SET_KEYS = {"amenities", "exclude_amenities", "cities"}


def normalize(filters):
    out = {}
    for k, v in filters.items():
        if k in ("amenities", "exclude_amenities"):
            out[k] = frozenset(x.lower() for x in v)
        elif k == "cities":
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def score(parser, validator, rows):
    n_ok, field_counts, failures, pat = 0, defaultdict(Counter), [], Counter()
    valid_n = 0
    for r in rows:
        pq = parser.parse_detailed(r["query"])
        got, exp = normalize(pq.filters), normalize(r["expected"])
        ok = got == exp
        n_ok += ok
        pat.update(pq.patterns_used())
        valid_n += validator.validate_query(pq.filters)[0]
        for k in set(got) | set(exp):
            c = field_counts[k]
            if k in got and k in exp and got[k] == exp[k]:
                c["tp"] += 1
            else:
                if k in got:
                    c["fp"] += 1
                if k in exp:
                    c["fn"] += 1
        if not ok:
            failures.append({"query": r["query"], "cleaned": pq.cleaned, "expected": r["expected"],
                             "got": pq.filters, "ignored": pq.ignored})
    return n_ok, field_counts, failures, pat, valid_n


def report(name, rows, parser, validator, show_errors):
    n_ok, fc, failures, pat, valid_n = score(parser, validator, rows)
    n = len(rows)
    print(f"\n== {name}: {n_ok}/{n} queries exactly right = {n_ok / n:.1%}")
    print(f"   parsed filters that pass SchemaValidator: {valid_n}/{n}")
    print(f"   {'filter':<18}{'right':>6}{'missed/wrong':>14}{'spurious':>10}")
    for k in sorted(fc):
        c = fc[k]
        print(f"   {k:<18}{c['tp']:>6}{c['fn']:>14}{c['fp']:>10}")
    if show_errors or failures and len(failures) <= 12:
        for f in failures:
            print(f"\n   MISS: {f['query']!r}\n         cleaned : {f['cleaned']!r}\n"
                  f"         expected: {f['expected']}\n         got     : {f['got']}")
    return {"queries": n, "exact": n_ok, "accuracy": round(n_ok / n, 4),
            "valid_after_parse": valid_n,
            "fields": {k: dict(v) for k, v in fc.items()},
            "failures": len(failures), "patterns_used": sorted(pat)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", action="store_true", help="score the held-out queries instead")
    ap.add_argument("--final", action="store_true", help="required with --heldout (one-time scoring)")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--save", action="store_true", help="write docs/week4_results.json")
    args = ap.parse_args()

    gold = json.load(open(GOLD))
    parser = QueryParser()
    validator = SchemaValidator(schema=parser.schema)

    if args.heldout and not args.final:
        sys.exit("The held-out queries are scored once. Add --final when the parser is frozen.")
    if args.heldout:
        res = {"heldout": report("held-out queries", gold["heldout"], parser, validator, True)}
    else:
        res = {"week1": report("Week 1 sample queries (tuning set)", gold["week1"], parser, validator, args.errors)}
    if args.save:
        old = json.load(open(RESULTS)) if RESULTS.exists() else {}
        old.update(res)
        RESULTS.parent.mkdir(exist_ok=True)
        json.dump(old, open(RESULTS, "w"), indent=1)
        print(f"\nsaved {RESULTS}")


if __name__ == "__main__":
    main()
