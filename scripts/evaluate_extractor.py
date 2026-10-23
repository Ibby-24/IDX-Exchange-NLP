"""Week 3: evaluate EntityExtractor against the hand-labeled gold set.

Run from the repo root::

    python scripts/evaluate_extractor.py                 # dev split (use while developing)
    python scripts/evaluate_extractor.py --errors        # list every dev error (ids + values only)
    python scripts/evaluate_extractor.py --split test --final   # the held-out test split: run ONCE

Inputs
    data/processed/listing_sample.csv   MLS sample (git-ignored; made by data_loading.py)
    data/processed/gold_labels.json     gold labels (committed; offsets + values, no remark text)
    data/processed/taxonomy.json        Week 1 taxonomy (made by taxonomy_builder.py)

Scoring (per listing, per entity)
    gold value g (None = nothing explicit is stated), predicted value p
    TP = g is not None and p == g
    FP = p is not None and p != g          (spurious value OR wrong value)
    FN = g is not None and p != g          (missing value OR wrong value)
    A wrong value therefore costs one FP and one FN.
    Listings whose gold is {"ambiguous": true} are skipped for that entity.
    Span check: a TP also counts as a "span hit" if the chosen mention overlaps a gold span.
Amenities are scored as sets of taxonomy term ids per listing on the labeled subset.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_extractor import NaiveBaseline  # noqa: E402
from entity_extractor import EntityExtractor  # noqa: E402
from text_cleaning import TextCleaner         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
GOLD = ROOT / "data" / "processed" / "gold_labels.json"
RESULTS = ROOT / "docs" / "week3_results.json"

NUMERIC = [("bedrooms", "bedrooms"), ("bathrooms", "bathrooms"), ("sqft", "sqft"), ("price", "price")]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def overlaps(span, spans):
    return any(span[0] < e and span[1] > s for s, e in spans)


def load_data():
    if not SAMPLE.exists():
        raise SystemExit(f"Missing {SAMPLE}. Run: python scripts/data_loading.py")
    if not GOLD.exists():
        raise SystemExit(f"Missing {GOLD}")
    df = pd.read_csv(SAMPLE)
    df["L_ListingID"] = df["L_ListingID"].astype(str)
    gold = json.load(open(GOLD))["listings"]
    return df.set_index("L_ListingID"), gold


def evaluate(split, df, gold, ex, cleaner, with_errors=False):
    counts = {e: Counter() for e, _ in NUMERIC}
    per_listing = {e: [] for e, _ in NUMERIC}       # (tp, fp, fn) tuples for bootstrap
    errors = []
    amen = Counter()
    amen_cat = defaultdict(Counter)
    amen_listing = []
    skipped = 0
    for lid, g in gold.items():
        if g["split"] != split:
            continue
        if lid not in df.index:
            skipped += 1
            continue
        text = cleaner.clean_text(df.loc[lid, "remarks"])
        if hashlib.sha1(text.encode()).hexdigest() != g["text_sha1"]:
            skipped += 1
            print(f"warning: cleaned text of {lid} differs from when it was labeled; skipped", file=sys.stderr)
            continue
        chosen = {
            "bedrooms": ex.select_bedroom_mention(text),
            "bathrooms": ex.select_bathroom_mention(text),
            "sqft": ex.select_sqft_mention(text),
            "price": ex.select_price_mention(text),
        }
        for ent, key in NUMERIC:
            if key not in g:
                continue
            gv = g[key]
            if gv is not None and "ambiguous" in gv:
                counts[ent]["ambiguous_skipped"] += 1
                continue
            gold_val = gv["value"] if gv else None
            m = chosen[ent]
            pred = m.value if m else None
            if pred is not None and float(pred).is_integer():
                pred = int(pred)
            tp = int(gold_val is not None and pred == gold_val)
            fp = int(pred is not None and pred != gold_val)
            fn = int(gold_val is not None and pred != gold_val)
            counts[ent]["tp"] += tp
            counts[ent]["fp"] += fp
            counts[ent]["fn"] += fn
            counts[ent]["listings"] += 1
            counts[ent]["gold_positive"] += int(gold_val is not None)
            if tp and gv.get("spans") and overlaps((m.start, m.end), gv["spans"]):
                counts[ent]["span_hit"] += 1
            per_listing[ent].append((tp, fp, fn))
            if (fp or fn) and with_errors:
                kind = "wrong value" if fp and fn else ("missed" if fn else "spurious")
                errors.append((lid, ent, kind, gold_val, pred, m.role if m else None))
        if "amenities" in g:
            gold_ids = set(g["amenities"])
            pred_ids = {a["id"] for a in ex.extract_amenities(text)}
            tp, fp, fn = len(gold_ids & pred_ids), len(pred_ids - gold_ids), len(gold_ids - pred_ids)
            amen.update({"tp": tp, "fp": fp, "fn": fn, "listings": 1})
            amen_listing.append((tp, fp, fn))
            cat_of = {t["id"]: t["category"] for t in ex._terms}
            for i in gold_ids & pred_ids:
                amen_cat[cat_of[i]]["tp"] += 1
            for i in pred_ids - gold_ids:
                amen_cat[cat_of[i]]["fp"] += 1
                if with_errors:
                    errors.append((lid, "amenity", "false positive", None, i, None))
            for i in gold_ids - pred_ids:
                amen_cat[cat_of[i]]["fn"] += 1
                if with_errors:
                    errors.append((lid, "amenity", "missed", i, None, None))
    return counts, per_listing, amen, amen_cat, amen_listing, errors, skipped


def bootstrap_f1(rows, n=1000, seed=7):
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        tp, fp, fn = (sum(x[i] for x in s) for i in range(3))
        scores.append(prf(tp, fp, fn)[2])
    scores.sort()
    return scores[int(0.025 * n)], scores[int(0.975 * n) - 1]


def structured_agreement(df, ex, cleaner):
    """Independent cross-check: extractor vs the MLS structured beds/baths (all sample rows).

    The structured bathroom column turned out to be a whole number (2.5 baths is stored as 3),
    so bathrooms are also compared after rounding the extracted value up.
    """
    res = {"bedrooms": Counter(), "bathrooms": Counter()}
    for lid, row in df.iterrows():
        text = cleaner.clean_text(row["remarks"])
        for ent, col, fn in (("bedrooms", "beds", ex.extract_bedrooms), ("bathrooms", "baths", ex.extract_bathrooms)):
            pred, ref = fn(text), row[col]
            if pred is None or pd.isna(ref):
                continue
            res[ent]["both"] += 1
            res[ent]["exact"] += int(float(pred) == float(ref))
            res[ent]["rounded_up"] += int(math.ceil(float(pred)) == float(ref))
    return res


def report(split, counts, per_listing, amen, amen_cat, amen_listing, skipped):
    lines = []
    lines.append(f"\n=== {split.upper()} split ===" + (f"   ({skipped} listings skipped)" if skipped else ""))
    lines.append(f"{'entity':<11}{'n':>5}{'gold+':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>8}{'R':>8}{'F1':>8}   95% CI (F1)     span-hit")
    tot = Counter()
    out = {}
    for ent, _ in NUMERIC:
        c = counts[ent]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        lo, hi = bootstrap_f1(per_listing[ent])
        hit = f"{c['span_hit']}/{c['tp']}" if c["tp"] else "-"
        lines.append(f"{ent:<11}{c['listings']:>5}{c['gold_positive']:>7}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}"
                     f"{p:>8.3f}{r:>8.3f}{f:>8.3f}   [{lo:.2f}, {hi:.2f}]   {hit}")
        tot.update({k: c[k] for k in ("tp", "fp", "fn")})
        out[ent] = {"n": c["listings"], "gold_positive": c["gold_positive"], "tp": c["tp"], "fp": c["fp"],
                    "fn": c["fn"], "precision": p, "recall": r, "f1": f, "f1_ci95": [lo, hi],
                    "ambiguous_skipped": c["ambiguous_skipped"]}
    p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
    lines.append(f"{'NUMERIC':<11}{'':>5}{'':>7}{tot['tp']:>5}{tot['fp']:>5}{tot['fn']:>5}{p:>8.3f}{r:>8.3f}{f:>8.3f}   (micro over the 4 numeric entities)")
    out["numeric_micro"] = {"tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"], "precision": p, "recall": r, "f1": f}
    if amen["listings"]:
        ap, ar, af = prf(amen["tp"], amen["fp"], amen["fn"])
        lo, hi = bootstrap_f1(amen_listing)
        lines.append(f"{'amenities':<11}{amen['listings']:>5}{'':>7}{amen['tp']:>5}{amen['fp']:>5}{amen['fn']:>5}{ap:>8.3f}{ar:>8.3f}{af:>8.3f}   [{lo:.2f}, {hi:.2f}]   (term-id sets, {amen['listings']} listings)")
        for cat, c in sorted(amen_cat.items()):
            cp, cr, cf = prf(c["tp"], c["fp"], c["fn"])
            lines.append(f"    {cat:<24}TP {c['tp']:>3}  FP {c['fp']:>3}  FN {c['fn']:>3}   P {cp:.2f}  R {cr:.2f}  F1 {cf:.2f}")
        out["amenities"] = {"listings": amen["listings"], "tp": amen["tp"], "fp": amen["fp"], "fn": amen["fn"],
                            "precision": ap, "recall": ar, "f1": af, "f1_ci95": [lo, hi]}
        tp_all, fp_all, fn_all = tot["tp"] + amen["tp"], tot["fp"] + amen["fp"], tot["fn"] + amen["fn"]
        op, orr, of = prf(tp_all, fp_all, fn_all)
        lines.append(f"{'ALL':<11}{'':>5}{'':>7}{tp_all:>5}{fp_all:>5}{fn_all:>5}{op:>8.3f}{orr:>8.3f}{of:>8.3f}   (micro, numeric + amenities)")
        out["all_micro"] = {"precision": op, "recall": orr, "f1": of}
    return "\n".join(lines), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required for the test split (run it once)")
    ap.add_argument("--errors", action="store_true", help="list each error (listing id, entity, gold, predicted)")
    ap.add_argument("--save", action="store_true", help="write docs/week3_results.json")
    ap.add_argument("--baseline", action="store_true", help="score the naive baseline instead of EntityExtractor")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("The test split is held out. Re-run with --final when the extractor is frozen.")

    df, gold = load_data()
    cleaner = TextCleaner()
    ex = NaiveBaseline() if args.baseline else EntityExtractor()
    if not args.baseline:
        ex._load_taxonomy()
    counts, per_listing, amen, amen_cat, amen_listing, errors, skipped = evaluate(
        args.split, df, gold, ex, cleaner, with_errors=args.errors)
    if args.baseline:
        print("*** NAIVE BASELINE (starter-style regexes), not EntityExtractor ***")
    text, out = report(args.split, counts, per_listing, amen, amen_cat, amen_listing, skipped)
    print(text)
    if args.errors:
        print("\nErrors (listing id, entity, kind, gold, predicted, role of chosen mention):")
        for e in sorted(errors):
            print("  ", e)
    if not args.baseline:
        agree = structured_agreement(df, ex, cleaner)
        print("\nAgreement with structured MLS fields (all sample rows with both values; independent check):")
        for ent, c in agree.items():
            if c["both"]:
                print(f"  {ent:<10} exact {c['exact']}/{c['both']} = {c['exact'] / c['both']:.1%}"
                      f"   (extracted value rounded up: {c['rounded_up'] / c['both']:.1%})")
        out["structured_agreement"] = {k: dict(v) for k, v in agree.items()}
    if args.save:
        RESULTS.parent.mkdir(exist_ok=True)
        existing = json.load(open(RESULTS)) if RESULTS.exists() else {}
        existing[args.split + ("_baseline" if args.baseline else "")] = out
        json.dump(existing, open(RESULTS, "w"), indent=2)
        print(f"\nsaved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
