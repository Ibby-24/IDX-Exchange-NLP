"""Week 6: evaluate the SignalExtractor.

Run from the repo root::

    python scripts/evaluate_signals.py                    # dev split (use while developing)
    python scripts/evaluate_signals.py --errors           # list every dev error (ids + signal names only)
    python scripts/evaluate_signals.py --split test --final    # held-out test split: run ONCE
    python scripts/evaluate_signals.py --no-db            # never try MySQL (use only CSVs you already have)

What is scored
  Free text (target 75%+)   33 yes/no signals x 120 hand-labeled listings (data/processed/signal_gold.json).
                            Headline = micro F1 over all signals (pooled TP/FP/FN); also per family
                            (condition / financing / location) and per signal. The two fields hoa and
                            mello_roos are scored as classification accuracy on listings that state them.
  Structured fields (90%+)  bedrooms, bathrooms, sqft, price from the Week 3 gold set (data/processed/gold_labels.json,
                            301 listings), read through the *new* pipeline (SignalExtractor.extract_signals).
                            The Week 3 EntityExtractor is not retuned in Week 6.

Texts: the gold files hold ids, labels and a SHA-1 of the cleaned text, never remark text. The texts are loaded
by listing id from the CSVs you already have (listing_sample.csv, listing_sample_eval.csv, listing_sample_signals.csv)
and, for ids missing there, fetched from MySQL and cached in listing_sample_signals.csv (git-ignored; contains remark
text; never commit it). If fewer than 90% of the gold listings can be loaded, the run stops with instructions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_extractor import (ALL_SIGNALS, FAMILIES, FAMILY_OF, SignalExtractor)  # noqa: E402
from build_eval_corpus import read_sample, fetch_remarks, write_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SIGNAL_GOLD = PROC / "signal_gold.json"
WEEK3_GOLD = PROC / "gold_labels.json"
TEXT_SOURCES = [PROC / "listing_sample.csv", PROC / "listing_sample_eval.csv", PROC / "listing_sample_signals.csv"]
CACHE = PROC / "listing_sample_signals.csv"
RESULTS = ROOT / "docs" / "week6_results.json"
MIN_COVERAGE = 0.9
NUMERIC = ["bedrooms", "bathrooms", "sqft", "price"]
TARGET_STRUCTURED = 0.90
TARGET_FREETEXT = 0.75


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ------------------------------------------------------------------------------ texts
def load_texts(ids: List[str], use_db: bool = True, sources=None, cache: Path = CACHE, conn_args=None) -> Dict[str, str]:
    """{id: raw remarks} for as many ids as can be found. Falls back to MySQL for the missing ones."""
    texts: Dict[str, str] = {}
    for p in (sources if sources is not None else TEXT_SOURCES):
        for k, v in read_sample(p).items():
            texts.setdefault(k, v)
    missing = [i for i in ids if i not in texts]
    if missing and use_db:
        fetched = fetch_from_db(missing, conn_args)
        if fetched:
            texts.update(fetched)
            merged = read_sample(cache)
            merged.update(fetched)
            write_corpus(merged, cache)
            print(f"fetched {len(fetched)} listings from MySQL and cached them in {cache.name}", file=sys.stderr)
    return texts


def fetch_from_db(ids, conn_args=None) -> Dict[str, str]:
    import os
    try:
        import mysql.connector
    except ImportError:
        print("note: mysql-connector-python is not installed; skipping the database.", file=sys.stderr)
        return {}
    a = conn_args or argparse.Namespace(host=os.environ.get("DB_HOST", "localhost"),
                                        user=os.environ.get("DB_USER", "root"),
                                        password=os.environ.get("DB_PASSWORD", "root"),
                                        database=os.environ.get("DB_NAME", "real_estate"))
    try:
        conn = mysql.connector.connect(host=a.host, user=a.user, password=a.password, database=a.database)
    except Exception as e:  # noqa: BLE001
        print(f"note: could not reach MySQL ({str(e)[:80]}); continuing with the CSVs only.", file=sys.stderr)
        return {}
    try:
        return fetch_remarks(conn, list(ids))
    finally:
        conn.close()


def check_coverage(found: int, total: int, what: str, allow_low: bool):
    if total and found / total < MIN_COVERAGE and not allow_low:
        sys.exit(
            f"STOP: only {found} of {total} {what} listings ({found / total:.0%}) could be loaded by id.\n"
            "The gold labels refer to specific listings, so scores on a partial set mean nothing. Start the MySQL\n"
            "container (docker compose up -d), wait for 'ready for connections', and run this command again; it fetches\n"
            "the missing listings by id and caches them in data/processed/listing_sample_signals.csv (git-ignored).\n"
            "(--allow-low-coverage overrides this check.)")


# ------------------------------------------------------------------------------ free-text signals
def score_signals(split, gold, texts, ex, cleaner_hash=True):
    per_sig = {s: Counter() for s in ALL_SIGNALS}
    fields = {"hoa": Counter(), "mello_roos": Counter()}
    per_listing = []          # (tp, fp, fn) for bootstrap
    errors = []
    used, skipped = 0, 0
    for lid, g in gold.items():
        if g["split"] != split:
            continue
        raw = texts.get(lid)
        if raw is None:
            skipped += 1
            continue
        cleaned = ex.entity_extractor.clean(raw)
        if g.get("text_sha1") and hashlib.sha1(cleaned.encode()).hexdigest() != g["text_sha1"]:
            print(f"warning: cleaned text of {lid} differs from when it was labeled; skipped", file=sys.stderr)
            skipped += 1
            continue
        used += 1
        pred, pf = _predict(ex, cleaned)
        gold_set = set(g["signals"])
        ltp = lfp = lfn = 0
        for s in ALL_SIGNALS:
            in_g, in_p = s in gold_set, s in pred
            if in_g and in_p:
                per_sig[s]["tp"] += 1; ltp += 1
            elif in_p:
                per_sig[s]["fp"] += 1; lfp += 1
                errors.append((lid, s, "false positive"))
            elif in_g:
                per_sig[s]["fn"] += 1; lfn += 1
                errors.append((lid, s, "missed"))
            per_sig[s]["gold_pos"] += int(in_g)
        per_listing.append((ltp, lfp, lfn))
        for fld in ("hoa", "mello_roos"):
            gv, pv = g.get(fld), pf[fld]
            if gv is not None or pv is not None:
                fields[fld]["n"] += 1
                fields[fld]["correct"] += int(gv == pv)
                if gv != pv:
                    errors.append((lid, fld, f"gold={gv} pred={pv}"))
    return per_sig, fields, per_listing, errors, used, skipped


def _predict(ex, cleaned):
    by_family, fields, _ = ex.signals_from_text(cleaned, clean=False)
    return {s for ss in by_family.values() for s in ss}, fields


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


def summarize_signals(per_sig):
    def agg(sigs):
        c = Counter()
        for s in sigs:
            c.update({k: per_sig[s][k] for k in ("tp", "fp", "fn", "gold_pos")})
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "gold_positives": c["gold_pos"]}
    out = {"overall": agg(ALL_SIGNALS), "families": {f: agg(ss) for f, ss in FAMILIES.items()}, "signals": {}}
    for s in ALL_SIGNALS:
        out["signals"][s] = agg([s])
        out["signals"][s]["family"] = FAMILY_OF[s]
    return out


# ------------------------------------------------------------------------------ structured fields
def score_structured(split, gold3, texts, ex):
    counts = {e: Counter() for e in NUMERIC}
    errors = []
    skipped = 0
    for lid, g in gold3.items():
        if g["split"] != split:
            continue
        raw = texts.get(lid)
        if raw is None:
            skipped += 1
            continue
        cleaned = ex.entity_extractor.clean(raw)
        if hashlib.sha1(cleaned.encode()).hexdigest() != g["text_sha1"]:
            skipped += 1
            continue
        out = ex.extract_signals({"L_ListingID": lid, "L_Remarks": raw})["entities"]
        for e in NUMERIC:
            if e not in g:
                continue
            gv = g[e]
            if gv is not None and "ambiguous" in gv:
                counts[e]["ambiguous_skipped"] += 1
                continue
            gold_val = gv["value"] if gv else None
            pred = out[e]
            if pred is not None and float(pred).is_integer():
                pred = int(pred)
            tp = int(gold_val is not None and pred == gold_val)
            fp = int(pred is not None and pred != gold_val)
            fn = int(gold_val is not None and pred != gold_val)
            counts[e].update({"tp": tp, "fp": fp, "fn": fn, "listings": 1, "gold_positive": int(gold_val is not None),
                              "exact": int(pred == gold_val)})
            if fp or fn:
                errors.append((lid, e, f"gold={gold_val} pred={pred}"))
    return counts, errors, skipped


def summarize_structured(counts):
    out = {}
    tot = Counter()
    for e, c in counts.items():
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        out[e] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                  "exact_match_accuracy": round(c["exact"] / c["listings"], 3) if c["listings"] else None,
                  "listings": c["listings"], "gold_positives": c["gold_positive"],
                  "tp": c["tp"], "fp": c["fp"], "fn": c["fn"]}
        tot.update({k: c[k] for k in ("tp", "fp", "fn", "listings", "exact")})
    p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
    out["overall"] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                      "exact_match_accuracy": round(tot["exact"] / tot["listings"], 3) if tot["listings"] else None,
                      "tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"]}
    return out


# ------------------------------------------------------------------------------ report
def fmt(x):
    return f"{x * 100:5.1f}%"


def report(split, sig, flds, per_listing, used, skipped, struct, sused):
    L = []
    L.append(f"\n=== {split.upper()} split ===")
    L.append(f"free-text signals: {used} listings scored" + (f" ({skipped} skipped)" if skipped else ""))
    o = sig["overall"]
    lo, hi = bootstrap_f1(per_listing)
    L.append(f"\nFREE TEXT (target {TARGET_FREETEXT:.0%}+)   micro P {fmt(o['precision'])}  R {fmt(o['recall'])}  F1 {fmt(o['f1'])}"
             f"   (95% CI for F1 {fmt(lo)} - {fmt(hi)})   gold positives {o['gold_positives']}")
    L.append(f"{'family':<12}{'P':>8}{'R':>8}{'F1':>8}{'gold+':>7}")
    for f, v in sig["families"].items():
        L.append(f"{f:<12}{fmt(v['precision']):>8}{fmt(v['recall']):>8}{fmt(v['f1']):>8}{v['gold_positives']:>7}")
    L.append(f"\n{'signal':<22}{'P':>8}{'R':>8}{'F1':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    for s, v in sig["signals"].items():
        if v["gold_positives"] == 0 and v["fp"] == 0:
            L.append(f"{s:<22}{'  (no gold positives, no false alarms)':>38}")
        else:
            L.append(f"{s:<22}{fmt(v['precision']):>8}{fmt(v['recall']):>8}{fmt(v['f1']):>8}{v['tp']:>5}{v['fp']:>5}{v['fn']:>5}")
    for fld, c in flds.items():
        if c["n"]:
            L.append(f"field {fld}: {c['correct']}/{c['n']} correct ({c['correct'] / c['n']:.0%}) on listings where gold or prediction has a value")
    L.append(f"\nSTRUCTURED (target {TARGET_STRUCTURED:.0%}+)   Week 3 gold, {sused} listings, through SignalExtractor")
    L.append(f"{'field':<12}{'P':>8}{'R':>8}{'F1':>8}{'exact-match acc':>17}{'gold+':>7}")
    for e in NUMERIC + ["overall"]:
        v = struct[e]
        acc = v["exact_match_accuracy"]
        L.append(f"{e:<12}{fmt(v['precision']):>8}{fmt(v['recall']):>8}{fmt(v['f1']):>8}"
                 f"{(fmt(acc) if acc is not None else '-'):>17}{v.get('gold_positives', ''):>7}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required for --split test: the held-out run happens once")
    ap.add_argument("--errors", action="store_true", help="list every error (listing id, signal, kind)")
    ap.add_argument("--no-db", action="store_true", help="never try MySQL")
    ap.add_argument("--allow-low-coverage", action="store_true")
    ap.add_argument("--save", action="store_true", help="write this split's results into docs/week6_results.json")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.final:
        sys.exit("The test split is held out. Run it once, when you are done tuning: --split test --final")
    for p in (SIGNAL_GOLD, WEEK3_GOLD):
        if not p.exists():
            sys.exit(f"Missing {p}")
    sgold = json.load(open(SIGNAL_GOLD))["listings"]
    w3gold = json.load(open(WEEK3_GOLD))["listings"]
    ids = sorted({i for i, g in sgold.items() if g["split"] == args.split} |
                 {i for i, g in w3gold.items() if g["split"] == args.split})
    texts = load_texts(ids, use_db=not args.no_db)
    s_ids = [i for i, g in sgold.items() if g["split"] == args.split]
    w_ids = [i for i, g in w3gold.items() if g["split"] == args.split]
    check_coverage(sum(i in texts for i in s_ids), len(s_ids), "signal-gold", args.allow_low_coverage)
    check_coverage(sum(i in texts for i in w_ids), len(w_ids), "Week 3 gold", args.allow_low_coverage)

    ex = SignalExtractor()
    per_sig, flds, per_listing, errs, used, skipped = score_signals(args.split, sgold, texts, ex)
    sig = summarize_signals(per_sig)
    counts, serrs, sskipped = score_structured(args.split, w3gold, texts, ex)
    struct = summarize_structured(counts)
    sused = max((c["listings"] for c in counts.values()), default=0)
    print(report(args.split, sig, flds, per_listing, used, skipped, struct, sused))
    if args.errors:
        print("\nERRORS (free text)")
        for lid, s, kind in sorted(errs):
            print(f"  {lid}  {s:<22} {kind}")
        print("\nERRORS (structured)")
        for lid, e, kind in sorted(serrs):
            print(f"  {lid}  {e:<10} {kind}")
    if args.save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        prev = json.load(open(RESULTS)) if RESULTS.exists() else {}
        prev[args.split] = {"free_text": sig, "fields": {k: dict(v) for k, v in flds.items()},
                            "structured": struct, "listings_scored": used, "structured_listings": sused}
        json.dump(prev, open(RESULTS, "w"), indent=1)
        print(f"saved {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
