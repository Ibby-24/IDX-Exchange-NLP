"""Week 7: evaluate the query intent classifier.

Run from the repo root::

    python scripts/evaluate_intent.py                    # dev split: fit on train, score dev (use while developing)
    python scripts/evaluate_intent.py --compare          # dev: classifier vs majority class vs a hand-written rule baseline vs ablations
    python scripts/evaluate_intent.py --errors           # list every dev mistake (the queries are invented; no MLS text)
    python scripts/evaluate_intent.py --split test --final   # held-out: fit on train+dev, score test. Run ONCE.
    python scripts/evaluate_intent.py --extra my_queries.csv # score queries YOU labeled (csv: query,label), fit on everything

What is reported
  accuracy (target 80%+ on the held-out test), macro-F1, per-class precision/recall/F1, confusion matrix,
  bootstrap 95% interval for accuracy, accuracy on the flagged "hard" queries, accuracy by writing style,
  confidence behaviour: mean confidence vs accuracy, expected calibration error, and accuracy/coverage when the
  classifier is allowed to say "uncertain" below a threshold; robustness to upper-casing, dropped punctuation and
  typos.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_intent import (DATA, LABELS, QueryIntentClassifier, build_pipeline, normalize_query)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "week7_results.json"
TARGET = 0.80
THRESHOLDS = [0.4, 0.5, 0.55, 0.6, 0.7, 0.8]


# ------------------------------------------------------------------------------ metrics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def metrics(gold: Sequence[str], pred: Sequence[str]) -> dict:
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    per = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        p_, r_, f_ = prf(tp, fp, fn)
        per[c] = {"precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f_, 3), "support": tp + fn}
    conf = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    return {"n": n, "accuracy": round(acc, 4), "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
            "per_class": per, "confusion": conf}


def bootstrap_accuracy(correct: Sequence[int], n_boot: int = 2000, seed: int = 11):
    if not len(correct):
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct)
    vals = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def calibration(conf: np.ndarray, correct: np.ndarray, bins: int = 5) -> dict:
    """Expected calibration error with equal-width bins over [1/3, 1]."""
    edges = np.linspace(1 / 3, 1.0, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & ((conf < hi) if i < bins - 1 else (conf <= hi))
        if m.sum():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            rows.append({"range": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()), "mean_confidence": round(float(conf[m].mean()), 3),
                         "accuracy": round(float(correct[m].mean()), 3)})
    return {"ece": round(float(ece), 4), "mean_confidence": round(float(conf.mean()), 3),
            "accuracy": round(float(correct.mean()), 3), "bins": rows}


def coverage_table(conf: np.ndarray, correct: np.ndarray) -> list:
    out = []
    for t in THRESHOLDS:
        keep = conf >= t
        out.append({"threshold": t, "coverage": round(float(keep.mean()), 3),
                    "accuracy_when_answering": round(float(correct[keep].mean()), 3) if keep.any() else None,
                    "accuracy_when_uncertain": round(float(correct[~keep].mean()), 3) if (~keep).any() else None})
    return out


# ------------------------------------------------------------------------------ perturbations
def typo(q: str, rng: random.Random) -> str:
    words = q.split()
    cands = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not cands:
        return q
    i = rng.choice(cands)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


PERTURBATIONS = {
    "UPPER CASE": lambda q, rng: q.upper(),
    "no punctuation": lambda q, rng: re.sub(r"[^\w\s]", "", q),
    "one typo": typo,
}


# ------------------------------------------------------------------------------ baselines
class MajorityBaseline:
    def fit(self, X, y):
        self.label = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.label] * len(X)


_TIMING = re.compile(r"\b(this (week|weekend)|today|tomorrow|tmrw|tonight|asap|right away|immediately|available now|"
                     r"avail(?:able)? now|ready now|open house|open hse|open houses|just listed|new listings?|newly listed|"
                     r"price reduced|reduced price|close in \d+|can close|move in (now|this)|offer deadline|still available|"
                     r"schedule|showing|tour|viewing|pre-?approved|cash (offer|buyer)|make an offer|ready to (buy|offer|sign)|"
                     r"contact (me|an agent|agent)|ready to close|av now)\b", re.I)
_INFO = re.compile(r"^\s*(what|which|how|is|are|does|do|can|will|why|should)\b|\?|\b(compare|vs\.?|versus|pros and cons|pros cons|"
                   r"average|avg|typical|median|cost of|how much|rates?|ratings?|trends?|best|safest|rules|programs?|guide|"
                   r"low (hoa|property tax(es)?|taxes|crime)|no mello|mello-?roos|property tax(es)?|hoa fees?|insurance|"
                   r"commute|school district)\b", re.I)
_MONEY = re.compile(r"\b(under|below|max|up to|<)\s?\$?\d|\$?\d[\d,.]*\s?(k|m|million)\b", re.I)
_SPEC = re.compile(r"\b(\d+\s?(bed|bd|br|bath|ba|bedroom|bedrooms)|pool|garage|move[- ]in ready|turnkey|fha|va|seller financing|"
                   r"assumable|adu|sqft|square f)", re.I)


class RuleBaseline:
    """The tie-break rules of docs/intent_guidelines.md written as regular expressions. No training."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        out = []
        for q in X:
            if _TIMING.search(q):
                out.append("high_intent_inquiry")
            elif _INFO.search(q):
                out.append("researching")
            elif _MONEY.search(q) and _SPEC.search(q):
                out.append("high_intent_inquiry")
            else:
                out.append("browsing")
        return out


class PipelineModel:
    def __init__(self, **kw):
        self.pipe = build_pipeline(**kw)

    def fit(self, X, y):
        self.pipe.fit(list(X), list(y))
        return self

    def predict(self, X):
        return [str(p) for p in self.pipe.predict(list(X))]


# Non-housing text. There is no "other" class, so the classifier must flag these as uncertain rather than label them.
# The last three contain a real California city, which the in-scope check cannot tell apart from a housing search.
OUT_OF_SCOPE = ["what is the weather in Paris", "pizza near me", "how do I bake bread", "flights to New York next week",
                "hello", "asdf qwerty", "who won the game last night", "python tutorial for beginners",
                "stock market today", "best phone 2026", "how to lose weight fast", "movie times tonight",
                "translate hello to Italian", "used cars in Irvine", "jobs in San Diego", "restaurants in Pasadena"]

COMPARE = {
    "majority class": lambda: MajorityBaseline(),
    "rule baseline (guidelines as regexes)": lambda: RuleBaseline(),
    "LR words only": lambda: PipelineModel(char=False, parser_features=False),
    "LR words + chars": lambda: PipelineModel(char=True, parser_features=False),
    "LR words + chars + parser features (final)": lambda: PipelineModel(char=True, parser_features=True),
}


# ------------------------------------------------------------------------------ data
def load_rows(path: Path = DATA):
    return json.load(open(path))["queries"]


def read_extra(path: Path):
    """Queries to score: a csv with columns query,label or a json file shaped like intent_stress.json."""
    rows = []
    if str(path).lower().endswith(".json"):
        items = json.load(open(path))["queries"]
        for r in items:
            if r["intent"] not in LABELS:
                sys.exit(f"unknown label {r['intent']!r} in {path}; use one of {LABELS}")
            rows.append({"query": r["query"], "intent": r["intent"]})
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = (r.get("query") or "").strip()
            lab = (r.get("label") or r.get("intent") or "").strip()
            if q and lab:
                if lab not in LABELS:
                    sys.exit(f"unknown label {lab!r} in {path}; use one of {LABELS}")
                rows.append({"query": q, "intent": lab})
    return rows


# ------------------------------------------------------------------------------ report
def evaluate(train_rows, test_rows, threshold: float, label: str):
    clf = QueryIntentClassifier(uncertain_below=threshold).train([r["query"] for r in train_rows],
                                                                 [r["intent"] for r in train_rows])
    q = [r["query"] for r in test_rows]
    gold = [r["intent"] for r in test_rows]
    det = clf.predict_detailed_many(q)
    pred = [d["intent"] for d in det]
    conf = np.array([d["confidence"] for d in det])
    correct = np.array([int(g == p) for g, p in zip(gold, pred)])
    m = metrics(gold, pred)
    lo, hi = bootstrap_accuracy(list(correct))
    m["accuracy_ci95"] = [round(lo, 3), round(hi, 3)]
    hard = [i for i, r in enumerate(test_rows) if r.get("hard")]
    m["hard"] = {"n": len(hard), "accuracy": round(float(correct[hard].mean()), 3) if hard else None}
    m["easy"] = {"n": len(correct) - len(hard), "accuracy": round(float(np.delete(correct, hard).mean()), 3)
                 if len(correct) > len(hard) else None}
    by_style = defaultdict(list)
    for i, r in enumerate(test_rows):
        by_style[r.get("style", "?")].append(correct[i])
    m["by_style"] = {s: {"n": len(v), "accuracy": round(float(np.mean(v)), 3)} for s, v in sorted(by_style.items())}
    by_src = defaultdict(list)
    for i, r in enumerate(test_rows):
        by_src[r.get("source", "?")].append(correct[i])
    m["by_source"] = {s: {"n": len(v), "accuracy": round(float(np.mean(v)), 3)} for s, v in sorted(by_src.items())}
    m["calibration"] = calibration(conf, correct)
    m["coverage"] = coverage_table(conf, correct)
    m["uncertain_flag"] = {"threshold": threshold, "n_uncertain": int((conf < threshold).sum()),
                           "accuracy_uncertain": round(float(correct[conf < threshold].mean()), 3) if (conf < threshold).any() else None,
                           "accuracy_confident": round(float(correct[conf >= threshold].mean()), 3) if (conf >= threshold).any() else None}
    oos = clf.predict_detailed_many(OUT_OF_SCOPE)
    m["out_of_scope"] = {"n": len(oos), "flagged_uncertain": sum(d["uncertain"] for d in oos),
                         "not_flagged": [d["query"] for d in oos if not d["uncertain"]]}
    m["in_scope_false_on_labeled"] = int(sum(not d["in_scope"] for d in det))
    rng = random.Random(5)
    rob = {}
    for name, fn in PERTURBATIONS.items():
        accs = []
        for rep in range(5 if name == "one typo" else 1):
            r2 = random.Random(100 + rep)
            pp = clf.predict_detailed_many([fn(x, r2) for x in q])
            accs.append(np.mean([d["intent"] == g for d, g in zip(pp, gold)]))
        rob[name] = round(float(np.mean(accs)), 3)
    m["robustness_accuracy"] = rob
    errors = [{"query": r["query"], "gold": g, "predicted": p, "confidence": round(float(c), 2), "hard": bool(r.get("hard"))}
              for r, g, p, c in zip(test_rows, gold, pred, conf) if g != p]
    return m, errors, clf


def fmt(x):
    return f"{x * 100:5.1f}%"


def print_report(label: str, m: dict, errors, show_errors: bool):
    lo, hi = m["accuracy_ci95"]
    print(f"\n=== {label} ===  n={m['n']}")
    print(f"accuracy {fmt(m['accuracy'])}  (target {TARGET:.0%}+; 95% interval {fmt(lo)} to {fmt(hi)})   macro-F1 {fmt(m['macro_f1'])}")
    print(f"{'class':<22}{'P':>8}{'R':>8}{'F1':>8}{'n':>5}")
    for c, v in m["per_class"].items():
        print(f"{c:<22}{fmt(v['precision']):>8}{fmt(v['recall']):>8}{fmt(v['f1']):>8}{v['support']:>5}")
    print("confusion (rows = gold, columns = predicted: browsing / researching / high_intent):")
    for g in LABELS:
        print(f"  {g:<22}" + "".join(f"{m['confusion'][g][p]:>5}" for p in LABELS))
    print(f"hard queries: {m['hard']['n']} -> {fmt(m['hard']['accuracy']) if m['hard']['accuracy'] is not None else '-'}   "
          f"easy queries: {m['easy']['n']} -> {fmt(m['easy']['accuracy']) if m['easy']['accuracy'] is not None else '-'}")
    print("by style: " + ", ".join(f"{s} {fmt(v['accuracy'])} (n={v['n']})" for s, v in m["by_style"].items()))
    print("by source: " + ", ".join(f"{s} {fmt(v['accuracy'])} (n={v['n']})" for s, v in m["by_source"].items()))
    cal = m["calibration"]
    print(f"confidence: mean {cal['mean_confidence']:.2f} vs accuracy {cal['accuracy']:.2f}; expected calibration error {cal['ece']:.3f}")
    print("accuracy when only answering above a confidence threshold:")
    print(f"  {'threshold':<11}{'coverage':>9}{'acc answered':>14}{'acc uncertain':>15}")
    for r in m["coverage"]:
        a = f"{r['accuracy_when_answering'] * 100:.1f}%" if r["accuracy_when_answering"] is not None else "-"
        b = f"{r['accuracy_when_uncertain'] * 100:.1f}%" if r["accuracy_when_uncertain"] is not None else "-"
        print(f"  {r['threshold']:<11}{r['coverage'] * 100:>8.0f}%{a:>14}{b:>15}")
    print("robustness (accuracy after changing every query): " + ", ".join(f"{k} {fmt(v)}" for k, v in m["robustness_accuracy"].items()))
    o = m["out_of_scope"]
    print(f"non-housing text (no 'other' class): {o['flagged_uncertain']}/{o['n']} flagged uncertain; "
          f"labeled queries wrongly called out of scope: {m['in_scope_false_on_labeled']}")
    if o["not_flagged"]:
        print("  not flagged: " + "; ".join(o["not_flagged"]))
    if show_errors:
        print("\nERRORS")
        for e in sorted(errors, key=lambda e: e["gold"]):
            print(f"  [{e['gold'][:4]} -> {e['predicted'][:4]} {e['confidence']:.2f}{' hard' if e['hard'] else ''}] {e['query']}")


def cross_validate(rows, k: int = 5, seed: int = 3) -> dict:
    """Group k-fold over train+dev (near-duplicate groups stay together)."""
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["group"]].append(i)
    gids = sorted(groups)
    random.Random(seed).shuffle(gids)
    folds = [gids[i::k] for i in range(k)]
    accs = []
    for f in range(k):
        te = {i for g in folds[f] for i in groups[g]}
        tr = [rows[i] for i in range(len(rows)) if i not in te]
        ts = [rows[i] for i in sorted(te)]
        clf = QueryIntentClassifier().train([r["query"] for r in tr], [r["intent"] for r in tr])
        pred = [d["intent"] for d in clf.predict_detailed_many([r["query"] for r in ts])]
        accs.append(np.mean([p == r["intent"] for p, r in zip(pred, ts)]))
    return {"folds": k, "mean_accuracy": round(float(np.mean(accs)), 4), "fold_accuracies": [round(float(a), 3) for a in accs]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required for --split test: the held-out run happens once")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--compare", action="store_true", help="dev only: baselines and ablations")
    ap.add_argument("--cv", action="store_true", help="5-fold group cross-validation on train+dev")
    ap.add_argument("--extra", type=Path, default=None,
                    help="queries you labeled yourself: csv with columns query,label (or a json like data/processed/intent_stress.json)")
    ap.add_argument("--threshold", type=float, default=QueryIntentClassifier().uncertain_below)
    ap.add_argument("--save", action="store_true", help="write this run into docs/week7_results.json")
    args = ap.parse_args(argv)

    rows = load_rows()
    if args.extra:
        extra = read_extra(args.extra)
        clf_rows = [r for r in rows if r["split"] in ("train", "dev")]
        m, errs, _ = evaluate(clf_rows, extra, args.threshold, "extra")
        rb = RuleBaseline().predict([r["query"] for r in extra])
        rule_acc = sum(p == r["intent"] for p, r in zip(rb, extra)) / len(extra)
        print_report(f"EXTRA QUERIES ({args.extra.name}), model fit on train+dev ({len(clf_rows)} queries)", m, errs, True)
        print(f"rule baseline on the same queries: {fmt(rule_acc)}")
        if args.save:
            _save("extra:" + args.extra.name, {"classifier": m, "rule_baseline_accuracy": round(rule_acc, 4), "errors": errs})
        return 0
    if args.split == "test" and not args.final:
        sys.exit("The test split is held out. Run it once, when you are done tuning: --split test --final")

    if args.split == "dev":
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "dev"]
    else:
        train = [r for r in rows if r["split"] in ("train", "dev")]
        test = [r for r in rows if r["split"] == "test"]

    if args.compare:
        print(f"\nCOMPARISON on the {args.split} split (n={len(test)}), fit on {len(train)} queries")
        table = {}
        for name, mk in COMPARE.items():
            model = mk().fit([r["query"] for r in train], [r["intent"] for r in train])
            pred = model.predict([r["query"] for r in test])
            mm = metrics([r["intent"] for r in test], pred)
            table[name] = {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]}
            print(f"  {name:<46} accuracy {fmt(mm['accuracy'])}   macro-F1 {fmt(mm['macro_f1'])}")
        if args.save:
            _save(args.split, {"comparison": table})

    m, errs, clf = evaluate(train, test, args.threshold, args.split)
    print_report(f"{args.split.upper()} split, fit on {len(train)} queries", m, errs, args.errors)
    out = {"classifier": m, "errors_count": len(errs), "fit_on": len(train)}
    if args.cv:
        cv = cross_validate([r for r in rows if r["split"] in ("train", "dev")])
        print(f"\n5-fold group CV on train+dev: mean accuracy {fmt(cv['mean_accuracy'])}  folds {cv['fold_accuracies']}")
        out["cross_validation"] = cv
    if args.save:
        _save(args.split, out)
    return 0


def _save(split: str, payload: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(RESULTS)) if RESULTS.exists() else {}
    prev.setdefault(split, {}).update(payload)
    json.dump(prev, open(RESULTS, "w"), indent=1)
    print(f"saved {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
