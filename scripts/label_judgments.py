"""Week 5: judge the search results that have no relevance judgment yet, then re-run evaluate_search.py.

Why: evaluate_search.py counts a result nobody judged as "not relevant". A system whose top 10 contains many
unjudged listings (a new encoder such as MiniLM will surface listings the original pool never contained)
is therefore scored too low. Judging those listings removes that bias.

    python scripts/label_judgments.py export --encoder minilm     # writes data/processed/listing_sample_unjudged.csv
    (fill the grade column with 0, 1 or 2 in a spreadsheet; keep the file out of git)
    python scripts/label_judgments.py import data/processed/listing_sample_unjudged.csv

    python scripts/label_judgments.py interactive --encoder minilm   # judge in the terminal instead

Grades: 2 = clearly satisfies the query, 1 = partly (related, weaker, only implied), 0 = does not.
Judge each listing against the query text only, without looking at which system returned it.

Privacy: the export contains remark text, so it is named listing_sample_*.csv to match the existing
.gitignore pattern. Never commit it. The judgments file (relevance_judgments.json) stores only the
query, the listing id and the grade.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_search import JUDGMENTS, load_judgments  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data" / "processed" / "listing_sample_unjudged.csv"
VALID_GRADES = {0, 1, 2}


# ----------------------------------------------------------------------------- pure helpers
def find_unjudged(rankings: Dict[str, List[str]], queries: List[dict], top: int = 10) -> List[Tuple[str, str]]:
    """(query_id, listing_id) pairs in any system's top-N that have no judgment, in a stable shuffled order.

    ``rankings`` maps system name -> {query_id: [listing ids]}. Pairs are de-duplicated across systems and
    shuffled with a fixed seed so the order does not reveal which system returned a listing.
    """
    judged = {q["id"]: q["judgments"] for q in queries}
    seen, pairs = set(), []
    for per_query in rankings.values():
        for qid, ranked in per_query.items():
            for d in ranked[:top]:
                key = (qid, str(d))
                if qid in judged and str(d) not in judged[qid] and key not in seen:
                    seen.add(key)
                    pairs.append(key)
    random.Random(0).shuffle(pairs)
    return pairs


def parse_grade(value) -> int:
    """'2', ' 1 ', '0.0' -> int. Anything else raises ValueError."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty grade")
    number = float(text)
    if number != int(number) or int(number) not in VALID_GRADES:
        raise ValueError(f"grade must be 0, 1 or 2 (got {value!r})")
    return int(number)


def merge_grades(queries: List[dict], new: Iterable[Tuple[str, str, int]], overwrite: bool = False) -> dict:
    """Add (query_id, listing_id, grade) triples. Existing different grades are kept unless overwrite=True."""
    by_id = {q["id"]: q for q in queries}
    report = {"added": 0, "unchanged": 0, "conflicts": [], "changed": 0, "unknown_query": []}
    for qid, lid, g in new:
        q = by_id.get(qid)
        if q is None:
            report["unknown_query"].append(qid)
            continue
        lid = str(lid)
        old = q["judgments"].get(lid)
        if old is None:
            q["judgments"][lid] = g
            report["added"] += 1
        elif old == g:
            report["unchanged"] += 1
        elif overwrite:
            q["judgments"][lid] = g
            report["changed"] += 1
        else:
            report["conflicts"].append((qid, lid, old, g))
    return report


def read_graded_csv(path: Path) -> Tuple[List[Tuple[str, str, int]], List[str]]:
    """Rows with a filled grade column -> triples. Blank grades are skipped; bad grades are reported."""
    triples, problems = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), 2):
            raw = (row.get("grade") or "").strip()
            if not raw:
                continue
            try:
                triples.append((row["query_id"], str(row["listing_id"]), parse_grade(raw)))
            except (ValueError, KeyError) as e:
                problems.append(f"line {n}: {e}")
    return triples, problems


def save_judgments(queries: List[dict], path: Path = JUDGMENTS):
    with open(path) as f:
        data = json.load(f)
    data["queries"] = [{"id": q["id"], "query": q["query"], "type": q["type"],
                        "judgments": dict(sorted(q["judgments"].items()))} for q in queries]
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


# ----------------------------------------------------------------------------- commands
def collect_pairs(args, queries):
    from evaluate_search import build_systems, evaluate_system
    systems, _ = build_systems(args)
    rankings = {}
    for name, fn in systems.items():
        _, rk = evaluate_system(fn, queries, args.top)
        rankings[name] = rk
    return find_unjudged(rankings, queries, args.top)


def cmd_export(args):
    from semantic_search import load_corpus
    queries = load_judgments(args.judgments)
    by_id = {q["id"]: q for q in queries}
    pairs = collect_pairs(args, queries)
    ids, texts = load_corpus(args.corpus)
    remarks = dict(zip(ids, texts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", "listing_id", "remarks", "grade"])
        for qid, lid in pairs:
            w.writerow([qid, by_id[qid]["query"], lid, remarks.get(lid, ""), ""])
    print(f"{len(pairs)} unjudged (query, listing) pairs -> {args.out}")
    print("Fill the grade column (0/1/2), then: python scripts/label_judgments.py import", args.out)
    print("Do NOT commit this file (it contains remark text).")


def cmd_import(args):
    queries = load_judgments(args.judgments)
    triples, problems = read_graded_csv(args.file)
    for p in problems:
        print("skipped:", p)
    report = merge_grades(queries, triples, args.overwrite)
    save_judgments(queries, args.judgments)
    print(f"added {report['added']}, unchanged {report['unchanged']}, changed {report['changed']}, "
          f"conflicts {len(report['conflicts'])}, unknown queries {len(report['unknown_query'])}")
    for qid, lid, old, new in report["conflicts"]:
        print(f"  conflict {qid} {lid}: kept {old}, file says {new} (use --overwrite to replace)")


def cmd_interactive(args):
    from semantic_search import load_corpus
    queries = load_judgments(args.judgments)
    by_id = {q["id"]: q for q in queries}
    pairs = collect_pairs(args, queries)
    ids, texts = load_corpus(args.corpus)
    remarks = dict(zip(ids, texts))
    print(f"{len(pairs)} pairs to judge. Grades: 2 clearly, 1 partly, 0 no.  s = skip, q = save and quit.\n")
    added = 0
    for n, (qid, lid) in enumerate(pairs, 1):
        print(f"[{n}/{len(pairs)}] QUERY: {by_id[qid]['query']}\n{remarks.get(lid, '')}\n")
        while True:
            ans = input("grade (0/1/2/s/q): ").strip().lower()
            if ans == "q":
                save_judgments(queries, args.judgments)
                print(f"saved {added} new judgments")
                return
            if ans == "s":
                break
            try:
                by_id[qid]["judgments"][lid] = parse_grade(ans)
                added += 1
                break
            except ValueError as e:
                print(" ", e)
    save_judgments(queries, args.judgments)
    print(f"saved {added} new judgments")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "interactive"):
        s = sub.add_parser(name)
        s.add_argument("--encoder", default="lsa")
        s.add_argument("--chunk-words", type=int, default=100)
        s.add_argument("--overlap", type=int, default=20)
        s.add_argument("--top", type=int, default=10)
        s.add_argument("--no-whole", action="store_true", default=True)
        s.add_argument("--corpus", type=Path, default=None)
        s.add_argument("--allow-low-coverage", action="store_true")
        if name == "export":
            s.add_argument("--out", type=Path, default=DEFAULT_EXPORT)
    imp = sub.add_parser("import")
    imp.add_argument("file", type=Path)
    imp.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    {"export": cmd_export, "import": cmd_import, "interactive": cmd_interactive}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
