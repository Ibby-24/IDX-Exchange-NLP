"""Week 5: build the listing corpus that the relevance judgments refer to.

    python scripts/build_eval_corpus.py            # needs the MySQL container running

Why this exists: relevance_judgments.json says things like "for the query 'wet bar', listing 1176314410 is a
2". That only means something if listing 1176314410 is among the listings being searched. The 1,000-listing
sample in data/processed/listing_sample.csv is a random draw, so two people (or two runs of data_loading.py)
get different listings and the judged ones may be missing. Then every score is 0 (the first MiniLM run showed
exactly that: 99% of the results were "unjudged").

This script fetches the judged listings from MySQL by id and adds your sample listings as the other
"background" listings the searchers have to rank against. Result: data/processed/listing_sample_eval.csv
(columns L_ListingID, remarks). evaluate_search.py, label_judgments.py and benchmark_latency.py use it
automatically when it exists.

The output contains remark text. Its name matches the git-ignored pattern listing_sample*.csv. Never commit it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS = ROOT / "data" / "processed" / "relevance_judgments.json"
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
OUT = ROOT / "data" / "processed" / "listing_sample_eval.csv"
BATCH = 200


def judged_ids(path: Path = JUDGMENTS) -> List[str]:
    """Every listing id that has a judgment, sorted, without duplicates."""
    with open(path) as f:
        data = json.load(f)
    return sorted({str(k) for q in data["queries"] for k in q["judgments"]})


def build_fetch_sql(n: int) -> str:
    """SELECT with n bound placeholders. Ids are never written into the statement text."""
    if n < 1:
        raise ValueError("need at least one id")
    return "SELECT L_ListingID, L_Remarks FROM rets_property WHERE L_ListingID IN (" + ",".join(["%s"] * n) + ")"


def fetch_remarks(conn, ids: Sequence[str], qmark: bool = False, batch: int = BATCH) -> Dict[str, str]:
    """{listing id: remarks} for the ids found. ``qmark=True`` for sqlite3 (tests); MySQL uses %s."""
    found: Dict[str, str] = {}
    cur = conn.cursor()
    for i in range(0, len(ids), batch):
        part = list(ids[i:i + batch])
        sql = build_fetch_sql(len(part))
        if qmark:
            sql = sql.replace("%s", "?")
        cur.execute(sql, part)
        for lid, text in cur.fetchall():
            if text and str(text).strip():
                found[str(lid)] = str(text).strip()
    return found


def read_sample(path: Path) -> Dict[str, str]:
    """{id: remarks} from a listing_sample-style CSV (columns L_ListingID, remarks). Empty dict if missing."""
    out: Dict[str, str] = {}
    if not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("remarks") or "").strip()
            if t:
                out[str(row["L_ListingID"])] = t
    return out


def merge_corpus(judged: Dict[str, str], background: Dict[str, str]) -> Dict[str, str]:
    """Judged listings first, then background listings that are not already in. Order is stable."""
    merged = dict(judged)
    for k, v in background.items():
        merged.setdefault(k, v)
    return merged


def write_corpus(rows: Dict[str, str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["L_ListingID", "remarks"])
        for k, v in rows.items():
            w.writerow([k, v])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS)
    ap.add_argument("--sample", type=Path, default=SAMPLE, help="background listings (default: your listing_sample.csv)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    ap.add_argument("--user", default=os.environ.get("DB_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("DB_PASSWORD", "root"))
    ap.add_argument("--database", default=os.environ.get("DB_NAME", "real_estate"))
    args = ap.parse_args(argv)

    ids = judged_ids(args.judgments)
    background = read_sample(args.sample)
    already = {i for i in ids if i in background}
    from build_schema import connect_or_exit
    conn = connect_or_exit(args)
    judged = fetch_remarks(conn, [i for i in ids if i not in already])
    conn.close()
    judged.update({i: background[i] for i in already})
    missing = [i for i in ids if i not in judged]

    corpus = merge_corpus(judged, background)
    write_corpus(corpus, args.out)
    print(f"judged listings: {len(ids)}   found: {len(judged)} ({len(already)} were already in your sample)   "
          f"missing: {len(missing)}")
    print(f"background listings from {args.sample.name}: {len(background)}")
    print(f"wrote {args.out}  ({len(corpus)} listings)")
    if missing:
        print(f"WARNING: {len(missing)} judged ids are not in the database (first few: {missing[:5]}). "
              "Are you using the same rets_property dump? Scores will be lower bounds.")
    print("Do NOT commit this file (it contains remark text).")
    return 0 if len(judged) / max(len(ids), 1) >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
