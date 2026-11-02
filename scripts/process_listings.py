"""Week 6: run the SignalExtractor over the whole rets_property table (or the sample CSV).

Run from the repo root::

    python scripts/process_listings.py --sample                  # the 1,000-listing CSV, no database needed
    python scripts/process_listings.py --limit 2000              # first 2,000 rows from MySQL (smoke test)
    python scripts/process_listings.py                           # the entire table (MySQL container must be running)
    python scripts/process_listings.py --compare-columns         # also compare extracted beds/baths/sqft/price with the DB columns
    python scripts/process_listings.py --resume                  # continue an interrupted full run

Output (both in data/processed/, git-ignored because they are derived from MLS data)
    listing_signals.jsonl        one JSON object per listing (schema in scripts/signal_extractor.py); no remark text
    listing_signals_summary.json counts and rates: how many listings have each signal, fill rate of each entity,
                                 optional agreement with the database columns, run time

Rows are read in id order in batches of --batch (keyset pagination: ``WHERE L_ListingID > last``), so memory stays
flat and an interrupted run can resume. Database defaults match the other scripts (localhost, root/root, real_estate;
override with --host/--user/--password/--database or DB_HOST, DB_USER, DB_PASSWORD, DB_NAME).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_extractor import ALL_SIGNALS, FAMILIES, SignalExtractor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
OUT = ROOT / "data" / "processed" / "listing_signals.jsonl"
SUMMARY = ROOT / "data" / "processed" / "listing_signals_summary.json"
BATCH = 2000

# database column -> role (same mapping as build_schema.py)
COLS = {"bedrooms": "L_Keyword2", "bathrooms": "LM_Dec_3", "sqft": "LM_Int2_3", "price": "L_SystemPrice"}


def build_batch_sql(with_columns: bool, batch: int, qmark: bool = False) -> str:
    """Keyset-paginated SELECT. The last id and the batch size are bound parameters, never pasted into the text."""
    cols = "L_ListingID, L_Remarks" + (", " + ", ".join(COLS.values()) if with_columns else "")
    ph = "?" if qmark else "%s"
    return (f"SELECT {cols} FROM rets_property WHERE L_ListingID > {ph} ORDER BY L_ListingID LIMIT {ph}")


def iter_db_rows(conn, with_columns: bool, batch: int = BATCH, start_after="", limit: Optional[int] = None,
                 qmark: bool = False) -> Iterator[dict]:
    """Yield dict rows (L_ListingID, L_Remarks[, DB columns]) in id order."""
    cur = conn.cursor()
    sql = build_batch_sql(with_columns, batch, qmark)
    last, seen = start_after, 0
    names = ["L_ListingID", "L_Remarks"] + (list(COLS.values()) if with_columns else [])
    while True:
        n = batch if limit is None else min(batch, limit - seen)
        if n <= 0:
            return
        cur.execute(sql, (last, n))
        rows = cur.fetchall()
        if not rows:
            return
        for r in rows:
            yield dict(zip(names, r))
        seen += len(rows)
        last = rows[-1][0]
        if len(rows) < n:
            return


def iter_sample_rows(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                return
            yield {"L_ListingID": row["L_ListingID"], "L_Remarks": row.get("remarks", ""),
                   "L_Keyword2": row.get("beds"), "LM_Dec_3": row.get("baths"),
                   "LM_Int2_3": row.get("sqft"), "L_SystemPrice": row.get("price")}


def _num(x):
    try:
        if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


class Summary:
    """Streaming counters: no per-listing state is kept."""

    def __init__(self):
        self.n = 0
        self.with_text = 0
        self.signal = Counter()
        self.family_any = Counter()
        self.fields = {"hoa": Counter(), "mello_roos": Counter()}
        self.entity_filled = Counter()
        self.amenity_listings = 0
        self.compare = {e: Counter() for e in COLS}

    def add(self, out: dict, row: dict, compare: bool):
        self.n += 1
        has_text = bool((row.get("L_Remarks") or "").strip())
        self.with_text += int(has_text)
        for fam, key in (("condition", "condition_keywords"), ("financing", "financing_terms"),
                         ("location", "location_features")):
            for s in out[key]:
                self.signal[s] += 1
            self.family_any[fam] += int(bool(out[key]))
        for fld, v in out["fields"].items():
            if v:
                self.fields[fld][v] += 1
        for e, v in out["entities"].items():
            self.entity_filled[e] += int(v is not None)
        self.amenity_listings += int(bool(out["amenities"]))
        if compare:
            for e, col in COLS.items():
                db, ex = _num(row.get(col)), _num(out["entities"][e])
                if db is None or ex is None:
                    c = self.compare[e]
                    c["db_missing" if db is None else "extracted_missing"] += 1
                    continue
                c = self.compare[e]
                c["both"] += 1
                c["exact"] += int(db == ex)
                if e == "bathrooms":      # the DB column stores whole numbers (2.5 baths -> 3); see Week 3 notes
                    c["exact_or_rounded_up"] += int(db == ex or db == math.ceil(ex))

    def as_dict(self, seconds: float) -> dict:
        n = max(self.n, 1)
        cmp_out = {}
        for e, c in self.compare.items():
            if c["both"]:
                key = "exact_or_rounded_up" if e == "bathrooms" else "exact"
                cmp_out[e] = {"both_present": c["both"], "agree": c[key], "agreement": round(c[key] / c["both"], 4),
                              "db_missing_extracted_present": c["db_missing"],
                              "db_present_extracted_missing": c["extracted_missing"]}
        return {
            "listings": self.n, "with_remarks": self.with_text, "seconds": round(seconds, 1),
            "listings_per_second": round(self.n / seconds, 1) if seconds else None,
            "signal_listing_counts": {s: self.signal[s] for s in ALL_SIGNALS},
            "signal_rates": {s: round(self.signal[s] / n, 4) for s in ALL_SIGNALS},
            "families_with_any_signal": {f: round(self.family_any[f] / n, 4) for f in FAMILIES},
            "fields": {k: dict(v) for k, v in self.fields.items()},
            "entity_fill_rate": {e: round(v / n, 4) for e, v in self.entity_filled.items()},
            "listings_with_amenities": round(self.amenity_listings / n, 4),
            "agreement_with_db_columns": cmp_out or None,
        }


def run(rows: Iterable[dict], out_path: Path, ex: SignalExtractor, compare: bool, mode: str = "w",
        progress_every: int = 5000) -> Summary:
    summ = Summary()
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode, encoding="utf-8") as f:
        for row in rows:
            out = ex.extract_signals(row)
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            summ.add(out, row, compare)
            if progress_every and summ.n % progress_every == 0:
                print(f"  {summ.n:,} listings  ({summ.n / (time.time() - t0):.0f}/s)", file=sys.stderr)
    summ.seconds = time.time() - t0
    return summ


def last_id(path: Path) -> str:
    """Last listing_id already written (for --resume). Empty string if nothing is there."""
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", "ignore").strip().splitlines()
    for line in reversed(tail):
        try:
            return str(json.loads(line)["listing_id"])
        except (ValueError, KeyError):
            continue
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="read data/processed/listing_sample.csv instead of MySQL")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many listings")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--compare-columns", action="store_true",
                    help="also compare extracted beds/baths/sqft/price with L_Keyword2, LM_Dec_3, LM_Int2_3, L_SystemPrice")
    ap.add_argument("--resume", action="store_true", help="continue after the last id already in --out (DB mode only)")
    ap.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    ap.add_argument("--user", default=os.environ.get("DB_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("DB_PASSWORD", "root"))
    ap.add_argument("--database", default=os.environ.get("DB_NAME", "real_estate"))
    args = ap.parse_args(argv)

    ex = SignalExtractor()
    if args.sample:
        if not SAMPLE.exists():
            sys.exit(f"Missing {SAMPLE}. Run: python scripts/data_loading.py")
        rows = iter_sample_rows(SAMPLE, args.limit)
        mode = "w"
        conn = None
    else:
        from build_schema import connect_or_exit
        conn = connect_or_exit(args)
        start = last_id(args.out) if args.resume else ""
        mode = "a" if args.resume and start else "w"
        if mode == "a":
            print(f"resuming after listing {start}", file=sys.stderr)
        rows = iter_db_rows(conn, args.compare_columns, args.batch, start, args.limit)

    summ = run(rows, args.out, ex, args.compare_columns, mode)
    if conn is not None:
        conn.close()
    result = summ.as_dict(summ.seconds)
    if mode == "a":
        result["note"] = "resumed run: counts cover only the listings processed in this invocation"
    with open(args.summary, "w") as f:
        json.dump(result, f, indent=1)
    print(f"processed {result['listings']:,} listings in {result['seconds']}s ({result['listings_per_second']}/s)")
    print(f"wrote {args.out}  and  {args.summary}")
    print("any signal: " + ", ".join(f"{f} {v:.0%}" for f, v in result["families_with_any_signal"].items()))
    if result["agreement_with_db_columns"]:
        for e, v in result["agreement_with_db_columns"].items():
            print(f"agreement with DB {COLS[e]:<14} {v['agreement']:.1%}  (n={v['both_present']:,})")
    print("Do NOT commit these outputs (derived from MLS data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
