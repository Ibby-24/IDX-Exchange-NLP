"""Week 4: type a search, see the filters, the validation result and the SQL - and optionally run it.

    python scripts/run_query.py "3 bed under 700k in Irvine"          # parse + validate + show SQL
    python scripts/run_query.py "3 bed under 700k in Irvine" --sample # also run it on the 1,000-row sample (no Docker needed)
    python scripts/run_query.py "3 bed under 700k in Irvine" --mysql  # also run it on your MySQL container

--sample loads data/processed/listing_sample.csv into an in-memory SQLite table with the same column
names as rets_property, so the generated SQL can be tried without a database. It prints listing ids,
cities, prices, beds and baths only - never remarks text.

--mysql uses mysql.connector with the same defaults as data_loading.py (localhost, root/root,
real_estate; override with --host/--user/--password/--database or DB_HOST, DB_USER, DB_PASSWORD, DB_NAME).
Run `python scripts/build_schema.py --from-db` first so the city list covers the whole database.
"""
import argparse
import csv
import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_parser import QueryParser          # noqa: E402
from schema_validator import SchemaValidator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"


# ------------------------------------------------------------------ SQLite stand-in for MySQL
def _regexp_like(text, pattern, flags="i"):
    """MySQL 8 REGEXP_LIKE(text, pattern, 'i') for SQLite."""
    if text is None:
        return None
    return 1 if re.search(pattern, text, re.I if "i" in (flags or "") else 0) else 0


def sqlite_connection():
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP_LIKE", 3, _regexp_like)
    conn.execute(
        "CREATE TABLE rets_property (L_ListingID TEXT, L_Address TEXT, L_City TEXT, L_SystemPrice REAL, "
        "L_Keyword2 INTEGER, LM_Dec_3 REAL, LM_Int2_3 INTEGER, L_Remarks TEXT)")
    return conn


def load_sample_into_sqlite(path=SAMPLE):
    """listing_sample.csv columns -> rets_property column names. Square footage is not in the sample (NULL)."""
    conn = sqlite_connection()
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            def num(key):
                try:
                    return float(r[key])
                except (KeyError, TypeError, ValueError):
                    return None
            beds = num("beds")
            rows.append((r.get("L_ListingID"), r.get("L_Address"), (r.get("L_City") or "").strip(),
                         num("price"), int(beds) if beds is not None else None, num("baths"), None,
                         r.get("remarks")))
    conn.executemany("INSERT INTO rets_property VALUES (?,?,?,?,?,?,?,?)", rows)
    return conn


def to_qmark(sql):
    """mysql.connector uses %s placeholders, sqlite3 uses ?."""
    return sql.replace("%s", "?")


def execute_sqlite(conn, sql, params):
    return conn.execute(to_qmark(sql), params).fetchall()


def execute_mysql(sql, params, args):
    from build_schema import connect_or_exit
    conn = connect_or_exit(args)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows


# ------------------------------------------------------------------ CLI
def show_rows(rows, columns):
    print(f"\n{len(rows)} row(s)   columns: {', '.join(columns)}")
    for r in rows[:10]:
        print("  ", r)
    if len(rows) > 10:
        print(f"   ... {len(rows) - 10} more")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="the search text, in quotes")
    ap.add_argument("--sample", action="store_true", help="run on the sample CSV via in-memory SQLite")
    ap.add_argument("--mysql", action="store_true", help="run on the MySQL container")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--schema", type=Path, default=None)
    ap.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    ap.add_argument("--user", default=os.environ.get("DB_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("DB_PASSWORD", "root"))
    ap.add_argument("--database", default=os.environ.get("DB_NAME", "real_estate"))
    args = ap.parse_args(argv)

    parser = QueryParser(schema_path=args.schema)
    validator = SchemaValidator(schema=parser.schema)
    res = parser.search(args.query, validator=validator, limit=args.limit)

    print(f"query    : {res['query']}")
    print(f"cleaned  : {res['cleaned']}")
    print(f"filters  : {res['filters']}")
    for note in res["ignored"]:
        print(f"note     : {note}")
    if not res["valid"]:
        print("valid    : NO")
        for e in res["errors"]:
            print(f"  - {e}")
        return 1
    print("valid    : yes")
    print(f"sql      : {res['sql']}")
    print(f"params   : {res['params']}")

    columns = res["sql"].split(" FROM ")[0].replace("SELECT ", "").split(", ")
    if args.sample:
        if not SAMPLE.exists():
            sys.exit(f"{SAMPLE} not found (run data_loading.py first).")
        show_rows(execute_sqlite(load_sample_into_sqlite(), res["sql"], res["params"]), columns)
    if args.mysql:
        show_rows(execute_mysql(res["sql"], res["params"], args), columns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
