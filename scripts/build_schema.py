"""Week 4: build data/processed/schema.json (the list of valid cities and the allowed ranges).

    python scripts/build_schema.py                 # from data/processed/listing_sample.csv (1,000-listing sample)
    python scripts/build_schema.py --from-db       # from MySQL: every city in rets_property (use this one)

Why --from-db matters: the 1,000-listing sample contains only some of the cities. A city that is real
but missing from the sample would be rejected by the validator. The database has them all.

Database connection (same defaults as data_loading.py): host localhost, user root, password root,
database real_estate. Override with --host/--user/--password/--database or the environment variables
DB_HOST, DB_USER, DB_PASSWORD, DB_NAME.

schema.json contains only: city names, per-column observed min/max (numbers), the allowed limits, the
column whitelist. No listing text, no addresses, no ids. Whether to commit it is a team decision
(it is small and not listing data, but it is derived from MLS data); `build_schema.py` recreates it.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema_validator import DEFAULT_LIMITS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
OUT = ROOT / "data" / "processed" / "schema.json"

# Placeholder values that MLS data uses instead of a city. They are not places a user can search for.
NOT_A_CITY = {"", "other", "n/a", "na", "none", "unknown", "null"}

# search-filter column -> real column in rets_property (the only columns the SQL builder may touch)
COLUMNS = {
    "city": "L_City",
    "price": "L_SystemPrice",
    "bedrooms": "L_Keyword2",
    "bathrooms": "LM_Dec_3",
    "sqft": "LM_Int2_3",
    "remarks": "L_Remarks",
}


def tidy(name):
    return " ".join(str(name).split())


def pick_spelling(counts):
    """Several spellings of one city ("Palm Desert", "palm desert") -> the most common, ties -> as written."""
    by_key = {}
    for name, n in counts.items():
        by_key.setdefault(name.lower(), []).append((n, name))
    return sorted(max(v)[1] for v in by_key.values())


def cities_from_counts(raw_counts):
    counts = Counter()
    for name, n in raw_counts.items():
        name = tidy(name) if name is not None else ""
        if name.lower() in NOT_A_CITY:
            continue
        counts[name] += n
    return pick_spelling(counts)


def from_csv(path):
    if not path.exists():
        sys.exit(f"{path} not found. Run data_loading.py first, or use --from-db.")
    raw = Counter()
    ranges = {"price": [], "bedrooms": [], "bathrooms": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw[row.get("L_City")] += 1
            for key, col in (("price", "price"), ("bedrooms", "beds"), ("bathrooms", "baths")):
                try:
                    ranges[key].append(float(row[col]))
                except (KeyError, TypeError, ValueError):
                    pass
    observed = {k: [min(v), max(v)] for k, v in ranges.items() if v}
    return cities_from_counts(raw), observed, f"csv sample ({path.name})"


def connect_or_exit(args):
    """Connect to MySQL, or stop with a plain explanation instead of a traceback."""
    try:
        import mysql.connector
    except ImportError:
        sys.exit("mysql-connector-python is not installed (pip install mysql-connector-python).")
    try:
        return mysql.connector.connect(host=args.host, user=args.user, password=args.password,
                                       database=args.database)
    except mysql.connector.Error as e:
        sys.exit(
            f"Could not connect to MySQL at {args.host} (database '{args.database}'): {e}\n"
            "  - 'Can't connect ... (111)' means the container is not running:\n"
            "      docker compose up -d      then wait for 'ready for connections' in\n"
            "      docker compose logs -f mysql\n"
            "  - 'Access denied' means a wrong user/password: pass --user/--password or set DB_USER/DB_PASSWORD.\n"
            "  - 'Unknown database' means the dump is not imported yet (see the README setup).\n"
            "  - To skip the database for now, run without --from-db (uses the sample CSV).")


def from_db(args):
    conn = connect_or_exit(args)
    cur = conn.cursor()
    cur.execute(f"SELECT {COLUMNS['city']}, COUNT(*) FROM rets_property GROUP BY {COLUMNS['city']}")
    raw = Counter()
    for city, n in cur.fetchall():
        raw[city] += int(n)
    observed = {}
    for key in ("price", "bedrooms", "bathrooms", "sqft"):
        col = COLUMNS[key]
        cur.execute(f"SELECT MIN({col}), MAX({col}) FROM rets_property WHERE {col} IS NOT NULL")
        lo, hi = cur.fetchone()
        if lo is not None:
            observed[key] = [float(lo), float(hi)]
    conn.close()
    return cities_from_counts(raw), observed, "mysql rets_property"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-db", action="store_true", help="read cities from MySQL instead of the sample CSV")
    ap.add_argument("--csv", type=Path, default=SAMPLE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    ap.add_argument("--user", default=os.environ.get("DB_USER", "root"))
    ap.add_argument("--password", default=os.environ.get("DB_PASSWORD", "root"))
    ap.add_argument("--database", default=os.environ.get("DB_NAME", "real_estate"))
    args = ap.parse_args()

    cities, observed, source = from_db(args) if args.from_db else from_csv(args.csv)
    schema = {
        "table": "rets_property",
        "source": source,
        "columns": COLUMNS,
        "limits": DEFAULT_LIMITS,
        "observed": observed,
        "city_count": len(cities),
        "cities": cities,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(schema, f, indent=1)
    print(f"wrote {args.out}  ({len(cities)} cities, source: {source})")
    for key, (lo, hi) in observed.items():
        print(f"  observed {key}: {lo:g} .. {hi:g}   allowed: {DEFAULT_LIMITS.get(key)}")


if __name__ == "__main__":
    main()
