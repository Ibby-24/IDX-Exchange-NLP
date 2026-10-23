"""Week 4: SchemaValidator - checks parsed search filters before any SQL is built.

    from schema_validator import SchemaValidator
    v = SchemaValidator()                       # reads data/processed/schema.json
    ok, errors = v.validate_query({"city": "Irvin", "bedrooms": 3, "price_max": 50})
    # (False, ["Unknown city 'Irvin'. Did you mean 'Irvine'?",
    #          "price_max 50 is below the minimum allowed price (100000)."])

What it catches
  * a city that is not in the database (with a "did you mean" suggestion)
  * numbers outside the allowed range (price 100000-10000000, bedrooms 1-10, ... see ``DEFAULT_LIMITS``)
  * a minimum larger than its maximum ("min 5 bedrooms, max 3")
  * wrong types (text where a number is expected, True/False, NaN, infinity)
  * filter names this system does not know (so nothing unexpected reaches the SQL builder)
  * a query with no usable filter at all ("hello")

The limits come from schema.json (``limits`` section) so the team can change them without
touching code. ``DEFAULT_LIMITS`` are the values from the Week 4 spec.
"""
from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "data" / "processed" / "schema.json"

# Week 4 spec: price 100000-10000000, bedrooms 1-10. Bathrooms and square feet are our additions.
DEFAULT_LIMITS = {
    "price": [100_000, 10_000_000],
    "bedrooms": [1, 10],
    "bathrooms": [1, 10],
    "sqft": [200, 50_000],
}

# filter name -> (limit group, human label)
NUMERIC_FILTERS = {
    "price_min": ("price", "price"),
    "price_max": ("price", "price"),
    "bedrooms": ("bedrooms", "bedrooms"),
    "bedrooms_min": ("bedrooms", "bedrooms"),
    "bedrooms_max": ("bedrooms", "bedrooms"),
    "bathrooms": ("bathrooms", "bathrooms"),
    "bathrooms_min": ("bathrooms", "bathrooms"),
    "bathrooms_max": ("bathrooms", "bathrooms"),
    "sqft_min": ("sqft", "square feet"),
    "sqft_max": ("sqft", "square feet"),
}
TEXT_FILTERS = {"city"}
LIST_FILTERS = {"cities", "amenities", "exclude_amenities"}
KNOWN_FILTERS = set(NUMERIC_FILTERS) | TEXT_FILTERS | LIST_FILTERS

# (minimum filter, maximum filter, label) pairs that must satisfy min <= max
RANGE_PAIRS = [
    ("price_min", "price_max", "price"),
    ("bedrooms_min", "bedrooms_max", "bedrooms"),
    ("bathrooms_min", "bathrooms_max", "bathrooms"),
    ("sqft_min", "sqft_max", "square feet"),
]


def is_number(value) -> bool:
    """A real, finite int/float. bool is rejected (True == 1 in Python) and so are NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


class SchemaValidator:
    def __init__(self, schema_path: Optional[Path] = None, schema: Optional[dict] = None):
        if schema is None:
            path = Path(schema_path) if schema_path else DEFAULT_SCHEMA
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Build it with: python scripts/build_schema.py "
                    "(add --from-db to read the cities from MySQL).")
            with open(path) as f:
                schema = json.load(f)
        self.schema = schema
        self.cities: List[str] = list(schema["cities"])
        self._city_lookup = {c.lower(): c for c in self.cities}
        limits = dict(DEFAULT_LIMITS)
        limits.update(schema.get("limits", {}))
        self.limits = limits

    # ------------------------------------------------------------------ cities
    def canonical_city(self, name: str) -> Optional[str]:
        """The database spelling of *name* (case/spacing-insensitive), or None."""
        if not isinstance(name, str):
            return None
        return self._city_lookup.get(" ".join(name.lower().split()))

    def suggest_cities(self, name: str, n: int = 3) -> List[str]:
        if not isinstance(name, str):
            return []
        key = " ".join(name.lower().split())
        close = difflib.get_close_matches(key, list(self._city_lookup), n=n, cutoff=0.75)
        # "Big Bear" is not a typo of "Big Bear City" by edit distance alone, so also offer prefix matches
        prefix = [k for k in self._city_lookup if len(key) >= 4 and (k.startswith(key + " "))]
        out = []
        for k in prefix + close:
            c = self._city_lookup[k]
            if c not in out:
                out.append(c)
        return out[:n]

    def _city_error(self, name) -> Optional[str]:
        if not isinstance(name, str) or not name.strip():
            return f"City must be a non-empty text value, got {name!r}."
        if self.canonical_city(name) is not None:
            return None
        sugg = self.suggest_cities(name)
        hint = " Did you mean " + " or ".join(f"'{s}'" for s in sugg) + "?" if sugg else ""
        return f"Unknown city {name!r}: not in the database.{hint}"

    # ---------------------------------------------------------------- validate
    def validate_query(self, filters: Dict) -> Tuple[bool, List[str]]:
        """Return (is_valid, errors). Never raises on bad input; bad input is reported as errors."""
        errors: List[str] = []
        if not isinstance(filters, dict):
            return False, [f"Filters must be a dict, got {type(filters).__name__}."]

        unknown = sorted(str(k) for k in filters if k not in KNOWN_FILTERS)
        if unknown:
            errors.append("Unknown filter(s): " + ", ".join(unknown) + ".")

        usable = {k: v for k, v in filters.items() if k in KNOWN_FILTERS and v not in (None, [], "")}
        if not usable and not unknown:
            errors.append("No search criteria found (try a city, price, bedrooms, bathrooms or an amenity).")

        # cities
        if "city" in usable:
            e = self._city_error(usable["city"])
            if e:
                errors.append(e)
        if "cities" in usable:
            cities = usable["cities"]
            if not isinstance(cities, list):
                errors.append("cities must be a list of city names.")
            else:
                for c in cities:
                    e = self._city_error(c)
                    if e:
                        errors.append(e)

        # numbers
        for key, (group, label) in NUMERIC_FILTERS.items():
            if key not in usable:
                continue
            value = usable[key]
            if not is_number(value):
                errors.append(f"{key} must be a finite number, got {value!r}.")
                continue
            lo, hi = self.limits[group]
            if value < lo:
                errors.append(f"{key} {_fmt(value)} is below the minimum allowed {label} ({_fmt(lo)}).")
            elif value > hi:
                errors.append(f"{key} {_fmt(value)} is above the maximum allowed {label} ({_fmt(hi)}).")

        # min <= max
        for lo_key, hi_key, label in RANGE_PAIRS:
            lo_v, hi_v = usable.get(lo_key), usable.get(hi_key)
            if is_number(lo_v) and is_number(hi_v) and lo_v > hi_v:
                errors.append(f"{lo_key} ({_fmt(lo_v)}) is greater than {hi_key} ({_fmt(hi_v)}).")
        if "bedrooms" in usable and is_number(usable["bedrooms"]):
            b = usable["bedrooms"]
            lo_v, hi_v = usable.get("bedrooms_min"), usable.get("bedrooms_max")
            if (is_number(lo_v) and b < lo_v) or (is_number(hi_v) and b > hi_v):
                errors.append("bedrooms conflicts with bedrooms_min/bedrooms_max.")

        # amenity lists
        for key in ("amenities", "exclude_amenities"):
            if key in usable:
                v = usable[key]
                if not isinstance(v, list) or not all(isinstance(a, str) and a.strip() for a in v):
                    errors.append(f"{key} must be a list of non-empty text values.")
        if all(k in usable for k in ("amenities", "exclude_amenities")) and \
                isinstance(usable["amenities"], list) and isinstance(usable["exclude_amenities"], list):
            both = sorted(set(usable["amenities"]) & set(usable["exclude_amenities"]))
            if both:
                errors.append("Amenity both required and excluded: " + ", ".join(both) + ".")

        return (not errors), errors


def _fmt(x) -> str:
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
