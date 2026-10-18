"""Week 2: clean the Week 1 sample and report what changed.

Run from the repo root:
    python scripts/generate_clean_sample.py

Reads   data/processed/listing_sample.csv        (created by scripts/data_loading.py)
Writes  data/processed/listing_sample_clean.csv  (adds a `remarks_clean` column; git-ignored, MLS data)
"""
import re
from pathlib import Path

import pandas as pd

from text_cleaning import TextCleaner

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "listing_sample.csv"
OUT_PATH = ROOT / "data" / "processed" / "listing_sample_clean.csv"

#noise patterns expected by the cleaner to remove: name -> regex
NOISE_PATTERNS = {
    "curly quotes": r"[\u2019\u2018\u201c\u201d]",
    "em/en dashes": r"[\u2013\u2014]",
    "zero-width characters": r"[\u200b\u200c\u200d]",
    "embedded newlines": r"\n",
    "double spaces": r"  ",
    "markdown ** markers": r"\*\*",
    "bullets": r"\u2022",
    "'sq ft' style units": r"(?i)\bsq\.?[ /]?ft\b|\bsqft\b",
    "'A/C' shorthand": r"\bA/C\b",
    "'approx.' shorthand": r"\bapprox\.",
    "'w/' shorthand": r"\bw/",
    "$ + k/m price shorthand": r"\$\d+(?:\.\d+)?\s?[kKmM]\b",
}


def count(series, pattern):
    return int(series.str.contains(pattern, regex=True).sum())


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run: python scripts/data_loading.py")

    cleaner = TextCleaner()
    df = pd.read_csv(IN_PATH)
    df["remarks_clean"] = df["remarks"].map(cleaner.clean_text)

    changed = int((df["remarks"] != df["remarks_clean"]).sum())
    not_idempotent = int((df["remarks_clean"].map(cleaner.clean_text) != df["remarks_clean"]).sum())

    print(f"Rows cleaned:             {len(df)}")
    print(f"Remarks changed:          {changed} ({changed / len(df):.1%})")
    print(f"Avg length before/after:  {df['remarks'].str.len().mean():.0f} -> {df['remarks_clean'].str.len().mean():.0f}")
    print(f"Changed by a 2nd pass:    {not_idempotent} (must be 0)")
    print("\nNoise patterns (remarks containing each), before -> after:")
    for name, pattern in NOISE_PATTERNS.items():
        print(f"  {name:<26} {count(df['remarks'], pattern):>4} -> {count(df['remarks_clean'], pattern):>4}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
