"""Week 2: TextCleaner - standardizes messy MLS listing remarks and search queries.

Used for both data sources:
  * listing remarks  (curly quotes, em dashes, double spaces, "sq ft" variants, A/C, approx., ...)
  * search queries   ("3br 2ba under 450k w/ pool")

Every rule below came from a pattern found in the 1,000-remark sample (see
docs/week2_profiling_report.md) and is pinned by tests/test_week2.py.

Design principles
  1. Conservative: only change text when the meaning is unambiguous.
     Protected acronyms (HOA, ADU, HVAC, RV, EV, BBQ, ...) are never expanded.
  2. Whole words only: "br" inside "brick" or "ac" inside "across" is never touched.
  3. Idempotent: clean_text(clean_text(x)) == clean_text(x).
"""
import html
import json
import math
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

#words that, when they directly precede "450k" / "1.2m" / "1.5 million", mark it as a price.
PRICE_CONTEXT = re.compile(r"(?i)\b(?:under|over|below|above|around|at|up to|about|than|max)\s+$")

STOPWORDS = set(
    """a an the and or of to in on at for with from by is are was were be been being it its this that these
    those as your you our we their his her has have had will can into over under up out all any each also
    but not no more most very just than then there which who whom while within without one two three four
    five per via only even well plus other another such through throughout every both many much new""".split()
)


class TextCleaner:
    def __init__(self, abbreviations_path=None):
        path = (
            Path(abbreviations_path)
            if abbreviations_path
            else Path(__file__).resolve().parents[1] / "config" / "abbreviations.json"
        )
        with open(path) as f:
            cfg = json.load(f)
        self.standalone = {k.lower(): v for k, v in cfg["standalone"].items()}
        self.after_number = {k.lower(): v for k, v in cfg["after_number"].items()}
        self.protected = {k.lower() for k in cfg["protected"]}

        # Compile once. Longest keys first so "w/o" and "w/d" win over "w/".
        standalone_keys = sorted(self.standalone, key=len, reverse=True)
        self._standalone_re = re.compile(
            r"(?i)(?<![\w/])("
            + "|".join(re.escape(k) for k in standalone_keys)
            + r")(?:\.(?=\s+[0-9a-z]))?(?!\w)"
        )
        number_keys = sorted(self.after_number, key=len, reverse=True)
        # No \b before the abbreviation: "2br" has no word boundary between "2" and "b".
        self._after_number_re = re.compile(
            r"(?i)(\d+(?:\.\d+)?)\s?(" + "|".join(number_keys) + r")s?\b"
        )

    #=========================PIPELINE
    def clean_text(self, text):
        """Run the full pipeline. None/NaN -> "". Order matters (see comments)."""
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return ""
        text = str(text)
        steps = (
            self.strip_markup,
            self.normalize_unicode,
            self.normalize_whitespace,
            self.normalize_numbers,
            self.normalize_prices,
            self.normalize_slashes, 
            self.normalize_measurements,
            self.expand_abbreviations,
            self.normalize_whitespace,
        )
        for step in steps:
            text = step(text)
        return text.strip()

    #=============================NORMALIZERS
    def normalize_unicode(self, text):
        """Curly quotes, dashes, invisible characters, fractions, mojibake."""

        def repair(match):
            try:
                return match.group(0).encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return match.group(0)

        text = re.sub(r"[\u00C2-\u00DF][\u0080-\u00BF]", repair, text)

        replacements = {
            "\u2019": "'", "\u2018": "'",
            "\u201c": '"', "\u201d": '"', 
            "\u00bd": "1/2", "\u00bc": "1/4", "\u00be": "3/4",
            "\u00b1": "+/-",
            "\u00ae": "", "\u2122": "", "\u00a9": "",
            "\u00a0": " ",
            "\u2022": "-",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[\u00C3\u00C2]+(?=[^A-Za-z]|$)", "", text)
        text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
        text = re.sub(r"\s*[\u2013\u2014]\s*", " - ", text)
        return unicodedata.normalize("NFKC", text)

    def normalize_whitespace(self, text):
        """Collapse every run of whitespace (spaces, tabs, newlines) to one space and strip."""
        return re.sub(r"\s+", " ", text).strip()

    def strip_markup(self, text):
        """Remove HTML tags, decode entities, remove markdown bold markers."""
        #tag must start with a letter or "/": a lone "<" as in "under <$500k" is not a tag.
        text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
        text = html.unescape(text)
        #replace with a SPACE: "***PRICE IMPROVEMENT***Welcome" must not become "IMPROVEMENTWelcome".
        text = re.sub(r"\*{2,}", " ", text)
        return text

    def normalize_numbers(self, text):
        """Remove thousands separators: "$1,250,000" -> "$1250000"."""
        #exactly three digits after each comma, and the number can't be part of a longer comma list
        return re.sub(
            r"(?<![\d,])\d{1,3}(?:,\d{3})+(?!\d)",
            lambda m: m.group(0).replace(",", ""),
            text,
        )

    def normalize_prices(self, text):
        """"$1.2m" -> "$1200000", "under 450k" -> "under 450000".

        Converts only with a "$" or right after a price word, so "FHA 203K loan", "0.2m" (miles)
        and "3 million tourists" are left alone. Decimal avoids float truncation
        (int(1.001 * 1e6) == 1000999).
        """
        pattern = re.compile(r"(?<![\w.])(\$)?(\d+(?:\.\d+)?)\s?(k|m|million)\b", re.I)

        def convert(match):
            dollar, number, unit = match.group(1), match.group(2), match.group(3).lower()
            if not dollar and not PRICE_CONTEXT.search(text[: match.start()]):
                return match.group(0)
            multiplier = 1000 if unit == "k" else 1_000_000
            value = int((Decimal(number) * multiplier).to_integral_value())
            return f"{dollar or ''}{value}"

        return pattern.sub(convert, text)

    def normalize_measurements(self, text):
        """"2,000 sqft" / "1,610 SF" / "750 sq. ft." -> "<number> square feet".

        "sf" only counts directly after a number, so "SF Bay Area" is untouched. A "+" or "+/-"
        between the number and unit is kept ("1,400+ sq ft" -> "1400+ square feet").
        """
        pattern = re.compile(
            r"(?i)(?<![\d,.])"
            r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
            r"(\+/-|\+)?\s?"
            r"(?:sq\.?[\s/]?ft|sqft|sq\.?[\s/]?feet|square\s(?:feet|foot|ft)|sf)"
            r"(?:\.(?=\s+[0-9a-z]))?(?![A-Za-z])" 
        )
        return pattern.sub(
            lambda m: f"{m.group(1).replace(',', '')}{m.group(2) or ''} square feet", text
        )

    def expand_abbreviations(self, text):
        """Expand abbreviations from config/abbreviations.json."""
        # "3br", "2 BA", "1.5ba" -> "3 bedroom", "2 bathroom" (always singular)
        text = self._after_number_re.sub(
            lambda m: f"{m.group(1)} {self.after_number[m.group(2).lower()]}", text
        )
        # "2 bedroom/2 bathroom" -> "2 bedroom 2 bathroom"
        text = re.sub(r"(bedroom|bathroom)\s*/\s*(?=\d)", r"\1 ", text)
        # Whole-word abbreviations ("A/C", "LVP", "approx.", ...)
        text = self._standalone_re.sub(lambda m: self.standalone[m.group(1).lower()], text)
        # "w/" glued to the next word: "w/quartz" -> "with quartz"
        return re.sub(r"(?i)(?<![\w/])w/(?=[a-z])", "with ", text)

    def normalize_slashes(self, text):
        """"pool/spa" -> "pool spa". Both sides need 2+ letters, so 24/7, 1/2, $80/month, N/A, A/C stay."""
        return re.sub(r"(?<=[A-Za-z]{2})/(?=[A-Za-z]{2})", " ", text)

    #============================PROFILING
    def _extract_top_ngrams(self, series, n=2, top=20):
        """Most common n-grams as [(ngram, count), ...] (lowercase alphabetic tokens, no stopwords)."""
        counts = Counter()
        for text in series.dropna():
            words = [
                w for w in re.findall(r"[a-z]+", str(text).lower())
                if w not in STOPWORDS and len(w) > 2
            ]
            counts.update(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
        return counts.most_common(top)

    def _detect_abbreviations(self, series, top=40):
        """Abbreviation-like tokens -> {token: count} (lowercase keys), most common first.

        Detects four shapes:
          * slash forms with a 1-2 letter side: "w/o", "a/c", "w/d", "sq/ft"; plus "w/" before a space or word
          * period abbreviations followed by more sentence: "approx. 4,200", "sq. ft. lot"
          * number-attached shorthand: "3br", "2 BA", "1,610 SF"
          * ALL-CAPS acronyms ("HOA", "ADU") - but not shouted ordinary words ("HOME", "PRICE"),
            which are skipped when the word is mostly written in lowercase elsewhere.
        """
        texts = series.dropna().astype(str).tolist()
        counts = Counter()

        lowercase_counts = Counter()
        for text in texts:
            lowercase_counts.update(re.findall(r"\b[a-z]{2,5}\b", text))

        number_attached = re.compile(
            r"(?i)\b\d[\d,]*\+?\s?(" + "|".join(sorted(set(self.after_number) | {"sf", "sqft"}, key=len, reverse=True)) + r")s?\b"
        )
        for text in texts:
            for token in re.findall(r"(?i)\b[a-z]{1,4}/[a-z]{1,4}\b", text):
                left, right = token.split("/")
                if len(left) <= 2 or len(right) <= 2:
                    counts[token.lower()] += 1
            counts["w/"] += len(re.findall(r"(?i)(?<![\w/])w/(?!o\b|d\b)", text))

            for token in re.findall(r"\b[A-Za-z]{2,6}\.(?=\s+[a-z0-9])", text):
                counts[token.lower()] += 1

            for match in number_attached.finditer(text):
                counts[match.group(1).lower()] += 1
            remaining = number_attached.sub(" ", text)

            for token in re.findall(r"\b[A-Z]{2,5}\b", remaining):
                if lowercase_counts[token.lower()] <= 2:
                    counts[token.lower()] += 1

        counts = Counter({k: v for k, v in counts.items() if v > 0})
        return dict(counts.most_common(top))

    def profile_column(self, df, column_name):
        """Describe a text column without modifying the DataFrame."""
        series = df[column_name]
        non_null = series.dropna().astype(str)
        lengths = non_null.str.len()
        return {
            "rows": int(len(series)),
            "null_rate": float(series.isnull().mean()),
            "avg_length": float(lengths.mean()),
            "median_length": float(lengths.median()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "common_terms": self._extract_top_ngrams(series),
            "price_mentions": int(non_null.str.contains(r"\$\d", regex=True).sum()),
            "has_html": int(non_null.str.contains("<", regex=False).sum()),
            "non_ascii": int(non_null.map(lambda t: any(ord(c) > 127 for c in t)).sum()),
            "double_spaces": int(non_null.str.contains(r"  +", regex=True).sum()),
            "newlines": int(non_null.str.contains("\n", regex=False).sum()),
            "common_abbreviations": self._detect_abbreviations(series),
        }
