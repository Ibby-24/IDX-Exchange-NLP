import json
import re
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = namedtuple("M", "value start end role")


class NaiveBaseline:
    def __init__(self, taxonomy_path=None):
        path = Path(taxonomy_path) if taxonomy_path else ROOT / "data" / "processed" / "taxonomy.json"
        self._terms = json.load(open(path))["terms"]

    def clean(self, text):
        return text

    @staticmethod
    def _first(pattern, text, conv=float):
        m = re.search(pattern, text, re.I)
        if not m:
            return None
        return M(conv(m.group(1).replace(",", "")), m.start(), m.end(), "main")

    def select_bedroom_mention(self, text):
        return self._first(r"(\d+)\s*(?:bed|br|bedroom)s?", text)

    def select_bathroom_mention(self, text):
        return self._first(r"(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)s?", text)

    def select_sqft_mention(self, text):
        return self._first(r"(\d[\d,]*)\s*(?:square feet|sq\.? ?ft)", text)

    def select_price_mention(self, text):
        return self._first(r"\$?(\d{5,})", text)

    def extract_amenities(self, text):
        low = text.lower().replace("-", " ")
        return [{"id": t["id"]} for t in self._terms if t["term"].lower().replace("-", " ") in low]
