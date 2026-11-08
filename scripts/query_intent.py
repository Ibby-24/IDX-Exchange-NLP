"""Week 7: search query intent classification (browsing / researching / high_intent_inquiry).

Intent is inferred from the *wording of the query only*. It is a guess about what the language signals, not a
measurement of the person (see docs/week7_report.md, "Limitations").

Use from Python::

    from query_intent import QueryIntentClassifier, understand
    clf = QueryIntentClassifier.load_or_train()          # trains from data/processed/intent_queries.json if no saved model
    clf.predict("homes with open houses this weekend")   # -> ("high_intent_inquiry", 0.81)
    clf.predict_detailed("3 bed under 700k in Irvine")   # -> dict with probabilities, uncertain flag, margin
    understand("3 bed under 700k in Irvine")             # Week 4 parser filters + intent + suggested handling

From the command line (repo root)::

    python scripts/query_intent.py train                       # fit on train+dev, save data/models/intent_model.joblib
    python scripts/query_intent.py train --all                 # fit on every labeled query (do this after the final test run)
    python scripts/query_intent.py predict "condos near UC Irvine with low HOA"
    python scripts/query_intent.py understand "new listings in Irvine under 900k"

Model: TF-IDF on words (1-2 grams) and characters (2-5 grams, so typos and shorthand still match), plus a few
numbers from the Week 4 QueryParser (how many filters it found, whether it saw a price, bedrooms, bathrooms, a city,
amenities) and a question-mark flag, fed to a multinomial logistic regression. The spec's starter code maps
``probas.argmax()`` to a fixed label list; scikit-learn orders classes alphabetically (browsing, high_intent_inquiry,
researching), so that mapping would swap two labels. This class always uses ``model.classes_``.

Confidence: the top class probability. ``uncertain`` is True when it is below ``uncertain_below`` (default 0.65,
chosen from cross-validated predictions on train+dev) or when the query does not look like a housing search at all
(``in_scope`` is False). Callers should treat uncertain results as "not sure" rather than act on the label.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "intent_queries.json"
MODEL_PATH = ROOT / "data" / "models" / "intent_model.joblib"

LABELS = ["browsing", "researching", "high_intent_inquiry"]
DEFAULT_UNCERTAIN_BELOW = 0.65

# What a search layer could do with each intent. A suggestion only: nothing in the SQL builder changes.
SUGGESTED_HANDLING = {
    "browsing": {"mode": "explore", "note": "show a broad result set; missing filters are fine", "sort": "newest"},
    "researching": {"mode": "inform", "note": "likely a question about an area or costs; consider answering or "
                    "semantic search (Week 5) rather than a strict filter list"},
    "high_intent_inquiry": {"mode": "act", "note": "apply the parsed filters strictly; surface newest listings, "
                            "open houses and price changes"},
    "uncertain": {"mode": "explore", "note": "the classifier is not sure; behave like browsing and do not act on the label"},
}

# Words that make a query plausibly about housing. Used only for the "in scope" check below.
_DOMAIN = re.compile(r"\b(home|homes|house|houses|housing|condo|condos|townhome|townhomes|townhouse|townhouses|apartment|"
                     r"apartments|property|properties|listing|listings|real estate|realtor|agent|mortgage|hoa|mello|lot|lots|"
                     r"cabin|cabins|duplex|loft|lofts|cottage|cottages|bungalow|bungalows|estate|rent|rental|rentals|"
                     r"neighborhood|neighborhoods|bed|beds|bedroom|bedrooms|bath|baths|bathroom|bathrooms|br|ba|bd|sqft|"
                     r"open house|open houses|fha|adu|buy|buying|sell|selling|offer|tour|showing|school|schools|"
                     r"property tax|tax|taxes|escrow|closing|appraisal|foreclosure|foreclosures|fixer|ranch|villa|mansion|for sale|"
                     r"homebuyer|homebuyers|homeowner|homeowners|wildfire|flood|earthquake|per square foot|square foot|sq ft)\b", re.I)

_PRICE = re.compile(r"\$?\s?\d[\d,]*(?:\.\d+)?\s?(?:k|m|mm|million|thousand)\b", re.I)


def normalize_query(q: str) -> str:
    """Light, reversible-in-spirit normalisation for the vectoriser: price amounts become one token."""
    q = "" if q is None else str(q)
    q = _PRICE.sub(" pricetoken ", q)
    return re.sub(r"\s+", " ", q).strip()


def _prep(s: str) -> str:
    """Module-level (not a lambda) so the fitted model can be pickled."""
    return normalize_query(s).lower()


# ------------------------------------------------------------------------------ parser features
class ParserFeatures(BaseEstimator, TransformerMixin):
    """A handful of dense numbers per query from the Week 4 QueryParser, as a sparse matrix.

    Columns: n_filters, has_city, has_price, has_beds, has_baths, has_sqft, n_amenities, has_question_mark, n_tokens.
    If the parser cannot be built (no schema.json yet) the parser columns are zeros and the rest still work.
    """

    NAMES = ["n_filters", "has_city", "has_price", "has_beds", "has_baths", "has_sqft", "n_amenities",
             "has_question_mark", "log_tokens"]

    def __init__(self, use_parser: bool = True, scale: float = 1.0):
        self.use_parser = use_parser
        self.scale = scale
        self._parser = None

    def fit(self, X, y=None):
        return self

    def _get_parser(self):
        if self._parser is None and self.use_parser:
            try:
                from query_parser import QueryParser
                self._parser = QueryParser()
            except Exception:  # noqa: BLE001 - features degrade, never crash the classifier
                self.use_parser = False
        return self._parser

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parser"] = None              # the parser is rebuilt after loading
        return state

    def features(self, q: str) -> List[float]:
        q = "" if q is None else str(q)
        f: Dict = {}
        p = self._get_parser()
        if p is not None:
            try:
                f = p.parse(q)
            except Exception:  # noqa: BLE001
                f = {}
        has = lambda *ks: float(any(k in f for k in ks))  # noqa: E731
        return [float(len(f)), has("city"), has("price_min", "price_max"), has("bedrooms", "bedrooms_min", "bedrooms_max"),
                has("bathrooms", "bathrooms_min", "bathrooms_max"), has("sqft_min", "sqft_max", "sqft"),
                float(len(f.get("amenities", []))), float("?" in q), float(np.log1p(len(q.split())))]

    def transform(self, X):
        rows = np.array([self.features(q) for q in X], dtype=float) * self.scale
        return sparse.csr_matrix(rows)


# ------------------------------------------------------------------------------ the classifier
def build_pipeline(C: float = 6.0, char: bool = True, parser_features: bool = True, word_ngrams=(1, 2),
                   parser_scale: float = 0.5, max_features: Optional[int] = None) -> Pipeline:
    parts = [("word", TfidfVectorizer(ngram_range=word_ngrams, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b",
                                      max_features=max_features, preprocessor=_prep))]
    if char:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                              preprocessor=_prep)))
    if parser_features:
        parts.append(("parser", ParserFeatures(scale=parser_scale)))
    return Pipeline([("features", FeatureUnion(parts)),
                     ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


class QueryIntentClassifier:
    """Predict browsing / researching / high_intent_inquiry from the words of a search query."""

    labels = LABELS

    def __init__(self, C: float = 6.0, char: bool = True, parser_features: bool = True,
                 uncertain_below: float = DEFAULT_UNCERTAIN_BELOW, parser_scale: float = 0.5):
        self.config = {"C": C, "char": char, "parser_features": parser_features, "parser_scale": parser_scale}
        self.uncertain_below = uncertain_below
        self.pipeline = build_pipeline(C=C, char=char, parser_features=parser_features, parser_scale=parser_scale)
        self.trained = False

    # -- training
    def train(self, queries: Sequence[str], labels: Sequence[str]) -> "QueryIntentClassifier":
        bad = sorted(set(labels) - set(LABELS))
        if bad:
            raise ValueError(f"unknown intent labels: {bad}")
        if len(set(labels)) < 2:
            raise ValueError("need at least two different labels to train")
        self.pipeline.fit(list(queries), list(labels))
        self.trained = True
        return self

    # -- prediction
    def _probabilities(self, queries: Sequence[str]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("train() or load() before predict()")
        return self.pipeline.predict_proba(list(queries))

    @property
    def classes_(self) -> List[str]:
        return [str(c) for c in self.pipeline.named_steps["clf"].classes_]

    def in_scope(self, query: str) -> bool:
        """Heuristic: does the query look like a housing search at all? True if the Week 4 parser found any filter
        or the query contains a housing word. There is no 'other' class, so without this a question about the
        weather would still get a confident-looking intent."""
        q = "" if query is None else str(query)
        if _DOMAIN.search(q):
            return True
        for name, tr in self.pipeline.named_steps["features"].transformer_list:
            if name == "parser":
                parser = tr._get_parser()
                if parser is None:
                    return False
                try:
                    f = parser.parse(q)
                except Exception:  # noqa: BLE001
                    return False
                known = {c.lower() for c in getattr(parser, "cities", [])}
                # the parser also returns names it does not know ("in Paris"); those are not evidence of a housing search
                return any(k != "city" for k in f) or str(f.get("city", "")).lower() in known
        return False

    def predict_detailed_many(self, queries: Sequence[str]) -> List[dict]:
        probs = self._probabilities(queries)
        classes = self.classes_
        out = []
        for q, p in zip(queries, probs):
            in_scope = self.in_scope(q)
            order = np.argsort(-p)
            top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            conf = float(p[top])
            out.append({
                "query": q,
                "intent": classes[top],
                "confidence": round(conf, 4),
                "margin": round(float(p[top] - p[second]), 4),
                "second_choice": classes[second],
                "probabilities": {c: round(float(p[i]), 4) for i, c in enumerate(classes)},
                "in_scope": in_scope,
                "uncertain": conf < self.uncertain_below or not in_scope,
            })
        return out

    def predict_detailed(self, query: str) -> dict:
        return self.predict_detailed_many([query])[0]

    def predict(self, query: str) -> Tuple[str, float]:
        """(intent, confidence): the same return shape as the spec's starter class."""
        d = self.predict_detailed(query)
        return d["intent"], d["confidence"]

    # -- persistence
    def save(self, path: Path = MODEL_PATH) -> Path:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "config": self.config, "uncertain_below": self.uncertain_below}, path)
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "QueryIntentClassifier":
        import joblib
        blob = joblib.load(path)
        obj = cls(**blob["config"], uncertain_below=blob["uncertain_below"])
        obj.pipeline = blob["pipeline"]
        obj.trained = True
        return obj

    @classmethod
    def load_or_train(cls, path: Path = MODEL_PATH, data_path: Path = DATA, splits=("train", "dev")) -> "QueryIntentClassifier":
        if Path(path).exists():
            try:
                return cls.load(path)
            except Exception:  # noqa: BLE001 - stale or incompatible file: retrain
                pass
        q, y = load_labeled(data_path, splits)
        return cls().train(q, y)


def load_labeled(path: Path = DATA, splits=("train", "dev")) -> Tuple[List[str], List[str]]:
    rows = json.load(open(path))["queries"]
    rows = [r for r in rows if r["split"] in splits]
    return [r["query"] for r in rows], [r["intent"] for r in rows]


# ------------------------------------------------------------------------------ integration with Week 4
def understand(query: str, classifier: Optional[QueryIntentClassifier] = None, parser=None, validator=None,
               limit: int = 50) -> dict:
    """Week 4 parse + validation + SQL, plus the intent of the wording and a suggested way to handle it."""
    from query_parser import QueryParser
    from schema_validator import SchemaValidator
    classifier = classifier or QueryIntentClassifier.load_or_train()
    parser = parser or QueryParser()
    validator = validator or SchemaValidator(schema=parser.schema)
    res = parser.search(query, validator=validator, limit=limit)
    intent = classifier.predict_detailed(query)
    handling = SUGGESTED_HANDLING["uncertain" if intent["uncertain"] else intent["intent"]]
    return {
        "query": query,
        "filters": res["filters"],
        "valid": res["valid"],
        "errors": res["errors"],
        "sql": res["sql"] if res["valid"] else None,
        "params": res["params"] if res["valid"] else None,
        "intent": intent["intent"],
        "confidence": intent["confidence"],
        "uncertain": intent["uncertain"],
        "probabilities": intent["probabilities"],
        "suggested_handling": handling,
    }


# ------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="fit on the labeled queries and save the model")
    t.add_argument("--all", action="store_true", help="include the test split (only after the final test run)")
    t.add_argument("--out", type=Path, default=MODEL_PATH)
    p = sub.add_parser("predict", help="intent, confidence and probabilities for one query")
    p.add_argument("query")
    u = sub.add_parser("understand", help="Week 4 parser filters + intent + suggested handling")
    u.add_argument("query")
    args = ap.parse_args(argv)

    if args.cmd == "train":
        splits = ("train", "dev", "test") if args.all else ("train", "dev")
        q, y = load_labeled(DATA, splits)
        clf = QueryIntentClassifier().train(q, y)
        path = clf.save(args.out)
        print(f"trained on {len(q)} queries ({', '.join(splits)}); saved {path}")
        return 0

    clf = QueryIntentClassifier.load_or_train()
    if args.cmd == "predict":
        d = clf.predict_detailed(args.query)
        print(f"intent     : {d['intent']}" + ("   (UNCERTAIN)" if d["uncertain"] else ""))
        print(f"confidence : {d['confidence']:.2f}   margin over {d['second_choice']}: {d['margin']:.2f}")
        for c, pr in sorted(d["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"  {c:<22}{pr:.2f}")
        return 0
    res = understand(args.query, clf)
    print(f"query      : {res['query']}")
    print(f"filters    : {res['filters']}")
    print(f"valid      : {'yes' if res['valid'] else 'NO ' + str(res['errors'])}")
    print(f"intent     : {res['intent']}  (confidence {res['confidence']:.2f})" + ("  UNCERTAIN" if res["uncertain"] else ""))
    print(f"handling   : {res['suggested_handling']['mode']} - {res['suggested_handling']['note']}")
    if res["sql"]:
        print(f"sql        : {res['sql']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
