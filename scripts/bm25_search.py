"""Week 5: BM25 keyword search - the baseline that semantic search is compared against.

    from bm25_search import BM25Searcher
    bm = BM25Searcher().build_index(remarks, ids=listing_ids)
    bm.search("RV parking", top_k=5)          # same SearchHit objects as SemanticSearcher

BM25 (Okapi) scores a listing by the query words it contains. A word counts more when it is rare in the
collection (idf) and when it appears often in the listing, with diminishing returns (k1) and a
correction for long listings (b):

    score(q, d) = sum over query words w of  idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
    idf(w)      = ln(1 + (N - n_w + 0.5) / (n_w + 0.5))        n_w = listings containing w

It never sees meaning: "peaceful" and "quiet" are different words. That is the gap embeddings are
supposed to close, and the gap this baseline lets us measure.

Preprocessing is identical to the semantic side where it can be: Week 2 TextCleaner, lowercase, stopwords
removed, plural endings stripped ("pools" -> "pool", "communities" -> "community").
Written from scratch (about 60 lines) so there is no extra dependency and the formula is testable.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from semantic_search import SearchHit

# Standard English function words. "no"/"not" are removed too, which is why BM25 cannot tell "no HOA" from "HOA".
STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its just me more most my no nor not of off on once only or other our out over own
same she should so some such than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Strip a plural ending. Deliberately tiny: it must not turn "glass" into "glas" or "bonus" into "bonu"."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, use_stem: bool = True) -> List[str]:
    out = []
    for t in TOKEN.findall((text or "").lower()):
        if t in STOPWORDS or (len(t) == 1 and not t.isdigit()):
            continue
        out.append(stem(t) if use_stem else t)
    return out


class BM25Searcher:
    def __init__(self, k1: float = 1.5, b: float = 0.75, use_stem: bool = True, clean: bool = True, cleaner=None):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b, self.use_stem, self.clean = k1, b, use_stem, clean
        self._cleaner = cleaner
        self.ids: List = []
        self.postings: Dict[str, List] = {}
        self.doc_len: List[int] = []
        self.avg_len = 0.0
        self.idf: Dict[str, float] = {}

    def _tokens(self, text: Optional[str]) -> List[str]:
        text = "" if text is None else str(text)
        if self.clean and text.strip():
            if self._cleaner is None:
                from text_cleaning import TextCleaner
                self._cleaner = TextCleaner()
            text = self._cleaner.clean_text(text)
        return tokenize(text, self.use_stem)

    def build_index(self, texts: Sequence[str], ids: Optional[Sequence] = None) -> "BM25Searcher":
        texts = list(texts)
        ids = list(range(len(texts))) if ids is None else list(ids)
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if not texts:
            raise ValueError("nothing to index")
        postings = defaultdict(list)
        lengths = []
        for d, text in enumerate(texts):
            toks = self._tokens(text)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((d, tf))
        n = len(texts)
        self.ids, self.doc_len, self.postings = ids, lengths, dict(postings)
        self.avg_len = sum(lengths) / n or 1.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        return self

    def scores(self, query: str) -> Dict[int, float]:
        """Score of every listing that contains at least one query word."""
        acc: Dict[int, float] = defaultdict(float)
        for term in set(self._tokens(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for d, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[d] / self.avg_len
                acc[d] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return acc

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        if not self.ids:
            raise RuntimeError("call build_index() first")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [SearchHit(self.ids[d], s, r, d) for r, (d, s) in enumerate(ranked, 1)]
