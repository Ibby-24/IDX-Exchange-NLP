"""Week 5: SemanticSearcher - sentence embeddings + a FAISS index for meaning-based listing search.

    from semantic_search import SemanticSearcher, SentenceTransformerEncoder
    s = SemanticSearcher(SentenceTransformerEncoder("all-MiniLM-L6-v2"), chunk_words=100, chunk_overlap=20)
    s.build_index(remarks, ids=listing_ids)
    for hit in s.search("quiet street, good for a young family", top_k=5):
        print(hit.rank, hit.listing_id, round(hit.score, 3))

Command line (from the repo root)::

    python scripts/semantic_search.py build --encoder minilm            # sample CSV -> data/models/semantic_index/
    python scripts/semantic_search.py search "room for horses" --top 5  # query the saved index (add --show for excerpts)

How it works
  1. Remarks and queries go through the Week 2 TextCleaner (same cleaning for both sides).
  2. Long remarks are optionally split into passages of ~chunk_words words. all-MiniLM-L6-v2 reads at most
     256 word pieces and silently drops the rest; the sample's median remark is ~1,260 characters, so
     without chunking the end of many remarks is never seen by the model. A listing's score is the
     score of its best passage.
  3. Every passage becomes a unit-length vector (384 numbers for MiniLM). With unit vectors, inner
     product = cosine similarity, which is what ``faiss.IndexFlatIP`` computes.
  4. IndexFlatIP is an exact search (no approximation). Measured here: about 4 ms for 40,000 passage vectors
     (10,000 listings) and 30 ms for 200,000. Approximate indexes (IVF/HNSW) only pay off around a million
     vectors. Without FAISS installed the class falls back to a plain numpy matrix product (same results;
     see scripts/benchmark_latency.py for the timing of both).

Encoders (all expose ``encode(list_of_text) -> float32 array``)
  * SentenceTransformerEncoder  the real thing; needs ``pip install sentence-transformers`` and one
                                internet download of the model (~90 MB for MiniLM) on first use.
  * LSAEncoder                  TF-IDF + truncated SVD ("latent semantic analysis"). A classical, offline
                                stand-in that also gives dense vectors. It is NOT a transformer; it exists so
                                the pipeline can be tested and compared without downloading a model.

Nothing is stored except vectors, listing ids and settings. Remark text is never written to the index.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "processed" / "listing_sample.csv"
EVAL_CORPUS = ROOT / "data" / "processed" / "listing_sample_eval.csv"   # built by build_eval_corpus.py
DEFAULT_INDEX_DIR = ROOT / "data" / "models" / "semantic_index"

MODEL_ALIASES = {
    "minilm": "all-MiniLM-L6-v2",      # 384 dims, fast; the curriculum's model
    "mpnet": "all-mpnet-base-v2",      # 768 dims, slower, usually a bit better
}


class SearchHit(NamedTuple):
    listing_id: object
    score: float
    rank: int          # 1 = best
    index: int         # position of the listing in the list given to build_index


# ----------------------------------------------------------------------------- helpers
def l2_normalize(a) -> np.ndarray:
    """Rows scaled to length 1 (all-zero rows stay zero). float32 C-contiguous, as FAISS wants."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(a / norms, dtype=np.float32)


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def chunk_text(text: str, max_words: int = 100, overlap_words: int = 0) -> List[str]:
    """Split *text* into passages of at most ``max_words`` words, on sentence boundaries when possible.

    ``overlap_words``: the last sentences of a passage (up to that many words) are repeated at the
    start of the next one, so a feature mentioned across a boundary is still seen whole.
    A sentence longer than max_words is cut by words. Short texts come back as one passage.
    """
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("need max_words >= 1 and 0 <= overlap_words < max_words")
    text = (text or "").strip()
    if not text:
        return []
    if len(text.split()) <= max_words:
        return [text]
    sentences = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        words = s.split()
        while len(words) > max_words:
            sentences.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if words:
            sentences.append(" ".join(words))
    chunks, cur, n = [], [], 0
    for s in sentences:
        k = len(s.split())
        if cur and n + k > max_words:
            chunks.append(" ".join(cur))
            keep, m = [], 0
            for t in reversed(cur):
                tk = len(t.split())
                if m + tk > overlap_words:
                    break
                keep.insert(0, t)
                m += tk
            cur, n = keep, m
        cur.append(s)
        n += k
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def load_sample(path: Path = SAMPLE, id_col: str = "L_ListingID", text_col: str = "remarks"):
    """(ids, remarks) from the sample CSV, skipping rows without remarks."""
    if not Path(path).exists():
        sys.exit(f"{path} not found. Run data_loading.py first (Week 1).")
    ids, texts = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get(text_col) or "").strip()
            if t:
                ids.append(str(row[id_col]))
                texts.append(t)
    return ids, texts


def default_corpus_path() -> Path:
    """The corpus the Week 5 evaluation searches: listing_sample_eval.csv when it exists, else the sample."""
    return EVAL_CORPUS if EVAL_CORPUS.exists() else SAMPLE


def load_corpus(path: Optional[Path] = None):
    """(ids, remarks) for the evaluation scripts. Same format as load_sample; path defaults to default_corpus_path()."""
    return load_sample(Path(path) if path else default_corpus_path())


# ----------------------------------------------------------------------------- encoders
class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model. The model is loaded on first use."""
    needs_fit = False
    kind = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64, device: Optional[str] = None):
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. On a laptop without a GPU install the small CPU "
                    "build of PyTorch first, then the library:\n"
                    "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                    "  pip install sentence-transformers faiss-cpu") from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())    # the method was renamed in newer sentence-transformers releases

    @property
    def max_seq_length(self) -> int:
        return int(self._load().max_seq_length)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        out = model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                           show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)

    def config(self) -> dict:
        return {"kind": self.kind, "model": self.model_name}


class LSAEncoder:
    """TF-IDF + truncated SVD. Fitted on the passages it will index. Offline stand-in, not a transformer."""
    needs_fit = True
    kind = "lsa"
    name = "lsa-tfidf-svd"

    def __init__(self, n_components: int = 384):
        self.n_components = n_components
        self.dim = n_components
        self._tfidf = None
        self._svd = None

    def fit(self, texts: Sequence[str]):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), stop_words="english",
                                      max_features=60000)
        x = self._tfidf.fit_transform(texts)
        k = max(2, min(self.n_components, x.shape[0] - 1, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=0)
        self._svd.fit(x)
        self.dim = k
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._svd is None:
            raise RuntimeError("LSAEncoder must be fitted (build_index does this)")
        return np.asarray(self._svd.transform(self._tfidf.transform(list(texts))), dtype=np.float32)

    def config(self) -> dict:
        return {"kind": self.kind, "n_components": self.n_components}

    def save(self, path: Path):
        import joblib
        joblib.dump({"tfidf": self._tfidf, "svd": self._svd, "dim": self.dim}, path)

    @classmethod
    def load(cls, path: Path, n_components: int = 384):
        import joblib   # only load files you created yourself: joblib/pickle can run code
        d = joblib.load(path)
        enc = cls(n_components)
        enc._tfidf, enc._svd, enc.dim = d["tfidf"], d["svd"], d["dim"]
        return enc


def make_encoder(kind: str, **kw):
    """'minilm' | 'mpnet' | any sentence-transformers model name | 'lsa'."""
    if kind == "lsa":
        return LSAEncoder(**kw)
    return SentenceTransformerEncoder(kind, **kw)


# ----------------------------------------------------------------------------- the searcher
class SemanticSearcher:
    def __init__(self, encoder=None, chunk_words: Optional[int] = None, chunk_overlap: int = 0,
                 clean: bool = True, use_faiss: bool = True, cleaner=None):
        self.encoder = encoder if encoder is not None else SentenceTransformerEncoder()
        self.chunk_words = chunk_words or None
        self.chunk_overlap = chunk_overlap
        self.clean = clean
        self._cleaner = cleaner
        self._want_faiss = use_faiss
        self.embeddings: Optional[np.ndarray] = None     # (n_passages, dim), unit length
        self.owner: Optional[np.ndarray] = None          # passage -> listing position
        self.ids: List = []
        self.listings: List[str] = []                    # original remark text (memory only; never saved)
        self.index = None
        self.backend = "numpy"

    # ------------------------------------------------------------------ text preparation
    def _prep(self, text: Optional[str]) -> str:
        text = "" if text is None else str(text)
        if not self.clean or not text.strip():
            return text.strip()
        if self._cleaner is None:
            from text_cleaning import TextCleaner
            self._cleaner = TextCleaner()
        return self._cleaner.clean_text(text)

    def _passages(self, text: str) -> List[str]:
        if self.chunk_words:
            return chunk_text(text, self.chunk_words, self.chunk_overlap)
        return [text] if text.strip() else []

    # ------------------------------------------------------------------ build / search
    def build_index(self, texts: Sequence[str], ids: Optional[Sequence] = None) -> "SemanticSearcher":
        texts = list(texts)
        if ids is None:
            ids = list(range(len(texts)))
        ids = list(ids)
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        passages, owner = [], []
        for i, t in enumerate(texts):
            for p in self._passages(self._prep(t)):
                passages.append(p)
                owner.append(i)
        if not passages:
            raise ValueError("nothing to index: every text is empty")
        if getattr(self.encoder, "needs_fit", False):
            self.encoder.fit(passages)
        emb = l2_normalize(self.encoder.encode(passages))
        self.listings, self.ids = texts, ids
        self._set_vectors(emb, np.asarray(owner, dtype=np.int64))
        return self

    def _set_vectors(self, emb: np.ndarray, owner: np.ndarray):
        self.embeddings, self.owner = emb, owner
        faiss = _faiss() if self._want_faiss else None
        if faiss is not None:
            self.index = faiss.IndexFlatIP(emb.shape[1])    # inner product == cosine for unit vectors
            self.index.add(emb)
            self.backend = "faiss-flat-ip"
        else:
            self.index, self.backend = None, "numpy"

    def _knn(self, q: np.ndarray, k: int):
        k = min(k, len(self.embeddings))
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            scores, idx = scores[0], idx[0]
            keep = idx >= 0
            return scores[keep], idx[keep]
        sims = (self.embeddings @ q[0])
        if k < len(sims):
            top = np.argpartition(-sims, k - 1)[:k]
        else:
            top = np.arange(len(sims))
        top = top[np.argsort(-sims[top], kind="stable")]
        return sims[top], top

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        if self.embeddings is None:
            raise RuntimeError("call build_index() or load() first")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        query = self._prep(query)
        if not query.strip():
            return []
        q = l2_normalize(self.encoder.encode([query]))
        n_listings = len(self.ids)
        k_fetch = min(len(self.embeddings), max(top_k * 4, top_k))
        while True:
            scores, idx = self._knn(q, k_fetch)
            best: Dict[int, float] = {}
            for s, p in zip(scores, idx):
                doc = int(self.owner[p])
                if doc not in best:                  # results come best-first, so the first hit is the max
                    best[doc] = float(s)
            if len(best) >= min(top_k, n_listings) or k_fetch >= len(self.embeddings):
                break
            k_fetch = min(len(self.embeddings), k_fetch * 2)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [SearchHit(self.ids[d], s, r, d) for r, (d, s) in enumerate(ranked, 1)]

    def stats(self) -> dict:
        return {"listings": len(self.ids), "passages": int(len(self.embeddings)),
                "dim": int(self.embeddings.shape[1]), "backend": self.backend,
                "encoder": getattr(self.encoder, "name", "?"), "chunk_words": self.chunk_words,
                "chunk_overlap": self.chunk_overlap}

    # ------------------------------------------------------------------ save / load
    def save(self, directory: Path = DEFAULT_INDEX_DIR) -> Path:
        """Vectors + ids + settings. Remark text is NOT saved. The FAISS index is rebuilt on load (it is flat)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        meta = {"version": 1, "encoder": self.encoder.config(), "chunk_words": self.chunk_words,
                "chunk_overlap": self.chunk_overlap, "clean": self.clean, "ids": [str(i) for i in self.ids],
                "owner": self.owner.tolist(), "dim": int(self.embeddings.shape[1]),
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f)
        if isinstance(self.encoder, LSAEncoder):
            self.encoder.save(directory / "encoder.joblib")
        return directory

    @classmethod
    def load(cls, directory: Path = DEFAULT_INDEX_DIR, encoder=None, use_faiss: bool = True) -> "SemanticSearcher":
        directory = Path(directory)
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"no index in {directory}. Build one with: python scripts/semantic_search.py build")
        meta = json.load(open(directory / "meta.json"))
        if encoder is None:
            cfg = meta["encoder"]
            if cfg["kind"] == "lsa":
                encoder = LSAEncoder.load(directory / "encoder.joblib", cfg.get("n_components", 384))
            else:
                encoder = SentenceTransformerEncoder(cfg["model"])
        s = cls(encoder, chunk_words=meta["chunk_words"], chunk_overlap=meta["chunk_overlap"],
                clean=meta["clean"], use_faiss=use_faiss)
        emb = np.load(directory / "embeddings.npy")
        if emb.shape[1] != meta["dim"]:
            raise ValueError("embeddings.npy does not match meta.json")
        s.ids = meta["ids"]
        s._set_vectors(np.ascontiguousarray(emb, dtype=np.float32), np.asarray(meta["owner"], dtype=np.int64))
        return s


# ----------------------------------------------------------------------------- command line
def _rows_from_db(limit: int, args):
    from build_schema import connect_or_exit
    conn = connect_or_exit(args)
    cur = conn.cursor()
    cur.execute("SELECT L_ListingID, L_Remarks FROM rets_property "
                "WHERE L_Remarks IS NOT NULL AND CHAR_LENGTH(L_Remarks) >= 50 LIMIT %s", (int(limit),))
    rows = cur.fetchall()
    conn.close()
    return [str(r[0]) for r in rows], [r[1] for r in rows]


def main(argv=None):
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="encode listings and save an index")
    b.add_argument("--encoder", default="minilm", help="minilm | mpnet | lsa | any sentence-transformers model")
    b.add_argument("--source", choices=["csv", "db"], default="csv")
    b.add_argument("--limit", type=int, default=1000, help="rows to read when --source db")
    b.add_argument("--chunk-words", type=int, default=100, help="0 = embed each remark whole")
    b.add_argument("--overlap", type=int, default=20)
    b.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    b.add_argument("--host", default=os.environ.get("DB_HOST", "localhost"))
    b.add_argument("--user", default=os.environ.get("DB_USER", "root"))
    b.add_argument("--password", default=os.environ.get("DB_PASSWORD", "root"))
    b.add_argument("--database", default=os.environ.get("DB_NAME", "real_estate"))

    s = sub.add_parser("search", help="query a saved index")
    s.add_argument("query")
    s.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--show", action="store_true", help="print the first 200 characters of each remark (from the sample CSV)")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        ids, texts = load_sample() if args.source == "csv" else _rows_from_db(args.limit, args)
        enc = make_encoder(args.encoder)
        searcher = SemanticSearcher(enc, chunk_words=args.chunk_words or None, chunk_overlap=args.overlap)
        t0 = time.time()
        searcher.build_index(texts, ids)
        out = searcher.save(args.out)
        print(f"indexed {searcher.stats()} in {time.time() - t0:.1f}s -> {out}")
        return 0
    searcher = SemanticSearcher.load(args.index_dir)
    t0 = time.perf_counter()
    hits = searcher.search(args.query, args.top)
    ms = (time.perf_counter() - t0) * 1000
    texts = {}
    if args.show and SAMPLE.exists():
        texts = dict(zip(*load_sample()))
    print(f"{len(hits)} result(s) in {ms:.1f} ms  [{searcher.backend}]")
    for h in hits:
        line = f"{h.rank:>2}. {h.listing_id}  score {h.score:.3f}"
        if h.listing_id in texts:
            line += "\n      " + texts[h.listing_id][:200].replace("\n", " ")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
