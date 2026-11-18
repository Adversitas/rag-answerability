"""Passage retrieval: BM25, dense bi-encoder and Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .data import Passage

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = (
    "a an the of to in on at by for with from and or is are was were be been what which who "
    "whom when where why how did does do that this these those it its as"
)
_STOP = frozenset(_STOP_WORDS.split())
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    name = "bm25"

    def __init__(self, passages: list[Passage], k1: float = 0.9, b: float = 0.4):
        self.passages = passages
        docs = [tokenize(f"{p.title} {p.text}") for p in passages]
        self.k1, self.b = k1, b
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.len.mean()
        self.tf = [Counter(d) for d in docs]
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tf):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.passages))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avgdl)
        for t in set(tokenize(query)):
            for i in self.postings.get(t, ()):
                f = self.tf[i][t]
                out[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
        return out

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.scores(q) for q in queries])


class Dense:
    def __init__(self, passages: list[Passage], model, name: str = "dense", prefix: str = ""):
        self.passages, self.model, self.name, self.prefix = passages, model, name, prefix
        self.matrix = self._enc([f"{p.title}. {p.text}" for p in passages])

    def _enc(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=32, normalize_embeddings=True), dtype=np.float32
        )

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        return self._enc([self.prefix + q for q in queries]) @ self.matrix.T


def rrf(score_rows: list[np.ndarray], k: int = 60) -> np.ndarray:
    fused = np.zeros_like(score_rows[0], dtype=float)
    for s in score_rows:
        ranks = np.empty(len(s), dtype=int)
        ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += 1.0 / (k + ranks)
    return fused


class Hybrid:
    name = "hybrid"

    def __init__(self, lexical: BM25, dense: Dense):
        self.lexical, self.dense = lexical, dense

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        lex = self.lexical.scores_batch(queries)
        den = self.dense.scores_batch(queries)
        return np.stack([rrf([a, b]) for a, b in zip(lex, den, strict=True)])


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-scores, axis=1)[:, :k]
    return idx, np.take_along_axis(scores, idx, axis=1)
