"""SQuAD 2.0 scoring (re-implementation of the official script) plus selective-QA metrics."""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def exact(pred: str, golds: list[str]) -> float:
    """Empty prediction is correct only for unanswerable questions (empty gold list)."""
    if not golds:
        return float(pred == "")
    return max(float(normalize(pred) == normalize(g)) for g in golds)


def f1(pred: str, golds: list[str]) -> float:
    if not golds:
        return float(pred == "")
    if pred == "":
        return 0.0
    best = 0.0
    p = normalize(pred).split()
    for g in golds:
        gt = normalize(g).split()
        common = Counter(p) & Counter(gt)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p), same / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def squad_scores(preds: list[str], golds: list[list[str]]) -> dict:
    em = np.array([exact(p, g) for p, g in zip(preds, golds, strict=True)])
    fs = np.array([f1(p, g) for p, g in zip(preds, golds, strict=True)])
    has = np.array([bool(g) for g in golds])
    answered = np.array([p != "" for p in preds])
    return {
        "em": float(em.mean()),
        "f1": float(fs.mean()),
        "has_ans_f1": float(fs[has].mean()) if has.any() else float("nan"),
        "no_ans_acc": float((~answered[~has]).mean()) if (~has).any() else float("nan"),
        # the number a product owner cares about: how often we answer a question that has
        # no answer in the documents (a confident hallucination)
        "hallucination_rate": float(answered[~has].mean()) if (~has).any() else float("nan"),
        "answer_rate": float(answered.mean()),
    }


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Selective prediction: sort by confidence, answer the top fraction, measure accuracy."""
    order = np.argsort(-confidence)
    c = correct[order].astype(float)
    n = len(c)
    acc_at = np.cumsum(c) / np.arange(1, n + 1)
    risk = 1 - acc_at
    out = {"aurc": float(risk.mean())}
    for cov in (0.25, 0.5, 0.8):
        k = max(1, int(round(cov * n)))
        out[f"acc@{int(cov * 100)}%cov"] = float(acc_at[k - 1])
    return out


def ece(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (prob >= lo) & (prob < hi) if hi < 1 else (prob >= lo) & (prob <= hi)
        if m.any():
            total += m.mean() * abs(prob[m].mean() - label[m].mean())
    return float(total)
