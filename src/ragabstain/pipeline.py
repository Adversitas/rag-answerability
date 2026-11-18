"""Retrieve -> read -> decide whether to answer at all.

Every policy is tuned on the *calibration* half and evaluated on the *test* half, so the
reported numbers never benefit from a threshold chosen on the questions they are scored on.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import ece, f1, risk_coverage, squad_scores
from .retrieval import tokenize, top_k

FEATURES = [
    "margin",
    "span_score",
    "null_score",
    "span_prob",
    "ret_top1",
    "ret_gap",
    "chosen_rank",
    "answer_tokens",
    "question_overlap",
]


def sentence_with(text: str, start: int) -> str:
    left = max(text.rfind(". ", 0, start), 0)
    right = text.find(". ", start)
    return text[left : right if right != -1 else len(text)]


def run_reader(questions, passages, retriever, reader, k: int = 3) -> list[dict]:
    t0 = time.perf_counter()
    scores = retriever.scores_batch([q.text for q in questions])
    idx, vals = top_k(scores, k)
    t_ret = (time.perf_counter() - t0) / len(questions)

    pairs = [(q.text, passages[j].text) for q, row in zip(questions, idx, strict=True) for j in row]
    t0 = time.perf_counter()
    outs = reader.read(pairs)
    t_read = (time.perf_counter() - t0) / len(questions)

    rows = []
    for qi, q in enumerate(questions):
        cand = outs[qi * k : (qi + 1) * k]
        best = int(np.argmax([c.margin for c in cand]))
        o = cand[best]
        passage = passages[idx[qi, best]].text
        pos = passage.find(o.answer) if o.answer else -1
        sent = sentence_with(passage, pos) if pos >= 0 else ""
        qt = set(tokenize(q.text))
        rows.append(
            {
                "qid": q.qid,
                "split": q.split,
                "answerable": q.answerable,
                "golds": q.answers,
                "answer": o.answer,
                "gold_retrieved": bool(q.pid in idx[qi]),
                "gold_rank1": bool(idx[qi, 0] == q.pid),
                "margin": o.margin,
                "span_score": o.span_score,
                "null_score": o.null_score,
                "span_prob": o.span_prob,
                "ret_top1": float(vals[qi, 0]),
                "ret_gap": float(vals[qi, 0] - vals[qi, 1]) if k > 1 else 0.0,
                "chosen_rank": best + 1,
                "answer_tokens": len(o.answer.split()),
                "question_overlap": len(qt & set(tokenize(sent))) / max(1, len(qt)),
                "span_f1": f1(o.answer, q.answers) if q.answerable else 0.0,
            }
        )
    for r in rows:
        r["ms_retrieval"] = t_ret * 1000
        r["ms_reader"] = t_read * 1000
    return rows


def _x(rows) -> np.ndarray:
    return np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)


def _tune_threshold(conf: np.ndarray, rows: list[dict]) -> float:
    """Threshold on a confidence score that maximises SQuAD 2.0 F1 on the given rows."""
    best_t, best_f = 0.0, -1.0
    for t in np.unique(np.quantile(conf, np.linspace(0, 1, 201))):
        preds = [r["answer"] if c >= t else "" for r, c in zip(rows, conf, strict=True)]
        s = squad_scores(preds, [r["golds"] for r in rows])["f1"]
        if s > best_f:
            best_t, best_f = float(t), s
    return best_t


def evaluate_policies(rows: list[dict]) -> dict:
    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]
    golds = [r["golds"] for r in test]
    # target of the learned gate: "the reader's span would be judged correct"
    y_cal = np.array([r["span_f1"] >= 0.5 for r in calib])
    y_test = np.array([r["span_f1"] >= 0.5 for r in test])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(_x(calib), y_cal)
    p_cal, p_test = clf.predict_proba(_x(calib))[:, 1], clf.predict_proba(_x(test))[:, 1]

    m_cal = np.array([r["margin"] for r in calib])
    m_test = np.array([r["margin"] for r in test])
    t_margin = _tune_threshold(m_cal, calib)
    t_gate = _tune_threshold(p_cal, calib)

    policies = {
        "always answer": [r["answer"] for r in test],
        "reader null-score (default, margin > 0)": [
            r["answer"] if r["margin"] > 0 else "" for r in test
        ],
        "reader null-score (tuned threshold)": [
            r["answer"] if m >= t_margin else "" for r, m in zip(test, m_test, strict=True)
        ],
        "learned answerability gate": [
            r["answer"] if p >= t_gate else "" for r, p in zip(test, p_test, strict=True)
        ],
    }
    out = {"policies": {}, "thresholds": {"margin": t_margin, "gate": t_gate}}
    for name, preds in policies.items():
        out["policies"][name] = squad_scores(preds, golds)
    out["selective"] = {
        "reader margin": risk_coverage(m_test, y_test),
        "learned gate": risk_coverage(p_test, y_test),
    }
    out["calibration_ece"] = {"learned gate": ece(p_test, y_test.astype(float))}
    coef = clf[-1].coef_[0]
    out["gate_coefficients"] = {f: float(c) for f, c in zip(FEATURES, coef, strict=True)}
    out["retrieval"] = {
        "answerable_gold_in_topk": float(
            np.mean([r["gold_retrieved"] for r in test if r["answerable"]])
        ),
        "answerable_gold_at_1": float(np.mean([r["gold_rank1"] for r in test if r["answerable"]])),
    }
    out["latency_ms"] = {
        "retrieval": float(np.mean([r["ms_retrieval"] for r in rows])),
        "reader_k_passages": float(np.mean([r["ms_reader"] for r in rows])),
    }
    out["n_test"] = len(test)
    out["gate_test_probs"] = [float(p) for p in p_test]
    return out
