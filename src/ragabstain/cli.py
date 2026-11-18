"""``ragabstain`` CLI: prepare | retrieval | run | llm | report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as data_mod
from .metrics import squad_scores
from .report import write_report
from .retrieval import BGE_QUERY_PREFIX, BM25, Dense, Hybrid, top_k

EMBEDDER = "BAAI/bge-small-en-v1.5"
READER = "deepset/tinyroberta-squad2"


def make_retriever(kind: str, passages):
    bm25 = BM25(passages)
    if kind == "bm25":
        return bm25
    from sentence_transformers import SentenceTransformer

    dense = Dense(
        passages, SentenceTransformer(EMBEDDER, device="cpu"), "bge-small", BGE_QUERY_PREFIX
    )
    return dense if kind == "dense" else Hybrid(bm25, dense)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragabstain", description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="download SQuAD 2.0 dev, sample 2,000 questions")
    pr.add_argument("--raw", type=Path, default=Path("data/raw/dev-v2.0.json"))
    r = sub.add_parser("retrieval", help="passage recall@k on answerable questions")
    r.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"])
    run = sub.add_parser("run", help="retrieve + read + abstention policies")
    run.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    run.add_argument("-k", type=int, default=3)
    run.add_argument("--oracle", action="store_true", help="read the gold passage only")
    llm = sub.add_parser("llm", help="generative reader (local LLM) on test questions")
    llm.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    llm.add_argument("-n", type=int, default=300, help="test questions (balanced)")
    llm.add_argument("--retriever", default="hybrid")
    sub.add_parser("report", help="rebuild reports/results.md")
    a = p.parse_args(argv)
    a.reports.mkdir(parents=True, exist_ok=True)

    if a.cmd == "prepare":
        raw = data_mod.download(a.raw)
        passages, questions = data_mod.parse(json.loads(raw.read_text(encoding="utf-8")))
        data_mod.save(passages, data_mod.sample(questions), a.data)
        print(f"{len(passages)} passages, 2000 sampled questions -> {a.data}")
        return
    if a.cmd == "report":
        print(write_report(a.reports))
        return

    passages, questions = data_mod.load(a.data)
    if a.cmd == "retrieval":
        ans = [q for q in questions if q.answerable]
        res = {}
        for kind in a.retrievers:
            ret = make_retriever(kind, passages)
            idx, _ = top_k(ret.scores_batch([q.text for q in ans]), 10)
            gold = np.array([q.pid for q in ans])[:, None]
            res[kind] = {
                f"recall@{k}": float((idx[:, :k] == gold).any(1).mean()) for k in (1, 3, 5, 10)
            }
            print(kind, res[kind])
        name = "retrieval_bm25.json" if a.retrievers == ["bm25"] else "retrieval.json"
        (a.reports / name).write_text(json.dumps(res, indent=1))
    elif a.cmd == "run":
        from .pipeline import evaluate_policies, run_reader
        from .reader import Reader

        reader = Reader(READER)
        if a.oracle:

            class Oracle:
                def scores_batch(self, qs):
                    s = np.zeros((len(qs), len(passages)))
                    for i, q in enumerate(questions):
                        s[i, q.pid] = 1.0
                    return s

            rows = run_reader(questions, passages, Oracle(), reader, k=1)
            tag = "oracle"
        else:
            rows = run_reader(
                questions, passages, make_retriever(a.retriever, passages), reader, k=a.k
            )
            tag = a.retriever
        res = evaluate_policies(rows)
        res["retriever"], res["k"] = tag, 1 if a.oracle else a.k
        (a.reports / f"abstention_{tag}.json").write_text(json.dumps(res, indent=1))
        (a.reports / f"rows_{tag}.json").write_text(json.dumps(rows))
        for name, s in res["policies"].items():
            print(f"{name:42s} F1={s['f1']:.3f} halluc={s['hallucination_rate']:.3f}")
    elif a.cmd == "llm":
        from .llm import HFChatModel, answer_all

        test = [q for q in questions if q.split == "test"]
        half = a.n // 2
        subset = [q for q in test if q.answerable][:half] + [q for q in test if not q.answerable][
            :half
        ]
        ret = make_retriever(a.retriever, passages)
        idx, _ = top_k(ret.scores_batch([q.text for q in subset]), 1)
        model = HFChatModel(a.model)
        preds, lat = answer_all(model, subset, passages, idx[:, 0].tolist())
        golds = [q.answers for q in subset]
        res = {
            "model": a.model,
            "retriever": a.retriever,
            "n": len(subset),
            "llm": squad_scores(preds, golds),
            "median_latency_s": float(np.median(lat)),
        }
        # same questions, extractive reader + learned gate, for a like-for-like comparison
        rows_path = a.reports / f"rows_{a.retriever}.json"
        ab_path = a.reports / f"abstention_{a.retriever}.json"
        if rows_path.exists() and ab_path.exists():
            rows = {r["qid"]: r for r in json.loads(rows_path.read_text())}
            ab = json.loads(ab_path.read_text())
            test_rows = [r for r in rows.values() if r["split"] == "test"]
            probs = dict(zip([r["qid"] for r in test_rows], ab["gate_test_probs"], strict=True))
            t = ab["thresholds"]["gate"]
            gated = [rows[q.qid]["answer"] if probs[q.qid] >= t else "" for q in subset]
            res["reader+gate_same_questions"] = squad_scores(gated, golds)
            # gate in front of the LLM: only spend a generation when the gate says answerable
            combo = [pr if probs[q.qid] >= t else "" for pr, q in zip(preds, subset, strict=True)]
            res["gate+llm"] = squad_scores(combo, golds)
            res["gate+llm_share_of_llm_calls"] = float(np.mean([probs[q.qid] >= t for q in subset]))
        res["examples"] = [
            {"question": q.text, "gold": q.answers, "llm": pr}
            for q, pr in list(zip(subset, preds, strict=True))[:: max(1, len(subset) // 12)]
        ]
        slug = a.model.split("/")[-1]
        (a.reports / f"llm_{slug}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
