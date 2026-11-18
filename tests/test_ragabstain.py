import json
from dataclasses import replace

import numpy as np
import pytest

from ragabstain import data as data_mod
from ragabstain.llm import parse_answer
from ragabstain.metrics import ece, exact, f1, normalize, risk_coverage, squad_scores
from ragabstain.pipeline import evaluate_policies, run_reader, sentence_with
from ragabstain.reader import ReaderOutput, best_span
from ragabstain.report import write_report
from ragabstain.retrieval import BM25, rrf, top_k

SQUAD = {
    "data": [
        {
            "title": "Normandy",
            "paragraphs": [
                {
                    "context": "The Normans were in Normandy in the 10th and 11th centuries.",
                    "qas": [
                        {
                            "id": "a",
                            "question": "When were the Normans in Normandy?",
                            "answers": [{"text": "10th and 11th centuries", "answer_start": 31}],
                            "is_impossible": False,
                        },
                        {
                            "id": "b",
                            "question": "Who gave their name to Normandy in 1000?",
                            "answers": [],
                            "is_impossible": True,
                        },
                    ],
                }
            ],
        },
        {
            "title": "Oxygen",
            "paragraphs": [
                {
                    "context": "Oxygen is a chemical element with symbol O.",
                    "qas": [
                        {
                            "id": "c",
                            "question": "What is the symbol of oxygen?",
                            "answers": [{"text": "O", "answer_start": 41}],
                        }
                    ],
                }
            ],
        },
    ]
}


def test_parse_and_sample(tmp_path):
    passages, questions = data_mod.parse(SQUAD)
    assert [p.title for p in passages] == ["Normandy", "Oxygen"]
    assert [q.answerable for q in questions] == [True, False, True]
    sample = data_mod.sample(questions, n_per_class=1)
    assert sorted(q.split for q in sample) == ["calib", "calib"]
    data_mod.save(passages, sample, tmp_path)
    p2, q2 = data_mod.load(tmp_path)
    assert p2 == passages and q2 == sample


def test_official_squad_normalisation_and_scores():
    assert normalize("The  Normans!") == "normans"
    assert exact("the Normans", ["Normans"]) == 1.0
    assert f1("10th century", ["10th and 11th centuries"]) == pytest.approx(0.333, abs=1e-3)
    assert exact("", []) == 1.0 and f1("something", []) == 0.0
    s = squad_scores(["Normans", "", "x"], [["Normans"], [], []])
    assert s["em"] == pytest.approx(2 / 3)
    assert s["hallucination_rate"] == 0.5 and s["no_ans_acc"] == 0.5


def test_selective_metrics():
    conf = np.array([0.9, 0.8, 0.1, 0.2])
    correct = np.array([1, 1, 0, 0])
    rc = risk_coverage(conf, correct)
    assert rc["acc@50%cov"] == 1.0
    worse = risk_coverage(-conf, correct)
    assert rc["aurc"] < worse["aurc"]
    assert ece(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0


def test_bm25_and_fusion():
    passages, _ = data_mod.parse(SQUAD)
    bm = BM25(passages)
    s = bm.scores_batch(["symbol of oxygen", "Normans centuries"])
    idx, _ = top_k(s, 1)
    assert idx[:, 0].tolist() == [1, 0]
    assert rrf([np.array([2.0, 1.0]), np.array([2.0, 1.0])]).argmax() == 0


def test_best_span_respects_context_and_length():
    start = np.array([5.0, 0.1, 3.0, 0.2, 0.0])
    end = np.array([5.0, 0.0, 0.5, 2.5, 0.1])
    offsets = [(0, 0), (0, 3), (4, 8), (9, 12), (13, 15)]
    mask = np.array([False, True, True, True, True])  # position 0 is [CLS]/question
    score, a, z, prob = best_span(start, end, offsets, mask)
    assert (a, z) == (4, 12) and score == pytest.approx(5.5) and 0 < prob <= 1


class FakeReader:
    """Answers when the passage contains the question's last word, else prefers no-answer."""

    def read(self, pairs):
        out = []
        for q, c in pairs:
            key = q.rstrip("?").split()[-1]
            if key.lower() in c.lower() and "gave" not in q:
                span = "10th and 11th centuries" if "Normans" in c else "O"
                out.append(ReaderOutput(span, 5.0, 1.0, 0.9))
            else:
                out.append(ReaderOutput(c.split()[0], 0.5, 3.0, 0.3))
        return out


def test_pipeline_and_policies_on_toy_data():
    passages, questions = data_mod.parse(SQUAD)
    questions = [
        replace(q, split="calib" if i % 2 == 0 else "test") for i, q in enumerate(questions * 10)
    ]
    rows = run_reader(questions, passages, BM25(passages), FakeReader(), k=2)
    assert {r["chosen_rank"] for r in rows} <= {1, 2}
    res = evaluate_policies(rows)
    pol = res["policies"]
    assert pol["always answer"]["hallucination_rate"] == 1.0
    assert pol["reader null-score (default, margin > 0)"]["hallucination_rate"] == 0.0
    assert len(res["gate_test_probs"]) == res["n_test"]


def test_sentence_and_llm_parsing():
    text = "First one. The answer is here. Last."
    assert "answer is here" in sentence_with(text, text.find("answer"))
    assert parse_answer("UNANSWERABLE") == ""
    assert parse_answer("The passage does not say.") == ""
    assert parse_answer(' "10th century".\nextra') == "10th century"


def test_report(tmp_path):
    s = {
        "f1": 0.5,
        "em": 0.4,
        "has_ans_f1": 0.6,
        "no_ans_acc": 0.7,
        "hallucination_rate": 0.3,
        "answer_rate": 0.5,
    }
    res = {
        "retriever": "bm25",
        "k": 3,
        "n_test": 4,
        "policies": {"always answer": s},
        "selective": {
            "reader margin": {"aurc": 0.2, "acc@25%cov": 1.0, "acc@50%cov": 0.9, "acc@80%cov": 0.8}
        },
        "calibration_ece": {"learned gate": 0.05},
        "latency_ms": {"retrieval": 1.0, "reader_k_passages": 20.0},
    }
    (tmp_path / "abstention_bm25.json").write_text(json.dumps(res))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "| always answer | 0.500 |" in text and "AURC" in text
