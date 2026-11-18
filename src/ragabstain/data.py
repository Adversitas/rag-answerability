"""SQuAD 2.0 dev set as an open-book RAG benchmark with unanswerable questions.

The corpus is every paragraph of the official dev set (1,204 Wikipedia paragraphs). About half
of the 11,873 questions are *unanswerable*: written by crowd workers to look answerable from
the paragraph, with a plausible but wrong span nearby. That is exactly the situation in which a
RAG system hallucinates, which makes it a good stress test for abstention.
"""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


@dataclass
class Passage:
    pid: int
    title: str
    text: str


@dataclass
class Question:
    qid: str
    text: str
    pid: int  # paragraph the question was written against
    answers: list[str] = field(default_factory=list)  # empty = unanswerable
    split: str = "test"  # calib | test

    @property
    def answerable(self) -> bool:
        return bool(self.answers)


def download(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path


def parse(squad: dict) -> tuple[list[Passage], list[Question]]:
    passages, questions = [], []
    for article in squad["data"]:
        title = article["title"].replace("_", " ")
        for para in article["paragraphs"]:
            pid = len(passages)
            passages.append(Passage(pid, title, para["context"]))
            for qa in para["qas"]:
                answers = sorted({a["text"] for a in qa["answers"]})
                questions.append(Question(qa["id"], qa["question"].strip(), pid, answers))
    return passages, questions


def sample(questions: list[Question], n_per_class: int = 1000, seed: int = 0) -> list[Question]:
    """Balanced sample, then a 50/50 calibration/test split stratified by answerability."""
    rng = random.Random(seed)
    pos = [q for q in questions if q.answerable]
    neg = [q for q in questions if not q.answerable]
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = []
    for group in (pos[:n_per_class], neg[:n_per_class]):
        for i, q in enumerate(group):
            q.split = "calib" if i % 2 == 0 else "test"
            out.append(q)
    return out


def save(passages: list[Passage], questions: list[Question], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "passages.json").write_text(
        json.dumps([asdict(p) for p in passages], ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "questions.json").write_text(
        json.dumps([asdict(q) for q in questions], ensure_ascii=False, indent=0), encoding="utf-8"
    )


def load(data_dir: Path) -> tuple[list[Passage], list[Question]]:
    ps = json.loads((data_dir / "passages.json").read_text(encoding="utf-8"))
    qs = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    return [Passage(**p) for p in ps], [Question(**q) for q in qs]
