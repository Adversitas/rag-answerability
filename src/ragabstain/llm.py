"""Generative reader: a small open-weight LLM answers from the retrieved passage or abstains.

The prompt asks for a short span copied from the passage, or the single word UNANSWERABLE.
Running it on the same test questions as the extractive reader gives a like-for-like
comparison of hallucination rate, accuracy and latency.
"""

from __future__ import annotations

import re
import time

PROMPT = (
    "Answer the question using only the passage. Reply with a short answer copied from the "
    "passage (a few words). If the passage does not contain the answer, reply exactly "
    "UNANSWERABLE.\n\nPassage ({title}): {passage}\n\nQuestion: {question}\nAnswer:"
)
_ABSTAIN = re.compile(
    r"\bunanswerable\b|not (?:mentioned|provided|stated)|does not (?:say|contain)", re.I
)


def parse_answer(text: str) -> str:
    """Map raw model output to a SQuAD prediction ('' means abstain)."""
    text = text.strip().split("\n")[0].strip()
    if not text or _ABSTAIN.search(text):
        return ""
    return text.strip(" .\"'")


class HFChatModel:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None):
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # float32 on CPU: bf16 matmuls are an order of magnitude slower without AMX support
        dtype = torch.float32 if device == "cpu" else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int = 24) -> str:
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)


def answer_all(model, questions, passages, top1: list[int]) -> tuple[list[str], list[float]]:
    preds, latencies = [], []
    for q, pid in zip(questions, top1, strict=True):
        p = passages[pid]
        t0 = time.perf_counter()
        raw = model.generate(PROMPT.format(title=p.title, passage=p.text, question=q.text))
        latencies.append(time.perf_counter() - t0)
        preds.append(parse_answer(raw))
    return preds, latencies
