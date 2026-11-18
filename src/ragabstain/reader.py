"""Extractive reader trained on SQuAD 2.0: returns the best span *and* the no-answer score.

For each (question, passage) pair the model gives start/end logits over the passage tokens
plus the logits of the [CLS] position, which SQuAD 2.0 models use to say "no answer here".
The gap between the best span score and that null score is the classic abstention signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReaderOutput:
    answer: str
    span_score: float  # start_logit + end_logit of the best valid span
    null_score: float  # start_logit[CLS] + end_logit[CLS]
    span_prob: float  # softmax probability mass of the chosen span among top candidates

    @property
    def margin(self) -> float:
        return self.span_score - self.null_score


def best_span(start, end, offsets, context_mask, max_len: int = 30, top: int = 20):
    """Highest-scoring (start <= end, length <= max_len) span inside the context."""
    s_idx = [i for i in np.argsort(-start)[: top * 2] if context_mask[i]][:top]
    e_idx = [i for i in np.argsort(-end)[: top * 2] if context_mask[i]][:top]
    cands = [
        (start[s] + end[e], s, e) for s in s_idx for e in e_idx if s <= e and e - s + 1 <= max_len
    ]
    if not cands:
        return None
    cands.sort(reverse=True)
    scores = np.array([c[0] for c in cands[:top]])
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    score, s, e = cands[0]
    return float(score), offsets[s][0], offsets[e][1], float(probs[0])


class Reader:
    def __init__(self, model_name: str = "deepset/tinyroberta-squad2", device: str | None = None):
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device).eval()
        self.device = device

    def read(
        self, pairs: list[tuple[str, str]], batch_size: int = 32, max_length: int = 384
    ) -> list[ReaderOutput]:
        """Read (question, passage) pairs. Long passages are windowed (stride 128)."""
        outputs: list[ReaderOutput | None] = [None] * len(pairs)
        for b0 in range(0, len(pairs), batch_size):
            chunk = pairs[b0 : b0 + batch_size]
            enc = self.tok(
                [q for q, _ in chunk],
                [c for _, c in chunk],
                truncation="only_second",
                max_length=max_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
            )
            sample_map = enc.pop("overflow_to_sample_mapping").tolist()
            offsets = enc.pop("offset_mapping").tolist()
            with self.torch.no_grad():
                out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
            starts = out.start_logits.cpu().numpy()
            ends = out.end_logits.cpu().numpy()
            for f, si in enumerate(sample_map):
                seq_ids = enc.sequence_ids(f)
                mask = np.array([sid == 1 for sid in seq_ids])
                null = float(starts[f][0] + ends[f][0])
                found = best_span(starts[f], ends[f], offsets[f], mask)
                i = b0 + si
                prev = outputs[i]
                if found is None:
                    cand = ReaderOutput("", -1e9, null, 0.0)
                else:
                    score, a, z, prob = found
                    cand = ReaderOutput(chunk[si][1][a:z], score, null, prob)
                # across windows keep the best span; the null score is the most confident
                # "no answer" (min over windows, as in the HF SQuAD 2.0 post-processing)
                if prev is None:
                    outputs[i] = cand
                else:
                    best = cand if cand.span_score > prev.span_score else prev
                    outputs[i] = ReaderOutput(
                        best.answer, best.span_score, min(prev.null_score, null), best.span_prob
                    )
        return outputs  # type: ignore[return-value]
