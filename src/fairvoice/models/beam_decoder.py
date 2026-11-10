"""
src/fairvoice/models/beam_decoder.py — CTC Prefix Beam Search Decoder.

Implements pure-Python CTC Prefix Beam Search over Wav2Vec2 emission log-probabilities.
Allows generating Top-K candidate transcriptions for downstream language model rescoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import Wav2Vec2Processor


def log_add(a: float, b: float) -> float:
    """Numerically stable log addition: log(exp(a) + exp(b))."""
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


@dataclass
class BeamHypothesis:
    """Represents a candidate decoded hypothesis from CTC Beam Search."""
    text: str
    token_ids: tuple[int, ...]
    acoustic_score: float
    total_score: float = 0.0
    rank: int = 1


class CTCBeamSearchDecoder:
    """
    Fast CTC Prefix Beam Search Decoder for Wav2Vec2 models.

    Parameters
    ----------
    processor : Wav2Vec2Processor
        The Hugging Face processor with tokenizer vocabulary.
    beam_width : int
        Number of active beams to maintain at each time step (default: 10).
    blank_id : int
        Token ID for CTC blank / pad token (default: 1 for Wav2Vec2).
    prune_threshold : float
        Top cumulative probability threshold for frame pruning (default: 0.99).
    top_k_per_frame : int
        Maximum number of character candidates to expand per frame (default: 10).
    """

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        beam_width: int = 10,
        blank_id: int | None = None,
        top_k_per_frame: int = 10,
    ) -> None:
        self.processor = processor
        self.beam_width = beam_width
        self.top_k_per_frame = top_k_per_frame
        self.blank_id = blank_id if blank_id is not None else processor.tokenizer.pad_token_id
        if self.blank_id is None:
            self.blank_id = 1

    def decode_logits(self, logits: torch.Tensor | np.ndarray) -> list[BeamHypothesis]:
        """
        Run CTC Prefix Beam Search on emission logits.

        Parameters
        ----------
        logits : torch.Tensor | np.ndarray
            Frame emissions of shape [time_steps, vocab_size] or [1, time_steps, vocab_size].

        Returns
        -------
        list[BeamHypothesis]
            Top ranked decoded candidate hypotheses with acoustic log-probabilities.
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)

        if logits.ndim == 3:
            logits = logits.squeeze(0)

        # Compute log probabilities: shape [T, V]
        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
        time_steps, vocab_size = log_probs.shape

        # Beams map: prefix tuple -> (p_blank, p_non_blank) in log-space
        # Initial empty prefix ()
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -float("inf"))}

        for t in range(time_steps):
            frame_probs = log_probs[t]

            # Prune frame to top-K characters to ensure fast execution
            top_k_indices = np.argsort(frame_probs)[-self.top_k_per_frame:][::-1]

            new_beams: dict[tuple[int, ...], tuple[float, float]] = {}

            for prefix, (p_b, p_nb) in beams.items():
                p_total = log_add(p_b, p_nb)

                for c in top_k_indices:
                    log_p = float(frame_probs[c])

                    if c == self.blank_id:
                        # Blank token updates current prefix's blank probability
                        curr_b, curr_nb = new_beams.get(prefix, (-float("inf"), -float("inf")))
                        new_beams[prefix] = (log_add(curr_b, p_total + log_p), curr_nb)
                    else:
                        end_char = prefix[-1] if len(prefix) > 0 else None

                        if c == end_char:
                            # Repeated character:
                            # 1. Extending with blank in between (p_b -> prefix + c)
                            new_prefix = prefix + (c,)
                            curr_b, curr_nb = new_beams.get(new_prefix, (-float("inf"), -float("inf")))
                            new_beams[new_prefix] = (curr_b, log_add(curr_nb, p_b + log_p))

                            # 2. Duplicate collapsed (p_nb stays on prefix)
                            curr_b, curr_nb = new_beams.get(prefix, (-float("inf"), -float("inf")))
                            new_beams[prefix] = (curr_b, log_add(curr_nb, p_nb + log_p))
                        else:
                            # New character extends prefix
                            new_prefix = prefix + (c,)
                            curr_b, curr_nb = new_beams.get(new_prefix, (-float("inf"), -float("inf")))
                            new_beams[new_prefix] = (curr_b, log_add(curr_nb, p_total + log_p))

            # Prune to top beam_width beams
            beam_scores = [
                (prefix, log_add(pb, pnb))
                for prefix, (pb, pnb) in new_beams.items()
            ]
            beam_scores.sort(key=lambda x: x[1], reverse=True)
            top_beams = beam_scores[: self.beam_width]

            beams = {prefix: new_beams[prefix] for prefix, _ in top_beams}

        # Final hypotheses compilation
        sorted_candidates = [
            (prefix, log_add(pb, pnb))
            for prefix, (pb, pnb) in beams.items()
        ]
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[BeamHypothesis] = []
        for rank, (prefix, score) in enumerate(sorted_candidates, start=1):
            if len(prefix) == 0:
                text = ""
            else:
                decoded_str = self.processor.decode(list(prefix), group_tokens=False)
                text = decoded_str.replace("<s>", "").replace("  ", " ").strip()

            results.append(BeamHypothesis(
                text=text,
                token_ids=prefix,
                acoustic_score=round(score, 4),
                rank=rank,
            ))

        return results

