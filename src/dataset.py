"""PyTorch ``Dataset`` for session sequences.

We follow the SASRec training formulation:

  * each session of length ``L`` produces ONE training example
  * the input is the first ``L-1`` clicks, **left-padded** to ``max_seq_len``
    with the padding id ``0``
  * the positive target at position ``t`` is the click that *actually* came
    next (so positives = input shifted left by one, then the final true item)
  * one negative item per non-pad position is sampled uniformly from items
    that do **not** occur anywhere in this session — feeds the BCE loss

The same class is reused for evaluation, where every prefix of the session
becomes a query (handled in ``evaluate.py``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import PROCESSED_DIR


PAD_ID = 0


def load_split(name: str) -> dict[int, list[int]]:
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_n_items() -> int:
    import json
    with open(PROCESSED_DIR / "meta.json") as f:
        return int(json.load(f)["n_items"])


# --------------------------------------------------------------------------- #
# Training dataset                                                            #
# --------------------------------------------------------------------------- #

class SessionTrainDataset(Dataset):
    """Yields (input_seq, pos_seq, neg_seq) triplets for SASRec training."""

    def __init__(
        self,
        sessions: dict[int, list[int]],
        n_items: int,
        max_seq_len: int,
        seed: int = 0,
    ) -> None:
        # Drop sessions that cannot supply at least one (input, target) pair.
        self._sessions: list[list[int]] = [
            s for s in sessions.values() if len(s) >= 2
        ]
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._sessions)

    # SASRec right-aligns the sequence: the most recent click is at the rightmost
    # position, older clicks pad on the left. The model's causal mask then makes
    # padding positions attend to nothing.
    def _right_align(self, seq: list[int]) -> np.ndarray:
        out = np.zeros(self.max_seq_len, dtype=np.int64)
        seq = seq[-self.max_seq_len:]
        out[-len(seq):] = seq
        return out

    def _sample_negative(self, exclude: set[int]) -> int:
        # Items live in [1, n_items]. Loop until we land outside the session.
        while True:
            cand = int(self._rng.integers(1, self.n_items + 1))
            if cand not in exclude:
                return cand

    def __getitem__(self, idx: int):
        sess = self._sessions[idx]
        # input: first L-1 clicks; pos target: clicks shifted left by one.
        input_clicks = sess[:-1]
        pos_clicks = sess[1:]

        inp = self._right_align(input_clicks)
        pos = self._right_align(pos_clicks)

        # Build one negative per non-pad position.
        exclude = set(sess)
        neg = np.zeros_like(inp)
        nonpad = np.flatnonzero(inp)
        for p in nonpad:
            neg[p] = self._sample_negative(exclude)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )


# --------------------------------------------------------------------------- #
# Evaluation dataset                                                          #
# --------------------------------------------------------------------------- #

class SessionEvalDataset(Dataset):
    """One example per non-trivial *prefix* of every session.

    Following the GRU4Rec / NARM evaluation protocol: for a session
    ``[c1, c2, ..., cN]`` we score the model on every prediction step
    ``c_{t+1} | c_1..c_t`` for ``t = 1..N-1``. Each item ``__getitem__``
    returns the right-aligned context and the held-out next item.
    """

    def __init__(
        self,
        sessions: dict[int, list[int]],
        max_seq_len: int,
    ) -> None:
        self.examples: list[tuple[list[int], int]] = []
        for s in sessions.values():
            if len(s) < 2:
                continue
            for t in range(1, len(s)):
                self.examples.append((s[:t], s[t]))
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ctx, target = self.examples[idx]
        out = np.zeros(self.max_seq_len, dtype=np.int64)
        ctx = ctx[-self.max_seq_len:]
        out[-len(ctx):] = ctx
        return torch.from_numpy(out), torch.tensor(target, dtype=torch.long)
