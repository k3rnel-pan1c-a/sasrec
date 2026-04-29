"""Two non-neural baselines required by the proposal.

  * **First-Order Markov Chain** — predicts the next click using the
    transition matrix ``P(j | i)`` estimated from training sessions only.
  * **Item-KNN (session-similarity flavour)** — for each query item, score
    candidates by cosine similarity of their *item-occurrence vectors* over
    sessions. This is the classic GRU4Rec baseline ("co-occurrence ignoring
    order").

Both baselines expose a ``score_batch(seq) -> (B, n_items + 1)`` method so
that ``evaluate.py`` can rank them against SASRec with the same protocol.

Sparse matrices keep memory bounded (n_items can be ~30 K).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# First-Order Markov                                                          #
# --------------------------------------------------------------------------- #

class MarkovBaseline:
    """``score = log P(j | last_click_i)`` estimated with add-1 smoothing."""

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.transitions: sp.csr_matrix | None = None

    def fit(self, sessions: dict[int, list[int]]) -> "MarkovBaseline":
        rows, cols, vals = [], [], []
        cnt: Counter = Counter()
        for s in sessions.values():
            for a, b in zip(s[:-1], s[1:]):
                cnt[(a, b)] += 1
        for (a, b), c in cnt.items():
            rows.append(a); cols.append(b); vals.append(c)

        # +1 size because index 0 is reserved for padding.
        N = self.n_items + 1
        m = sp.coo_matrix(
            (vals, (rows, cols)), shape=(N, N), dtype=np.float32
        ).tocsr()
        # Row-normalise to a probability distribution.
        row_sums = np.asarray(m.sum(axis=1)).ravel() + 1e-9
        d = sp.diags(1.0 / row_sums)
        self.transitions = (d @ m).tocsr()
        return self

    def score_batch(self, seqs: np.ndarray) -> np.ndarray:
        """seqs: (B, L) right-aligned; we look only at the last (rightmost)
        non-pad click."""
        last = seqs[:, -1]
        scores = np.asarray(self.transitions[last].todense())
        scores[:, 0] = -np.inf       # mask padding slot
        return scores


# --------------------------------------------------------------------------- #
# Item-KNN                                                                    #
# --------------------------------------------------------------------------- #

class ItemKNNBaseline:
    """Cosine similarity between items based on session co-occurrence.

    For a query session, we score each candidate item by the *sum* of its
    similarities to every item in the input sequence. This is the standard
    "Session-KNN" reduction used as a baseline in GRU4Rec.
    """

    def __init__(self, n_items: int):
        self.n_items = n_items
        self.sim: sp.csr_matrix | None = None

    def fit(self, sessions: dict[int, list[int]]) -> "ItemKNNBaseline":
        N = self.n_items + 1
        # Build a sparse session x item incidence matrix (binary: was the item
        # clicked at least once in this session?).
        rows, cols = [], []
        for s_idx, s in enumerate(sessions.values()):
            for itm in set(s):
                rows.append(s_idx); cols.append(itm)
        n_sess = len(sessions)
        M = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_sess, N),
        ).tocsr()

        # Item co-occurrence = M^T M; cosine via norm rescaling.
        cooc = (M.T @ M).astype(np.float32)
        norms = np.sqrt(np.asarray(cooc.diagonal()).ravel()) + 1e-9
        d = sp.diags(1.0 / norms)
        self.sim = (d @ cooc @ d).tocsr()
        # The padding row/col is implicitly already harmless: pad rows in the
        # query are all-zero, and we mask the pad column at scoring time.
        return self

    def score_batch(self, seqs: np.ndarray) -> np.ndarray:
        # Treat each sequence as a 0/1 indicator over items; the score is
        # row_indicator @ sim. Padding rows are 0 -> harmless.
        B, _ = seqs.shape
        rows, cols = [], []
        for b in range(B):
            unique = np.unique(seqs[b])
            unique = unique[unique != 0]
            rows.extend([b] * len(unique))
            cols.extend(unique.tolist())
        N = self.n_items + 1
        Q = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(B, N),
        ).tocsr()
        scores = np.asarray((Q @ self.sim).todense())
        scores[:, 0] = -np.inf
        # No seen-item masking — see evaluate_model() in src/train.py.
        return scores
