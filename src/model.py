"""SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Implemented from scratch in PyTorch. The architecture mirrors the paper:

    item embedding + learnable positional embedding
        -> dropout
        -> N x [ multi-head self-attention with causal mask
                 + residual + layer-norm
                 + position-wise feed-forward
                 + residual + layer-norm ]
        -> per-position output vector

Predictions reuse the item embedding matrix (weight tying) so the score for
item ``i`` at position ``t`` is the dot product
``output_t . item_embedding_i``. Training uses BCE between
``sigmoid(score_pos) - 1`` and ``sigmoid(score_neg) - 0``, ignoring positions
that come from padding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


class PointWiseFeedForward(nn.Module):
    """Two 1-d convolutions with ReLU in the middle, à la SASRec."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D). Conv1d wants (B, D, L).
        h = x.transpose(1, 2)
        h = self.dropout1(F.relu(self.fc1(h)))
        h = self.dropout2(self.fc2(h))
        return h.transpose(1, 2)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.ff = PointWiseFeedForward(hidden_dim, dropout)
        self.ln_ff = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # SASRec uses pre-LN style; LN before each sublayer + residual.
        h = self.ln_attn(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        h = self.ln_ff(x)
        x = x + self.ff(h)
        return x


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        n_blocks: int = 2,
        n_heads: int = 1,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # n_items + 1 to accommodate the padding id 0; padding_idx zeroes its
        # embedding and disables gradient flow through it.
        self.item_emb = nn.Embedding(
            n_items + 1, hidden_dim, padding_idx=PAD_ID
        )
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD_ID].zero_()

    # ------------------------------------------------------------------ #
    # Encoding                                                           #
    # ------------------------------------------------------------------ #

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """Run the stack and return per-position vectors of shape (B, L, D)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) * math.sqrt(self.hidden_dim)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Causal mask: position i may not attend to j > i. ``True`` blocks.
        causal = torch.triu(
            torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
        )
        # Padding mask: ``True`` where keys are padding and should be ignored.
        kpm = seq == PAD_ID

        for block in self.blocks:
            x = block(x, causal_mask=causal, key_padding_mask=kpm)

        return self.ln_out(x)

    # ------------------------------------------------------------------ #
    # Training-time scoring                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        seq: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor | None = None,
    ):
        """Two flavours, selected by whether ``neg`` is provided.

        BCE mode (``neg`` given): returns (pos_logits, neg_logits, valid),
        each (B, L). The training loop applies ``BCEWithLogitsLoss`` to the
        first two and masks by the third.

        CE mode (``neg=None``): returns (per_position_loss, valid), each
        (B, L). ``per_position_loss`` is the cross-entropy of the model's
        full-vocab distribution at each position against the true next item.
        The training loop just masks and averages — no further loss
        computation. This keeps the heavy ``logits @ embedding^T`` matmul
        inside the model so DataParallel only gathers small (B, L) tensors,
        not (B, L, V+1) logit tensors.
        """
        h = self.encode(seq)                      # (B, L, D)

        if neg is not None:
            # Original SASRec BCE recipe.
            pos_e = self.item_emb(pos)            # (B, L, D)
            neg_e = self.item_emb(neg)            # (B, L, D)
            pos_logits = (h * pos_e).sum(dim=-1)
            neg_logits = (h * neg_e).sum(dim=-1)
            valid = seq != PAD_ID
            return pos_logits, neg_logits, valid

        # Full cross-entropy over all items.
        logits = h @ self.item_emb.weight.T       # (B, L, V+1)
        B, L, V1 = logits.shape
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(B * L, V1),
            pos.reshape(B * L),
            reduction="none",
            ignore_index=PAD_ID,
        ).reshape(B, L)
        valid = pos != PAD_ID
        return ce, valid

    # Back-compat alias — kept so older snippets / tests still work.
    score_pairs = forward

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_next(self, seq: torch.Tensor) -> torch.Tensor:
        """Score every item for the *last* position of each sequence.

        Returns a tensor of shape (B, n_items + 1) — index 0 is the padding
        slot and is masked to -inf so it can never be the top-1 prediction.
        """
        h = self.encode(seq)[:, -1, :]            # (B, D)
        scores = h @ self.item_emb.weight.T       # (B, n_items + 1)
        scores[:, PAD_ID] = float("-inf")
        return scores
