"""Train SASRec on the YOOCHOOSE 1/64 split.

Training recipe (matches the original SASRec paper):

  * AdamW, lr 1e-3, weight decay 1e-2, betas (0.9, 0.98)
  * BCE-with-logits over (positive, negative) pairs at every non-pad step
  * one negative per position (cheap and surprisingly competitive vs full
    softmax according to the SASRec ablations)
  * gradient clipping at 5.0
  * early stopping on validation HR@10

Run with:
    python -m src.train
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, CONFIG, LOG_DIR
from src.dataset import SessionEvalDataset, SessionTrainDataset, \
    load_n_items, load_split
from src.evaluate import topk_metrics, aggregate
from src.model import SASRec


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(cfg, n_items: int):
    train_ds = SessionTrainDataset(
        load_split("train"), n_items=n_items,
        max_seq_len=cfg.data.max_seq_len, seed=cfg.train.seed,
    )
    valid_ds = SessionEvalDataset(load_split("valid"), cfg.data.max_seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    return train_dl, valid_dl


def evaluate_model(model: SASRec, valid_dl: DataLoader,
                   device: str, k: int) -> dict:
    """Evaluation runs on a single GPU — the eval batches are small and
    `predict_next` isn't wrapped by DataParallel anyway.

    NOTE: we deliberately do NOT mask out items already seen in the input
    session. On YOOCHOOSE the user often re-clicks the same item, so the
    Markov baseline (which has no such mask) would get a free advantage if
    we masked here. Only the padding slot is masked (already done inside
    `predict_next`).
    """
    base = model.module if isinstance(model, nn.DataParallel) else model
    base.eval()
    parts: list[dict] = []
    with torch.no_grad():
        for seq, target in valid_dl:
            seq = seq.to(device, non_blocking=True)
            scores = base.predict_next(seq).cpu().numpy()
            parts.append(topk_metrics(scores, target.numpy(), k=k))
    return aggregate(parts, k=k)


# --------------------------------------------------------------------------- #
# Training loop                                                               #
# --------------------------------------------------------------------------- #

def train() -> None:
    cfg = CONFIG
    set_seed(cfg.train.seed)
    device = (
        "cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    print(f"[train] device = {device} (visible GPUs: {n_gpus})")

    # If multiple GPUs are visible, scale the batch size linearly so each GPU
    # still sees the original per-device batch. DataParallel splits batches
    # along dim 0, so this gives ~linear speedup when the model fits per GPU.
    if n_gpus > 1:
        original_bs = cfg.train.batch_size
        cfg.train.batch_size = original_bs * n_gpus
        print(f"[train] scaling batch_size {original_bs} -> "
              f"{cfg.train.batch_size} for {n_gpus} GPUs (DataParallel)")

    n_items = load_n_items()
    print(f"[train] n_items = {n_items}")

    train_dl, valid_dl = make_dataloaders(cfg, n_items)
    print(f"[train] train batches = {len(train_dl)}, "
          f"valid examples = {len(valid_dl.dataset)}")

    model = SASRec(
        n_items=n_items,
        hidden_dim=cfg.model.hidden_dim,
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] SASRec parameters: {n_params:,}")

    if n_gpus > 1:
        # `score_pairs` returns a tuple of (B, L) tensors; DataParallel
        # concatenates along dim 0, which is exactly what we want.
        model = nn.DataParallel(model)
        print(f"[train] wrapped in DataParallel across {n_gpus} GPUs")

    # When wrapped, the underlying SASRec is at `model.module`. We keep a
    # handle so evaluation (which calls `predict_next` directly) and
    # checkpointing always touch the unwrapped state dict.
    base_model: SASRec = model.module if isinstance(model, nn.DataParallel) \
        else model

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.98),
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    use_ce = cfg.train.loss_type == "ce"
    print(f"[train] loss_type = {cfg.train.loss_type}")

    history: list[dict] = []
    best_hr = -1.0
    bad_rounds = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_pos = 0

        for seq, pos, neg in train_dl:
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            if use_ce:
                ce, valid = model(seq, pos)        # both (B, L)
                loss = (ce * valid.float()).sum()
                loss = loss / valid.float().sum().clamp(min=1.0)
            else:
                neg = neg.to(device, non_blocking=True)
                pos_logits, neg_logits, valid = model(seq, pos, neg)
                pos_targets = torch.ones_like(pos_logits)
                neg_targets = torch.zeros_like(neg_logits)
                loss_pos = bce(pos_logits, pos_targets)
                loss_neg = bce(neg_logits, neg_targets)
                loss = ((loss_pos + loss_neg) * valid.float()).sum()
                loss = loss / valid.float().sum().clamp(min=1.0)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip
            )
            optim.step()

            loss_sum += float(loss.item()) * valid.sum().item()
            n_pos += valid.sum().item()

        train_loss = loss_sum / max(n_pos, 1)
        elapsed = time.time() - t0

        if epoch % cfg.train.eval_every == 0:
            val = evaluate_model(model, valid_dl, device, k=cfg.train.top_k)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_HR": val[f"HR@{cfg.train.top_k}"],
                "valid_NDCG": val[f"NDCG@{cfg.train.top_k}"],
                "epoch_seconds": round(elapsed, 1),
            }
            history.append(row)
            print(f"[train] epoch {epoch:02d} | "
                  f"loss {train_loss:.4f} | "
                  f"HR@{cfg.train.top_k}={val[f'HR@{cfg.train.top_k}']:.4f} "
                  f"NDCG@{cfg.train.top_k}={val[f'NDCG@{cfg.train.top_k}']:.4f}"
                  f" | {elapsed:.1f}s")

            if val[f"HR@{cfg.train.top_k}"] > best_hr:
                best_hr = val[f"HR@{cfg.train.top_k}"]
                bad_rounds = 0
                # Always save the unwrapped state dict so evaluate.py (which
                # builds a plain SASRec) can load it on either 1 or 2 GPUs.
                torch.save(
                    {"model": base_model.state_dict(),
                     "epoch": epoch,
                     "metrics": val},
                    CKPT_DIR / "sasrec_best.pt",
                )
                print(f"[train]   new best HR -- saved checkpoint")
            else:
                bad_rounds += 1
                if bad_rounds >= cfg.train.patience:
                    print(f"[train] early stop after {bad_rounds} stale "
                          f"rounds")
                    break
        else:
            print(f"[train] epoch {epoch:02d} | loss {train_loss:.4f} | "
                  f"{elapsed:.1f}s")

    with open(LOG_DIR / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] history saved -> {LOG_DIR / 'train_history.json'}")


if __name__ == "__main__":
    train()
