"""Full-rank evaluation for SASRec and the baselines.

For every prefix in the held-out split we compute:

  * **HR@K** — was the true next item among the top-K predictions?
  * **NDCG@K** — discounted gain at the rank of the true next item.

Ranking is done over the *entire* item vocabulary (no negative sampling at
eval time), which is the protocol recommended by Krichene & Rendle (2020) and
now standard in the recommendation literature.

Run with:
    python -m src.evaluate --split test
    python -m src.evaluate --split test --model markov
    python -m src.evaluate --split test --model itemknn
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, CONFIG, LOG_DIR, PROCESSED_DIR
from src.dataset import SessionEvalDataset, load_n_items, load_split
from src.model import SASRec


# --------------------------------------------------------------------------- #
# Metric utilities                                                            #
# --------------------------------------------------------------------------- #

def topk_metrics(scores: np.ndarray, targets: np.ndarray, k: int) -> dict:
    """``scores``: (B, V), ``targets``: (B,)."""
    # argpartition is O(V) vs full sort's O(V log V).
    topk = np.argpartition(-scores, kth=k, axis=1)[:, :k]
    # then sort just those k columns by score for proper rank.
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, topk], axis=1)
    topk_sorted = topk[rows, order]

    hits = (topk_sorted == targets[:, None])
    hit = hits.any(axis=1).astype(np.float32)
    rank = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    ndcg = np.where(rank >= 0, 1.0 / np.log2(rank + 2.0), 0.0)
    return {
        "n": int(scores.shape[0]),
        "hit_sum": float(hit.sum()),
        "ndcg_sum": float(ndcg.sum()),
    }


def aggregate(parts: list[dict], k: int) -> dict:
    n = sum(p["n"] for p in parts)
    return {
        "n_examples": n,
        f"HR@{k}": sum(p["hit_sum"] for p in parts) / max(n, 1),
        f"NDCG@{k}": sum(p["ndcg_sum"] for p in parts) / max(n, 1),
    }


# --------------------------------------------------------------------------- #
# SASRec evaluation                                                           #
# --------------------------------------------------------------------------- #

def eval_sasrec(
    split: str = "test",
    ckpt: str = "sasrec_best.pt",
    batch_size: int = 512,
    k: int = 10,
) -> dict:
    cfg = CONFIG
    device = (
        "cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    n_items = load_n_items()
    ds = SessionEvalDataset(load_split(split), cfg.data.max_seq_len)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = SASRec(
        n_items=n_items,
        hidden_dim=cfg.model.hidden_dim,
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
    ).to(device)
    state = torch.load(CKPT_DIR / ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    parts: list[dict] = []
    with torch.no_grad():
        for seq, target in dl:
            seq = seq.to(device, non_blocking=True)
            scores = model.predict_next(seq).cpu().numpy()
            # No seen-item masking: see evaluate_model() in train.py for the
            # rationale (fair comparison against the Markov baseline, which
            # naturally allows next == previous on YOOCHOOSE).
            parts.append(topk_metrics(scores, target.numpy(), k=k))

    res = aggregate(parts, k=k)
    res["model"] = "sasrec"
    res["split"] = split
    return res


# --------------------------------------------------------------------------- #
# Baseline evaluation                                                         #
# --------------------------------------------------------------------------- #

def eval_baseline(model_name: str, split: str, k: int = 10,
                  batch_size: int = 512) -> dict:
    from src.baselines import ItemKNNBaseline, MarkovBaseline

    cfg = CONFIG
    n_items = load_n_items()
    train = load_split("train")

    if model_name == "markov":
        baseline = MarkovBaseline(n_items=n_items).fit(train)
    elif model_name == "itemknn":
        baseline = ItemKNNBaseline(n_items=n_items).fit(train)
    else:
        raise ValueError(f"unknown baseline {model_name}")

    ds = SessionEvalDataset(load_split(split), cfg.data.max_seq_len)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    parts: list[dict] = []
    for seq, target in dl:
        scores = baseline.score_batch(seq.numpy())
        parts.append(topk_metrics(scores, target.numpy(), k=k))

    res = aggregate(parts, k=k)
    res["model"] = model_name
    res["split"] = split
    return res


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sasrec", "markov", "itemknn"],
                   default="sasrec")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--ckpt", default="sasrec_best.pt")
    p.add_argument("--k", type=int, default=CONFIG.train.top_k)
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    if args.model == "sasrec":
        res = eval_sasrec(split=args.split, ckpt=args.ckpt,
                          batch_size=args.batch_size, k=args.k)
    else:
        res = eval_baseline(args.model, split=args.split,
                            batch_size=args.batch_size, k=args.k)

    out = LOG_DIR / f"eval_{args.model}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n[evaluate] saved to {out}")


if __name__ == "__main__":
    main()
