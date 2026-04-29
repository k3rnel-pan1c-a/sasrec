"""Live-demo session replay.

Pick a random session from the test split (or one supplied via --session),
walk through it click by click, and at each step print the model's top-K
next-item predictions next to the *actual* next click. Marks hits in green
and misses in red, and reports the running HR@K and NDCG@K at the end.

Usage during a live demo:
    python -m src.demo                     # one random session, k=10
    python -m src.demo --k 20              # top-20 predictions
    python -m src.demo --n 5               # 5 sessions in a row
    python -m src.demo --session 42        # a specific session id
    python -m src.demo --model markov      # baseline replay for contrast
    python -m src.demo --slow 1.0          # pause N seconds between steps
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.config import CKPT_DIR, CONFIG, PROCESSED_DIR
from src.dataset import load_n_items, load_split
from src.model import SASRec


# Terminal colors. ``no-color=True`` if stdout is being piped.
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    YEL = "\033[93m"
    CYAN = "\033[96m"
    END = "\033[0m"


def right_align(seq: list[int], L: int) -> np.ndarray:
    out = np.zeros(L, dtype=np.int64)
    seq = seq[-L:]
    out[-len(seq):] = seq
    return out


# --------------------------------------------------------------------------- #
# Scorer wrappers — uniform interface                                         #
# --------------------------------------------------------------------------- #

class SasrecScorer:
    name = "SASRec"

    def __init__(self, ckpt: str = "sasrec_best.pt"):
        n_items = load_n_items()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SASRec(
            n_items=n_items,
            hidden_dim=CONFIG.model.hidden_dim,
            n_blocks=CONFIG.model.n_blocks,
            n_heads=CONFIG.model.n_heads,
            max_seq_len=CONFIG.model.max_seq_len,
            dropout=0.0,
        ).to(device)
        state = torch.load(CKPT_DIR / ckpt, map_location=device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.L = CONFIG.model.max_seq_len

    @torch.no_grad()
    def topk(self, prefix: list[int], k: int) -> list[int]:
        seq = torch.from_numpy(right_align(prefix, self.L)).unsqueeze(0)
        seq = seq.to(self.device)
        scores = self.model.predict_next(seq).cpu().numpy()[0]
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])].tolist()


class BaselineScorer:
    def __init__(self, name: str):
        from src.baselines import ItemKNNBaseline, MarkovBaseline
        n_items = load_n_items()
        train = load_split("train")
        if name == "markov":
            self.b = MarkovBaseline(n_items).fit(train)
            self.name = "Markov"
        elif name == "itemknn":
            self.b = ItemKNNBaseline(n_items).fit(train)
            self.name = "Item-KNN"
        else:
            raise ValueError(name)
        self.L = CONFIG.data.max_seq_len

    def topk(self, prefix: list[int], k: int) -> list[int]:
        seq = right_align(prefix, self.L)[None, :]
        scores = self.b.score_batch(seq)[0]
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])].tolist()


# --------------------------------------------------------------------------- #
# Replay                                                                      #
# --------------------------------------------------------------------------- #

def replay_session(scorer, session: list[int], k: int, slow: float,
                   sess_idx: int) -> dict:
    print(f"\n{C.BOLD}{C.CYAN}=== Session {sess_idx} "
          f"(length {len(session)}) | model: {scorer.name} ==={C.END}\n")
    print(f"{C.DIM}Full click sequence: {session}{C.END}\n")

    hits = 0
    ranks: list[int] = []
    for t in range(1, len(session)):
        prefix = session[:t]
        target = session[t]
        preds = scorer.topk(prefix, k)

        in_topk = target in preds
        rank = preds.index(target) + 1 if in_topk else None
        if in_topk:
            hits += 1
            ranks.append(rank)
        else:
            ranks.append(0)

        # Pretty print
        prefix_str = " -> ".join(str(x) for x in prefix)
        print(f"  step {t:>2}: prefix [{prefix_str}]")
        print(f"           target = {C.BOLD}{target}{C.END}", end="  ")
        if in_topk:
            print(f"{C.GREEN}HIT @ rank {rank}{C.END}")
        else:
            print(f"{C.RED}MISS{C.END}")
        # Show top-k with the hit highlighted
        decorated = []
        for i, p in enumerate(preds, 1):
            tag = f"{i}:{p}"
            if p == target:
                tag = f"{C.GREEN}{C.BOLD}{tag}{C.END}"
            decorated.append(tag)
        print(f"           top{k}: " + "  ".join(decorated))
        if slow > 0:
            time.sleep(slow)

    n = len(session) - 1
    hr = hits / n
    ndcg = sum(
        (1.0 / np.log2(r + 1.0)) if r > 0 else 0.0 for r in ranks
    ) / n
    print(f"\n{C.YEL}Session summary: HR@{k} = {hr:.3f} | "
          f"NDCG@{k} = {ndcg:.3f}{C.END}")
    return {"n": n, "hits": hits, "ranks": ranks}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sasrec", "markov", "itemknn"],
                   default="sasrec")
    p.add_argument("--ckpt", default="sasrec_best.pt")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n", type=int, default=1,
                   help="how many random sessions to replay")
    p.add_argument("--session", type=int, default=None,
                   help="specific session id from the test split")
    p.add_argument("--min-len", type=int, default=4,
                   help="only sample sessions with at least this many clicks")
    p.add_argument("--slow", type=float, default=0.0,
                   help="seconds to pause between steps (for live demo flow)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    scorer = (
        SasrecScorer(ckpt=args.ckpt) if args.model == "sasrec"
        else BaselineScorer(args.model)
    )

    test = load_split("test")
    sids = list(test.keys())

    if args.session is not None:
        if args.session not in test:
            raise SystemExit(f"session {args.session} not in test split")
        chosen = [args.session]
    else:
        eligible = [s for s in sids if len(test[s]) >= args.min_len]
        if not eligible:
            eligible = sids
        chosen = random.sample(eligible, k=min(args.n, len(eligible)))

    totals = {"n": 0, "hits": 0, "ndcg_sum": 0.0}
    for sid in chosen:
        out = replay_session(
            scorer, test[sid], k=args.k, slow=args.slow, sess_idx=sid
        )
        totals["n"] += out["n"]
        totals["hits"] += out["hits"]
        totals["ndcg_sum"] += sum(
            (1.0 / np.log2(r + 1.0)) if r > 0 else 0.0 for r in out["ranks"]
        )

    if len(chosen) > 1:
        print(f"\n{C.BOLD}Aggregate over {len(chosen)} sessions: "
              f"HR@{args.k} = {totals['hits']/totals['n']:.3f} | "
              f"NDCG@{args.k} = {totals['ndcg_sum']/totals['n']:.3f}"
              f"{C.END}")


if __name__ == "__main__":
    main()
