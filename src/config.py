"""Central configuration for the in-session recommendation project.

All paths and hyperparameters live here so that scripts stay self-contained
and the user can tune behaviour without touching multiple files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path("/kaggle/working")
DATA_ROOT = Path(
    "/kaggle/input/datasets/chadgostopp/recsys-challenge-2015"
)

CLICKS_FILE = DATA_ROOT / "yoochoose-clicks.dat"
BUYS_FILE = DATA_ROOT / "yoochoose-buys.dat"

OUT_ROOT = PROJECT_ROOT / "outputs"
EDA_DIR = OUT_ROOT / "eda"
PROCESSED_DIR = OUT_ROOT / "processed"
CKPT_DIR = OUT_ROOT / "checkpoints"
LOG_DIR = OUT_ROOT / "logs"

for _p in (EDA_DIR, PROCESSED_DIR, CKPT_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data preprocessing                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class DataConfig:
    # YOOCHOOSE has 33M clicks. The original SASRec / GRU4Rec papers train on
    # only the most-recent 1/64 fraction; that is also what we adopt to keep
    # one epoch under a few minutes on a Kaggle T4.
    recent_fraction: float = 1 / 64

    # Sessions shorter than this provide no supervision signal.
    min_session_length: int = 2

    # Cold-start items hurt training and inflate the vocabulary. We drop any
    # item that appears fewer than this many times in the (filtered) data.
    min_item_support: int = 5

    # Truncate / left-pad to this many clicks per session.
    max_seq_len: int = 50

    # Chronological split by *quantile of session-start timestamp*. The last
    # ``test_quantile`` of session starts goes to test; the previous
    # ``valid_quantile`` goes to validation; the rest is training. We
    # deliberately avoid splitting by calendar day because the 1/64 slice of
    # YOOCHOOSE spans only ~2-3 days and the file ends mid-day, which would
    # leave the "last day" with almost no sessions.
    test_quantile: float = 0.05
    valid_quantile: float = 0.05


# --------------------------------------------------------------------------- #
# Model hyperparameters                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    hidden_dim: int = 64           # d in the SASRec paper
    n_blocks: int = 2              # stacked self-attention blocks
    n_heads: int = 1               # SASRec uses single-head on small d
    dropout: float = 0.2
    max_seq_len: int = 50          # must match DataConfig.max_seq_len


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    batch_size: int = 256
    # SASRec on YOOCHOOSE-1/64 wants long training: HR@10 climbs slowly past
    # epoch ~30 and plateaus around 100-150. With ~5s/epoch on 2 T4s we can
    # afford the full schedule.
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-2     # AdamW
    grad_clip: float = 5.0

    # Loss mode:
    #   "ce"  – cross-entropy over the full item vocabulary at every
    #           position. Equivalent to having (n_items - 1) negatives per
    #           position. This is the modern SASRec recipe (Petrov &
    #           Macdonald 2022) and consistently beats co-occurrence
    #           baselines on YOOCHOOSE.
    #   "bce" – the original Kang & McAuley 2018 BCE-with-one-negative.
    #           Kept available for the proposal-faithful ablation.
    loss_type: str = "ce"

    # Only used when loss_type == "bce".
    n_negatives: int = 1

    # Evaluate every N epochs on the validation split.
    eval_every: int = 1

    # Early stopping patience (in eval rounds). Loose enough to ride out
    # plateau-bouncing in mid-training.
    patience: int = 20

    # Reproducibility.
    seed: int = 42

    device: str = "cuda"           # falls back to cpu if unavailable
    num_workers: int = 2

    # Top-k for HR / NDCG.
    top_k: int = 10


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
