"""Build the train/valid/test splits used by every model in this project.

Pipeline (single pass over the click log, then in-memory work):

  1. Keep the last ``recent_fraction`` of the time window — standard practice
     for YOOCHOOSE since the original GRU4Rec / SASRec papers (the older
     half is stale and dominates run time).

  2. Iteratively prune sessions shorter than ``min_session_length`` and items
     with fewer than ``min_item_support`` clicks. The two filters interact,
     so we loop until the data is stable.

  3. Reindex item IDs into a contiguous ``[1, N]`` range. ``0`` is reserved
     for the padding token. The mapping is saved so we can decode predictions
     later.

  4. Split sessions chronologically by their *start day*: the very last day
     becomes the test set, the day before that becomes validation, the rest
     is training. Test/valid items not seen in train are dropped (cold-start
     items have no learned embedding).

  5. Sort each session by timestamp and serialise the resulting list-of-lists
     to ``outputs/processed/`` as pickled dicts. Downstream code never has to
     touch the raw log again.

Run with:
    python -m src.preprocess
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLICKS_FILE, PROCESSED_DIR, CONFIG


COLNAMES = ["session_id", "timestamp", "item_id", "category"]
CHUNK_SIZE = 2_000_000


# --------------------------------------------------------------------------- #
# Step 1 — load (recent fraction only)                                        #
# --------------------------------------------------------------------------- #

def load_recent(recent_fraction: float) -> pd.DataFrame:
    """Read the click log in chunks and keep only the most-recent slice.

    We compute the cutoff timestamp via a fast first pass over only the
    timestamp column, then re-read keeping rows past the cutoff. This is far
    cheaper than loading 1.4 GB twice.
    """
    print(f"[preprocess] step 1 — find timestamp cutoff for the most "
          f"recent {recent_fraction:.4f} of clicks")

    # First pass: gather every timestamp (as int seconds) so we can quantile.
    ts_chunks = []
    reader = pd.read_csv(
        CLICKS_FILE,
        names=COLNAMES,
        header=None,
        usecols=["timestamp"],
        dtype={"timestamp": "string"},
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        # YYYY-MM-DDTHH:MM:SS.SSSZ — view as datetime then to int seconds
        ts = pd.to_datetime(chunk["timestamp"], format="ISO8601",
                            utc=True).astype("int64") // 10**9
        ts_chunks.append(ts.to_numpy())
    ts_all = np.concatenate(ts_chunks)
    cutoff_int = int(np.quantile(ts_all, 1.0 - recent_fraction))
    cutoff = pd.Timestamp(cutoff_int, unit="s", tz="UTC")
    print(f"[preprocess]   cutoff = {cutoff} "
          f"(keeping clicks at or after this time)")
    del ts_all, ts_chunks

    # Second pass: actually load the rows we need.
    print("[preprocess] step 1b — read clicks past the cutoff")
    rows = []
    reader = pd.read_csv(
        CLICKS_FILE,
        names=COLNAMES,
        header=None,
        usecols=["session_id", "timestamp", "item_id"],
        dtype={"session_id": np.int64, "item_id": np.int64,
               "timestamp": "string"},
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        ts = pd.to_datetime(chunk["timestamp"], format="ISO8601", utc=True)
        keep = ts >= cutoff
        if keep.any():
            sub = chunk.loc[keep].copy()
            sub["timestamp"] = ts[keep].astype("int64") // 10**9
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    print(f"[preprocess]   loaded {len(df):,} rows")
    return df


# --------------------------------------------------------------------------- #
# Step 2 — iterative filtering                                                #
# --------------------------------------------------------------------------- #

def filter_sessions_and_items(
    df: pd.DataFrame,
    min_session_len: int,
    min_item_support: int,
) -> pd.DataFrame:
    print(f"[preprocess] step 2 — filter (min_session_len="
          f"{min_session_len}, min_item_support={min_item_support})")
    while True:
        before = len(df)
        item_support = df.groupby("item_id")["session_id"].count()
        good_items = item_support[item_support >= min_item_support].index
        df = df[df["item_id"].isin(good_items)]

        sess_len = df.groupby("session_id")["item_id"].count()
        good_sess = sess_len[sess_len >= min_session_len].index
        df = df[df["session_id"].isin(good_sess)]
        after = len(df)
        print(f"[preprocess]   {before:,} -> {after:,} rows")
        if before == after:
            break
    print(f"[preprocess]   final: {len(df):,} rows, "
          f"{df['session_id'].nunique():,} sessions, "
          f"{df['item_id'].nunique():,} items")
    return df


# --------------------------------------------------------------------------- #
# Step 3 — item reindexing                                                    #
# --------------------------------------------------------------------------- #

def reindex_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map the original item IDs into a contiguous [1, N] range.

    ``0`` stays free for the padding token used by the SASRec embedding.
    """
    print("[preprocess] step 3 — reindex items into a contiguous range")
    unique_items = np.sort(df["item_id"].unique())
    # offset by 1 so that 0 is reserved for padding
    item_to_idx = {int(orig): idx + 1 for idx, orig in enumerate(unique_items)}
    df = df.assign(item_idx=df["item_id"].map(item_to_idx).astype(np.int64))
    return df, item_to_idx


# --------------------------------------------------------------------------- #
# Step 4 — chronological split by session start day                           #
# --------------------------------------------------------------------------- #

def chronological_split(
    df: pd.DataFrame, test_quantile: float, valid_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by quantile of *session start* timestamps.

    Last ``test_quantile`` fraction of session starts -> test,
    previous ``valid_quantile`` fraction -> valid, rest -> train.
    Splits are by *session*: every click in a session lands in the same
    split, so a session that begins in train and continues into test still
    goes entirely to train.
    """
    print(f"[preprocess] step 4 — split by session-start quantile "
          f"(test={test_quantile:.2%}, valid={valid_quantile:.2%})")
    sess_start = df.groupby("session_id")["timestamp"].min()
    starts = sess_start.to_numpy()

    test_cut = np.quantile(starts, 1.0 - test_quantile)
    valid_cut = np.quantile(starts, 1.0 - test_quantile - valid_quantile)

    def _label(t: int) -> str:
        if t >= test_cut:
            return "test"
        if t >= valid_cut:
            return "valid"
        return "train"

    sess_split = sess_start.map(_label)
    sess_split.name = "split"

    df = df.merge(sess_split, left_on="session_id", right_index=True)
    train_df = df[df["split"] == "train"].drop(columns="split")
    valid_df = df[df["split"] == "valid"].drop(columns="split")
    test_df = df[df["split"] == "test"].drop(columns="split")
    print(f"[preprocess]   train: {len(train_df):,} rows / "
          f"{train_df['session_id'].nunique():,} sess, "
          f"valid: {len(valid_df):,} rows / "
          f"{valid_df['session_id'].nunique():,} sess, "
          f"test: {len(test_df):,} rows / "
          f"{test_df['session_id'].nunique():,} sess")
    return train_df, valid_df, test_df


def restrict_to_train_items(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_session_len: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop test/valid clicks on items not seen in train, then drop sessions
    that became too short."""
    train_items = set(train_df["item_idx"].unique().tolist())

    def _trim(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["item_idx"].isin(train_items)]
        sess_len = df.groupby("session_id")["item_idx"].count()
        good = sess_len[sess_len >= min_session_len].index
        return df[df["session_id"].isin(good)]

    return _trim(valid_df), _trim(test_df)


# --------------------------------------------------------------------------- #
# Step 5 — serialise as list-of-lists                                         #
# --------------------------------------------------------------------------- #

def to_session_lists(df: pd.DataFrame) -> dict[int, list[int]]:
    df = df.sort_values(["session_id", "timestamp"])
    return {
        int(sid): grp["item_idx"].tolist()
        for sid, grp in df.groupby("session_id", sort=False)
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    cfg = CONFIG.data
    t0 = time.time()

    df = load_recent(cfg.recent_fraction)
    df = filter_sessions_and_items(
        df, cfg.min_session_length, cfg.min_item_support
    )
    df, item_to_idx = reindex_items(df)

    train_df, valid_df, test_df = chronological_split(
        df, cfg.test_quantile, cfg.valid_quantile
    )
    valid_df, test_df = restrict_to_train_items(
        train_df, valid_df, test_df, cfg.min_session_length
    )

    train = to_session_lists(train_df)
    valid = to_session_lists(valid_df)
    test = to_session_lists(test_df)

    n_items = max(item_to_idx.values())   # ids are 1..N
    meta = {
        "n_items": int(n_items),
        "n_sessions": {
            "train": len(train), "valid": len(valid), "test": len(test),
        },
        "n_clicks": {
            "train": int(sum(len(s) for s in train.values())),
            "valid": int(sum(len(s) for s in valid.values())),
            "test": int(sum(len(s) for s in test.values())),
        },
        "config": {
            "recent_fraction": cfg.recent_fraction,
            "min_session_length": cfg.min_session_length,
            "min_item_support": cfg.min_item_support,
            "test_quantile": cfg.test_quantile,
            "valid_quantile": cfg.valid_quantile,
            "max_seq_len": cfg.max_seq_len,
        },
    }

    print("[preprocess] saving artefacts...")
    with open(PROCESSED_DIR / "train.pkl", "wb") as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROCESSED_DIR / "valid.pkl", "wb") as f:
        pickle.dump(valid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROCESSED_DIR / "test.pkl", "wb") as f:
        pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROCESSED_DIR / "item_to_idx.pkl", "wb") as f:
        pickle.dump(item_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[preprocess] done in {time.time() - t0:.1f}s")
    print(json.dumps(meta, indent=2))
    print(f"\n[preprocess] artefacts saved under {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
