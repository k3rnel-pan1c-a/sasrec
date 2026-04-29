"""Exploratory data analysis on the YOOCHOOSE click stream.

The raw clicks file is ~1.4 GB / 33 M rows. Strategy:

  * read in chunks of ~2 M rows so memory stays bounded
  * keep just the integer columns (session_id, item_id) and the timestamp
    *string* — full datetime parsing on 33 M rows is several minutes, but
    we only need date / hour, so we slice the ISO string instead
  * concatenate the chunk arrays at the end (~250 MB per array) and use
    ``np.unique(..., return_counts=True)`` to compute the distributions

Run with:
    python -m src.eda
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLICKS_FILE, EDA_DIR


CHUNK_SIZE = 2_000_000
COLNAMES = ["session_id", "timestamp", "item_id", "category"]


# --------------------------------------------------------------------------- #
# Streaming pass                                                              #
# --------------------------------------------------------------------------- #

def stream() -> dict:
    sessions_chunks: list[np.ndarray] = []
    items_chunks: list[np.ndarray] = []
    date_chunks: list[np.ndarray] = []      # 'YYYY-MM-DD' slice
    hour_chunks: list[np.ndarray] = []      # int 0..23
    weekday_chunks: list[np.ndarray] = []   # int 0..6
    category_counts: dict[str, int] = {}

    reader = pd.read_csv(
        CLICKS_FILE,
        names=COLNAMES,
        header=None,
        dtype={
            "session_id": np.int64,
            "item_id": np.int64,
            "timestamp": "string",
            "category": "string",
        },
        chunksize=CHUNK_SIZE,
    )

    n_rows = 0
    for i, chunk in enumerate(reader):
        n_rows += len(chunk)
        sessions_chunks.append(chunk["session_id"].to_numpy())
        items_chunks.append(chunk["item_id"].to_numpy())

        ts = chunk["timestamp"].to_numpy().astype("U24")
        # 'YYYY-MM-DDTHH:MM:SS.SSSZ' — slice without full parse
        date_arr = np.char.partition(ts, "T")[:, 0]      # YYYY-MM-DD
        hour_arr = np.array([int(s[11:13]) for s in ts], dtype=np.int8)
        date_chunks.append(date_arr)
        hour_chunks.append(hour_arr)

        # weekday needs a real datetime — do it at numpy speed in one shot
        wd = pd.to_datetime(date_arr).weekday  # pd.Index supports .weekday
        weekday_chunks.append(np.asarray(wd, dtype=np.int8))

        cc = chunk["category"].value_counts()
        for k, v in cc.items():
            category_counts[k] = category_counts.get(k, 0) + int(v)

        print(f"  chunk {i + 1}: rows so far = {n_rows:,}")

    return {
        "n_rows": n_rows,
        "sessions": np.concatenate(sessions_chunks),
        "items": np.concatenate(items_chunks),
        "dates": np.concatenate(date_chunks),
        "hours": np.concatenate(hour_chunks),
        "weekdays": np.concatenate(weekday_chunks),
        "category_counts": category_counts,
    }


# --------------------------------------------------------------------------- #
# Plot helpers                                                                #
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, name: str) -> Path:
    path = EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_session_length_hist(session_lengths: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    clipped = np.clip(session_lengths, 1, 30)
    ax.hist(clipped, bins=np.arange(1, 32) - 0.5, color="#2b6cb0",
            edgecolor="black")
    ax.set_xlabel("Session length (clicks, clipped at 30)")
    ax.set_ylabel("Number of sessions")
    ax.set_yscale("log")
    ax.set_title("Distribution of session lengths")
    return _save(fig, "session_length_hist.png")


def plot_item_popularity(item_counts: np.ndarray) -> Path:
    sorted_counts = np.sort(item_counts)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(np.arange(1, len(sorted_counts) + 1), sorted_counts,
              color="#c53030")
    ax.set_xlabel("Item rank (log)")
    ax.set_ylabel("Click count (log)")
    ax.set_title("Item popularity follows a long tail (log-log)")
    return _save(fig, "item_popularity_loglog.png")


def plot_daily_clicks(daily: dict[str, int]) -> Path:
    days = sorted(daily.keys())
    counts = [daily[d] for d in days]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.to_datetime(days), counts, color="#2f855a")
    ax.set_ylabel("Clicks per day")
    ax.set_title("Daily click volume across the YOOCHOOSE collection window")
    fig.autofmt_xdate()
    return _save(fig, "daily_clicks.png")


def plot_hourly_clicks(hourly: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(24), hourly, color="#6b46c1")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Clicks")
    ax.set_title("Click volume by hour of day")
    return _save(fig, "hourly_clicks.png")


def plot_weekday_clicks(weekday: np.ndarray) -> Path:
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, weekday, color="#dd6b20")
    ax.set_ylabel("Clicks")
    ax.set_title("Click volume by weekday")
    return _save(fig, "weekday_clicks.png")


def plot_category_pie(cats: dict[str, int], top_k: int = 8) -> Path:
    items = sorted(cats.items(), key=lambda kv: -kv[1])
    top = items[:top_k]
    other = sum(c for _, c in items[top_k:])
    labels = [str(k) for k, _ in top] + ["other"]
    sizes = [c for _, c in top] + [other]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Click breakdown by category (top {top_k} + other)")
    return _save(fig, "category_pie.png")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"[EDA] reading {CLICKS_FILE}...")
    agg = stream()

    print("[EDA] computing distributions...")
    _, session_lengths = np.unique(agg["sessions"], return_counts=True)
    _, item_counts = np.unique(agg["items"], return_counts=True)

    unique_dates, daily = np.unique(agg["dates"], return_counts=True)
    daily_dict = {str(d): int(c) for d, c in zip(unique_dates, daily)}
    hour_clicks = np.bincount(agg["hours"], minlength=24)
    weekday_clicks = np.bincount(agg["weekdays"], minlength=7)

    summary = {
        "n_clicks": int(agg["n_rows"]),
        "n_sessions": int(len(session_lengths)),
        "n_unique_items": int(len(item_counts)),
        "n_unique_categories": int(len(agg["category_counts"])),
        "time_range": {
            # np.unique returns sorted strings; min/max via indexing avoids
            # the "no ufunc loop for U-dtype" issue in newer numpy.
            "start": str(unique_dates[0]),
            "end": str(unique_dates[-1]),
        },
        "session_length": {
            "mean": float(session_lengths.mean()),
            "median": float(np.median(session_lengths)),
            "p95": float(np.percentile(session_lengths, 95)),
            "p99": float(np.percentile(session_lengths, 99)),
            "max": int(session_lengths.max()),
            "frac_len_ge_2": float((session_lengths >= 2).mean()),
            "frac_len_ge_5": float((session_lengths >= 5).mean()),
        },
        "item_popularity": {
            "mean_clicks": float(item_counts.mean()),
            "median_clicks": float(np.median(item_counts)),
            "max_clicks": int(item_counts.max()),
            "frac_items_with_lt_5_clicks": float(
                (item_counts < 5).mean()
            ),
            "top20pct_click_share": float(
                np.sort(item_counts)[::-1][
                    : max(1, int(0.2 * len(item_counts)))
                ].sum()
                / item_counts.sum()
            ),
        },
    }

    plot_paths = {
        "session_length_hist": str(plot_session_length_hist(session_lengths)),
        "item_popularity_loglog": str(plot_item_popularity(item_counts)),
        "daily_clicks": str(plot_daily_clicks(daily_dict)),
        "hourly_clicks": str(plot_hourly_clicks(hour_clicks)),
        "weekday_clicks": str(plot_weekday_clicks(weekday_clicks)),
        "category_pie": str(plot_category_pie(agg["category_counts"])),
    }
    summary["plots"] = plot_paths

    with open(EDA_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[EDA] summary:")
    print(json.dumps(summary, indent=2))
    print(f"\n[EDA] artefacts saved under {EDA_DIR}")


if __name__ == "__main__":
    main()
