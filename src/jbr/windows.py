"""Reaction windows: which comments fall after each batted ball, per stream.

Main window per stream: [play end + start_s, play end + that stream's 90%-arrived
lag], cut at the next play's end + next_play_cut_s. Each row keeps its offset so
robustness windows are just filters on the same table.
"""

from __future__ import annotations

import json

import pandas as pd

from jbr import config

WIDEST_S = 200  # covers the 180 s robustness window from start_s = 15


def stream_tail_lags() -> dict[str, float]:
    lags = json.loads((config.IMPORTED / "reaction_lag.json").read_text(encoding="utf-8"))
    return {stream: v["tail_lag_s"] for stream, v in lags.items()}


def next_play_gaps(bip: pd.DataFrame, plays: pd.DataFrame) -> pd.Series:
    """Seconds from each batted ball's play end to the next play's end (NaN for the last)."""
    ends = plays.sort_values("end_time_utc")["end_time_utc"].reset_index(drop=True)
    idx = ends.searchsorted(bip["play_end_ts"], side="right")
    nxt = pd.Series([ends[i] if i < len(ends) else pd.NaT for i in idx], index=bip.index)
    return (nxt - bip["play_end_ts"]).dt.total_seconds()


def candidates(bip: pd.DataFrame, comments: pd.DataFrame, plays: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Every (batted ball, comment) pair inside the widest window, with window flags."""
    w = cfg["window"]
    tails = stream_tail_lags()
    gaps = next_play_gaps(bip, plays)
    game = comments[comments["thread_type"] == "game"]
    rows = []
    for i, b in bip.iterrows():
        offset = (game["created_utc"] - b["play_end_ts"]).dt.total_seconds()
        inside = game[(offset >= w["start_s"]) & (offset < w["start_s"] + WIDEST_S)]
        rows.append(inside.assign(bip_id=b["bip_id"], offset_s=offset[inside.index], next_gap_s=gaps[i]))
    out = pd.concat(rows, ignore_index=True)
    return flag_windows(out, tails, cfg)


def flag_windows(df: pd.DataFrame, tails: dict[str, float], cfg: dict) -> pd.DataFrame:
    w = cfg["window"]
    start = w["start_s"]
    cut = (df["next_gap_s"] + w["next_play_cut_s"]).fillna(float("inf"))
    tail = df["stream"].map(tails)
    df = df.copy()
    df["in_main"] = (df["offset_s"] >= start) & (df["offset_s"] < tail) & (df["offset_s"] < cut)
    df["in_nocut"] = (df["offset_s"] >= start) & (df["offset_s"] < tail)
    for length in w["robustness_lengths_s"]:
        df[f"in_{length}s"] = (df["offset_s"] >= start) & (df["offset_s"] < start + length) & (df["offset_s"] < cut)
    return df
