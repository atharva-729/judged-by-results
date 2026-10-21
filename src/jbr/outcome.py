"""Phase 2: does the result (hit or out) move fan sentiment more than contact quality (xBA)?

Comment-level OLS, own fans as the reference fanbase:
  sentiment ~ hit + xba + |dWP| + fanbase + inning_bucket + hit x fanbase + xba x fanbase
CIs come from resampling whole batted balls. Outcome bias is the hit effect at fixed
xBA; process sensitivity is the xBA effect (0 -> 1) at a fixed result. Both are on
the same 0-1 scale, so their ratio says how much more the result counts.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from jbr import config, stats

TERMS = ["hit", "xba", "abs_dwp", "opp", "neu", "inn_mid", "inn_late",
         "hit:opp", "hit:neu", "xba:opp", "xba:neu"]
CATEGORIES = ["unlucky_out", "expected_out", "lucky_hit", "expected_hit"]
FANBASES = ["own", "opponent", "neutral"]


def prepare(react: pd.DataFrame, bip: pd.DataFrame, role: str, window: str = "in_main",
            sentiment: str = "sentiment_jev") -> pd.DataFrame:
    df = react[(react["role"] == role) & react[window]].merge(bip, on="bip_id", how="inner")
    df = df[df["xba"].notna() & df[sentiment].notna()].copy()
    df["y"] = df[sentiment]
    df["abs_dwp"] = df["delta_wp_batting_team"].abs()
    df["opp"] = (df["fanbase"] == "opponent").astype(float)
    df["neu"] = (df["fanbase"] == "neutral").astype(float)
    df["inn_mid"] = df["inning"].between(4, 6).astype(float)
    df["inn_late"] = (df["inning"] >= 7).astype(float)
    return df


def drop_thin(df: pd.DataFrame, min_n: int) -> tuple[pd.DataFrame, list[str]]:
    n = df.groupby("bip_id").size()
    thin = n[n < min_n].index.tolist()
    return df[~df["bip_id"].isin(thin)], thin


def effects(df: pd.DataFrame) -> pd.Series:
    """Hit and xBA effects for each fanbase, from one fit."""
    b = stats.ols(df, "y", TERMS)
    out = {"own_hit": b["hit"], "own_xba": b["xba"],
           "opponent_hit": b["hit"] + b["hit:opp"], "opponent_xba": b["xba"] + b["xba:opp"],
           "neutral_hit": b["hit"] + b["hit:neu"], "neutral_xba": b["xba"] + b["xba:neu"],
           "abs_dwp": b["abs_dwp"]}
    out["own_ratio"] = out["own_hit"] / out["own_xba"] if out["own_xba"] != 0 else np.nan
    return pd.Series(out)


def model(df: pd.DataFrame, n_boot: int = stats.N_BOOT) -> dict:
    point = effects(df)
    draws = stats.cluster_bootstrap(df, "bip_id", effects, n_boot)
    res = {}
    for k, v in point.items():
        lo, hi = stats.ci(draws[k])
        res[k] = {"est": float(v), "lo": lo, "hi": hi}
    res["n_comments"] = int(len(df))
    res["n_bip"] = int(df["bip_id"].nunique())
    res["share_ratio_draws_positive_xba"] = float((draws["own_xba"] > 0).mean())
    return res


def per_ball_model(df: pd.DataFrame, n_boot: int = stats.N_BOOT) -> dict:
    """Same question on own-fan per-ball means, each ball counted once."""
    own = df[df["fanbase"] == "own"]
    balls = own.groupby("bip_id").agg(y=("y", "mean"), hit=("hit", "first"), xba=("xba", "first"),
                                      abs_dwp=("abs_dwp", "first"), inn_mid=("inn_mid", "first"),
                                      inn_late=("inn_late", "first")).reset_index()
    terms = ["hit", "xba", "abs_dwp", "inn_mid", "inn_late"]
    fit = lambda d: stats.ols(d, "y", terms)[["hit", "xba"]]
    point = fit(balls)
    draws = stats.cluster_bootstrap(balls, "bip_id", fit, n_boot)
    return {k: {"est": float(point[k]), "lo": stats.ci(draws[k])[0], "hi": stats.ci(draws[k])[1]}
            for k in ("hit", "xba")} | {"n_bip": int(len(balls))}


def contrast(df: pd.DataFrame) -> dict:
    """Mean sentiment by luck category and fanbase."""
    out = {}
    for fb in FANBASES:
        sub = df[df["fanbase"] == fb]
        out[fb] = {c: stats.mean_ci(sub[sub["luck_category"] == c], "y", "bip_id", 1000) for c in CATEGORIES}
    return out


def luck_awareness(df: pd.DataFrame) -> dict:
    out = {}
    for fb in [*FANBASES, "all"]:
        sub = df if fb == "all" else df[df["fanbase"] == fb]
        out[fb] = {c: {"n": int((s := sub[sub["luck_category"] == c]).shape[0]),
                       "n_process": int(s["mentions_process"].sum()),
                       "share": float(s["mentions_process"].mean()) if len(s) else None}
                   for c in CATEGORIES}
    return out


USER_RE = re.compile(r"/?u/[A-Za-z0-9_-]+")
EMBED_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")   # markdown links and ![gif](...) embeds
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def excerpt(text: str, n: int = 90) -> str:
    """A short trimmed excerpt with usernames, links and embeds removed."""
    text = EMBED_RE.sub("[image]", str(text))
    text = URL_RE.sub("[link]", text)
    text = USER_RE.sub("[user]", " ".join(text.split()))
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def ball_table(df: pd.DataFrame, bip: pd.DataFrame, thin: list[str]) -> list[dict]:
    rows = []
    for _, b in bip.iterrows():
        sub = df[df["bip_id"] == b["bip_id"]]
        own = sub[sub["fanbase"] == "own"].sort_values("y")
        row = {k: (None if pd.isna(b[k]) else b[k]) for k in
               ["bip_id", "inning", "half", "batter", "batter_team", "pitcher", "event", "luck_category"]}
        row |= {k: (None if pd.isna(b[k]) else round(float(b[k]), 3)) for k in
                ["xba", "launch_speed", "launch_angle", "delta_wp_batting_team"]}
        row["description"] = b["description"]
        row["thin"] = b["bip_id"] in thin
        row["n"] = {fb: int((sub["fanbase"] == fb).sum()) for fb in FANBASES}
        row["mean"] = {fb: (round(float(s["y"].mean()), 3) if len(s := sub[sub["fanbase"] == fb]) else None)
                       for fb in FANBASES}
        row["process_share_own"] = round(float(own["mentions_process"].mean()), 3) if len(own) else None
        row["excerpts"] = ([excerpt(own.iloc[-1]["body"]), excerpt(own.iloc[0]["body"])] if len(own) >= 2 else [])
        rows.append(row)
    return rows


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or config.load()
    min_n = cfg["attribution"]["min_comments_per_bip"]
    react = pd.read_parquet(config.DATA / "reactions.parquet")
    bip = pd.read_parquet(config.DATA / "bip.parquet")

    main, thin = drop_thin(prepare(react, bip, "batter"), min_n)
    results = {"n_bip_total": int(len(bip)), "n_bip_with_xba": int(bip["xba"].notna().sum()),
               "thin_bip": thin, "main": model(main), "per_ball": per_ball_model(main),
               "contrast": contrast(main), "luck_awareness": luck_awareness(main)}

    # Pitcher side: fanbase is relative to the pitcher's team.
    pitch, pthin = drop_thin(prepare(react, bip, "pitcher"), min_n)
    results["pitcher"] = model(pitch, 1000) | {"thin_bip": pthin}

    # Robustness: does the own-fan sign hold?
    biggest = bip.loc[bip["delta_wp_batting_team"].abs().idxmax(), "bip_id"]
    variants = {
        "RoBERTa sentiment": drop_thin(prepare(react, bip, "batter", sentiment="sentiment_roberta"), min_n)[0],
        "60 s window": drop_thin(prepare(react, bip, "batter", window="in_60s"), min_n)[0],
        "180 s window": drop_thin(prepare(react, bip, "batter", window="in_180s"), min_n)[0],
        "No next-play cut": drop_thin(prepare(react, bip, "batter", window="in_nocut"), min_n)[0],
        "Name-match attribution only": drop_thin(prepare(react[react["name_match"]], bip, "batter"), min_n)[0],
        "Without the biggest WP swing": main[main["bip_id"] != biggest],
        "Without home runs": main[main["event"] != "home_run"],
    }
    rob = {"Main (Jev, main window)": {k: results["main"][k] for k in ("own_hit", "own_xba", "n_comments", "n_bip")}}
    for name, d in variants.items():
        m = model(d, 1000)
        rob[name] = {k: m[k] for k in ("own_hit", "own_xba", "n_comments", "n_bip")}
    pb = results["per_ball"]
    rob["Per-ball means (each ball once)"] = {"own_hit": pb["hit"], "own_xba": pb["xba"],
                                               "n_comments": None, "n_bip": pb["n_bip"]}
    results["robustness"] = rob
    results["biggest_wp_bip"] = biggest
    results["balls"] = ball_table(main, bip, thin)

    path = config.REPORTS / "outcome.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")
    return results


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(type(o))
