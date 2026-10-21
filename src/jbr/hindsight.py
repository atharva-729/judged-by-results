"""Phase 4: do postgame "it was obvious" claims match what fans believed at the time?

For each moment m that postgame claims point at:
  claimed  - how many claims say the result was decided by m, and how sure
  belief   - share of in-game predictions in the 10 minutes after m that picked
             the eventual winner, plus Jev mood by fanbase
  actual   - the eventual winner's win probability right after m
Moments are split by that win probability: early (real uncertainty), decided, or between.
"""

from __future__ import annotations

import json

import pandas as pd

from jbr import config
from jbr.outcome import _json

TEAMS = ("LAD", "TOR")


def winner(games: pd.DataFrame) -> str:
    g = games.iloc[0]
    return g["home_abbr"] if g["home_score"] > g["away_score"] else g["away_abbr"]


def band(wp: float, cfg: dict) -> str:
    h = cfg["hindsight"]
    lo, hi = h["early_wp_band"]
    if lo <= wp <= hi:
        return "early"
    return "decided" if wp >= h["decided_wp_min"] else "between"


def belief(preds: pd.DataFrame, start: pd.Timestamp, minutes: int, win: str) -> dict:
    """Share of in-game predictions in [start, start + minutes) that picked ``win``."""
    w = preds[(preds["created_utc"] >= start) & (preds["created_utc"] < start + pd.Timedelta(minutes=minutes))]
    w = w[w["winner"].isin(TEAMS)]
    out = {"n": int(len(w)), "share_winner": float((w["winner"] == win).mean()) if len(w) else None}
    for team in TEAMS:
        t = w[w["team"] == team]
        out[f"{team}_n"] = int(len(t))
        out[f"{team}_share_winner"] = float((t["winner"] == win).mean()) if len(t) else None
    return out


def mood(ticks: pd.DataFrame, teams: dict, start: pd.Timestamp, minutes: int) -> dict:
    w = ticks[(ticks["t"] >= start) & (ticks["t"] < start + pd.Timedelta(minutes=minutes)) & ticks["mood"].notna()]
    w = w.assign(team=w["stream"].map(teams))
    return {f"{team}_mood": (float(m) if pd.notna(m := w.loc[w["team"] == team, "mood"].mean()) else None)
            for team in TEAMS}


def same_user(claims_pg: pd.DataFrame, preds: pd.DataFrame, minutes: int) -> dict:
    """Users with a postgame claim about m and an in-game prediction near m (from 5 min before to 10 after)."""
    rows = []
    for c in claims_pg.dropna(subset=["author_hash", "moment_ts"]).itertuples():
        mine = preds[(preds["author_hash"] == c.author_hash) & preds["winner"].isin(TEAMS)]
        near = mine[(mine["created_utc"] >= c.moment_ts - pd.Timedelta(minutes=5)) &
                    (mine["created_utc"] < c.moment_ts + pd.Timedelta(minutes=minutes))]
        before = mine[mine["created_utc"] < c.moment_ts + pd.Timedelta(minutes=minutes)]
        rows.append({"author": c.author_hash, "claim_winner": c.winner, "moment_play": c.moment_play,
                     "near": near.iloc[-1]["winner"] if len(near) else None,
                     "latest": before.iloc[-1]["winner"] if len(before) else None})
    df = pd.DataFrame(rows).drop_duplicates(["author", "moment_play"]) if rows else pd.DataFrame(
        columns=["author", "claim_winner", "moment_play", "near", "latest"])

    def tally(col):
        d = df[df[col].notna()]
        return {"users": int(d["author"].nunique()), "pairs": int(len(d)),
                "said_same_then": int((d[col] == d["claim_winner"]).sum()),
                "said_other_team_then": int(((d[col] != d["claim_winner"]) & d["claim_winner"].notna()).sum())}

    return {"claimants": int(df["author"].nunique()), "near_moment": tally("near"),
            "latest_prediction_before": tally("latest")}


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or config.load()
    h = cfg["hindsight"]
    mins = h["belief_window_min"]
    imp = config.IMPORTED
    games = pd.read_parquet(imp / "games.parquet")
    plays = pd.read_parquet(imp / "plays.parquet").merge(pd.read_parquet(imp / "win_prob.parquet"),
                                                         on=["game_pk", "at_bat_index"])
    ticks = pd.read_parquet(imp / "ticks.parquet")
    comments = pd.read_parquet(imp / "comments.parquet", columns=["stream", "phase"])
    claims = pd.read_parquet(config.DATA / "claims.parquet")
    from jbr.claims import stream_teams

    teams = stream_teams()
    win = winner(games)
    is_home = win == games.iloc[0]["home_abbr"]
    plays["winner_wp"] = plays["home_wp_after"] if is_home else 1 - plays["home_wp_after"]
    plays["winner_wp_before"] = plays["home_wp_before"] if is_home else 1 - plays["home_wp_before"]

    preds = claims[(claims["kind"] == "ingame") & claims["main"]].sort_values("created_utc")
    pg = claims[(claims["kind"] == "postgame") & claims["main"]]
    mapped = pg[pg["moment_play"].notna()]

    moments = []
    by_play = plays.set_index("at_bat_index")
    for play, grp in mapped.groupby("moment_play"):
        p = by_play.loc[int(play)]
        ts = p["end_time_utc"]
        moments.append({
            "at_bat_index": int(play), "inning": int(p["inning"]), "half": p["half"],
            "description": p["description"], "ts": ts, "winner_wp": float(p["winner_wp"]),
            "winner_wp_before": float(p["winner_wp_before"]), "band": band(float(p["winner_wp"]), cfg),
            "claims": {"n": int(len(grp)),
                       "share_winner": float((grp["winner"] == win).mean()),
                       "by_team": grp["team"].fillna("neutral").value_counts().to_dict(),
                       "certainty": grp["certainty_final"].value_counts().to_dict(),
                       "how": grp["moment_how"].value_counts().to_dict()},
            "belief": belief(preds, ts, mins, win), **mood(ticks, teams, ts, mins),
        })
    moments.sort(key=lambda m: m["at_bat_index"])

    split = {}
    for b in ("early", "between", "decided"):
        ms = [m for m in moments if m["band"] == b]
        n_claims = sum(m["claims"]["n"] for m in ms)
        bel_n = sum(m["belief"]["n"] for m in ms)
        bel_w = sum((m["belief"]["share_winner"] or 0) * m["belief"]["n"] for m in ms)
        claimed = sum(m["claims"]["share_winner"] * m["claims"]["n"] for m in ms)
        split[b] = {"moments": len(ms), "claims": n_claims,
                    "claimed_share_winner": claimed / n_claims if n_claims else None,
                    "belief_n": bel_n, "belief_share_winner": bel_w / bel_n if bel_n else None,
                    "mean_winner_wp": (sum(m["winner_wp"] * m["claims"]["n"] for m in ms) / n_claims) if n_claims else None}

    # Timeline: win probability per play and in-game belief in 5-minute bins.
    first, last = plays["start_time_utc"].min(), plays["end_time_utc"].max()
    bins = pd.date_range(first.floor("5min"), last.ceil("5min"), freq="5min")
    timeline_belief = []
    for t in bins:
        b = belief(preds, t, 5, win)
        timeline_belief.append({"t": t, **b})
    timeline_wp = [{"t": r.end_time_utc, "winner_wp": float(r.winner_wp), "inning": int(r.inning), "half": r.half,
                    "at_bat_index": int(r.at_bat_index)} for r in plays.sort_values("end_time_utc").itertuples()]

    # By fanbase: how common are claims among each fanbase's postgame comments?
    post = comments[comments["phase"] == "postgame"].assign(team=lambda d: d["stream"].map(teams))
    fan = {}
    for team in (*TEAMS, None):
        key = team or "neutral"
        n_post = int((post["team"] == team).sum() if team else post["team"].isna().sum())
        c = pg[pg["team"] == team] if team else pg[pg["team"].isna()]
        p_ = preds[((preds["team"] == team) if team else preds["team"].isna()) & preds["winner"].isin(TEAMS)]
        pre6 = p_[p_["created_utc"] < plays.loc[plays["inning"] == 6, "start_time_utc"].min()]
        fan[key] = {"postgame_comments": n_post, "claims": int(len(c)),
                    "claims_per_100": round(100 * len(c) / n_post, 2) if n_post else None,
                    "claims_favour_winner": int((c["winner"] == win).sum()),
                    "ingame_predictions": int(len(p_)),
                    "pre6th_predictions": int(len(pre6)),
                    "pre6th_share_winner": float((pre6["winner"] == win).mean()) if len(pre6) else None}

    out = {"winner": win, "belief_window_min": mins, "bands": h, "moments": moments, "split": split,
           "same_user": same_user(mapped, preds, mins), "fanbase": fan,
           "n_claims_main": int(len(pg)), "n_claims_mapped": int(len(mapped)),
           "n_predictions_main": int(len(preds)),
           "timeline": {"wp": timeline_wp, "belief": timeline_belief,
                        "claims": [{"t": m["ts"], "n": m["claims"]["n"], "share_winner": m["claims"]["share_winner"],
                                    "label": f"{m['inning']} {m['half']}: {m['description'][:60]}"} for m in moments]}}
    (config.REPORTS / "hindsight.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=_json),
                                                   encoding="utf-8")
    return out
