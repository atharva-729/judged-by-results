"""Pull Statcast for the game and join batted balls to MLB Stats API plays.

Statcast's at_bat_number is 1-based; the MLB feed's at_bat_index is 0-based.
The join is checked on batter, pitcher and inning, not assumed.
"""

from __future__ import annotations

import pandas as pd

from jbr import config

STATCAST_COLUMNS = [
    "game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "batter", "pitcher",
    "events", "description", "des", "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "delta_home_win_exp", "home_score", "away_score", "post_home_score", "post_away_score",
]


def pull(cfg: dict) -> pd.DataFrame:
    """Statcast rows for the game, cached in data/statcast.parquet."""
    path = config.DATA / "statcast.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from pybaseball import statcast

    date = cfg["game"]["date"]
    df = statcast(start_dt=date, end_dt=date)
    df = df[df["game_pk"] == cfg["game"]["game_pk"]][STATCAST_COLUMNS].reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def batted_balls(sc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The last pitch of each plate appearance that ended with the ball in play."""
    bb = cfg["batted_balls"]
    in_play = bb["hits"] + bb["outs"] + bb["errors"] + bb["fielders_choice"]
    final = sc[sc["events"].notna()]
    return final[final["events"].isin(in_play)].sort_values("at_bat_number").reset_index(drop=True)


def luck_category(hit: int, xba: float | None, is_error: bool, is_fc: bool, cfg: dict) -> str | None:
    if xba is None or pd.isna(xba) or is_error or is_fc:
        return None
    luck = cfg["luck"]
    if hit:
        return "lucky_hit" if xba <= luck["lucky_hit_max_xba"] else "expected_hit"
    return "unlucky_out" if xba >= luck["unlucky_out_min_xba"] else "expected_out"


def join_plays(bip: pd.DataFrame, plays: pd.DataFrame, win_prob: pd.DataFrame) -> pd.DataFrame:
    """Attach play end time, names and win probability; raise if any row disagrees."""
    plays = plays.merge(win_prob, on=["game_pk", "at_bat_index"], validate="one_to_one")
    joined = bip.assign(at_bat_index=bip["at_bat_number"] - 1).merge(
        plays, on=["game_pk", "at_bat_index"], how="left", validate="one_to_one", suffixes=("", "_mlb")
    )
    missing = joined["end_time_utc"].isna()
    if missing.any():
        raise ValueError(f"no MLB play for at_bat_number {joined.loc[missing, 'at_bat_number'].tolist()}")
    for sc_col, mlb_col in [("batter", "batter_id"), ("pitcher", "pitcher_id"), ("inning", "inning_mlb")]:
        bad = joined[sc_col] != joined[mlb_col]
        if bad.any():
            raise ValueError(f"{sc_col} disagrees at at_bat_number {joined.loc[bad, 'at_bat_number'].tolist()}")
    return joined


def build_bip(sc: pd.DataFrame, plays: pd.DataFrame, win_prob: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per batted ball in play: data/bip.parquet."""
    bb = cfg["batted_balls"]
    j = join_plays(batted_balls(sc, cfg), plays, win_prob)
    top = j["inning_topbot"] == "Top"
    home, away = cfg["game"]["home"], cfg["game"]["away"]

    out = pd.DataFrame({
        "bip_id": j["game_pk"].astype(str) + "-" + j["at_bat_number"].astype(str),
        "game_pk": j["game_pk"],
        "at_bat_number": j["at_bat_number"],
        "inning": j["inning"],
        "half": j["inning_topbot"].str.lower(),
        "batter_id": j["batter"],
        "batter": j["batter_name"],
        "batter_team": top.map({True: away, False: home}),
        "pitcher_id": j["pitcher"],
        "pitcher": j["pitcher_name"],
        "pitcher_team": top.map({True: home, False: away}),
        "event": j["events"],
        "description": j["description_mlb"],
        "hit": j["events"].isin(bb["hits"]).astype(int),
        "is_error": j["events"].isin(bb["errors"]),
        "is_fielders_choice": j["events"].isin(bb["fielders_choice"]),
        "xba": j["estimated_ba_using_speedangle"].astype(float),
        "xwoba": j["estimated_woba_using_speedangle"].astype(float),
        "launch_speed": j["launch_speed"].astype(float),
        "launch_angle": j["launch_angle"].astype(float),
        "wp_before": j["home_wp_before"].where(~top, 1 - j["home_wp_before"]),
        "wp_after": j["home_wp_after"].where(~top, 1 - j["home_wp_after"]),
        "statcast_delta_home_wp": j["delta_home_win_exp"].astype(float),
        "play_end_ts": j["end_time_utc"],
    })
    # wp_before / wp_after are for the batting team.
    out["delta_wp_batting_team"] = out["wp_after"] - out["wp_before"]
    out["luck"] = out["hit"] - out["xba"]
    out["luck_category"] = [
        luck_category(h, x, e, f, cfg)
        for h, x, e, f in zip(out["hit"], out["xba"], out["is_error"], out["is_fielders_choice"])
    ]
    return out


def run() -> pd.DataFrame:
    cfg = config.load()
    sc = pull(cfg)
    plays = pd.read_parquet(config.IMPORTED / "plays.parquet")
    win_prob = pd.read_parquet(config.IMPORTED / "win_prob.parquet")
    bip = build_bip(sc, plays, win_prob, cfg)
    bip.to_parquet(config.DATA / "bip.parquet", index=False)
    return bip
