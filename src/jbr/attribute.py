"""Which windowed comments are about the batter (or pitcher), and how they feel.

Attribution, strongest first:
  jev    - Jev's subject tag is the player
  name   - name or nickname match (mlb-fan-live's baseline tags)
  reply  - a reply whose parent comment is about the player (weaker)
"""

from __future__ import annotations

import re

import pandas as pd

from jbr import config, windows

METHOD_ORDER = ["jev", "name", "reply"]


def process_pattern(phrases: list[str]) -> re.Pattern:
    alternatives = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(alternatives) + r")(?![\w-])", re.IGNORECASE)


def fanbase(stream_team: str | None, batter_team: str) -> str:
    if stream_team is None or pd.isna(stream_team):
        return "neutral"
    return "own" if stream_team == batter_team else "opponent"


def attribute(cand: pd.DataFrame, bip: pd.DataFrame, jev: pd.DataFrame, name: pd.DataFrame, role: str) -> pd.DataFrame:
    """Rows of ``cand`` about the bip's batter (role='batter') or pitcher, with the method used."""
    who = bip.set_index("bip_id")[role]
    target = cand["bip_id"].map(who)
    jev_subject = cand["comment_id"].map(jev.set_index("comment_id")["subject"])
    name_subject = cand["comment_id"].map(name.set_index("comment_id")["subject"])

    # A parent is about the player if either tag on it names the player.
    parent = cand["parent_id"].where(cand["parent_id"].str.startswith("t1_")).str[3:]
    parent_about = (parent.map(jev.set_index("comment_id")["subject"]) == target) | (
        parent.map(name.set_index("comment_id")["subject"]) == target
    )

    method = pd.Series(None, index=cand.index, dtype=object)
    method[parent_about] = "reply"
    method[name_subject == target] = "name"
    method[jev_subject == target] = "jev"
    keep = method.notna()
    out = cand[keep].copy()
    out["attribution_method"] = method[keep]
    out["name_match"] = (name_subject == target)[keep]
    out["role"] = role
    return out


def build(cfg: dict | None = None) -> pd.DataFrame:
    """data/reactions.parquet: one row per (bip, comment, role)."""
    cfg = cfg or config.load()
    imp = config.IMPORTED
    bip = pd.read_parquet(config.DATA / "bip.parquet")
    comments = pd.read_parquet(imp / "comments.parquet")
    plays = pd.read_parquet(imp / "plays.parquet")
    jev = pd.read_parquet(imp / "comment_tags.parquet")
    name = pd.read_parquet(imp / "comment_tags_baseline.parquet")
    roberta = pd.read_parquet(imp / "comment_scores.parquet").set_index("comment_id")["sentiment"]
    streams = {s["id"]: s["team"] for s in _game_yaml()["streams"]}

    cand = windows.candidates(bip, comments, plays, cfg)
    react = pd.concat([attribute(cand, bip, jev, name, r) for r in ("batter", "pitcher")],
                      ignore_index=True)

    team_col = react["role"].map({"batter": "batter_team", "pitcher": "pitcher_team"})
    teams = bip.set_index("bip_id")
    player_team = [teams.at[b, c] for b, c in zip(react["bip_id"], team_col)]
    react["fanbase"] = [fanbase(streams.get(s), t) for s, t in zip(react["stream"], player_team)]
    react["sentiment_jev"] = react["comment_id"].map(jev.set_index("comment_id")["sentiment"])
    react["sentiment_roberta"] = react["comment_id"].map(roberta)
    react["mentions_process"] = react["body"].str.contains(process_pattern(cfg["process_phrases"]))

    cols = ["bip_id", "role", "comment_id", "author_hash", "stream", "fanbase", "created_utc", "offset_s",
            "next_gap_s", "in_main", "in_nocut", *[f"in_{n}s" for n in cfg["window"]["robustness_lengths_s"]],
            "attribution_method", "name_match", "sentiment_jev", "sentiment_roberta", "mentions_process", "body"]
    react = react[cols].sort_values(["bip_id", "role", "created_utc"]).reset_index(drop=True)
    react.to_parquet(config.DATA / "reactions.parquet", index=False)
    return react


def _game_yaml() -> dict:
    import yaml

    return yaml.safe_load((config.IMPORTED / "game.yaml").read_text(encoding="utf-8"))
