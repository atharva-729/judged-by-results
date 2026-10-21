"""Copy the mlb-fan-live data this study reuses into data/imported/.

Read-only on mlb-fan-live: files are copied, never edited, and no bytecode is
written into that repo. Authors are re-hashed with a salt from .env, because
mlb-fan-live's author_hash is an unsalted SHA-256 and could be reversed from a
list of usernames. The raw-author table itself is not copied.

Run with: py -3.11 src/jbr/imports.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT.parent / "mlb-fan-live").resolve()
OUT = ROOT / "data" / "imported"
GAME_PK = 813027

# Copied as-is from data/processed/<name>/813027.parquet.
TABLES = [
    "comment_tags",           # Jev subject + sentiment per comment
    "comment_tags_baseline",  # name-matched subject + RoBERTa sentiment
    "comment_scores",         # RoBERTa neg/neu/pos per comment
    "ticks",                  # Jev window readings (per-stream mood over time)
    "ticks_baseline",
    "volume",                 # comments per 5 s per stream
    "plays",
    "play_events",
    "win_prob",
    "games",
    "threads",
    "moments_jev",
    "moments_baseline",
]
CONFIG = ["config/game.yaml", "config/nicknames.yaml"]
STATSAPI = [
    "api_v1_1_game_813027_feed_live__0a4ccdc0a6df.json",
    "api_v1_game_813027_winProbability__be458d74a0b4.json",
    "api_v1_schedule__2ffdae0a44be.json",
    "api_v1_schedule__87e52c5652f1.json",
]


def author_salt() -> str:
    """The salt from .env, created on first run."""
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    for line in lines:
        if line.startswith("JBR_AUTHOR_SALT="):
            return line.split("=", 1)[1].strip()
    salt = secrets.token_hex(32)
    lines.append(f"JBR_AUTHOR_SALT={salt}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return salt


def hash_author(author: str | None, salt: str) -> str | None:
    if not author or author == "[deleted]":
        return None
    return hmac.new(salt.encode(), author.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def import_comments(salt: str) -> None:
    comments = pd.read_parquet(SOURCE / f"data/processed/comments/{GAME_PK}.parquet")
    raw = pd.read_parquet(SOURCE / f"data/processed/comments_raw/{GAME_PK}.parquet", columns=["comment_id", "author"])
    authors = comments["comment_id"].map(raw.set_index("comment_id")["author"])
    assert authors.notna().all(), "every cleaned comment should have a raw row"
    comments["author_hash"] = authors.map(lambda a: hash_author(a, salt))
    comments.to_parquet(OUT / "comments.parquet", index=False)


def reaction_lag() -> dict:
    """Per-stream lag, exactly as mlb-fan-live's `lag` command computes it."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE / "src"))
    from fanpulse_live.analysis import lag

    proc = SOURCE / "data/processed"
    plays = pd.read_parquet(proc / f"plays/{GAME_PK}.parquet").merge(
        pd.read_parquet(proc / f"win_prob/{GAME_PK}.parquet"), on=["game_pk", "at_bat_index"]
    )
    comments = pd.read_parquet(proc / f"comments/{GAME_PK}.parquet")
    comments = comments[comments["thread_type"] == "game"]
    weights = plays["wp_delta"].abs()
    out = {}
    for stream, group in comments.groupby("stream"):
        curve = lag.reaction_curve(group["created_utc"], plays["end_time_utc"], weights)
        out[stream] = {**lag.lag_stats(curve), "curve": {int(k): v for k, v in curve.items()}}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "statsapi").mkdir(exist_ok=True)
    import_comments(author_salt())
    for name in TABLES:
        shutil.copy2(SOURCE / f"data/processed/{name}/{GAME_PK}.parquet", OUT / f"{name}.parquet")
    for path in CONFIG:
        shutil.copy2(SOURCE / path, OUT / Path(path).name)
    for name in STATSAPI:
        shutil.copy2(SOURCE / "data/raw/statsapi.mlb.com" / name, OUT / "statsapi" / name)
    (OUT / "reaction_lag.json").write_text(json.dumps(reaction_lag(), indent=2), encoding="utf-8")

    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"{f.stat().st_size:>10,}  {f.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
