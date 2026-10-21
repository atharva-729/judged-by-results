# Imported data

Everything in `data/imported/` comes from the sister repo `mlb-fan-live` (game_pk **813027**, 2025 World Series Game 1, LAD @ TOR). It was copied on 2026-10-21 by `py -3.11 src/jbr/imports.py`. The copy is read-only: `git status` in `mlb-fan-live` was clean before and after.

Source root: `../mlb-fan-live/`. Every parquet comes from `data/processed/<name>/813027.parquet` unless a different source is listed.

| File here | Source | Rows | What it is |
|---|---|---|---|
| `comments.parquet` | `comments/` + `comments_raw/` (author only) | 52,872 | Cleaned comments: thread, stream (LAD / TOR / neutral, flair-split in r/baseball), phase (pregame / in-game / postgame), `created_utc`, `parent_id`, body. **`author_hash` re-made here** (see below). |
| `comment_tags.parquet` | `comment_tags/` | 40,354 | Jev tags per comment: `subject` (player name, team, umpire, …), `sentiment` (−1…+1) and their confidences |
| `comment_tags_baseline.parquet` | `comment_tags_baseline/` | 45,572 | Baseline tags: subject from name and nickname matching, sentiment from RoBERTa |
| `comment_scores.parquet` | `comment_scores/` | 52,872 | RoBERTa `neg / neu / pos` and `sentiment` for every comment |
| `ticks.parquet` | `ticks/` | 12,175 | Jev window readings: per-stream mood every 5 s |
| `ticks_baseline.parquet` | `ticks_baseline/` | 12,175 | Same readings from the baseline |
| `volume.parquet` | `volume/` | 12,175 | Comments per 5 s per stream |
| `plays.parquet` | `plays/` | 78 | MLB Stats API plays: `at_bat_index`, inning, half, start/end time, event, batter, pitcher, score |
| `play_events.parquet` | `play_events/` | 375 | Per-pitch and other events with start/end times |
| `win_prob.parquet` | `win_prob/` | 78 | Home win probability before and after each play |
| `games.parquet` | `games/` | 1 | Teams, final score, start time |
| `threads.parquet` | `threads/` | 6 | The six Reddit threads (3 game, 3 postgame) |
| `moments_jev.parquet`, `moments_baseline.parquet` | `moments_jev/`, `moments_baseline/` | 12 each | Detected moments linked to plays |
| `reaction_lag.json` | computed (see below) | 5 streams | Measured reaction lag per stream |
| `game.yaml`, `nicknames.yaml` | `config/` | | Thread IDs, stream and flair rules; nickname → MLB person ID map |
| `statsapi/*.json` | `data/raw/statsapi.mlb.com/` | | Raw MLB `feed/live` (plays, per-pitch times), `winProbability`, and the two schedule responses used to look up `game_pk` |

## Author hashing

`mlb-fan-live` stores `author_hash` as an unsalted `sha256(username)[:12]`, which could be reversed from a list of usernames. Here it is replaced with `HMAC-SHA256(salt, username)[:16]`. The salt is `JBR_AUTHOR_SALT` in `.env`, which is gitignored. Raw usernames were read from `comments_raw` only to compute the new hash and were not copied. Coverage is the same: 52,872 of 52,872 comments are hashed, with 9,579 unique authors, as in the source. None of the new hashes match an old one.

## Reaction lag

`mlb-fan-live` never saved the lag as data; it only printed it and drew `reports/813027_reaction_lag.html`. `reaction_lag.json` reruns that repo's own `fanpulse_live.analysis.lag` functions with the same inputs as its `lag` command: game-thread comments, 78 plays, each play weighted by |Δ win probability|. Each stream entry has `baseline`, `peak_lag_s`, `tail_lag_s` (the point by which 90% of the excess reaction has arrived), `peak_x_baseline` and the full 5-second `curve` (offsets −60…175 s).

| Stream | Peak | 90% arrived by |
|---|---|---|
| r/Dodgers | 85 s | 155 s |
| r/Torontobluejays | 50 s | 145 s |
| r/baseball:LAD-flair | 110 s | 160 s |
| r/baseball:TOR-flair | 40 s | 135 s |
| r/baseball:neutral | 70 s | 150 s |

## Not imported

- `comments_raw`: it has raw usernames.
- Raw Reddit pulls (`data/raw/arctic-shift…`, `api.pullpush.io`) and the Jev response cache (`data/raw/jev`, 79 MB): everything needed is already in the processed tables.
- Per-player season stats (`api_v1_people_*`), hand labels (`data/labels`) and the moment summaries: none of the phases use them.
- `mlb-fan-pulse`: it covers a different game (813074, 2025 AL Wild Card Game 1, Red Sox @ Yankees), so nothing from it applies here.
