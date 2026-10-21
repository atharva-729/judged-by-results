# Judged by Results

**Do fans judge the swing or where the ball landed? And do they rewrite what they "always knew" after the game?**

A case study on one game, 2025 World Series Game 1 (Dodgers 4 @ Blue Jays 11, 24 Oct 2025), covering two classic biases:

1. **Outcome bias ("moral luck").** We judge a decision or action by its result, not by its quality. Statcast tells us how well every ball was actually hit (expected batting average, **xBA**), so we can separate **process** (contact quality) from **outcome** (hit or out) for every batted ball. Then we measure how fans reacted to the batter.
   - *Good process, bad outcome:* a 108 mph liner caught for an out.
   - *Bad process, good outcome:* a 62 mph bloop that drops for a hit.
2. **Hindsight bias.** After the game, fans say "it was over after the 6th" or "we never had a chance". We check what they **actually** said at that moment in the in-game thread, and what the win probability really was.

This is **not** fan sentiment tracking (which teams already do). Sentiment is only the measuring tool. The question is about a cognitive bias: *how far fan judgment follows luck instead of quality, and how far fan memory rewrites itself after the result.*

Context: an innovation-team study. It reuses the data already collected for `mlb-fan-live`, so **no new Reddit collection** (no Arctic Shift).

---

## Status (2026-10-21)

All phases except the optional Phase 5 have been run end to end. Results page: `docs/index.html`. Every change from this plan is in `reports/decisions_log.md`. The main ones:
- **Luck thresholds** are 0.40 / 0.30, not 0.50 / 0.25 (too few unlucky outs otherwise).
- **Phase 3 ran local-only** (keyword filter + local NLI). LLM Foundry isn't reachable from the laptop this was built on. Classifier A is built and tested but not run; see below.
- **The reaction window** is [play end + 15 s, the stream's 90%-arrived lag], cut at the next play's end + 15 s.

**Waiting on Atharva:** the spot-check. Open `reports/spotcheck.html`, answer the 50 yes/no items (about 15 minutes), press "Download answers", put `spotcheck_labels.json` in `reports/`, then run `jbr site`. Until then the page says "spot-check pending".

**Setup:** `py -3.11 -m pip install -e ".[dev]"`, then run steps with `jbr <step>` (or `py -3.11 -m jbr.cli <step>`). Steps in order: `import`, `statcast`, `windows`, `outcome`, `claims`, `hindsight`, `spotcheck`, `site`. Tests: `py -3.11 -m pytest`.

### Running with the LLM later

Classifier A (`src/jbr/llm.py`, prompts in `prompts/`) is wired in and tested against a fake API response. Every call is cached in `runs/llm_cache/`, and spend is tracked in `runs/spend.json` with a hard stop at `llm.daily_cap_usd` ($0.80).

1. Add `LLMFOUNDRY_TOKEN=...` to `.env`.
2. Check `llm.base_url`, `llm.model` and `llm.price_per_mtok` in `config.yaml`. They are set to LLM Foundry's OpenAI-compatible endpoint and a cheap model, but haven't been checked against the live service.
3. Run `jbr claims --with-llm`. About 1,450 filtered comments in batches of 20 is roughly 75 calls, well under $0.10 at the configured price. The main set then becomes the items where A and B agree, and A's `referenced_moment` / `referenced_inning` feed the moment mapping.
4. Run `jbr hindsight`, then `jbr spotcheck`, then `jbr site`. The spot-check sample is re-drawn from the new main set, so redo the Phase 3 half of the spot-check. The page's classifier table then shows A, B and agreement counts.

If the daily cap is reached, the run stops with `SpendCapReached`. Cached calls are kept, so rerunning the next day continues where it stopped.

## How to use this README (for Claude Code)

- Work **one phase at a time**. At each **CHECKPOINT**, stop, show what it asks for, and wait for a go-ahead.
- Don't start later phases early. Don't add dependencies that aren't listed here without asking.
- Keep it simple: Python, pandas/DuckDB, Parquet files, a small CLI, plots with plotly. No databases, no agent frameworks.
- If a sanity check fails, stop and report. Don't work around it silently.
- This is a **case study of one game**. Never write conclusions as if they generalise to all fans. Always say "in this game".

## Ground rules

**Do**
- Reuse the existing `mlb-fan-live` data (comments, Jev tags, reaction lag, play-by-play). Copy it into this repo's `data/imported/` and record the exact source paths in `data/IMPORT.md`. **Don't modify the `mlb-fan-live` repo.**
- Cache every LLM call (keyed by model + prompt hash). LLM calls go through **LLM Foundry** (`LLMFOUNDRY_TOKEN`) on the cheapest suitable model. Track spend in `runs/spend.json` and **stop at $0.80 a day**.
- Prefer free local models where they're good enough (a RoBERTa sentiment baseline, a local zero-shot NLI model). Use the LLM only on pre-filtered comments.
- Keep prompts in `prompts/*.md`.
- Report every number with its sample size, and use bootstrap confidence intervals for effects. With about 60 batted balls, intervals will be wide. Say so.
- Write tests for deterministic parts: joins, time windows, keyword filters, metric maths.

**Don't**
- Don't collect new Reddit data.
- Don't publish usernames, user IDs or full comment text. Analysis may link a username across threads internally (hashed), but outputs show **aggregates only**, plus short paraphrased or trimmed excerpts with no names.
- Don't hand-label at scale. The only manual step is a ~10-minute spot-check of classifier output (Phase 3).
- Don't tune thresholds or keyword lists after looking at the outcome-bias result. Fix them in Phase 1, and change them only with a logged reason.
- Don't present this as client-ready. Reddit data is fine for an internal prototype; any client use needs licensing or legal sign-off, as in `mlb-fan-live`.

---

## Data

**Already have** (from `mlb-fan-live`):

| What | Notes |
|---|---|
| ~52.9k cleaned in-game + postgame comments | r/baseball (Game Thread + Postgame, split by flair into LAD / TOR / neutral), r/Dodgers (Game Chat + Postgame), r/Torontobluejays (Game Thread + Postgame; pre-game comments excluded) |
| Comment timestamps, thread, stream, author, reply context | Author is needed only for the same-user analysis (Phase 4). Hash it on import |
| Jev comment tags (~40k comments) | Subject (who or what the comment is about) and sentiment |
| Jev window readings (~12k) | Per-stream mood over time |
| Measured reaction lag per stream | Use it to place reaction windows |
| MLB Stats API play-by-play | Plays, per-pitch timing, win probability per play |
| RoBERTa baseline sentiment, player name and nickname matching | Use as a second sentiment measure and for subject attribution |

Locate the actual files in `mlb-fan-live` (cleaned comments table, Phase 3 tagged comments, readings, lag report, plays table), copy them, and document them in `data/IMPORT.md`.

**New:**
- **Statcast** for this game via `pybaseball.statcast(start_dt='2025-10-24', end_dt='2025-10-24')`, filtered to the game's `game_pk`. Look up the `game_pk` from the MLB Stats API schedule; don't guess it. Key columns:
  - `game_pk`, `at_bat_number`, `pitch_number`, `inning`, `inning_topbot`, `batter`, `pitcher`, `events`, `description`, `des`;
  - `launch_speed`, `launch_angle`, `estimated_ba_using_speedangle` (**xBA**), `estimated_woba_using_speedangle` (**xwOBA**);
  - `delta_home_win_exp`, `home_score`, `away_score`.
- Join Statcast batted balls to the MLB Stats API plays (and so to play end times) on `game_pk` + `at_bat_number`. Verify the join rather than assuming it.

## Repo layout

```
judged-by-results/
  README.md  config.yaml
  data/imported/  data/IMPORT.md  data/statcast.parquet  data/bip.parquet
  prompts/
  src/jbr/
    imports.py      # copy + hash authors + document sources
    statcast.py     # pull + join batted balls to plays/timestamps
    windows.py      # reaction windows per batted ball per stream
    attribute.py    # which comments are about the batter / pitcher
    outcome.py      # outcome-bias analysis
    claims.py       # prediction + inevitability extraction
    hindsight.py    # hindsight analysis
    decisions.py    # (optional) manager decision case studies
    stats.py        # bootstrap, regression helpers
    cli.py
  reports/  docs/  tests/  runs/
```

CLI shape: `jbr <step>`, for example `jbr statcast`, `jbr windows`, `jbr outcome`, `jbr claims`, `jbr hindsight`, `jbr site`.

---

## Phase 0: Setup and data join

1. Scaffold the repo, config, LLM Foundry client with spend guard and cache, and a single smoke-test call.
2. Import the `mlb-fan-live` data into `data/imported/`. Hash author names with a salt kept in `.env` (gitignored). Write `data/IMPORT.md`.
3. Look up `game_pk`, pull Statcast, and keep **batted balls in play** (rows where `launch_speed` is not null and `events` is a ball-in-play result; exclude strikeouts, walks, HBP).
4. Join each batted ball to its play in the MLB Stats API data to get the **play end timestamp**, inning, batter and pitcher names, teams, and win probability before and after.
5. Build `data/bip.parquet`, one row per batted ball:
   - `bip_id`, `inning`, `half`, `batter`, `batter_team`, `pitcher`, `pitcher_team`, `event`;
   - `hit` (1 for single/double/triple/HR, 0 for an out; flag errors and fielder's choice separately);
   - `xba`, `xwoba`, `launch_speed`, `launch_angle`;
   - `luck = hit - xba`;
   - `wp_before`, `wp_after`, `delta_wp_batting_team`, `play_end_ts`.
6. **Luck categories** (thresholds fixed here, in config):
   - `unlucky_out`: out with xBA ≥ 0.50
   - `lucky_hit`: hit with xBA ≤ 0.25
   - `expected_hit`: hit with xBA > 0.25
   - `expected_out`: out with xBA < 0.50

**Don't:** look at any comment data in this phase.

### CHECKPOINT 0
Show:
- `game_pk`.
- Batted-ball count, with how many have xBA.
- The count per luck category.
- Join success rate (it must be 100%, or every miss explained).
- A table of all `unlucky_out` and `lucky_hit` rows with batter, inning and description.
- Final score check against the official box score.
- The LLM smoke-test cost.

---

## Phase 1: Reaction windows and attribution

**Goal:** for each batted ball, collect the comments reacting to **that batter's** at-bat result, per stream.

1. **Window:** comments with timestamp in `[play_end_ts + lag_stream, play_end_ts + lag_stream + 120 s]`, using each stream's measured reaction lag. If the next play ends inside the window, cut the window there, so reactions to later plays don't leak in. Log the window length.
2. **Attribution:** a comment counts as "about the batter" if its Jev subject tag is the batter, **or** it name-matches the batter (using the `mlb-fan-live` nickname map), **or** (weaker) it's a reply in a thread whose parent is about the batter. Store `attribution_method`. Do the same for the pitcher (a secondary analysis).
3. **Sentiment toward the batter:** primary = Jev sentiment. Secondary = RoBERTa baseline. Map both to −1…+1.
4. **Fanbase of each comment:** batter's own fans, opponent fans, or neutral (from stream and flair).
5. **Process-language dictionary** (fixed now, in `config.yaml`): phrases showing a fan noticed contact quality or luck, for example "hit it hard", "smoked", "barreled", "crushed", "right at him", "unlucky", "tough luck", "bloop", "duck snort", "seeing-eye", "lucky", "found a hole", "great contact", "weak contact", "jammed", "off the end of the bat". Flag `mentions_process` per comment.

Output `data/reactions.parquet`, one row per (bip, comment).

**Don't:** compute any outcome-bias numbers yet. Freeze windows, attribution and the dictionary first.

### CHECKPOINT 1
Show:
- Per batted ball: how many attributed comments per fanbase. Flag batted balls with fewer than 5 attributed comments; they'll be excluded or pooled.
- Attribution method mix.
- Jev vs RoBERTa sentiment agreement (correlation) on the attributed comments.
- 10 random (batted ball, comment) pairs, trimmed, so Atharva can eyeball that attribution looks right (about 5 minutes).

---

## Phase 2: Outcome bias analysis

**Core question:** holding contact quality fixed, how much does the *result* change how fans judge the batter?

1. **Per batted ball:** mean sentiment toward the batter, per fanbase (own, opponent, neutral), plus comment volume.
2. **Main model** (OLS, comment level, with bootstrap CIs by resampling batted balls):
   `sentiment ~ hit + xba + |delta_wp| + fanbase + inning_bucket` (plus `hit × fanbase` as an interaction).
   - **Outcome bias** = the effect of `hit` while controlling for `xba`.
   - **Process sensitivity** = the effect of `xba` while controlling for `hit`.
   - The headline ratio is outcome effect ÷ process effect, with a CI.
   - `|delta_wp|` is there because results also change the game state. Without it, "fans like hits" would just be "fans like winning".
3. **The 2×2 contrast** (descriptive, easy to show): mean own-fan sentiment for `lucky_hit` vs `unlucky_out` vs `expected_hit` vs `expected_out`. If fans judged process, unlucky outs would be treated like hits. If they judge outcome, unlucky outs are treated like weak outs.
4. **Luck awareness:** the share of comments with `mentions_process`, by luck category and fanbase. Do fans *notice* when a hard-hit ball was caught? Own fans vs opponents?
5. **Pitcher side (secondary):** does sentiment toward the pitcher follow hits allowed or the xBA allowed?
6. **Robustness:**
   - rerun with RoBERTa sentiment;
   - with 60 s and 180 s windows;
   - name-match attribution only;
   - leaving out the single highest-leverage play (the biggest WP swing);
   - leaving out home runs (their xBA is high and the outcome is certain, so they inflate both effects).

**Don't:** add covariates or drop batted balls after seeing results without logging why in `reports/decisions_log.md`.

### CHECKPOINT 2
Show:
- The main model table with CIs.
- The 2×2 chart.
- Luck-awareness rates.
- Pitcher results.
- The robustness table: does the sign hold?
- A one-paragraph plain-English verdict, worded as "in this game…".

---

## Phase 3: Predictions and inevitability claims

**Goal:** find (a) in-game **predictions** ("this game is over", "we're coming back", "we've got this"), and (b) postgame **inevitability claims** ("it was over after the 6th", "knew it from the first pitch", "never had a chance", "should have seen this coming").

1. **Keyword pre-filter** (fixed list in config), so only a few thousand comments reach a classifier. Log recall risk: run the classifier on a random 300 unfiltered comments, and report how many positives the filter would have missed.
2. **Classifier A:** LLM Foundry, cheapest model, batched (around 20 comments per call), JSON output.
   - **In-game comments:** `{is_prediction, predicted_winner: LAD|TOR|null, confidence: low|mid|high}`
   - **Postgame comments:** `{is_inevitability_claim, favoured: LAD|TOR, referenced_moment: free text, referenced_inning: int|null, certainty: low|mid|high}`
3. **Classifier B:** a free local zero-shot NLI model (for example a DeBERTa/BART MNLI model) on the same items, for `is_prediction` / `is_inevitability_claim` and direction.
4. **Agreement** between A and B is the main quality signal. Keep items where both agree for the main analysis, and use all A-positives for a sensitivity check.
5. **Spot-check:** Atharva answers yes or no on 40 random A-positive items (20 predictions, 20 claims). That's about 10 minutes, and it gives a rough precision figure. No other manual labelling.
6. Map each `referenced_moment` to a play and timestamp (by inning, plus event keywords such as "grand slam" or "the pitching change"). If it can't be mapped, set it to null.

### CHECKPOINT 3
Show:
- Counts after the filter, after A, after B, and with both agreeing.
- Filter missed-positive estimate.
- Spot-check precision.
- Spend so far.
- Top referenced moments.
- 10 trimmed examples of each type.

---

## Phase 4: Hindsight analysis

**Core question:** do postgame claims of "it was obvious" match what fans actually believed at the time?

1. **Hindsight gap, crowd level:** for each referenced moment *m*:
   - **claimed certainty:** share of postgame claims saying the result was decided by *m*, and their certainty;
   - **in-game belief at *m*:** share of in-game predictions in the 10 minutes after *m* that predicted the eventual winner, plus Jev mood readings;
   - **actual win probability at *m*** for the eventual winner.

   Plot all three on the game timeline.
2. **Same-user check:** for users (hashed) who posted both an in-game prediction near *m* and a postgame claim about *m*, compare what they said then with what they claimed later. Report aggregate counts only, for example "of N users who claimed they knew after the 3rd, X had predicted the other team at that time".
3. **By fanbase:** winners (TOR) vs losers (LAD). The literature reports "retroactive pessimism" in losers (the loss feels inevitable afterwards). Is that visible here?
4. **Early vs late moments:**
   - Claims about moments when the win probability was near 50% are the interesting ones, since that's real hindsight.
   - Claims about moments after the game was effectively decided are mostly *accurate*.
   - Split them, and report the gap separately.

**Caveat to state on the page:** Game 1 was a blowout, so late-game claims are largely correct. The hindsight signal lives in the early-game claims, and there may be few of them.

### CHECKPOINT 4
Show:
- The timeline chart (claimed certainty vs in-game belief vs win probability).
- The same-user table.
- The fanbase comparison.
- The early-vs-late split.
- A one-paragraph verdict.

---

## Phase 5 (optional): Manager decisions, as qualitative case studies

For each clear manager decision (pitching change, pinch hitter, intentional walk, steal attempt), show:
- the win probability before;
- the result;
- the win probability after;
- the reaction before and after the result toward the manager (subject tags: the manager's name, or "Roberts/Schneider", "the manager").

No statistics, just 3–5 short stories, labelled "illustrative, n = 1 each". Skip it if there are fewer than 3 clean decisions.

---

## Phase 6: Demo page (GitHub Pages)

`docs/index.html`, single file, no build step, data from `reports/*.json`:
1. **The hook:** two batted balls side by side, a hard-hit out and a bloop hit, with their xBA and what fans said (trimmed, no names).
2. **The scatter:** each batted ball as a dot. x = xBA (how well it was hit), y = own-fan sentiment toward the batter, colour = hit or out. If fans judged process, the colours would mix; if they judge outcome, the colours split. Hover for details.
3. **The 2×2 chart** and the main model result in plain words, with CIs.
4. **Luck awareness:** "only X% of reactions to unlucky outs mentioned how hard it was hit".
5. **The hindsight timeline:** win probability vs what fans believed at the time vs what they claimed afterwards.
6. **Method and limits:**
   - one game;
   - a blowout;
   - Reddit users aren't all fans;
   - the classifiers are imperfect (with precision from the spot-check);
   - xBA is itself a model;
   - internal prototype data rules.
7. Sources and credits: MLB Stats API, Statcast/Baseball Savant, Reddit threads named, and the `mlb-fan-live` lineage.

**Don't:** show usernames, full comments, or claims beyond "in this game".

### CHECKPOINT 6
Page renders offline, every number comes from `reports/`, the limits section is visible, and no usernames appear (grep the built page for hashed and raw author fields).

---

## Rough timeline

| Phase | Time |
|---|---|
| 0 Setup + Statcast join | ½ day |
| 1 Windows + attribution | 1 day |
| 2 Outcome bias | 1 day |
| 3 Claims extraction | 1 day |
| 4 Hindsight | 1 day |
| 5 Manager decisions (optional) | ½ day |
| 6 Demo page | 1 day |

About **5–6 working days**. LLM spend: a few dollars at most, over several days, within the daily cap.

## Out of scope (v1)
- More games (it's the natural v2; the pipeline should take a `game_pk` so it can be rerun).
- New Reddit collection.
- NFL (no comment data yet).
- Fine-tuning classifiers.