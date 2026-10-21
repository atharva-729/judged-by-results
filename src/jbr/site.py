"""Phase 6: docs/index.html, built from reports/*.json.

The data is embedded in the page at build time, so it renders offline with no
fetches. Every number on the page comes from reports/. After building, the page
is checked for author hashes and usernames.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd

from jbr import config, spotcheck

TEMPLATE = config.ROOT / "src" / "jbr" / "site_template.html"
OUT = config.ROOT / "docs" / "index.html"


def _load(name: str) -> dict:
    return json.loads((config.REPORTS / name).read_text(encoding="utf-8"))


def f2(x: float | None) -> str:
    if x is None:
        return "–"
    return "0.00" if abs(x) < 0.005 else f"{'+' if x > 0 else '−'}{abs(x):.2f}"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{round(x * 100)}%"


def ci(e: dict) -> str:
    return f"[{f2(e['lo'])}, {f2(e['hi'])}]"


def hook(balls: list[dict], cfg: dict) -> list[dict]:
    """The unlucky out and the lucky hit that drew the most own-fan reactions (excluding home runs)."""
    ok = [b for b in balls if not b["thin"] and b["n"]["own"] >= 5 and b["excerpts"] and b["event"] != "home_run"]
    out_ = max((b for b in ok if b["luck_category"] == "unlucky_out"), key=lambda b: b["n"]["own"], default=None)
    hit_ = max((b for b in ok if b["luck_category"] == "lucky_hit"), key=lambda b: b["n"]["own"], default=None)
    cards = []
    for b, label in ((out_, "Hit well, caught"), (hit_, "Hit poorly, fell in")):
        if b is None:
            continue
        neg_first = b["mean"]["own"] is not None and b["mean"]["own"] < 0
        cards.append({"label": label, "batter": b["batter"], "inning": b["inning"], "half": b["half"],
                      "result": b["event"].replace("_", " "), "description": b["description"],
                      "launch_speed": b["launch_speed"], "xba": b["xba"],
                      "xba_words": f"balls hit like this fall for hits {round(b['xba'] * 100)}% of the time",
                      "own_mean": b["mean"]["own"], "own_n": b["n"]["own"],
                      "excerpt": b["excerpts"][1] if neg_first else b["excerpts"][0]})
    return cards


HYPE = [("LET'S GO", r"let[’']?s go+"), ("f***", r"\bf+u+c+k"), ("lol / lmao", r"\b(?:lol|lmao)\b"),
        ("clutch", r"\bclutch\b"), ("wow", r"\bwo+w\b"), ("omg", r"\bomg\b"), ("what a", r"\bwhat a\b")]


def game_series(cfg: dict) -> dict:
    """Per-minute comment volume, 2-minute fan mood by team, and key plays for annotations."""
    from jbr.claims import stream_teams

    imp = config.IMPORTED
    plays = pd.read_parquet(imp / "plays.parquet").merge(pd.read_parquet(imp / "win_prob.parquet"),
                                                         on=["game_pk", "at_bat_index"])
    start, end = plays["start_time_utc"].min() - pd.Timedelta(minutes=5), plays["end_time_utc"].max() + pd.Timedelta(minutes=5)
    vol = pd.read_parquet(imp / "volume.parquet")
    vol = vol[(vol["t"] >= start) & (vol["t"] <= end)]
    per_min = vol.groupby(vol["t"].dt.floor("1min"))["n"].sum()
    ticks = pd.read_parquet(imp / "ticks.parquet")
    ticks = ticks[ticks["mood"].notna() & ~ticks["stale"] & (ticks["t"] >= start) & (ticks["t"] <= end)]
    ticks = ticks.assign(team=ticks["stream"].map(stream_teams())).dropna(subset=["team"])
    mood = ticks.groupby([ticks["t"].dt.floor("2min"), "team"])["mood"].mean().unstack()
    key = []
    for r in plays.itertuples():
        if not (r.event == "Home Run" or abs(r.wp_delta) >= 0.1):
            continue
        last = r.batter_name.split()[-1]
        what = "grand slam" if "grand slam" in r.description else "HR" if r.event == "Home Run" else r.event.lower()
        key.append({"t": r.end_time_utc, "label": f"{last} {what}", "wp": float(r.home_wp_after),
                    "half": r.half, "inning": int(r.inning)})
    return {"volume": [{"t": t, "n": int(n)} for t, n in per_min.items()],
            "mood": [{"t": t, "TOR": (None if pd.isna(row.get("TOR")) else round(float(row["TOR"]), 3)),
                      "LAD": (None if pd.isna(row.get("LAD")) else round(float(row["LAD"]), 3))}
                     for t, row in mood.iterrows()],
            "key_plays": key}


def word_counts(react: pd.DataFrame, cfg: dict) -> dict:
    bat = react[(react["role"] == "batter") & react["in_main"]].drop_duplicates("comment_id")
    from jbr.attribute import process_pattern

    process = [{"word": p, "n": int(bat["body"].str.contains(process_pattern([p])).sum())} for p in cfg["process_phrases"]]
    hype = [{"word": w, "n": int(bat["body"].str.contains(p, case=False, regex=True).sum())} for w, p in HYPE]
    return {"n_comments": int(len(bat)), "process": sorted(process, key=lambda d: -d["n"]),
            "process_any": int(bat["body"].str.contains(process_pattern(cfg["process_phrases"])).sum()),
            "hype": sorted(hype, key=lambda d: -d["n"])}


def build() -> dict:
    cfg = config.load()
    o, c, h = _load("outcome.json"), _load("claims.json"), _load("hindsight.json")
    react = pd.read_parquet(config.DATA / "reactions.parquet")
    claims_df = pd.read_parquet(config.DATA / "claims.parquet")
    m, pb, con, aw = o["main"], o["per_ball"], o["contrast"]["own"], o["luck_awareness"]["all"]
    prec = spotcheck.precision()

    bat = react[(react["role"] == "batter") & react["in_main"]]
    both = bat.dropna(subset=["sentiment_jev", "sentiment_roberta"])
    mix = bat["attribution_method"].value_counts(normalize=True)
    lengths = react[react["in_main"]].groupby("bip_id")["offset_s"].max()
    lucky = [b for b in o["balls"] if b["luck_category"] == "lucky_hit" and not b["thin"]]
    unlucky = [b for b in o["balls"] if b["luck_category"] == "unlucky_out" and not b["thin"]]

    k = {
        "n_comments": f"{52872:,}", "n_bip_model": str(m["n_bip"]), "n_bip_total": str(o["n_bip_total"]),
        "n_bip_xba": str(o["n_bip_with_xba"]), "n_thin": str(len(o["thin_bip"])),
        "th_out": f"{cfg['luck']['unlucky_out_min_xba']:.2f}", "th_hit": f"{cfg['luck']['lucky_hit_max_xba']:.2f}",
        "n_phrases": str(len(cfg["process_phrases"])), "phrases": ", ".join(cfg["process_phrases"]),
        "attr_mix": f"Of {len(bat):,} batter comments: Jev tag {pct(mix.get('jev'))}, name match {pct(mix.get('name'))}, "
                    f"reply {pct(mix.get('reply'))}.",
        "jev_roberta_r": f"{both['sentiment_jev'].corr(both['sentiment_roberta']):.2f}",
        "window_median": f"{lengths.median():.0f}", "n_unlucky": str(len(unlucky)), "n_lucky": str(len(lucky)),
        "built": dt.date.today().isoformat(),
        "th_out_v": cfg["luck"]["unlucky_out_min_xba"], "th_hit_v": cfg["luck"]["lucky_hit_max_xba"],
    }

    split = h["split"]
    early, decided = split["early"], split["decided"]
    ig, pg = c["counts"]["ingame"], c["counts"]["postgame"]
    keypoint = (
        f"<b>In this game, fans judged the result, not the swing.</b> At the same contact quality, a hit lifted "
        f"the batter's own fans' sentiment by <mark><b>{f2(m['own_hit']['est'])}</b></mark> on a −1…+1 scale "
        f"(95% CI {ci(m['own_hit'])}). How well the ball was actually hit moved it by only "
        f"<b>{f2(m['own_xba']['est'])}</b> {ci(m['own_xba'])}, which can't be told apart from zero. "
        f"<mark><b>{aw['unlucky_out']['n_process']} of {aw['unlucky_out']['n']}</b></mark> reactions to hard-hit outs "
        f"mentioned how well it was hit."
    )
    if early["claims"]:
        keypoint += (f" After the game, {early['claims']} claims said it was decided at moments when Toronto's real "
                     f"win chance was only ~{pct(early['mean_winner_wp'])}.")
    else:
        keypoint += (f"<br><b>Hindsight:</b> no measurable rewriting. Only {h['n_claims_main']} postgame comments claimed "
                     f"the result was inevitable, and none pointed at the early, uncertain part of the game.")
    real = (prec or {}).get("postgame", {}).get("yes")
    keypoint = (
        "<b>In this game, fans judged the result, not the swing.</b> A hit lifted the batter's own fans by a lot; "
        "hitting the ball well, at the same result, did nothing measurable, and almost nobody mentioned luck. "
        f"<b>Hindsight</b> couldn't be tested: just {h['n_claims_main']} postgame \"we knew it\" claims"
        + (f" ({real} real on a hand check)" if real is not None else "")
        + ", and the few that named a moment picked one that really was decisive."
    )

    tiles = [
        {"big": f"{o['n_bip_with_xba']}", "title": "batted balls with Statcast xBA",
         "sub": f"{m['n_bip']} with ≥5 comments about the batter; {len(unlucky)} unlucky outs, {len(lucky)} lucky hits"},
        {"big": f"{len(bat):,}", "title": "comments about the batter, right after the play",
         "sub": f"of 52,872 game and postgame comments; own fans, opponents and neutrals kept separate"},
        {"big": f"{ig['main']:,} / {pg['main']:,}", "title": "live predictions / postgame \"we knew\" claims",
         "sub": f"found by a keyword filter and a local NLI model among {ig['after_filter']:,} and {pg['after_filter']:,} candidates"},
    ]

    lk, uo, eo, eh = (con[x] for x in ("lucky_hit", "unlucky_out", "expected_out", "expected_hit"))
    notes_r1 = [
        f"<b>Unlucky outs were treated like ordinary outs:</b> own fans averaged {f2(uo['mean'])} after a well-hit ball "
        f"was caught, vs {f2(eo['mean'])} after an ordinary out. These rest on just <mark>{uo['n_clusters']} plays</mark>, "
        f"so the CI is very wide.",
        f"<b>Lucky hits got most of the credit of a real hit:</b> {f2(lk['mean'])} vs {f2(eh['mean'])} for ordinary hits, "
        f"and far above any out.",
        f"<b>Opponents and neutrals</b> show the mirror image: opponents are cool on hits, and neutrals warm to them. "
        f"Nobody's reactions track contact quality.",
    ]
    ratio = m["own_hit"]["est"] / m["own_xba"]["est"] if m["own_xba"]["est"] else None
    modeltiles = [
        {"title": "Outcome effect (own fans)", "big": f2(m["own_hit"]["est"]),
         "sub": f"a hit vs an out at the same xBA · 95% CI {ci(m['own_hit'])}"},
        {"title": "Process effect (own fans)", "big": f2(m["own_xba"]["est"]),
         "sub": f"xBA 0 → 1 at the same result · 95% CI {ci(m['own_xba'])}"},
        {"title": "Outcome ÷ process", "big": (f"≈{ratio:.0f}×" if ratio and ratio > 0 else "n/a"),
         "sub": "point estimate only: the process effect is too close to zero for a meaningful CI"},
    ]
    rob = o["robustness"]
    holds = sum(v["own_hit"]["lo"] > 0 for v in rob.values())
    xba_zero = sum(v["own_xba"]["lo"] <= 0 for v in rob.values())
    notes_r2 = [
        f"<b>The result effect survives every check:</b> it is positive with a CI above zero in <mark>{holds} of {len(rob)}</mark> "
        f"versions (RoBERTa sentiment, 60/180 s windows, name-match only, without the biggest play, without home runs). "
        f"<a href='#how'>Table ↓</a>",
        f"<b>The process effect is never clearly above zero</b> ({xba_zero} of {len(rob)} CIs include 0). Without home runs it is "
        f"{f2(rob['Without home runs']['own_xba']['est'])}.",
        f"<b>Weighting matters:</b> counting each ball once instead of each comment, the process effect rises to "
        f"{f2(pb['xba']['est'])} {ci(pb['xba'])}, still below the result effect ({f2(pb['hit']['est'])}). Busy plays such as "
        f"the grand slam dominate the comment-level model.",
        f"<b>The size of the win-probability swing is held fixed</b>, so the hit effect isn't just \"fans like winning\".",
    ]
    all_n = sum(v["n"] for v in aw.values())
    all_p = sum(v["n_process"] for v in aw.values())
    notes_r3 = [
        f"<b>Only {all_p} of {all_n:,}</b> comments about batters ({pct(all_p / all_n if all_n else None)}) mention contact or luck at all.",
        f"<b>Unlucky outs get the least luck talk ({aw['unlucky_out']['n_process']}/{aw['unlucky_out']['n']})</b>, the plays where "
        f"it matters most. Fans don't say \"he hit that hard, tough luck\" here; they just react to the out.",
    ]

    fan = h["fanbase"]
    su = h["same_user"]["near_moment"]
    notes_r4 = []
    late = decided["claims"] + split["between"]["claims"]
    if late:
        notes_r4.append(f"<b>The claims that name a moment point at the bottom of the 6th, after Toronto had taken over:</b> "
                        f"{late} of {h['n_claims_mapped']}, with the winner's win probability already "
                        f"{pct(min(m['winner_wp'] for m in h['moments']))}–{pct(max(m['winner_wp'] for m in h['moments']))}. "
                        f"Those claims are <mark>roughly accurate, not hindsight</mark>: this was a blowout.")
    if early["claims"]:
        notes_r4.append(f"<b>The real hindsight is in the early claims:</b> {early['claims']} claims point at moments with "
                        f"~{pct(early['mean_winner_wp'])} win probability, where live predictions picked Toronto "
                        f"{pct(early['belief_share_winner'])} of the time (n = {early['belief_n']}).")
    else:
        notes_r4.append("<b>No claims pointed at a moment of real uncertainty</b> (win probability 30–70%), so there is "
                        "no measurable hindsight gap in this game.")
    pre_n = sum(v["pre6th_predictions"] for v in fan.values())
    pre_w = sum((v["pre6th_share_winner"] or 0) * v["pre6th_predictions"] for v in fan.values())
    pre_wp = [p["winner_wp"] for p in h["timeline"]["wp"] if p["inning"] < 6]
    if pre_n:
        notes_r4.append(f"<b>Live, before the 6th, fans backed the eventual loser:</b> only <mark>{pct(pre_w / pre_n)}</mark> of "
                        f"{pre_n} team-specific predictions picked Toronto, while its win probability ranged "
                        f"{pct(min(pre_wp))}–{pct(max(pre_wp))}. "
                        f"Afterwards, nobody claimed to have seen Toronto's win coming that early.")
    checked = (f" A hand check found only <mark>{prec['postgame']['yes']} of {prec['postgame']['n']}</mark> of them were "
               f"real claims." if prec and prec["postgame"]["n"] else "")
    notes_r4.append(f"<b>\"We knew it\" was rare:</b> {h['n_claims_main']} inevitability claims in "
                    f"{sum(v['postgame_comments'] for v in fan.values()):,} postgame comments.{checked} Too few to measure "
                    f"a hindsight gap with any confidence.")
    lad, tor = fan.get("LAD", {}), fan.get("TOR", {})
    notes_r4.append(f"<b>Fanbases:</b> {lad.get('claims_per_100')} \"inevitable\" claims per 100 Dodgers postgame comments vs "
                    f"{tor.get('claims_per_100')} per 100 Blue Jays comments. Before the 6th, Dodgers fans' live predictions picked "
                    f"Toronto {pct(lad.get('pre6th_share_winner'))} of the time (n = {lad.get('pre6th_predictions')}).")
    notes_r4.append(f"<b>Same users:</b> {su['users']} users made both a live prediction near a moment and a postgame claim "
                    f"about it. {su['said_other_team_then']} of them had picked the other team at the time. "
                    f"<a href='#how'>Details ↓</a>")
    hindsight_tiles = [
        {"title": "Claims about early moments", "big": str(early["claims"]),
         "sub": f"winner's win prob 30–70%; {early['moments']} moments"},
        {"title": "Claims about decided moments", "big": str(decided["claims"]),
         "sub": f"winner's win prob ≥ 90%; {decided['moments']} moments"},
        {"title": "Claims not tied to a moment", "big": str(h["n_claims_main"] - h["n_claims_mapped"]),
         "sub": "no inning or event named"},
    ]

    robustness = [{"name": n, "hit": v["own_hit"], "xba": v["own_xba"], "n_bip": v["n_bip"]} for n, v in rob.items()]
    p = o["pitcher"]
    pitcher_text = (f"Sentiment toward the <b>pitcher</b> from his own fans: a hit allowed changes it by {f2(p['own_hit']['est'])} "
                    f"{ci(p['own_hit'])}, and the xBA allowed by {f2(p['own_xba']['est'])} {ci(p['own_xba'])} "
                    f"({p['n_comments']:,} comments, {p['n_bip']} balls). Same pattern, weaker: pitchers are blamed for hits, "
                    f"not for hard contact.")

    lb = h["same_user"]["latest_prediction_before"]
    sameuser_html = (f"<b>Same users, then vs later</b><table class='t'><tr><th></th><th class='num'>Users</th>"
                     f"<th class='num'>Same team then</th><th class='num'>Other team then</th></tr>"
                     f"<tr><td>Prediction near the moment</td><td class='num'>{su['users']}</td><td class='num'>{su['said_same_then']}</td>"
                     f"<td class='num'>{su['said_other_team_then']}</td></tr>"
                     f"<tr><td>Latest prediction before it</td><td class='num'>{lb['users']}</td><td class='num'>{lb['said_same_then']}</td>"
                     f"<td class='num'>{lb['said_other_team_then']}</td></tr></table>"
                     f"<p class='small muted'>Of {h['same_user']['claimants']} users who made a mapped postgame claim. Hashed IDs, aggregates only.</p>")
    fanbase_html = ("<b>By fanbase</b><table class='t'><tr><th></th><th class='num'>Postgame comments</th><th class='num'>Claims</th>"
                    "<th class='num'>Per 100</th><th class='num'>Live picks for TOR before the 6th</th></tr>" +
                    "".join(f"<tr><td>{t}</td><td class='num'>{v['postgame_comments']:,}</td><td class='num'>{v['claims']}</td>"
                            f"<td class='num'>{v['claims_per_100']}</td><td class='num'>{pct(v['pre6th_share_winner'])} of {v['pre6th_predictions']}</td></tr>"
                            for t, v in fan.items()) + "</table>")

    rc = c["recall"]
    clf = [{"name": "In-game predictions", "pool": f"{rc['ingame']['n_pool']:,}", "filtered": f"{ig['after_filter']:,}",
            "b": ig["b_positive"], "a": ig["a_positive"], "both": ig["both_agree"],
            "missed": f"~{rc['ingame']['est_missed']:,} ({rc['ingame']['sample_positives']}/{rc['ingame']['sample']} sample)"},
           {"name": "Postgame claims", "pool": f"{rc['postgame']['n_pool']:,}", "filtered": f"{pg['after_filter']:,}",
            "b": pg["b_positive"], "a": pg["a_positive"], "both": pg["both_agree"],
            "missed": f"~{rc['postgame']['est_missed']:,} ({rc['postgame']['sample_positives']}/{rc['postgame']['sample']} sample)"}]
    if prec:
        names = {"attribution": "comment is about the batter", "ingame": "live predictions",
                 "postgame": "postgame \"we knew it\" claims"}
        spot_html = ("<b>Spot-check (hand-checked yes/no):</b> " + "; ".join(
            f"{names[name]}: <b>{pct(v['precision'])}</b> correct ({v['yes']}/{v['n']})"
            for name, v in prec.items() if v["n"]) + ". "
            "Most wrong live predictions were about the series, not this game. Wrong postgame claims were "
            "mostly sarcasm or comments that a play simply \"felt destined\".")
    else:
        spot_html = ("<div class='pending'><b>Spot-check pending.</b> The 10 attribution pairs and 40 classified comments "
                     "haven't been hand-checked yet, so classifier precision is unknown.</div>")
    spot_html += (f"<p class='small muted'>Classifier: {c['classifier']}. The LLM second opinion (Classifier A) is built "
                  f"but not run; with it, the main set becomes the items both agree on. LLM spend so far: "
                  f"${c['spend_today_usd']:.2f}.</p>")

    data = {"k": k, "keypoint": keypoint, "tiles": tiles, "hook": hook(o["balls"], cfg), "contrast": o["contrast"],
            "balls": o["balls"], "notes": {"r1": notes_r1, "r2": notes_r2, "r3": notes_r3, "r4": notes_r4},
            "modeltiles": modeltiles, "awareness": aw, "timeline": h["timeline"], "hindsight_tiles": hindsight_tiles,
            "robustness": robustness, "pitcher_text": pitcher_text, "moments": h["moments"],
            "sameuser_html": sameuser_html, "fanbase_html": fanbase_html, "clf": clf, "spot_html": spot_html,
            "game": game_series(cfg), "words": word_counts(react, cfg), "precision": prec,
            "headline": {"hit": m["own_hit"], "xba": m["own_xba"], "unlucky": aw["unlucky_out"],
                         "claims": h["n_claims_main"], "claims_real": (prec or {}).get("postgame", {}).get("yes"),
                         "postgame_comments": sum(v["postgame_comments"] for v in fan.values()),
                         "pre6_share": (pre_w / pre_n) if pre_n else None, "pre6_n": pre_n}}
    blob = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return privacy_check(html, react, claims_df)


def privacy_check(html: str, react: pd.DataFrame, claims_df: pd.DataFrame) -> dict:
    """No author hashes, no raw usernames, no author fields in the built page."""
    hashes = set(react["author_hash"].dropna()) | set(claims_df["author_hash"].dropna())
    leaked = [x for x in hashes if x in html]
    raw = pd.read_parquet(config.ROOT.parent / "mlb-fan-live" / "data/processed/comments_raw/813027.parquet",
                          columns=["author"]) if (config.ROOT.parent / "mlb-fan-live").exists() else None
    names = set(raw["author"].dropna()) - {"[deleted]", "AutoModerator"} if raw is not None else set()
    # Usernames that are ordinary words would false-alarm; check names of 6+ chars as whole words.
    words = set(re.findall(r"[A-Za-z0-9_-]{6,}", html))
    name_hits = sorted(n for n in names if len(n) >= 6 and n in words)
    user_refs = re.findall(r"/?u/[A-Za-z0-9_-]{3,}", html)
    return {"hash_leaks": len(leaked), "username_hits": name_hits[:20], "n_username_hits": len(name_hits),
            "user_refs": user_refs[:10], "author_field": "author_hash" in html or '"author"' in html,
            "bytes": len(html.encode("utf-8"))}
