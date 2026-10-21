"""Phase 3: in-game predictions and postgame inevitability claims.

Keyword pre-filter -> Classifier B (local zero-shot NLI, always run) and, when
--with-llm is given, Classifier A (LLM Foundry). With both, the main set is the
items where A and B agree; with B only, it is B's positives.
Claimed moments are mapped to plays by event keywords, then by inning.
"""

from __future__ import annotations

import json
import random
import re

import pandas as pd

from jbr import config

FAN = {"LAD": "Dodgers fan", "TOR": "Blue Jays fan", None: "neutral baseball fan"}
HYPOTHESES = {
    "ingame": ["The writer predicts the final result of this game.",
               "The writer expects the Blue Jays to win this game.",
               "The writer expects the Dodgers to win this game."],
    "postgame": ["The writer says the result of this game was obvious or decided long before the end.",
                 "The writer says the Blue Jays were always going to win this game.",
                 "The writer says the Dodgers were always going to win this game."],
}
WHEN = {"ingame": "during the game", "postgame": "after the game"}
ORDINALS = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3, "4th": 4, "fourth": 4,
            "5th": 5, "fifth": 5, "6th": 6, "sixth": 6, "7th": 7, "seventh": 7, "8th": 8, "eighth": 8,
            "9th": 9, "ninth": 9}
_ORD = "|".join(ORDINALS)
AFTER_INNING = re.compile(rf"\b(?:after|by|since|end of) the ({_ORD})\b", re.I)
INNING = re.compile(rf"\b(?:(?:in|bottom of|top of|of) the ({_ORD})\b|({_ORD}) inning\b)", re.I)


def keyword_filter(patterns: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(patterns) + ")", re.IGNORECASE)


def stream_teams() -> dict[str, str | None]:
    import yaml

    game = yaml.safe_load((config.IMPORTED / "game.yaml").read_text(encoding="utf-8"))
    return {s["id"]: s["team"] for s in game["streams"]}


def certainty(score: float, bands: dict) -> str:
    return "high" if score >= bands["high"] else "mid" if score >= bands["mid"] else "low"


# ---- moments -----------------------------------------------------------------

def special_plays(plays: pd.DataFrame) -> dict[str, int]:
    """Named plays: first play, and the first batter after Snell left in the 6th."""
    out = {"FIRST_PLAY": int(plays["at_bat_index"].min())}
    b6 = plays[(plays["inning"] == 6) & (plays["half"] == "bottom")].sort_values("at_bat_index")
    if len(b6):
        first = b6.iloc[0]["pitcher_name"]
        changed = b6[b6["pitcher_name"] != first]
        if len(changed):
            out["PITCHING_CHANGE_6"] = int(changed.iloc[0]["at_bat_index"])
    return out


def map_moment(text: str | None, inning: int | None, plays: pd.DataFrame, cfg: dict) -> tuple[int | None, str | None]:
    """(at_bat_index, how) for the moment a claim points at, or (None, None)."""
    text = text or ""
    specials = special_plays(plays)
    for m in cfg["claims"]["moments"]:
        if re.search(m["pattern"], text, re.I):
            if m["find"] in specials:
                return specials[m["find"]], m["find"].lower()
            hit = plays[plays["description"].str.contains(m["find"], case=False, regex=False)]
            if len(hit):
                return int(hit.iloc[0]["at_bat_index"]), m["find"].lower()
    after = AFTER_INNING.search(text)
    if after:
        inning = ORDINALS[after.group(1).lower()]
        last = plays[plays["inning"] == inning]
        return (int(last["at_bat_index"].max()), f"end of inning {inning}") if len(last) else (None, None)
    if inning is None:
        found = INNING.search(text)
        inning = ORDINALS[(found.group(1) or found.group(2)).lower()] if found else None
    if inning is not None:
        inn = plays[plays["inning"] == inning]
        if len(inn):
            return int(inn.loc[inn["wp_delta"].abs().idxmax(), "at_bat_index"]), f"biggest swing of inning {inning}"
    return None, None


def referenced_inning(text: str) -> int | None:
    m = AFTER_INNING.search(text) or INNING.search(text)
    if not m:
        return None
    return ORDINALS[next(g for g in m.groups() if g).lower()]


# ---- classifiers -------------------------------------------------------------

def nli_scores(texts: list[str], kind: str, fans: list[str], cfg: dict) -> pd.DataFrame:
    """Entailment scores for the three hypotheses of ``kind``."""
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=cfg["claims"]["nli"]["model"], device=-1)
    premises = [f'A {f} wrote {WHEN[kind]}: "{t[:400]}"' for t, f in zip(texts, fans)]
    hyps = HYPOTHESES[kind]
    rows = []
    for res in clf(premises, candidate_labels=hyps, hypothesis_template="{}", multi_label=True, batch_size=32):
        s = dict(zip(res["labels"], res["scores"]))
        rows.append({"b_score": s[hyps[0]], "b_tor": s[hyps[1]], "b_lad": s[hyps[2]]})
    return pd.DataFrame(rows)


NLI_CACHE = config.DATA / "claims_nli.parquet"


def cached_nli(todo: pd.DataFrame, kind: str, cfg: dict) -> pd.DataFrame:
    """NLI scores for ``todo``, reusing data/claims_nli.parquet (keyed by comment, kind and model)."""
    model = cfg["claims"]["nli"]["model"]
    cache = pd.read_parquet(NLI_CACHE) if NLI_CACHE.exists() else pd.DataFrame(
        columns=["comment_id", "kind", "model", "b_score", "b_tor", "b_lad"])
    have = cache[(cache["kind"] == kind) & (cache["model"] == model)].set_index("comment_id")
    missing = todo[~todo["comment_id"].isin(have.index)]
    if len(missing):
        new = nli_scores(missing["body"].tolist(), kind, [FAN[t] for t in missing["team"]], cfg)
        new.insert(0, "comment_id", missing["comment_id"].to_numpy())
        new.insert(1, "kind", kind)
        new.insert(2, "model", model)
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(NLI_CACHE, index=False)
        have = cache[(cache["kind"] == kind) & (cache["model"] == model)].set_index("comment_id")
    return have.loc[todo["comment_id"], ["b_score", "b_tor", "b_lad"]].reset_index(drop=True)


# Which team a live prediction backs, from its wording and the score when posted.
# Checked in this order; "we"/"our" phrases need a fan's team.
FAN_WINS = re.compile(r"\b(we got this|got this one|we('?ll| will| gonna| going to) win|we win|it'?s ours|"
                      r"this one'?s ours|(we'?re|we are) winning this)\b", re.I)
FAN_LOSES = re.compile(r"\b(we('?ll| will| gonna| going to) lose|we lose|(we'?re|we are) (done|cooked|toast|finished|"
                       r"losing this)|can'?t win|no chance|no way we|it'?s theirs|this one'?s theirs)\b", re.I)
COMEBACK = re.compile(r"\b(come ?back|coming back|not over|still time|plenty of (time|game))\b", re.I)
OVER = re.compile(r"\b(over|done|finished|ballgame|ball game|in the bag|put (it|this) away|lock(ed)?|cooked|blow ?out)\b", re.I)


def score_at(t: pd.Timestamp, plays: pd.DataFrame, home: str, away: str) -> tuple[str | None, str | None]:
    """(leader, trailer) from the last play that ended before ``t``; (None, None) when tied."""
    done = plays[plays["end_time_utc"] < t]
    if done.empty:
        return None, None
    last = done.loc[done["end_time_utc"].idxmax()]
    if last["home_score"] == last["away_score"]:
        return None, None
    return (home, away) if last["home_score"] > last["away_score"] else (away, home)


def rule_direction(text: str, fan_team: str | None, leader: str | None, trailer: str | None) -> str | None:
    other = {"LAD": "TOR", "TOR": "LAD"}
    if fan_team and FAN_WINS.search(text):
        return fan_team
    if fan_team and FAN_LOSES.search(text):
        return other[fan_team]
    if COMEBACK.search(text):
        return trailer
    if OVER.search(text):
        return leader
    return None


def label_b(scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    nli = cfg["claims"]["nli"]
    out = scores.copy()
    out["b_positive"] = out["b_score"] >= nli["positive_min"]
    team = out[["b_tor", "b_lad"]].idxmax(axis=1).map({"b_tor": "TOR", "b_lad": "LAD"})
    out["b_winner"] = team.where(out[["b_tor", "b_lad"]].max(axis=1) >= nli["direction_min"])
    out.loc[~out["b_positive"], "b_winner"] = None
    out["b_certainty"] = [certainty(s, nli["certainty_bands"]) if p else None
                          for s, p in zip(out["b_score"], out["b_positive"])]
    return out


def run_llm(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Classifier A on the filtered items; returns columns a_positive, a_winner, a_certainty, a_moment, a_inning."""
    from jbr import llm

    fan_short = {"LAD": "LAD fan", "TOR": "TOR fan", None: "neutral"}
    out = []
    for kind, prompt in (("ingame", "ingame_prediction"), ("postgame", "postgame_claim")):
        sub = df[df["kind"] == kind]
        items = [{"id": r.comment_id, "fan": fan_short[r.team], "text": r.body[:400]} for r in sub.itertuples()]
        labels = llm.classify(items, prompt, cfg)
        for cid in sub["comment_id"]:
            lab = labels.get(cid, {})
            if kind == "ingame":
                out.append({"comment_id": cid, "a_positive": bool(lab.get("is_prediction")),
                            "a_winner": lab.get("predicted_winner"), "a_certainty": lab.get("confidence"),
                            "a_moment": None, "a_inning": None})
            else:
                out.append({"comment_id": cid, "a_positive": bool(lab.get("is_inevitability_claim")),
                            "a_winner": lab.get("favoured"), "a_certainty": lab.get("certainty"),
                            "a_moment": lab.get("referenced_moment"), "a_inning": lab.get("referenced_inning")})
    return pd.DataFrame(out)


# ---- pipeline ----------------------------------------------------------------

def build(with_llm: bool = False, cfg: dict | None = None) -> dict:
    cfg = cfg or config.load()
    c = cfg["claims"]
    comments = pd.read_parquet(config.IMPORTED / "comments.parquet")
    plays = pd.read_parquet(config.IMPORTED / "plays.parquet").merge(
        pd.read_parquet(config.IMPORTED / "win_prob.parquet"), on=["game_pk", "at_bat_index"])
    teams = stream_teams()
    comments["team"] = comments["stream"].map(teams)

    frames, recall = [], {}
    rng = random.Random(20251024)
    for kind, phase, key in (("ingame", "in-game", "ingame_filter"), ("postgame", "postgame", "postgame_filter")):
        pool = comments[comments["phase"] == phase].copy()
        pool["kind"] = kind
        pool["passes_filter"] = pool["body"].str.contains(keyword_filter(c[key]))
        rest = pool[~pool["passes_filter"]]
        sample_ids = set(rng.sample(sorted(rest["comment_id"]), min(c["recall_sample"], len(rest))))
        pool["in_recall_sample"] = pool["comment_id"].isin(sample_ids)
        todo = pool[pool["passes_filter"] | pool["in_recall_sample"]].reset_index(drop=True)
        scores = label_b(cached_nli(todo, kind, cfg), cfg)
        todo = pd.concat([todo, scores], axis=1)
        frames.append(todo)
        sample = todo[todo["in_recall_sample"]]
        k = int(sample["b_positive"].sum())
        recall[kind] = {"n_pool": int(len(pool)), "n_filtered": int(pool["passes_filter"].sum()),
                        "sample": int(len(sample)), "sample_positives": k,
                        "est_missed": round(k / max(len(sample), 1) * len(rest)),
                        "filtered_positives": int(todo.loc[todo["passes_filter"], "b_positive"].sum())}

    df = pd.concat(frames, ignore_index=True)
    filt = df[df["passes_filter"]].copy()

    if with_llm:
        filt = filt.merge(run_llm(filt, cfg), on="comment_id", how="left")
        filt["agree"] = filt["a_positive"].fillna(False).astype(bool) & filt["b_positive"]
        filt["main"] = filt["agree"]
        filt["certainty_final"] = filt["a_certainty"].where(filt["a_certainty"].notna(), filt["b_certainty"])
    else:
        for col in ("a_positive", "a_winner", "a_certainty", "a_moment", "a_inning", "agree"):
            filt[col] = None
        filt["main"] = filt["b_positive"]
        filt["certainty_final"] = filt["b_certainty"]

    # Team backed. In-game: wording + score at the time, then the LLM's answer, then NLI's.
    # Postgame: a claim that this game's result was inevitable backs the actual winner.
    g = pd.read_parquet(config.IMPORTED / "games.parquet").iloc[0]
    home, away = g["home_abbr"], g["away_abbr"]
    game_winner = home if g["home_score"] > g["away_score"] else away
    rule = [rule_direction(b, t if isinstance(t, str) else None, *score_at(ts, plays, home, away))
            for b, t, ts in zip(filt["body"], filt["team"], filt["created_utc"])]
    filt["rule_winner"] = rule
    a_win = filt["a_winner"].where(filt["a_winner"].isin(["LAD", "TOR"]))
    filt["winner"] = filt["rule_winner"].where(filt["rule_winner"].notna(), a_win)
    filt["winner"] = filt["winner"].where(filt["winner"].notna(), filt["b_winner"])
    filt.loc[filt["kind"] == "postgame", "winner"] = game_winner
    filt.loc[~filt["main"].astype(bool), "winner"] = None

    # Moments for postgame claims: LLM's moment text first when present, else the comment.
    filt["referenced_inning"] = filt["body"].map(referenced_inning)
    mapped = [map_moment(" ".join(x for x in (a, b) if isinstance(x, str)),
                         int(i) if pd.notna(i) else None, plays, cfg) if k == "postgame" else (None, None)
              for a, b, i, k in zip(filt["a_moment"], filt["body"], filt["a_inning"], filt["kind"])]
    filt["moment_play"] = [m[0] for m in mapped]
    filt["moment_how"] = [m[1] for m in mapped]
    ends = plays.set_index("at_bat_index")["end_time_utc"]
    filt["moment_ts"] = filt["moment_play"].map(ends)

    keep = ["comment_id", "kind", "stream", "team", "author_hash", "created_utc", "body", "b_score", "b_tor",
            "b_lad", "b_positive", "b_winner", "rule_winner", "b_certainty", "a_positive", "a_winner", "a_certainty",
            "a_moment", "a_inning", "agree", "main", "winner", "certainty_final", "referenced_inning",
            "moment_play", "moment_how", "moment_ts"]
    filt[keep].to_parquet(config.DATA / "claims.parquet", index=False)

    summary = summarise(filt, recall, plays, with_llm, cfg)
    (config.REPORTS / "claims.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
    write_spotcheck(filt, cfg)
    return summary


def summarise(filt: pd.DataFrame, recall: dict, plays: pd.DataFrame, with_llm: bool, cfg: dict) -> dict:
    from jbr import llm
    from jbr.outcome import excerpt

    out = {"classifier": "A+B (agreement)" if with_llm else "B only (local NLI)", "recall": recall,
           "spend_today_usd": llm.spent_today(), "counts": {}, "examples": {}, "top_moments": []}
    for kind in ("ingame", "postgame"):
        k = filt[filt["kind"] == kind]
        out["counts"][kind] = {
            "after_filter": int(len(k)), "b_positive": int(k["b_positive"].sum()),
            "a_positive": int(k["a_positive"].fillna(False).astype(bool).sum()) if with_llm else None,
            "both_agree": int(k["agree"].fillna(False).astype(bool).sum()) if with_llm else None,
            "main": int(k["main"].sum()),
            "by_winner": k[k["main"]]["winner"].fillna("none").value_counts().to_dict(),
            "by_team": k[k["main"]]["team"].fillna("neutral").value_counts().to_dict(),
        }
        ex = k[k["main"]].sample(min(10, int(k["main"].sum())), random_state=1)
        out["examples"][kind] = [{"team": t or "neutral", "winner": w, "text": excerpt(b, 120)}
                                 for t, w, b in zip(ex["team"], ex["winner"], ex["body"])]
    pg = filt[(filt["kind"] == "postgame") & filt["main"] & filt["moment_play"].notna()]
    desc = plays.set_index("at_bat_index")
    for play, n in pg["moment_play"].value_counts().head(8).items():
        p = desc.loc[int(play)]
        out["top_moments"].append({"at_bat_index": int(play), "n_claims": int(n), "inning": int(p["inning"]),
                                   "half": p["half"], "description": p["description"],
                                   "tor_wp_after": float(p["home_wp_after"])})
    out["postgame_main_unmapped"] = int(((filt["kind"] == "postgame") & filt["main"] & filt["moment_play"].isna()).sum())
    return out


def write_spotcheck(filt: pd.DataFrame, cfg: dict) -> None:
    from jbr.outcome import excerpt

    n = cfg["claims"]["spotcheck_per_kind"]
    items = []
    for kind, q in (("ingame", "Is this a prediction about who wins THIS game?"),
                    ("postgame", "Is this a claim that the result was obvious / decided early?")):
        pos = filt[(filt["kind"] == kind) & filt["main"]]
        for r in pos.sample(min(n, len(pos)), random_state=11).itertuples():
            items.append({"id": f"p3-{r.comment_id}", "kind": kind, "question": q,
                          "fan": FAN[r.team if isinstance(r.team, str) else None], "winner": r.winner,
                          "comment": excerpt(r.body, 240)})
    (config.REPORTS / "spotcheck_phase3.json").write_text(json.dumps(items, indent=2, ensure_ascii=False),
                                                          encoding="utf-8")
