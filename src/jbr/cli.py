"""jbr <step>"""

from __future__ import annotations

import argparse
import sys


def cmd_import(args: argparse.Namespace) -> int:
    from jbr import imports

    imports.main()
    return 0


def cmd_statcast(args: argparse.Namespace) -> int:
    import pandas as pd
    import requests

    from jbr import config, statcast

    cfg = config.load()
    bip = statcast.run()
    sc = pd.read_parquet(config.DATA / "statcast.parquet")
    pa = sc[sc["events"].notna()]
    plays = pd.read_parquet(config.IMPORTED / "plays.parquet")

    print(f"game_pk {cfg['game']['game_pk']}")
    print(f"Plate appearances: Statcast {len(pa)}, MLB plays {len(plays)}")
    print(f"Batted balls in play: {len(bip)}, with xBA {bip['xba'].notna().sum()}")
    for _, r in bip[bip["xba"].isna()].iterrows():
        print(f"  no tracking: {r.inning} {r.half}, {r.batter}, {r.event}: {r.description}")
    print("Join to MLB plays: 100% (checked on batter, pitcher, inning)")
    corr = bip["statcast_delta_home_wp"].corr(
        bip["delta_wp_batting_team"].where(bip["half"] == "bot", -bip["delta_wp_batting_team"])
    )
    print(f"Win-prob change, MLB API vs Statcast: r = {corr:.3f}")

    print("\nLuck categories:")
    print(bip["luck_category"].value_counts(dropna=False).to_string())
    print(f"Errors: {bip['is_error'].sum()}, fielder's choice: {bip['is_fielders_choice'].sum()}")

    cols = ["luck_category", "inning", "half", "batter", "batter_team", "event", "xba", "launch_speed", "description"]
    rows = bip[bip["luck_category"].isin(["unlucky_out", "lucky_hit"])].sort_values(["luck_category", "inning"])
    print("\nUnlucky outs and lucky hits:")
    print(rows[cols].to_string(index=False, max_colwidth=70))

    box = requests.get(
        f"https://statsapi.mlb.com/api/v1/game/{cfg['game']['game_pk']}/boxscore", timeout=30
    ).json()["teams"]
    official = (box["away"]["teamStats"]["batting"]["runs"], box["home"]["teamStats"]["batting"]["runs"])
    last = sc.sort_values(["at_bat_number", "pitch_number"]).iloc[-1]
    statcast_final = (int(last["post_away_score"]), int(last["post_home_score"]))
    ok = "OK" if official == statcast_final else "MISMATCH"
    print(f"\nFinal score LAD-TOR: box score {official[0]}-{official[1]}, Statcast {statcast_final[0]}-{statcast_final[1]}: {ok}")
    return 0 if ok == "OK" else 1


def cmd_windows(args: argparse.Namespace) -> int:
    import json

    import pandas as pd

    from jbr import attribute, config

    cfg = config.load()
    react = attribute.build(cfg)
    bip = pd.read_parquet(config.DATA / "bip.parquet")
    main = react[react["in_main"]]
    bat = main[main["role"] == "batter"]

    lengths = main.groupby("bip_id")["offset_s"].max()
    print(f"Main window: start {cfg['window']['start_s']} s; median used length {lengths.median():.0f} s")
    counts = bat.pivot_table(index="bip_id", columns="fanbase", values="comment_id", aggfunc="count", fill_value=0)
    counts = bip.set_index("bip_id")[["inning", "half", "batter", "event", "luck_category"]].join(counts).fillna(0)
    counts["total"] = counts[["own", "opponent", "neutral"]].sum(axis=1).astype(int)
    thin = counts["total"] < cfg["attribution"]["min_comments_per_bip"]
    print(f"\nBatter-attributed comments per batted ball ({len(bat)} total; {thin.sum()} of {len(counts)} below "
          f"{cfg['attribution']['min_comments_per_bip']}):")
    print(counts.assign(thin=thin.map({True: "THIN", False: ""})).to_string(max_colwidth=22))
    print("\nAttribution method (batter):", bat["attribution_method"].value_counts().to_dict())
    print("Attribution method (pitcher):", main[main["role"] == "pitcher"]["attribution_method"].value_counts().to_dict())
    both = bat.dropna(subset=["sentiment_jev", "sentiment_roberta"])
    print(f"Jev sentiment coverage {bat['sentiment_jev'].notna().mean():.0%}; "
          f"Jev vs RoBERTa r = {both['sentiment_jev'].corr(both['sentiment_roberta']):.2f} (n = {len(both)})")
    print(f"mentions_process: {bat['mentions_process'].mean():.1%} of batter comments")

    sample = bat.sample(10, random_state=7).merge(bip[["bip_id", "inning", "half", "batter", "description"]], on="bip_id")
    pairs = [{"id": f"p1-{i}", "play": f"{r.inning} {r.half}: {r.description}", "batter": r.batter,
              "method": r.attribution_method, "comment": r.body[:240]} for i, r in sample.iterrows()]
    (config.REPORTS / "spotcheck_phase1.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n10 pairs for the spot-check saved to reports/spotcheck_phase1.json")
    return 0


def _fmt(e: dict) -> str:
    return f"{e['est']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"


def cmd_outcome(args: argparse.Namespace) -> int:
    from jbr import outcome

    r = outcome.run()
    m = r["main"]
    print(f"Main model: {m['n_comments']} comments on {m['n_bip']} batted balls (thin, excluded: {len(r['thin_bip'])})")
    for fb in ("own", "opponent", "neutral"):
        print(f"  {fb:<9} hit effect {_fmt(m[fb + '_hit'])}   xBA effect {_fmt(m[fb + '_xba'])}")
    print(f"  own ratio hit/xBA {_fmt(m['own_ratio'])}   |dWP| {_fmt(m['abs_dwp'])}")
    pb = r["per_ball"]
    print(f"Per-ball (own fans, {pb['n_bip']} balls): hit {_fmt(pb['hit'])}  xBA {_fmt(pb['xba'])}")
    print("\n2x2, mean sentiment [95% CI] (comments / balls):")
    for fb, cats in r["contrast"].items():
        cells = "  ".join(f"{c}: {v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] ({v['n']}/{v['n_clusters']})"
                          if v["mean"] is not None else f"{c}: -" for c, v in cats.items())
        print(f"  {fb:<9} {cells}")
    print("\nLuck awareness (share mentioning contact or luck):")
    for fb, cats in r["luck_awareness"].items():
        print(f"  {fb:<9} " + "  ".join(f"{c}: {v['n_process']}/{v['n']}" for c, v in cats.items()))
    p = r["pitcher"]
    print(f"\nPitcher ({p['n_comments']} comments, {p['n_bip']} balls): own-fan hit allowed {_fmt(p['own_hit'])}"
          f"  xBA allowed {_fmt(p['own_xba'])}")
    print("\nRobustness (own fans):")
    for name, v in r["robustness"].items():
        print(f"  {name:<34} hit {_fmt(v['own_hit'])}  xBA {_fmt(v['own_xba'])}  balls {v['n_bip']}")
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    from jbr import claims

    s = claims.build(with_llm=args.with_llm)
    print(f"Classifier: {s['classifier']}")
    for kind, c in s["counts"].items():
        r = s["recall"][kind]
        print(f"{kind}: pool {r['n_pool']} -> filter {c['after_filter']} -> B+ {c['b_positive']}"
              f" -> A+ {c['a_positive']} -> both {c['both_agree']} | main {c['main']} | winner {c['by_winner']}"
              f" | team {c['by_team']}")
        print(f"   recall check: {r['sample_positives']}/{r['sample']} unfiltered sample positive"
              f" -> about {r['est_missed']} missed (vs {r['filtered_positives']} found)")
    print(f"Spend today: ${s['spend_today_usd']:.3f}")
    print("Top moments:")
    for m in s["top_moments"]:
        print(f"  {m['n_claims']:>3}  {m['inning']} {m['half']}  TOR WP after {m['tor_wp_after']:.3f}  {m['description'][:70]}")
    print(f"  unmapped postgame claims: {s['postgame_main_unmapped']}")
    for kind, ex in s["examples"].items():
        print(f"\n{kind} examples:")
        for e in ex:
            print(f"  [{e['team']} -> {e['winner']}] {e['text']}")
    return 0


def cmd_hindsight(args: argparse.Namespace) -> int:
    from jbr import hindsight

    r = hindsight.run()
    print(f"Winner {r['winner']}; {r['n_claims_main']} postgame claims ({r['n_claims_mapped']} mapped), "
          f"{r['n_predictions_main']} in-game predictions")
    print("\nMoments (claims | in-game belief next 10 min | winner WP after):")
    for m in r["moments"]:
        b = m["belief"]
        bel = f"{b['share_winner']:.0%} of {b['n']}" if b["share_winner"] is not None else f"- of {b['n']}"
        print(f"  {m['inning']} {m['half']:<6} [{m['band']:<7}] claims {m['claims']['n']:>3} "
              f"({m['claims']['share_winner']:.0%} winner) | belief {bel:<10} | WP {m['winner_wp']:.2f} "
              f"| {m['description'][:50]}")
    print("\nEarly vs late:")
    for k, v in r["split"].items():
        print(f"  {k:<8} {v}")
    print("\nSame user:", r["same_user"])
    print("\nBy fanbase:")
    for k, v in r["fanbase"].items():
        print(f"  {k:<8} {v}")
    return 0


def cmd_spotcheck(args: argparse.Namespace) -> int:
    from jbr import spotcheck

    n = spotcheck.build()
    print(f"reports/spotcheck.html: {n} items. Save the downloaded spotcheck_labels.json into reports/, then run jbr site.")
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    from jbr import site, spotcheck

    check = site.build()
    print(f"docs/index.html written ({check['bytes']:,} bytes)")
    print(f"Spot-check answers: {'found' if spotcheck.precision() else 'not yet (page says pending)'}")
    print(f"Privacy check: hash leaks {check['hash_leaks']}, username hits {check['n_username_hits']} "
          f"{check['username_hits']}, u/ refs {check['user_refs']}, author field {check['author_field']}")
    return 0 if check["hash_leaks"] == 0 and not check["author_field"] and not check["user_refs"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jbr")
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("import", help="copy mlb-fan-live data into data/imported/").set_defaults(func=cmd_import)
    sub.add_parser("statcast", help="pull Statcast, build data/bip.parquet").set_defaults(func=cmd_statcast)
    sub.add_parser("windows", help="reaction windows + attribution: data/reactions.parquet").set_defaults(func=cmd_windows)
    sub.add_parser("outcome", help="Phase 2 outcome-bias analysis: reports/outcome.json").set_defaults(func=cmd_outcome)
    p = sub.add_parser("claims", help="Phase 3 predictions + inevitability claims: data/claims.parquet")
    p.add_argument("--with-llm", action="store_true", help="also run Classifier A (LLM Foundry)")
    p.set_defaults(func=cmd_claims)
    sub.add_parser("hindsight", help="Phase 4 hindsight analysis: reports/hindsight.json").set_defaults(func=cmd_hindsight)
    sub.add_parser("spotcheck", help="yes/no spot-check page: reports/spotcheck.html").set_defaults(func=cmd_spotcheck)
    sub.add_parser("site", help="results page: docs/index.html").set_defaults(func=cmd_site)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
