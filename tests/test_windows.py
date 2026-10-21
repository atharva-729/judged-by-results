import pandas as pd

from jbr import attribute, config, windows

CFG = config.load()
T0 = pd.Timestamp("2025-10-25 00:10", tz="UTC")


def at(seconds):
    return T0 + pd.Timedelta(seconds=seconds)


def test_next_play_gaps():
    bip = pd.DataFrame({"play_end_ts": [at(0), at(300)]})
    plays = pd.DataFrame({"end_time_utc": [at(0), at(100), at(300)]})
    gaps = windows.next_play_gaps(bip, plays)
    assert gaps.iloc[0] == 100 and pd.isna(gaps.iloc[1])


def test_flag_windows_start_tail_and_next_play_cut():
    df = pd.DataFrame({
        "stream": "s", "offset_s": [10, 15, 100, 120, 140, 170], "next_gap_s": 100.0,
    })
    f = windows.flag_windows(df, {"s": 150}, CFG)
    # start 15; next play ends at 100, so cut at 115; tail at 150.
    assert f["in_main"].tolist() == [False, True, True, False, False, False]
    assert f["in_nocut"].tolist() == [False, True, True, True, True, False]
    assert f["in_60s"].tolist() == [False, True, False, False, False, False]


def test_last_play_has_no_cut():
    df = pd.DataFrame({"stream": "s", "offset_s": [140], "next_gap_s": [float("nan")]})
    assert windows.flag_windows(df, {"s": 150}, CFG)["in_main"].item()


def test_fanbase():
    assert attribute.fanbase("TOR", "TOR") == "own"
    assert attribute.fanbase("LAD", "TOR") == "opponent"
    assert attribute.fanbase(None, "TOR") == "neutral"


def test_process_pattern_matches_whole_phrases_only():
    p = attribute.process_pattern(CFG["process_phrases"])
    assert p.search("he SMOKED that ball")
    assert p.search("right at him, unlucky")
    assert p.search("seeing-eye single")
    assert not p.search("flared up")          # 'flare' inside another word
    assert not p.search("smokedsalmon")


def test_attribution_prefers_jev_then_name_then_reply():
    bip = pd.DataFrame({"bip_id": ["b"], "batter": ["Kirk"]})
    cand = pd.DataFrame({
        "bip_id": "b", "comment_id": ["c1", "c2", "c3", "c4"],
        "parent_id": ["t3_x", "t3_x", "t1_c2", "t1_zz"],
    })
    jev = pd.DataFrame({"comment_id": ["c1", "c2"], "subject": ["Kirk", "other"]})
    name = pd.DataFrame({"comment_id": ["c1", "c2"], "subject": ["Kirk", "Kirk"]})
    out = attribute.attribute(cand, bip, jev, name, "batter")
    assert dict(zip(out["comment_id"], out["attribution_method"])) == {"c1": "jev", "c2": "name", "c3": "reply"}


def test_excerpt_strips_users_links_and_embeds():
    from jbr.outcome import excerpt

    out = excerpt("u/someone look ![gif](giphy|abc) https://i.redd.it/x.gif nice", 200)
    assert "someone" not in out and "redd.it" not in out and "giphy" not in out
    assert out == "[user] look [image] [link] nice"
