import pandas as pd
import pytest

from jbr import config, statcast

CFG = config.load()
T0 = pd.Timestamp("2025-10-25 00:10", tz="UTC")


def sc_rows(rows):
    base = {c: None for c in statcast.STATCAST_COLUMNS}
    return pd.DataFrame([{**base, "game_pk": 1, "pitch_number": 1, **r} for r in rows])


def plays_and_wp(n, batters=None):
    plays = pd.DataFrame({
        "game_pk": 1, "at_bat_index": range(n), "inning": 1, "half": "top",
        "end_time_utc": [T0 + pd.Timedelta(minutes=i) for i in range(n)],
        "description": [f"play {i}" for i in range(n)],
        "batter_id": batters or list(range(100, 100 + n)), "batter_name": "B",
        "pitcher_id": 9, "pitcher_name": "P",
    })
    wp = pd.DataFrame({"game_pk": 1, "at_bat_index": range(n), "home_wp_before": 0.6, "home_wp_after": 0.5})
    return plays, wp


OUT_MIN = CFG["luck"]["unlucky_out_min_xba"]
HIT_MAX = CFG["luck"]["lucky_hit_max_xba"]


@pytest.mark.parametrize("hit,xba,expected", [
    (0, OUT_MIN, "unlucky_out"), (0, OUT_MIN - 0.01, "expected_out"),
    (1, HIT_MAX, "lucky_hit"), (1, HIT_MAX + 0.01, "expected_hit"), (1, None, None),
])
def test_luck_category_thresholds(hit, xba, expected):
    assert statcast.luck_category(hit, xba, False, False, CFG) == expected


def test_errors_and_fielders_choice_get_no_category():
    assert statcast.luck_category(0, 0.9, True, False, CFG) is None
    assert statcast.luck_category(0, 0.9, False, True, CFG) is None


def test_batted_balls_drop_strikeouts_walks_and_fouls():
    sc = sc_rows([
        {"at_bat_number": 1, "events": "strikeout"},
        {"at_bat_number": 2, "events": "walk"},
        {"at_bat_number": 3, "events": None, "description": "foul", "launch_speed": 90.0},
        {"at_bat_number": 3, "events": "single", "pitch_number": 2},
        {"at_bat_number": 4, "events": "hit_by_pitch"},
    ])
    assert statcast.batted_balls(sc, CFG)["at_bat_number"].tolist() == [3]


def test_join_offsets_at_bat_number_by_one_and_flips_wp_for_away_batter():
    sc = sc_rows([{"at_bat_number": 2, "events": "single", "inning": 1, "inning_topbot": "Top",
                   "batter": 101, "pitcher": 9, "estimated_ba_using_speedangle": 0.2}])
    plays, wp = plays_and_wp(3)
    bip = statcast.build_bip(sc, plays, wp, CFG)
    row = bip.iloc[0]
    assert row["description"] == "play 1"
    assert row["play_end_ts"] == T0 + pd.Timedelta(minutes=1)
    assert row["batter_team"] == "LAD" and row["pitcher_team"] == "TOR"
    assert row["wp_before"] == pytest.approx(0.4) and row["delta_wp_batting_team"] == pytest.approx(0.1)
    assert row["luck"] == pytest.approx(0.8) and row["luck_category"] == "lucky_hit"


def test_join_raises_when_batter_disagrees():
    sc = sc_rows([{"at_bat_number": 1, "events": "field_out", "inning": 1, "inning_topbot": "Top",
                   "batter": 999, "pitcher": 9}])
    plays, wp = plays_and_wp(1)
    with pytest.raises(ValueError, match="batter"):
        statcast.build_bip(sc, plays, wp, CFG)
