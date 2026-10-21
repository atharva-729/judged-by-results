import pandas as pd

from jbr import config, hindsight

CFG = config.load()
T0 = pd.Timestamp("2025-10-25 01:00", tz="UTC")


def at(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def test_band():
    assert hindsight.band(0.5, CFG) == "early"
    assert hindsight.band(0.95, CFG) == "decided"
    assert hindsight.band(0.8, CFG) == "between"


def test_belief_counts_only_the_window():
    preds = pd.DataFrame({"created_utc": [at(-1), at(1), at(5), at(11)],
                          "winner": ["TOR", "LAD", "TOR", "TOR"], "team": ["TOR", "LAD", "TOR", "TOR"]})
    b = hindsight.belief(preds, T0, 10, "TOR")
    assert b["n"] == 2 and b["share_winner"] == 0.5
    assert b["LAD_n"] == 1 and b["LAD_share_winner"] == 0.0


def test_same_user_compares_then_and_later():
    claims_pg = pd.DataFrame({"author_hash": ["u1", "u2", "u3"], "winner": ["TOR", "TOR", "TOR"],
                              "moment_play": [5, 5, 5], "moment_ts": [T0, T0, T0]})
    preds = pd.DataFrame({"author_hash": ["u1", "u2", "u3"], "created_utc": [at(2), at(-30), at(3)],
                          "winner": ["LAD", "TOR", "TOR"], "team": ["LAD", "TOR", "TOR"]})
    s = hindsight.same_user(claims_pg, preds, 10)
    assert s["near_moment"] == {"users": 2, "pairs": 2, "said_same_then": 1, "said_other_team_then": 1}
    assert s["latest_prediction_before"]["users"] == 3
