import json

import pandas as pd
import pytest

from jbr import claims, config, llm

CFG = config.load()
T0 = pd.Timestamp("2025-10-25 00:10", tz="UTC")

PLAYS = pd.DataFrame({
    "at_bat_index": [0, 1, 2, 3, 4, 5],
    "inning": [1, 1, 6, 6, 6, 7],
    "half": ["top", "bottom", "bottom", "bottom", "bottom", "top"],
    "description": ["Betts grounds out.", "Springer flies out.", "Kirk singles.",
                    "Addison Barger hits a grand slam (3) to center field.", "Alejandro Kirk homers (4).",
                    "Shohei Ohtani homers (6)."],
    "pitcher_name": ["Yesavage", "Snell", "Snell", "Sheehan", "Sheehan", "Little"],
    "wp_delta": [0.01, -0.02, 0.05, 0.30, 0.01, -0.01],
    "end_time_utc": [T0 + pd.Timedelta(minutes=i) for i in range(6)],
})


@pytest.mark.parametrize("text,expected", [
    ("that grand slam ended it", (3, "grand slam")),
    ("Barger is a legend", (3, "grand slam")),
    ("once they pulled Snell it was over", (3, "pitching_change_6")),
    ("knew it from the first pitch", (0, "first_play")),
    ("it was over after the 6th", (4, "end of inning 6")),
    ("the 6th inning was insane", (3, "biggest swing of inning 6")),
    ("what a night", (None, None)),
])
def test_map_moment(text, expected):
    assert claims.map_moment(text, None, PLAYS, CFG) == expected


def test_map_moment_uses_llm_inning_when_text_has_none():
    assert claims.map_moment("we choked", 1, PLAYS, CFG) == (1, "biggest swing of inning 1")


def test_referenced_inning():
    assert claims.referenced_inning("over after the sixth") == 6
    assert claims.referenced_inning("that 6th inning") == 6
    assert claims.referenced_inning("my first time watching") is None


def test_filters_match_expected_phrases():
    ig = claims.keyword_filter(CFG["claims"]["ingame_filter"])
    pg = claims.keyword_filter(CFG["claims"]["postgame_filter"])
    assert ig.search("this game is over") and ig.search("We're cooked") and not ig.search("nice catch")
    assert pg.search("knew it after the 6th") and pg.search("never had a chance") and not pg.search("LETS GO")


def test_label_b_thresholds():
    scores = pd.DataFrame({"b_score": [0.9, 0.97, 0.5], "b_tor": [0.8, 0.3, 0.9], "b_lad": [0.1, 0.2, 0.1]})
    out = claims.label_b(scores, CFG)
    assert out["b_positive"].tolist() == [True, True, False]
    assert out["b_winner"].iloc[0] == "TOR"
    assert out["b_winner"].iloc[1:].isna().all()
    assert out["b_certainty"].tolist()[:2] == ["mid", "high"]


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.content}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


@pytest.fixture
def tmp_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm, "SPEND", tmp_path / "spend.json")
    monkeypatch.setenv("LLMFOUNDRY_TOKEN", "test-token")
    return tmp_path


def test_llm_classify_caches_and_tracks_spend(tmp_runs):
    calls = []

    def post(url, headers, json, timeout):
        calls.append((url, headers))
        return FakeResponse('```json\n[{"id": "c1", "is_prediction": true, "predicted_winner": "TOR", '
                            '"confidence": "high"}]\n```')

    items = [{"id": "c1", "fan": "TOR fan", "text": "we've got this"}]
    out = llm.classify(items, "ingame_prediction", CFG, post=post)
    assert out["c1"]["predicted_winner"] == "TOR"
    assert calls[0][0].endswith("/chat/completions")
    assert calls[0][1]["Authorization"] == f"Bearer test-token:{CFG['llm']['project']}"
    assert llm.spent_today() == pytest.approx((1000 * 0.10 + 200 * 0.40) / 1e6)

    llm.classify(items, "ingame_prediction", CFG, post=post)  # second run: from cache
    assert len(calls) == 1


def test_llm_spend_cap_stops_calls(tmp_runs):
    llm.add_spend(CFG["llm"]["daily_cap_usd"])
    with pytest.raises(llm.SpendCapReached):
        llm.complete("anything", CFG, post=lambda **k: pytest.fail("should not call"))


def test_prompts_have_placeholder():
    for name in ("ingame_prediction", "postgame_claim"):
        assert "{{comments}}" in (config.ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def test_rule_direction():
    assert claims.rule_direction("we got this", "LAD", "TOR", "LAD") == "LAD"
    assert claims.rule_direction("we're cooked", "LAD", None, None) == "TOR"
    assert claims.rule_direction("they're coming back", None, "LAD", "TOR") == "TOR"
    assert claims.rule_direction("game over. Ohtani struck out", "LAD", "TOR", "LAD") == "TOR"
    assert claims.rule_direction("it's over", None, None, None) is None
    assert claims.rule_direction("what a pitch", "TOR", "TOR", "LAD") is None


def test_score_at():
    plays = pd.DataFrame({"end_time_utc": [T0, T0 + pd.Timedelta(minutes=5)],
                          "home_score": [0, 0], "away_score": [0, 2]})
    assert claims.score_at(T0 - pd.Timedelta(minutes=1), plays, "TOR", "LAD") == (None, None)
    assert claims.score_at(T0 + pd.Timedelta(minutes=1), plays, "TOR", "LAD") == (None, None)
    assert claims.score_at(T0 + pd.Timedelta(minutes=6), plays, "TOR", "LAD") == ("LAD", "TOR")
