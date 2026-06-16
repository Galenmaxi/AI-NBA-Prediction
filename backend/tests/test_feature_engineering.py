import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from ml.feature_engineering import build_team_features


def make_team_df(
    wl_sequence: list[str],
    team_id: int = 1,
    pts_sequence: list[int] | None = None,
) -> pd.DataFrame:
    n = len(wl_sequence)
    if pts_sequence is None:
        pts_sequence = [100 + i for i in range(n)]
    base_date = date(2024, 10, 1)
    return pd.DataFrame({
        "team_id": [team_id] * n,
        "season": ["2024-25"] * n,
        "game_id": [f"G{team_id}{i:03d}" for i in range(n)],
        "game_date": pd.to_datetime([base_date + timedelta(days=i * 2) for i in range(n)]),
        "home_away": ["HOME" if i % 2 == 0 else "AWAY" for i in range(n)],
        "wl": wl_sequence,
        "pts": pts_sequence,
    })


def test_build_team_features_adds_is_home():
    df = make_team_df(["W", "L"])
    result = build_team_features(df)
    assert result.iloc[0]["is_home"] == 1
    assert result.iloc[1]["is_home"] == 0


def test_build_team_features_first_game_win_pct_is_nan():
    df = make_team_df(["W", "W", "W"])
    result = build_team_features(df)
    assert pd.isna(result.iloc[0]["win_pct_last10"])


def test_build_team_features_win_pct_uses_only_past_games():
    """Game 5 (a loss) must have win_pct = 1.0 from its prior 4 wins."""
    df = make_team_df(["W", "W", "W", "W", "L"])
    result = build_team_features(df)
    assert result.iloc[4]["win_pct_last10"] == pytest.approx(1.0)
    assert result.iloc[4]["wl"] == "L"


def test_build_team_features_pts_avg_last5_uses_only_past():
    pts = [100, 110, 120, 130, 140, 999]
    df = make_team_df(["W"] * 6, pts_sequence=pts)
    result = build_team_features(df)
    expected = (100 + 110 + 120 + 130 + 140) / 5
    assert result.iloc[5]["pts_avg_last5"] == pytest.approx(expected)


def test_build_team_features_rest_days_first_game():
    df = make_team_df(["W", "L"])
    result = build_team_features(df)
    assert result.iloc[0]["rest_days"] == pytest.approx(7.0)


def test_build_team_features_rest_days_subsequent():
    df = make_team_df(["W", "L"])  # games 2 days apart (timedelta days=2)
    result = build_team_features(df)
    assert result.iloc[1]["rest_days"] == pytest.approx(2.0)


def test_build_team_features_season_win_pct_uses_only_past():
    df = make_team_df(["W", "W", "L", "W"])
    result = build_team_features(df)
    # game 4 sees history: W, W, L → 2/3 ≈ 0.667
    assert result.iloc[3]["season_win_pct"] == pytest.approx(2 / 3, abs=0.01)


def test_build_team_features_preserves_original_columns():
    df = make_team_df(["W", "L", "W"])
    result = build_team_features(df)
    for col in ["team_id", "season", "game_id", "game_date", "wl", "pts"]:
        assert col in result.columns


def test_build_team_features_two_teams_independent():
    team1 = make_team_df(["W", "W", "W"], team_id=1)
    team2 = make_team_df(["L", "L", "L"], team_id=2)
    result = build_team_features(pd.concat([team1, team2], ignore_index=True))
    t1 = result[result["team_id"] == 1]
    t2 = result[result["team_id"] == 2]
    assert t1.iloc[2]["win_pct_last10"] == pytest.approx(1.0)
    assert t2.iloc[2]["win_pct_last10"] == pytest.approx(0.0)
