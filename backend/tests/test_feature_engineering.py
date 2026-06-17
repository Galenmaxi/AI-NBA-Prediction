import pytest
import pandas as pd
from datetime import date, timedelta

from ml.feature_engineering import (
    build_team_features,
    build_player_features,
    build_win_model_dataset,
    build_player_star_dataset,
    build_player_stats_dataset,
    build_total_model_dataset,
    train_test_split_by_date,
    TEAM_FEATURE_COLS,
    PLAYER_FEATURE_COLS,
    TOTAL_FEATURE_COLS,
)


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


# ── Player feature tests ──────────────────────────────────────────────────────

def make_player_df(pts_sequence: list[int], player_id: int = 100) -> pd.DataFrame:
    n = len(pts_sequence)
    base_date = date(2024, 10, 1)
    return pd.DataFrame({
        "player_id": [player_id] * n,
        "season": ["2024-25"] * n,
        "game_id": [f"PG{player_id}{i:03d}" for i in range(n)],
        "game_date": pd.to_datetime([base_date + timedelta(days=i * 2) for i in range(n)]),
        "wl": ["W"] * n,
        "pts": pts_sequence,
        "reb": [5] * n,
        "ast": [3] * n,
        "stl": [1] * n,
        "blk": [0] * n,
        "min": [30.0] * n,
    })


def test_build_player_features_first_game_pts_avg_is_nan():
    df = make_player_df([30, 25, 20])
    result = build_player_features(df)
    assert pd.isna(result.iloc[0]["pts_avg_last5"])


def test_build_player_features_pts_avg_uses_only_past():
    pts = [10, 20, 30, 40, 50, 999]
    df = make_player_df(pts)
    result = build_player_features(df)
    # game 6 (index 5): shift(1) then rolling(5) sees games 1-5: [10,20,30,40,50]
    expected = (10 + 20 + 30 + 40 + 50) / 5
    assert result.iloc[5]["pts_avg_last5"] == pytest.approx(expected)


def test_build_player_features_adds_all_expected_columns():
    df = make_player_df([20, 25, 30])
    result = build_player_features(df)
    expected_cols = {
        "pts_avg_last5", "pts_avg_last10",
        "reb_avg_last5", "reb_avg_last10",
        "ast_avg_last5", "ast_avg_last10",
        "stl_avg_last5", "blk_avg_last5", "min_avg_last5",
    }
    assert expected_cols.issubset(set(result.columns))


def test_build_player_features_two_players_independent():
    p1 = make_player_df([100, 100, 100], player_id=1)
    p2 = make_player_df([10, 10, 10], player_id=2)
    result = build_player_features(pd.concat([p1, p2], ignore_index=True))
    r1 = result[result["player_id"] == 1]
    r2 = result[result["player_id"] == 2]
    assert r1.iloc[2]["pts_avg_last5"] == pytest.approx(100.0)
    assert r2.iloc[2]["pts_avg_last5"] == pytest.approx(10.0)


# ── Dataset builder + split tests ─────────────────────────────────────────────

def make_full_team_df(n_games: int = 20) -> pd.DataFrame:
    rows = []
    base = date(2024, 10, 1)
    for team_id in [1, 2]:
        for i in range(n_games):
            rows.append({
                "team_id": team_id,
                "season": "2024-25",
                "game_id": f"G{team_id}{i:03d}",
                "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
                "home_away": "HOME" if team_id == 1 else "AWAY",
                "wl": "W" if i % 2 == 0 else "L",
                "pts": 100 + i,
            })
    return pd.DataFrame(rows)


def make_full_player_df(n_games: int = 20) -> pd.DataFrame:
    rows = []
    base = date(2024, 10, 1)
    for player_id, base_pts in [(1, 30), (2, 20), (3, 15)]:
        for i in range(n_games):
            rows.append({
                "player_id": player_id,
                "season": "2024-25",
                "game_id": f"PG{player_id}{i:03d}",
                "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
                "wl": "W" if i % 2 == 0 else "L",
                "pts": base_pts + i % 5,
                "reb": 5 + i % 3,
                "ast": 3 + i % 4,
                "stl": 1,
                "blk": 0,
                "min": 30.0,
                "team_id": 1,
            })
    return pd.DataFrame(rows)


def test_build_win_model_dataset_returns_expected_columns():
    features = build_team_features(make_full_team_df(15))
    result = build_win_model_dataset(features)
    for col in TEAM_FEATURE_COLS + ["target", "game_id", "team_id", "game_date", "season"]:
        assert col in result.columns, f"Missing column: {col}"


def test_build_win_model_dataset_target_is_binary():
    features = build_team_features(make_full_team_df(15))
    result = build_win_model_dataset(features)
    assert set(result["target"].unique()).issubset({0, 1})


def test_build_win_model_dataset_drops_nan_rows():
    """First game per team has NaN rolling features and must be excluded."""
    features = build_team_features(make_full_team_df(5))
    result = build_win_model_dataset(features)
    assert result[TEAM_FEATURE_COLS].isna().sum().sum() == 0


def test_build_player_star_dataset_has_is_star_column():
    features = build_player_features(make_full_player_df(15))
    result = build_player_star_dataset(features)
    assert "is_star" in result.columns
    assert set(result["is_star"].unique()).issubset({0, 1})


def test_build_player_stats_dataset_returns_expected_columns():
    features = build_player_features(make_full_player_df(15))
    result = build_player_stats_dataset(features)
    for col in PLAYER_FEATURE_COLS + ["pts", "reb", "ast"]:
        assert col in result.columns, f"Missing column: {col}"


def test_train_test_split_by_date_correct_seasons():
    rows = []
    for season, n in [("2023-24", 10), ("2024-25", 10), ("2025-26", 5)]:
        for i in range(n):
            rows.append({"season": season, "game_id": f"{season}-{i}"})
    df = pd.DataFrame(rows)
    train, test = train_test_split_by_date(df, test_season="2025-26")
    assert set(train["season"].unique()) == {"2023-24", "2024-25"}
    assert set(test["season"].unique()) == {"2025-26"}


def test_train_test_split_no_overlap():
    rows = [{"season": "2023-24", "game_id": f"A{i}"} for i in range(5)]
    rows += [{"season": "2025-26", "game_id": f"B{i}"} for i in range(5)]
    df = pd.DataFrame(rows)
    train, test = train_test_split_by_date(df)
    assert len(set(train["game_id"]) & set(test["game_id"])) == 0


# ── Game total dataset tests ───────────────────────────────────────────────────

def make_paired_team_df(n_games: int = 15) -> pd.DataFrame:
    """Home team (id=1) + away team (id=2) sharing the same game_id per game."""
    base = date(2024, 10, 1)
    rows = []
    for i in range(n_games):
        game_id = f"GAME{i:04d}"
        rows.append({
            "team_id": 1, "season": "2024-25", "game_id": game_id,
            "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
            "home_away": "HOME", "wl": "W", "pts": 110 + i,
        })
        rows.append({
            "team_id": 2, "season": "2024-25", "game_id": game_id,
            "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
            "home_away": "AWAY", "wl": "L", "pts": 100 + i,
        })
    return pd.DataFrame(rows)


def test_total_feature_cols_has_home_and_away_prefixes():
    for col in TEAM_FEATURE_COLS:
        assert f"home_{col}" in TOTAL_FEATURE_COLS
        assert f"away_{col}" in TOTAL_FEATURE_COLS


def test_build_total_model_dataset_has_total_target():
    df = make_paired_team_df(n_games=15)
    team_feat = build_team_features(df)
    result = build_total_model_dataset(team_feat)
    assert "total" in result.columns
    assert (result["total"] > 0).all()
    assert set(TOTAL_FEATURE_COLS).issubset(result.columns)
