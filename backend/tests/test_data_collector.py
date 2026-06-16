import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from ml.data_collector import (
    clean_team_game_logs,
    fetch_team_game_logs,
    clean_player_game_logs,
    fetch_player_game_logs,
    SEASONS,
)


def make_raw_team_df() -> pd.DataFrame:
    return pd.DataFrame({
        "SEASON_YEAR": ["2022-23", "2022-23"],
        "TEAM_ID": [1610612738, 1610612752],
        "TEAM_ABBREVIATION": ["BOS", "NYK"],
        "TEAM_NAME": ["Boston Celtics", "New York Knicks"],
        "GAME_ID": ["0022300001", "0022300001"],
        "GAME_DATE": ["OCT 18, 2022", "OCT 18, 2022"],
        "MATCHUP": ["BOS vs. NYK", "NYK @ BOS"],
        "WL": ["W", "L"],
        "MIN": [240.0, 240.0],
        "PTS": [120, 100],
        "FGM": [45, 38], "FGA": [90, 85], "FG_PCT": [0.5, 0.447],
        "FG3M": [12, 8], "FG3A": [30, 25], "FG3_PCT": [0.4, 0.32],
        "FTM": [18, 16], "FTA": [22, 20], "FT_PCT": [0.818, 0.8],
        "OREB": [10, 8], "DREB": [35, 30], "REB": [45, 38],
        "AST": [28, 22], "TOV": [12, 15], "STL": [8, 6], "BLK": [5, 4],
        "PLUS_MINUS": [20.0, -20.0],
    })


def test_clean_team_game_logs_adds_home_away():
    result = clean_team_game_logs(make_raw_team_df())
    bos = result[result["team_id"] == 1610612738].iloc[0]
    nyk = result[result["team_id"] == 1610612752].iloc[0]
    assert bos["home_away"] == "HOME"
    assert nyk["home_away"] == "AWAY"


def test_clean_team_game_logs_renames_season_column():
    result = clean_team_game_logs(make_raw_team_df())
    assert "season" in result.columns
    assert "SEASON_YEAR" not in result.columns


def test_clean_team_game_logs_lowercases_all_columns():
    result = clean_team_game_logs(make_raw_team_df())
    for col in result.columns:
        assert col == col.lower(), f"Column '{col}' should be lowercase"


def test_clean_team_game_logs_parses_game_date():
    result = clean_team_game_logs(make_raw_team_df())
    assert pd.api.types.is_datetime64_any_dtype(result["game_date"])


def test_clean_team_game_logs_drops_rows_missing_game_id():
    raw = make_raw_team_df()
    raw.loc[0, "GAME_ID"] = None
    result = clean_team_game_logs(raw)
    assert len(result) == 1


def test_seasons_has_three_entries_in_correct_format():
    assert len(SEASONS) == 3
    for s in SEASONS:
        assert len(s) == 7 and s[4] == "-", f"Season '{s}' must be in YYYY-YY format"


@patch("ml.data_collector.TeamGameLogs")
def test_fetch_team_game_logs_calls_api_with_correct_args(mock_cls):
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = [make_raw_team_df()]
    mock_cls.return_value = mock_instance

    result = fetch_team_game_logs("2022-23")

    mock_cls.assert_called_once_with(season_nullable="2022-23", timeout=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


@patch("ml.data_collector.TeamGameLogs")
def test_fetch_team_game_logs_returns_cleaned_df(mock_cls):
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = [make_raw_team_df()]
    mock_cls.return_value = mock_instance

    result = fetch_team_game_logs("2022-23")

    assert "home_away" in result.columns
    assert "season" in result.columns
    for col in result.columns:
        assert col == col.lower()


# ── Player log tests ──────────────────────────────────────────────────────────

def make_raw_player_df() -> pd.DataFrame:
    return pd.DataFrame({
        "SEASON_YEAR": ["2022-23"],
        "PLAYER_ID": [1629029],
        "PLAYER_NAME": ["Jayson Tatum"],
        "NICKNAME": ["JT"],
        "TEAM_ID": [1610612738],
        "TEAM_ABBREVIATION": ["BOS"],
        "TEAM_NAME": ["Boston Celtics"],
        "GAME_ID": ["0022300001"],
        "GAME_DATE": ["OCT 18, 2022"],
        "MATCHUP": ["BOS vs. NYK"],
        "WL": ["W"],
        "MIN": [38.5],
        "FGM": [13], "FGA": [25], "FG_PCT": [0.52],
        "FG3M": [4], "FG3A": [10], "FG3_PCT": [0.4],
        "FTM": [2], "FTA": [2], "FT_PCT": [1.0],
        "OREB": [1], "DREB": [7], "REB": [8],
        "AST": [5], "TOV": [3], "STL": [2], "BLK": [1],
        "BLKA": [0], "PF": [2], "PFD": [3],
        "PTS": [32], "PLUS_MINUS": [15.0],
        "NBA_FANTASY_PTS": [55.0], "DD2": [1], "TD3": [0],
    })


def test_clean_player_game_logs_selects_correct_columns():
    result = clean_player_game_logs(make_raw_player_df())
    expected = {
        "season", "player_id", "player_name", "team_id", "team_abbreviation",
        "game_id", "game_date", "matchup", "home_away", "wl",
        "min", "pts", "reb", "ast", "stl", "blk", "tov",
        "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct", "plus_minus",
    }
    assert expected.issubset(set(result.columns))
    assert "nickname" not in result.columns
    assert "nba_fantasy_pts" not in result.columns


def test_clean_player_game_logs_adds_home_away():
    result = clean_player_game_logs(make_raw_player_df())
    assert result.iloc[0]["home_away"] == "HOME"


def test_clean_player_game_logs_handles_null_minutes():
    raw = make_raw_player_df()
    raw.loc[0, "MIN"] = None
    result = clean_player_game_logs(raw)
    assert pd.isna(result.iloc[0]["min"])


@patch("ml.data_collector.PlayerGameLogs")
def test_fetch_player_game_logs_calls_api_with_correct_args(mock_cls):
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = [make_raw_player_df()]
    mock_cls.return_value = mock_instance

    result = fetch_player_game_logs("2022-23")

    mock_cls.assert_called_once_with(season_nullable="2022-23", timeout=30)
    assert isinstance(result, pd.DataFrame)
    assert "home_away" in result.columns
