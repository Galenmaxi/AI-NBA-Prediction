import time
import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs, PlayerGameLogs

SEASONS: list[str] = ["2023-24", "2024-25", "2025-26"]
REQUEST_DELAY: float = 0.6

_TEAM_COLUMNS: list[str] = [
    "season", "team_id", "team_abbreviation", "team_name",
    "game_id", "game_date", "matchup", "home_away", "wl",
    "pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
    "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
    "ast", "tov", "stl", "blk", "plus_minus",
]

_PLAYER_COLUMNS: list[str] = [
    "season", "player_id", "player_name", "team_id", "team_abbreviation",
    "game_id", "game_date", "matchup", "home_away", "wl",
    "min", "pts", "reb", "ast", "stl", "blk", "tov",
    "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
    "ftm", "fta", "ft_pct", "plus_minus",
]


def clean_team_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"season_year": "season"})
    df["home_away"] = df["matchup"].apply(
        lambda x: "HOME" if "vs." in str(x) else "AWAY"
    )
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df.dropna(subset=["game_id", "team_id", "game_date"])
    return df[[c for c in _TEAM_COLUMNS if c in df.columns]]


def fetch_team_game_logs(season: str) -> pd.DataFrame:
    logs = TeamGameLogs(season_nullable=season, timeout=30)
    df = logs.get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    return clean_team_game_logs(df)


def clean_player_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"season_year": "season"})
    df["home_away"] = df["matchup"].apply(
        lambda x: "HOME" if "vs." in str(x) else "AWAY"
    )
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed")
    df = df.dropna(subset=["game_id", "player_id", "game_date", "player_name"])
    return df[[c for c in _PLAYER_COLUMNS if c in df.columns]]


def fetch_player_game_logs(season: str) -> pd.DataFrame:
    logs = PlayerGameLogs(season_nullable=season, timeout=30)
    df = logs.get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    return clean_player_game_logs(df)
