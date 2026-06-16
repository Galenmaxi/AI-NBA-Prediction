import pandas as pd

TEAM_FEATURE_COLS: list[str] = [
    "is_home",
    "rest_days",
    "win_pct_last10",
    "pts_avg_last5",
    "pts_avg_last10",
    "season_win_pct",
]

PLAYER_FEATURE_COLS: list[str] = [
    "pts_avg_last5", "pts_avg_last10",
    "reb_avg_last5", "reb_avg_last10",
    "ast_avg_last5", "ast_avg_last10",
    "stl_avg_last5",
    "blk_avg_last5",
    "min_avg_last5",
]


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling team features to a team game log DataFrame.

    Uses shift(1)-before-roll so no current-game data leaks into features.
    All rolling stats reflect only games played strictly before the current game.
    """
    df = df.copy()
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    df["is_home"] = (df["home_away"] == "HOME").astype(int)
    df["win"] = (df["wl"] == "W").astype(int)

    df["rest_days"] = (
        df.groupby("team_id")["game_date"]
        .transform(lambda x: x.diff().dt.days.fillna(7.0))
    )
    df["win_pct_last10"] = (
        df.groupby("team_id")["win"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    df["pts_avg_last5"] = (
        df.groupby("team_id")["pts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["pts_avg_last10"] = (
        df.groupby("team_id")["pts"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    df["season_win_pct"] = (
        df.groupby(["team_id", "season"])["win"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean().fillna(0.5))
    )

    return df


def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling player features to a player game log DataFrame.

    Uses shift(1)-before-roll for zero data leakage.
    NaN stats (DNP games) are preserved; callers should filter as needed.
    """
    df = df.copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    _roll_stats: dict[str, list[int]] = {
        "pts": [5, 10],
        "reb": [5, 10],
        "ast": [5, 10],
        "stl": [5],
        "blk": [5],
        "min": [5],
    }

    for stat, windows in _roll_stats.items():
        if stat not in df.columns:
            continue
        for w in windows:
            df[f"{stat}_avg_last{w}"] = (
                df.groupby("player_id")[stat]
                .transform(lambda x, window=w: x.shift(1).rolling(window, min_periods=1).mean())
            )

    return df
