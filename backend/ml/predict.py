from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    PLAYER_FEATURE_COLS,
    TEAM_FEATURE_COLS,
    TOTAL_FEATURE_COLS,
    build_player_features,
    build_team_features,
)
from ml.train_player_model import BEST_PLAYER_MODEL_PATH, STAT_MODEL_PATHS, STAT_TARGETS
from ml.train_total_model import TOTAL_MODEL_PATH
from ml.train_win_model import WIN_MODEL_PATH

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


def clear_model_cache() -> None:
    _model_cache.clear()


def _load(path: Path) -> object:
    key = str(path)
    if key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run training scripts first."
            )
        _model_cache[key] = joblib.load(path)
    return _model_cache[key]


def _add_team_dummy_row(df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
    """Append a synthetic upcoming-game row so shift(1) rolling includes all history."""
    last_date = df["game_date"].max()
    dummy = {
        "team_id": int(df["team_id"].iloc[0]),
        "season": df["season"].iloc[0],
        "game_id": "PREDICT",
        "game_date": last_date + pd.Timedelta(days=1),
        "home_away": "HOME" if is_home else "AWAY",
        "wl": "W",
        "pts": 0,
    }
    return pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)


def _add_player_dummy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append one synthetic upcoming-game row per player."""
    last_dates = df.groupby("player_id")["game_date"].max()
    name_map: dict = (
        df[["player_id", "player_name"]].drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
        .to_dict()
    ) if "player_name" in df.columns else {}
    dummies = []
    for player_id, last_date in last_dates.items():
        dummies.append({
            "player_id": player_id,
            "player_name": name_map.get(player_id, ""),
            "game_id": "PREDICT",
            "game_date": last_date + pd.Timedelta(days=1),
            "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0, "min": 0.0,
        })
    return pd.concat([df, pd.DataFrame(dummies)], ignore_index=True)


def predict_win_probability(
    home_team_df: pd.DataFrame,
    away_team_df: pd.DataFrame,
) -> dict[str, float]:
    """Return home and away win probabilities for an upcoming matchup.

    Args:
        home_team_df: Historical game logs for the home team.
            Required columns: team_id, season, game_id, game_date, home_away, wl, pts.
        away_team_df: Same for the away team.

    Returns:
        {"home_win_prob": float, "away_win_prob": float}

    Raises:
        FileNotFoundError: if the win model .joblib has not been trained yet.
    """
    model = _load(WIN_MODEL_PATH)

    home_with_dummy = _add_team_dummy_row(home_team_df, is_home=True)
    away_with_dummy = _add_team_dummy_row(away_team_df, is_home=False)

    home_feat = build_team_features(home_with_dummy)
    away_feat = build_team_features(away_with_dummy)

    home_row = (
        home_feat[home_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS]
        .iloc[0].values.reshape(1, -1)
    )
    away_row = (
        away_feat[away_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS]
        .iloc[0].values.reshape(1, -1)
    )

    home_prob = float(model.predict_proba(home_row)[0][1])
    away_prob = float(model.predict_proba(away_row)[0][1])

    return {
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
    }


def predict_best_player(players_df: pd.DataFrame) -> list[dict]:
    """Rank players by their probability of being the star performer.

    Args:
        players_df: Historical game logs for all players who will play.
            Required columns: player_id, player_name, game_id, game_date,
            pts, reb, ast, stl, blk, min.

    Returns:
        List of {"player_id": int, "player_name": str, "star_probability": float}
        sorted descending by star_probability.

    Raises:
        FileNotFoundError: if the best player model .joblib has not been trained yet.
    """
    model = _load(BEST_PLAYER_MODEL_PATH)

    df_with_dummy = _add_player_dummy_rows(players_df)
    feat = build_player_features(df_with_dummy)

    latest = feat[feat["game_id"] == "PREDICT"].dropna(subset=PLAYER_FEATURE_COLS).copy()
    if latest.empty:
        return []

    X = latest[PLAYER_FEATURE_COLS].values
    probs = model.predict_proba(X)[:, 1]

    results = [
        {
            "player_id": int(row["player_id"]),
            "player_name": str(row.get("player_name", "")),
            "star_probability": round(float(probs[idx]), 4),
        }
        for idx, (_, row) in enumerate(latest.iterrows())
    ]
    return sorted(results, key=lambda x: x["star_probability"], reverse=True)


def predict_game_total(
    home_team_df: pd.DataFrame,
    away_team_df: pd.DataFrame,
) -> dict[str, float]:
    """Predict total points scored in a game (home + away).

    Args:
        home_team_df: Historical game logs for the home team.
            Required columns: team_id, season, game_id, game_date, home_away, wl, pts.
        away_team_df: Same for the away team.

    Returns:
        {"predicted_total": float}

    Raises:
        FileNotFoundError: if the total model .joblib has not been trained yet.
    """
    model = _load(TOTAL_MODEL_PATH)

    home_with_dummy = _add_team_dummy_row(home_team_df, is_home=True)
    away_with_dummy = _add_team_dummy_row(away_team_df, is_home=False)

    home_feat = build_team_features(home_with_dummy)
    away_feat = build_team_features(away_with_dummy)

    home_row = home_feat[home_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS].iloc[0]
    away_row = away_feat[away_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS].iloc[0]

    combined = {f"home_{c}": home_row[c] for c in TEAM_FEATURE_COLS}
    combined.update({f"away_{c}": away_row[c] for c in TEAM_FEATURE_COLS})

    X = pd.DataFrame([combined])[TOTAL_FEATURE_COLS].values
    return {"predicted_total": round(float(model.predict(X)[0]), 1)}


def predict_player_stats(player_df: pd.DataFrame) -> dict[str, float]:
    """Predict pts/reb/ast for a player's next game.

    Args:
        player_df: Historical game logs for one player.
            Required columns: player_id, game_id, game_date, pts, reb, ast, stl, blk, min.

    Returns:
        {"pts": float, "reb": float, "ast": float}

    Raises:
        FileNotFoundError: if any stat model .joblib has not been trained yet.
        ValueError: if there is not enough history to build features.
    """
    df_with_dummy = _add_player_dummy_rows(player_df)
    feat = build_player_features(df_with_dummy)

    predict_rows = feat[feat["game_id"] == "PREDICT"]
    if predict_rows.empty or predict_rows[PLAYER_FEATURE_COLS].isna().any().any():
        raise ValueError("Not enough historical data to build features for this player.")

    X = predict_rows[PLAYER_FEATURE_COLS].iloc[0].values.reshape(1, -1)
    result: dict[str, float] = {}
    for stat in STAT_TARGETS:
        m = _load(STAT_MODEL_PATHS[stat])
        result[stat] = round(float(m.predict(X)[0]), 1)
    return result
