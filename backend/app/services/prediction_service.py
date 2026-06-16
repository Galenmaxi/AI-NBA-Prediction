from __future__ import annotations

import json
import os

import pandas as pd
from sqlalchemy.orm import Session

from app.models.player_game_log import PlayerGameLog
from app.models.team_game_log import TeamGameLog
from ml.predict import predict_best_player, predict_player_stats, predict_win_probability

_N_GAMES = 20
PREDICT_CACHE_TTL: int = 3600


def _get_redis():
    """Return a Redis client or None if REDIS_URL is not set / connection fails."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        import redis as redis_lib
        client = redis_lib.from_url(redis_url, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


def _cache_get(key: str) -> dict | None:
    r = _get_redis()
    if r is None:
        return None
    val = r.get(key)
    return json.loads(val) if val else None


def _cache_set(key: str, value: dict) -> None:
    r = _get_redis()
    if r is None:
        return
    r.setex(key, PREDICT_CACHE_TTL, json.dumps(value))


def _confidence(home_prob: float) -> str:
    spread = round(abs(home_prob - 0.5), 10)
    if spread >= 0.15:
        return "high"
    if spread >= 0.08:
        return "medium"
    return "low"


def get_win_probability(
    db: Session,
    home_team_id: int,
    away_team_id: int,
) -> dict:
    """Query recent team game logs and return win probability prediction.

    Returns:
        {"home_win_prob": float, "away_win_prob": float, "confidence": str}
    Raises:
        ValueError: if a team has fewer than 2 games in the DB.
        FileNotFoundError: propagated from predict if model not trained.
    """
    cache_key = f"win_prob:{home_team_id}:{away_team_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    def _fetch_team(team_id: int) -> pd.DataFrame:
        rows = (
            db.query(TeamGameLog)
            .filter(TeamGameLog.team_id == team_id)
            .order_by(TeamGameLog.game_date.desc())
            .limit(_N_GAMES)
            .all()
        )
        if len(rows) < 2:
            raise ValueError(
                f"Not enough game data for team {team_id} "
                f"(found {len(rows)} games, need at least 2)."
            )
        return pd.DataFrame([
            {
                "team_id": r.team_id,
                "season": r.season,
                "game_id": r.game_id,
                "game_date": pd.Timestamp(r.game_date),
                "home_away": r.home_away,
                "wl": r.wl,
                "pts": r.pts,
            }
            for r in rows
        ])

    home_df = _fetch_team(home_team_id)
    away_df = _fetch_team(away_team_id)

    probs = predict_win_probability(home_df, away_df)
    result = {**probs, "confidence": _confidence(probs["home_win_prob"])}
    _cache_set(cache_key, result)
    return result


def get_best_player(
    db: Session,
    home_team_id: int,
    away_team_id: int,
) -> list[dict]:
    """Predict which players are most likely to star in the game.

    Returns:
        List of {"player_id": int, "player_name": str, "star_probability": float}
        sorted descending by star_probability.
    Raises:
        ValueError: if combined player data is insufficient.
        FileNotFoundError: propagated from predict if model not trained.
    """
    def _fetch_team_players(team_id: int) -> list:
        return (
            db.query(PlayerGameLog)
            .filter(PlayerGameLog.team_id == team_id)
            .filter(PlayerGameLog.pts.isnot(None))
            .order_by(PlayerGameLog.game_date.desc())
            .limit(_N_GAMES * 20)
            .all()
        )

    home_rows = _fetch_team_players(home_team_id)
    away_rows = _fetch_team_players(away_team_id)
    all_rows = home_rows + away_rows

    if len(all_rows) < 5:
        raise ValueError(
            f"Not enough player data for teams {home_team_id} and {away_team_id}."
        )

    df = pd.DataFrame([
        {
            "player_id": r.player_id,
            "player_name": r.player_name,
            "game_id": r.game_id,
            "game_date": pd.Timestamp(r.game_date),
            "pts": r.pts, "reb": r.reb, "ast": r.ast,
            "stl": r.stl, "blk": r.blk, "min": r.min,
        }
        for r in all_rows
    ])
    return predict_best_player(df)


def get_player_stats(db: Session, player_id: int) -> dict[str, float]:
    """Predict pts/reb/ast for a player's next game.

    Returns:
        {"pts": float, "reb": float, "ast": float}
    Raises:
        ValueError: if the player has no game data in the DB.
        FileNotFoundError: propagated from predict if models not trained.
    """
    rows = (
        db.query(PlayerGameLog)
        .filter(PlayerGameLog.player_id == player_id)
        .filter(PlayerGameLog.pts.isnot(None))
        .order_by(PlayerGameLog.game_date.desc())
        .limit(_N_GAMES)
        .all()
    )
    if not rows:
        raise ValueError(f"No game data found for player {player_id}.")

    df = pd.DataFrame([
        {
            "player_id": r.player_id,
            "player_name": r.player_name,
            "game_id": r.game_id,
            "game_date": pd.Timestamp(r.game_date),
            "pts": r.pts, "reb": r.reb, "ast": r.ast,
            "stl": r.stl, "blk": r.blk, "min": r.min,
        }
        for r in rows
    ])
    return predict_player_stats(df)
