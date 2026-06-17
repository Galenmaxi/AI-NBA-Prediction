from __future__ import annotations

import pytest
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.base import Base
from app.models.team_game_log import TeamGameLog
from app.models.player_game_log import PlayerGameLog
from app.services.prediction_service import (
    get_win_probability,
    get_best_player,
    get_player_stats,
    get_game_total,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _add_team_games(db: Session, team_id: int, n: int = 15) -> None:
    base = date(2024, 10, 1)
    for i in range(n):
        db.add(TeamGameLog(
            season="2024-25",
            team_id=team_id,
            team_abbreviation="TST",
            team_name="Test Team",
            game_id=f"T{team_id}G{i:04d}",
            game_date=base + timedelta(days=i * 3),
            matchup="TST vs. OPP",
            home_away="HOME" if i % 2 == 0 else "AWAY",
            wl="W" if i % 2 == 0 else "L",
            pts=100 + i,
            fgm=40, fga=85, fg_pct=0.47,
            fg3m=12, fg3a=30, fg3_pct=0.40,
            ftm=8, fta=10, ft_pct=0.80,
            oreb=8, dreb=32, reb=40,
            ast=22, tov=12, stl=7, blk=5, plus_minus=5.0,
        ))
    db.commit()


def _add_player_games(db: Session, player_id: int, team_id: int, n: int = 15) -> None:
    base = date(2024, 10, 1)
    for i in range(n):
        db.add(PlayerGameLog(
            season="2024-25",
            player_id=player_id,
            player_name=f"Player {player_id}",
            team_id=team_id,
            team_abbreviation="TST",
            game_id=f"P{player_id}G{i:04d}",
            game_date=base + timedelta(days=i * 3),
            matchup="TST vs. OPP",
            home_away="HOME" if i % 2 == 0 else "AWAY",
            wl="W" if i % 2 == 0 else "L",
            min=32.0,
            pts=20 + i % 10,
            reb=5,
            ast=4,
            stl=1,
            blk=0,
            tov=2,
            fgm=8, fga=16, fg_pct=0.50,
            fg3m=2, fg3a=5, fg3_pct=0.40,
            ftm=2, fta=2, ft_pct=1.0,
            plus_minus=3.0,
        ))
    db.commit()


# --- get_win_probability ---

def test_get_win_probability_calls_predict_and_returns_result(db):
    _add_team_games(db, team_id=1, n=15)
    _add_team_games(db, team_id=2, n=15)

    with patch("app.services.prediction_service.predict_win_probability") as mock_pred:
        mock_pred.return_value = {"home_win_prob": 0.65, "away_win_prob": 0.35}
        result = get_win_probability(db, home_team_id=1, away_team_id=2)

    mock_pred.assert_called_once()
    assert result["home_win_prob"] == 0.65
    assert result["away_win_prob"] == 0.35


def test_get_win_probability_adds_confidence_field(db):
    _add_team_games(db, team_id=1)
    _add_team_games(db, team_id=2)

    with patch("app.services.prediction_service.predict_win_probability") as mock_pred:
        mock_pred.return_value = {"home_win_prob": 0.70, "away_win_prob": 0.30}
        result = get_win_probability(db, home_team_id=1, away_team_id=2)

    assert result["confidence"] == "high"


def test_get_win_probability_confidence_medium(db):
    _add_team_games(db, team_id=1)
    _add_team_games(db, team_id=2)

    with patch("app.services.prediction_service.predict_win_probability") as mock_pred:
        mock_pred.return_value = {"home_win_prob": 0.58, "away_win_prob": 0.42}
        result = get_win_probability(db, home_team_id=1, away_team_id=2)

    assert result["confidence"] == "medium"


def test_get_win_probability_confidence_low(db):
    _add_team_games(db, team_id=1)
    _add_team_games(db, team_id=2)

    with patch("app.services.prediction_service.predict_win_probability") as mock_pred:
        mock_pred.return_value = {"home_win_prob": 0.52, "away_win_prob": 0.48}
        result = get_win_probability(db, home_team_id=1, away_team_id=2)

    assert result["confidence"] == "low"


def test_get_win_probability_raises_if_team_has_no_data(db):
    _add_team_games(db, team_id=1)
    with pytest.raises(ValueError, match="Not enough game data"):
        get_win_probability(db, home_team_id=1, away_team_id=99)


# --- get_best_player ---

def test_get_best_player_returns_sorted_list(db):
    for player_id in range(1, 6):
        _add_player_games(db, player_id=player_id, team_id=1)

    with patch("app.services.prediction_service.predict_best_player") as mock_pred:
        mock_pred.return_value = [
            {"player_id": 1, "player_name": "Player 1", "star_probability": 0.8},
            {"player_id": 2, "player_name": "Player 2", "star_probability": 0.3},
        ]
        result = get_best_player(db, home_team_id=1, away_team_id=1)

    assert result[0]["player_id"] == 1
    assert result[0]["star_probability"] == 0.8


# --- get_player_stats ---

def test_get_player_stats_returns_stat_predictions(db):
    _add_player_games(db, player_id=7, team_id=1)

    with patch("app.services.prediction_service.predict_player_stats") as mock_pred:
        mock_pred.return_value = {"pts": 22.0, "reb": 5.0, "ast": 3.5}
        result = get_player_stats(db, player_id=7)

    assert result == {"pts": 22.0, "reb": 5.0, "ast": 3.5}


def test_get_player_stats_raises_if_player_has_no_data(db):
    with pytest.raises(ValueError, match="No game data found for player"):
        get_player_stats(db, player_id=999)


# --- get_game_total ---

def test_get_game_total_returns_predicted_total_and_confidence(db):
    _add_team_games(db, team_id=1)
    _add_team_games(db, team_id=2)

    with patch("app.services.prediction_service.predict_game_total") as mock_pred:
        mock_pred.return_value = {"predicted_total": 221.5}
        result = get_game_total(db, home_team_id=1, away_team_id=2)

    assert result["predicted_total"] == 221.5
    assert result["confidence"] in {"high", "medium", "low"}
