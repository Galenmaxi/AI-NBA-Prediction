import pytest
from datetime import date
from typing import Type
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.base import Base
from app.models.team_game_log import TeamGameLog
from app.models.player_game_log import PlayerGameLog
from scripts.seed_database import (
    insert_team_game_logs,
    insert_player_game_logs,
    count_rows,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def make_team_df() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2022-23", "2022-23"],
        "team_id": [1610612738, 1610612752],
        "team_abbreviation": ["BOS", "NYK"],
        "team_name": ["Boston Celtics", "New York Knicks"],
        "game_id": ["0022300001", "0022300001"],
        "game_date": [date(2022, 10, 18), date(2022, 10, 18)],
        "matchup": ["BOS vs. NYK", "NYK @ BOS"],
        "home_away": ["HOME", "AWAY"],
        "wl": ["W", "L"],
        "pts": [120, 100],
        "fgm": [45, 38], "fga": [90, 85], "fg_pct": [0.5, 0.447],
        "fg3m": [12, 8], "fg3a": [30, 25], "fg3_pct": [0.4, 0.32],
        "ftm": [18, 16], "fta": [22, 20], "ft_pct": [0.818, 0.8],
        "oreb": [10, 8], "dreb": [35, 30], "reb": [45, 38],
        "ast": [28, 22], "tov": [12, 15], "stl": [8, 6], "blk": [5, 4],
        "plus_minus": [20.0, -20.0],
    })


def make_player_df() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2022-23"],
        "player_id": [1629029],
        "player_name": ["Jayson Tatum"],
        "team_id": [1610612738],
        "team_abbreviation": ["BOS"],
        "game_id": ["0022300001"],
        "game_date": [date(2022, 10, 18)],
        "matchup": ["BOS vs. NYK"],
        "home_away": ["HOME"],
        "wl": ["W"],
        "min": [38.5],
        "pts": [32], "reb": [8], "ast": [5], "stl": [2], "blk": [1], "tov": [3],
        "fgm": [13], "fga": [25], "fg_pct": [0.52],
        "fg3m": [4], "fg3a": [10], "fg3_pct": [0.4],
        "ftm": [2], "fta": [2], "ft_pct": [1.0],
        "plus_minus": [15.0],
    })


def test_insert_team_game_logs_inserts_rows(db_session):
    inserted = insert_team_game_logs(db_session, make_team_df())
    assert inserted == 2
    assert db_session.query(TeamGameLog).count() == 2


def test_insert_team_game_logs_is_idempotent(db_session):
    df = make_team_df()
    insert_team_game_logs(db_session, df)
    second = insert_team_game_logs(db_session, df)
    assert second == 0
    assert db_session.query(TeamGameLog).count() == 2


def test_insert_team_game_logs_returns_zero_on_empty_df(db_session):
    empty = make_team_df().iloc[0:0]
    assert insert_team_game_logs(db_session, empty) == 0


def test_insert_player_game_logs_inserts_rows(db_session):
    inserted = insert_player_game_logs(db_session, make_player_df())
    assert inserted == 1
    assert db_session.query(PlayerGameLog).count() == 1


def test_insert_player_game_logs_is_idempotent(db_session):
    df = make_player_df()
    insert_player_game_logs(db_session, df)
    second = insert_player_game_logs(db_session, df)
    assert second == 0
    assert db_session.query(PlayerGameLog).count() == 1


def test_count_rows(db_session):
    insert_team_game_logs(db_session, make_team_df())
    assert count_rows(db_session, TeamGameLog) == 2
