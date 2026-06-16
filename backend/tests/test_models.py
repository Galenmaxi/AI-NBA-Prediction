import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.base import Base
from app.models.team_game_log import TeamGameLog
from app.models.player_game_log import PlayerGameLog


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def _make_team_log(**overrides) -> TeamGameLog:
    defaults = dict(
        season="2022-23",
        team_id=1610612738,
        team_abbreviation="BOS",
        team_name="Boston Celtics",
        game_id="0022300001",
        game_date=date(2022, 10, 18),
        matchup="BOS vs. NYK",
        home_away="HOME",
        wl="W",
        pts=120,
        fgm=45, fga=90, fg_pct=0.5,
        fg3m=12, fg3a=30, fg3_pct=0.4,
        ftm=18, fta=22, ft_pct=0.818,
        oreb=10, dreb=35, reb=45,
        ast=28, tov=12, stl=8, blk=5,
        plus_minus=20.0,
    )
    defaults.update(overrides)
    return TeamGameLog(**defaults)


def test_team_game_log_insert(db_session):
    log = _make_team_log()
    db_session.add(log)
    db_session.commit()

    result = db_session.query(TeamGameLog).filter_by(game_id="0022300001").first()
    assert result is not None
    assert result.team_abbreviation == "BOS"
    assert result.home_away == "HOME"
    assert result.pts == 120


def test_player_game_log_insert(db_session):
    log = PlayerGameLog(
        season="2022-23",
        player_id=1629029,
        player_name="Jayson Tatum",
        team_id=1610612738,
        team_abbreviation="BOS",
        game_id="0022300001",
        game_date=date(2022, 10, 18),
        matchup="BOS vs. NYK",
        home_away="HOME",
        wl="W",
        min=38.5,
        pts=32, reb=8, ast=5, stl=2, blk=1, tov=3,
        fgm=13, fga=25, fg_pct=0.52,
        fg3m=4, fg3a=10, fg3_pct=0.4,
        ftm=2, fta=2, ft_pct=1.0,
        plus_minus=15.0,
    )
    db_session.add(log)
    db_session.commit()

    result = db_session.query(PlayerGameLog).filter_by(
        game_id="0022300001", player_id=1629029
    ).first()
    assert result is not None
    assert result.player_name == "Jayson Tatum"
    assert result.pts == 32


def test_team_game_log_unique_constraint(db_session):
    db_session.add(_make_team_log())
    db_session.commit()

    db_session.add(_make_team_log())
    with pytest.raises(IntegrityError):
        db_session.commit()
