from datetime import date
from typing import Optional
from sqlalchemy import Integer, String, Float, Date, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class PlayerGameLog(Base):
    __tablename__ = "player_game_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[str] = mapped_column(String(10), nullable=False)
    player_id: Mapped[int] = mapped_column(Integer, nullable=False)
    player_name: Mapped[str] = mapped_column(String(100), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team_abbreviation: Mapped[str] = mapped_column(String(10), nullable=False)
    game_id: Mapped[str] = mapped_column(String(20), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    matchup: Mapped[str] = mapped_column(String(30), nullable=False)
    home_away: Mapped[str] = mapped_column(String(4), nullable=False)
    wl: Mapped[str] = mapped_column(String(1), nullable=False)
    min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ast: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stl: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    blk: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tov: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fgm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fga: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fg_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fg3m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fg3a: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fg3_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ftm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fta: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ft_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plus_minus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_game"),
    )
