from datetime import date
from sqlalchemy import Integer, String, Float, Date, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class TeamGameLog(Base):
    __tablename__ = "team_game_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    season: Mapped[str] = mapped_column(String(10), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team_abbreviation: Mapped[str] = mapped_column(String(10), nullable=False)
    team_name: Mapped[str] = mapped_column(String(100), nullable=False)
    game_id: Mapped[str] = mapped_column(String(20), nullable=False)
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    matchup: Mapped[str] = mapped_column(String(30), nullable=False)
    home_away: Mapped[str] = mapped_column(String(4), nullable=False)
    wl: Mapped[str] = mapped_column(String(1), nullable=False)
    pts: Mapped[int] = mapped_column(Integer, nullable=False)
    fgm: Mapped[int] = mapped_column(Integer, nullable=False)
    fga: Mapped[int] = mapped_column(Integer, nullable=False)
    fg_pct: Mapped[float] = mapped_column(Float, nullable=False)
    fg3m: Mapped[int] = mapped_column(Integer, nullable=False)
    fg3a: Mapped[int] = mapped_column(Integer, nullable=False)
    fg3_pct: Mapped[float] = mapped_column(Float, nullable=False)
    ftm: Mapped[int] = mapped_column(Integer, nullable=False)
    fta: Mapped[int] = mapped_column(Integer, nullable=False)
    ft_pct: Mapped[float] = mapped_column(Float, nullable=False)
    oreb: Mapped[int] = mapped_column(Integer, nullable=False)
    dreb: Mapped[int] = mapped_column(Integer, nullable=False)
    reb: Mapped[int] = mapped_column(Integer, nullable=False)
    ast: Mapped[int] = mapped_column(Integer, nullable=False)
    tov: Mapped[int] = mapped_column(Integer, nullable=False)
    stl: Mapped[int] = mapped_column(Integer, nullable=False)
    blk: Mapped[int] = mapped_column(Integer, nullable=False)
    plus_minus: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", name="uq_team_game"),
    )
