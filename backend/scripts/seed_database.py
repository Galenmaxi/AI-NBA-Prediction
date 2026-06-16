import os
import sys
import logging
from datetime import date
from typing import Type

import pandas as pd
from sqlalchemy.orm import Session
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from app.models.base import Base
from app.models.team_game_log import TeamGameLog
from app.models.player_game_log import PlayerGameLog
from app.models.database import create_tables, get_session
from ml.data_collector import fetch_team_game_logs, fetch_player_game_logs, SEASONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _to_date(val) -> date:
    return val.date() if hasattr(val, "date") else val


def _safe_int(val):
    return int(val) if pd.notna(val) else None


def _safe_float(val):
    return float(val) if pd.notna(val) else None


def insert_team_game_logs(session: Session, df: pd.DataFrame) -> int:
    inserted = 0
    for _, row in df.iterrows():
        exists = session.query(TeamGameLog).filter_by(
            game_id=str(row["game_id"]),
            team_id=int(row["team_id"]),
        ).first()
        if exists:
            continue
        session.add(TeamGameLog(
            season=str(row["season"]),
            team_id=int(row["team_id"]),
            team_abbreviation=str(row["team_abbreviation"]),
            team_name=str(row["team_name"]),
            game_id=str(row["game_id"]),
            game_date=_to_date(row["game_date"]),
            matchup=str(row["matchup"]),
            home_away=str(row["home_away"]),
            wl=str(row["wl"]),
            pts=int(row["pts"]),
            fgm=int(row["fgm"]), fga=int(row["fga"]), fg_pct=float(row["fg_pct"]),
            fg3m=int(row["fg3m"]), fg3a=int(row["fg3a"]), fg3_pct=float(row["fg3_pct"]),
            ftm=int(row["ftm"]), fta=int(row["fta"]), ft_pct=float(row["ft_pct"]),
            oreb=int(row["oreb"]), dreb=int(row["dreb"]), reb=int(row["reb"]),
            ast=int(row["ast"]), tov=int(row["tov"]), stl=int(row["stl"]),
            blk=int(row["blk"]), plus_minus=float(row["plus_minus"]),
        ))
        inserted += 1
    session.commit()
    return inserted


def insert_player_game_logs(session: Session, df: pd.DataFrame) -> int:
    inserted = 0
    for _, row in df.iterrows():
        exists = session.query(PlayerGameLog).filter_by(
            game_id=str(row["game_id"]),
            player_id=int(row["player_id"]),
        ).first()
        if exists:
            continue
        session.add(PlayerGameLog(
            season=str(row["season"]),
            player_id=int(row["player_id"]),
            player_name=str(row["player_name"]),
            team_id=int(row["team_id"]),
            team_abbreviation=str(row["team_abbreviation"]),
            game_id=str(row["game_id"]),
            game_date=_to_date(row["game_date"]),
            matchup=str(row["matchup"]),
            home_away=str(row["home_away"]),
            wl=str(row["wl"]),
            min=_safe_float(row.get("min")),
            pts=_safe_int(row.get("pts")),
            reb=_safe_int(row.get("reb")),
            ast=_safe_int(row.get("ast")),
            stl=_safe_int(row.get("stl")),
            blk=_safe_int(row.get("blk")),
            tov=_safe_int(row.get("tov")),
            fgm=_safe_int(row.get("fgm")),
            fga=_safe_int(row.get("fga")),
            fg_pct=_safe_float(row.get("fg_pct")),
            fg3m=_safe_int(row.get("fg3m")),
            fg3a=_safe_int(row.get("fg3a")),
            fg3_pct=_safe_float(row.get("fg3_pct")),
            ftm=_safe_int(row.get("ftm")),
            fta=_safe_int(row.get("fta")),
            ft_pct=_safe_float(row.get("ft_pct")),
            plus_minus=_safe_float(row.get("plus_minus")),
        ))
        inserted += 1
    session.commit()
    return inserted


def count_rows(session: Session, model: Type) -> int:
    return session.query(model).count()


def main() -> None:
    logger.info("Creating database tables...")
    create_tables()

    session = get_session()
    try:
        for season in SEASONS:
            logger.info(f"Fetching team game logs for {season}...")
            team_df = fetch_team_game_logs(season)
            n = insert_team_game_logs(session, team_df)
            logger.info(f"  Inserted {n} rows (total: {count_rows(session, TeamGameLog)})")

            logger.info(f"Fetching player game logs for {season}...")
            player_df = fetch_player_game_logs(season)
            n = insert_player_game_logs(session, player_df)
            logger.info(f"  Inserted {n} rows (total: {count_rows(session, PlayerGameLog)})")
    finally:
        session.close()

    with get_session() as s:
        logger.info(
            f"Seed complete — TeamGameLog: {count_rows(s, TeamGameLog)}, "
            f"PlayerGameLog: {count_rows(s, PlayerGameLog)}"
        )


if __name__ == "__main__":
    main()
