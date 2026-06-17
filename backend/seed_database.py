"""Seed the Railway PostgreSQL database with NBA game log data."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.database import create_tables, get_session
from app.models.player_game_log import PlayerGameLog
from app.models.team_game_log import TeamGameLog
from ml.data_collector import SEASONS, fetch_player_game_logs, fetch_team_game_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def seed_team_logs(session) -> None:
    for season in SEASONS:
        logger.info(f"Fetching team game logs for {season}…")
        df = fetch_team_game_logs(season)
        records = df.to_dict("records")
        if not records:
            logger.warning(f"No team data for {season}")
            continue
        stmt = pg_insert(TeamGameLog).values(records).on_conflict_do_nothing(
            constraint="uq_team_game"
        )
        session.execute(stmt)
        session.commit()
        logger.info(f"  ✓ {len(records)} team rows upserted for {season}")


def seed_player_logs(session) -> None:
    for season in SEASONS:
        logger.info(f"Fetching player game logs for {season}…")
        df = fetch_player_game_logs(season)
        records = df.to_dict("records")
        if not records:
            logger.warning(f"No player data for {season}")
            continue
        stmt = pg_insert(PlayerGameLog).values(records).on_conflict_do_nothing(
            constraint="uq_player_game"
        )
        session.execute(stmt)
        session.commit()
        logger.info(f"  ✓ {len(records)} player rows upserted for {season}")


if __name__ == "__main__":
    logger.info("Creating tables…")
    create_tables()
    logger.info("Tables ready.")

    session = get_session()
    try:
        seed_team_logs(session)
        seed_player_logs(session)
    finally:
        session.close()

    logger.info("Seeding complete.")
