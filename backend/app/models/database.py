import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from app.models.base import Base

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nba_user:nba_password@localhost:5432/nba_predictor",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def create_tables() -> None:
    Base.metadata.create_all(engine)
