# NBA AI Predictor — Phase 1: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable data pipeline that fetches 3 seasons of NBA team and player game logs from nba_api, stores them in PostgreSQL, and verifies data quality.

**Architecture:** A Python backend with three layers — SQLAlchemy models define the schema, `data_collector.py` handles nba_api fetching and cleaning, and `seed_database.py` orchestrates the full historical load. Docker Compose runs PostgreSQL locally; tests use SQLite in-memory so no Docker is required to run the test suite.

**Tech Stack:** Python 3.11, nba_api 1.4.1, pandas 2.1.4, SQLAlchemy 2.0, PostgreSQL 15 (Docker), pytest, python-dotenv

---

## File Map

| File | Responsibility |
|---|---|
| `docker-compose.yml` | Runs PostgreSQL 15 + Redis locally |
| `.env` / `.env.example` | Database connection string and credentials |
| `backend/requirements.txt` | Phase 1 Python dependencies |
| `backend/app/models/base.py` | SQLAlchemy `DeclarativeBase` |
| `backend/app/models/team_game_log.py` | `TeamGameLog` ORM model — one row per team per game |
| `backend/app/models/player_game_log.py` | `PlayerGameLog` ORM model — one row per player per game |
| `backend/app/models/database.py` | Engine + session factory + `create_tables()` |
| `backend/app/models/__init__.py` | Re-exports all models |
| `backend/ml/data_collector.py` | Fetches from nba_api, cleans to lowercase columns, derives `home_away` |
| `backend/scripts/seed_database.py` | Idempotent bulk loader — fetch all seasons, insert, skip duplicates |
| `backend/tests/test_models.py` | SQLite in-memory tests for insert and unique constraint |
| `backend/tests/test_data_collector.py` | Mocked tests for fetch and clean functions |
| `backend/tests/test_seed_database.py` | SQLite in-memory tests for insert and idempotency |

---

## Task 1: Project Skeleton

**Files:**
- Create: `docker-compose.yml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `.env`
- Create: `backend/requirements.txt`
- Create: placeholder `__init__.py` files in all backend packages
- Create: placeholder `.py` files for Phase 2/3 modules

- [ ] **Step 1: Create directory structure**

Run in PowerShell from `c:\Users\ASUS\OneDrive\Documents\AI_NBA_Prediction\nba-ai-predictor`:

```powershell
New-Item -ItemType Directory -Force -Path backend\app\models
New-Item -ItemType Directory -Force -Path backend\app\schemas
New-Item -ItemType Directory -Force -Path backend\app\routers
New-Item -ItemType Directory -Force -Path backend\app\services
New-Item -ItemType Directory -Force -Path backend\ml
New-Item -ItemType Directory -Force -Path backend\scripts
New-Item -ItemType Directory -Force -Path backend\tests
New-Item -ItemType Directory -Force -Path frontend
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-nba_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-nba_password}
      POSTGRES_DB: ${POSTGRES_DB:-nba_predictor}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-nba_user}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
*.pkl
*.joblib
mlruns/
.DS_Store
```

- [ ] **Step 4: Create `.env.example` and `.env`**

Both files get the same content for local dev:

```
DATABASE_URL=postgresql://nba_user:nba_password@localhost:5432/nba_predictor
POSTGRES_USER=nba_user
POSTGRES_PASSWORD=nba_password
POSTGRES_DB=nba_predictor
```

`.env` is git-ignored. `.env.example` is committed.

- [ ] **Step 5: Create `backend/requirements.txt`**

```
nba_api==1.4.1
pandas==2.1.4
numpy==1.26.2
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-dotenv==1.0.0
pytest==7.4.4
pytest-mock==3.12.2
```

- [ ] **Step 6: Create placeholder files for later phases**

`backend/app/main.py`:
```python
# TODO: Phase 3 - FastAPI entry point
```

`backend/ml/feature_engineering.py`:
```python
# TODO: Phase 2 - feature engineering
```

`backend/ml/train_win_model.py`:
```python
# TODO: Phase 2 - win probability model training
```

`backend/ml/train_player_model.py`:
```python
# TODO: Phase 2 - player model training
```

`backend/ml/predict.py`:
```python
# TODO: Phase 2/3 - inference logic
```

Create empty `__init__.py` files in:
- `backend/app/__init__.py`
- `backend/app/models/__init__.py`
- `backend/app/schemas/__init__.py`
- `backend/app/routers/__init__.py`
- `backend/app/services/__init__.py`
- `backend/ml/__init__.py`
- `backend/scripts/__init__.py`
- `backend/tests/__init__.py`

- [ ] **Step 7: Set up Python virtual environment**

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Verify:
```powershell
python -c "import nba_api; print('nba_api ok')"
python -c "import sqlalchemy; print('sqlalchemy ok')"
```

Expected: Both lines print `ok` with no errors.

- [ ] **Step 8: Initialize git and commit**

```powershell
cd ..
git init
git add docker-compose.yml .gitignore .env.example backend/
git commit -m "chore: initialize project structure with docker-compose and requirements"
```

---

## Task 2: Database Models

**Files:**
- Create: `backend/app/models/base.py`
- Create: `backend/app/models/team_game_log.py`
- Create: `backend/app/models/player_game_log.py`
- Create: `backend/app/models/database.py`
- Modify: `backend/app/models/__init__.py`
- Create: `backend/tests/test_models.py`

- [ ] **Step 1: Write failing tests for models**

`backend/tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd backend
.\venv\Scripts\activate
pytest tests/test_models.py -v
```

Expected: `ERROR` collecting — "No module named 'app.models.base'"

- [ ] **Step 3: Create `backend/app/models/base.py`**

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

- [ ] **Step 4: Create `backend/app/models/team_game_log.py`**

```python
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
```

- [ ] **Step 5: Create `backend/app/models/player_game_log.py`**

```python
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
```

- [ ] **Step 6: Create `backend/app/models/database.py`**

```python
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
```

- [ ] **Step 7: Update `backend/app/models/__init__.py`**

```python
from app.models.base import Base
from app.models.team_game_log import TeamGameLog
from app.models.player_game_log import PlayerGameLog

__all__ = ["Base", "TeamGameLog", "PlayerGameLog"]
```

- [ ] **Step 8: Run tests to verify they pass**

```powershell
pytest tests/test_models.py -v
```

Expected:
```
tests/test_models.py::test_team_game_log_insert PASSED
tests/test_models.py::test_player_game_log_insert PASSED
tests/test_models.py::test_team_game_log_unique_constraint PASSED

3 passed
```

- [ ] **Step 9: Commit**

```powershell
git add backend/app/models/ backend/tests/test_models.py
git commit -m "feat: add SQLAlchemy models for team and player game logs"
```

---

## Task 3: Data Collector — Team Game Logs

**Files:**
- Create: `backend/ml/data_collector.py`
- Create: `backend/tests/test_data_collector.py`

- [ ] **Step 1: Write failing tests**

`backend/tests/test_data_collector.py`:
```python
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from ml.data_collector import (
    clean_team_game_logs,
    fetch_team_game_logs,
    SEASONS,
)


def make_raw_team_df() -> pd.DataFrame:
    return pd.DataFrame({
        "SEASON_YEAR": ["2022-23", "2022-23"],
        "TEAM_ID": [1610612738, 1610612752],
        "TEAM_ABBREVIATION": ["BOS", "NYK"],
        "TEAM_NAME": ["Boston Celtics", "New York Knicks"],
        "GAME_ID": ["0022300001", "0022300001"],
        "GAME_DATE": ["OCT 18, 2022", "OCT 18, 2022"],
        "MATCHUP": ["BOS vs. NYK", "NYK @ BOS"],
        "WL": ["W", "L"],
        "MIN": [240.0, 240.0],
        "PTS": [120, 100],
        "FGM": [45, 38], "FGA": [90, 85], "FG_PCT": [0.5, 0.447],
        "FG3M": [12, 8], "FG3A": [30, 25], "FG3_PCT": [0.4, 0.32],
        "FTM": [18, 16], "FTA": [22, 20], "FT_PCT": [0.818, 0.8],
        "OREB": [10, 8], "DREB": [35, 30], "REB": [45, 38],
        "AST": [28, 22], "TOV": [12, 15], "STL": [8, 6], "BLK": [5, 4],
        "PLUS_MINUS": [20.0, -20.0],
    })


def test_clean_team_game_logs_adds_home_away():
    result = clean_team_game_logs(make_raw_team_df())
    bos = result[result["team_id"] == 1610612738].iloc[0]
    nyk = result[result["team_id"] == 1610612752].iloc[0]
    assert bos["home_away"] == "HOME"
    assert nyk["home_away"] == "AWAY"


def test_clean_team_game_logs_renames_season_column():
    result = clean_team_game_logs(make_raw_team_df())
    assert "season" in result.columns
    assert "SEASON_YEAR" not in result.columns


def test_clean_team_game_logs_lowercases_all_columns():
    result = clean_team_game_logs(make_raw_team_df())
    for col in result.columns:
        assert col == col.lower(), f"Column '{col}' should be lowercase"


def test_clean_team_game_logs_parses_game_date():
    result = clean_team_game_logs(make_raw_team_df())
    assert pd.api.types.is_datetime64_any_dtype(result["game_date"])


def test_clean_team_game_logs_drops_rows_missing_game_id():
    raw = make_raw_team_df()
    raw.loc[0, "GAME_ID"] = None
    result = clean_team_game_logs(raw)
    assert len(result) == 1


def test_seasons_has_three_entries_in_correct_format():
    assert len(SEASONS) == 3
    for s in SEASONS:
        assert len(s) == 7 and s[4] == "-", f"Season '{s}' must be in YYYY-YY format"


@patch("ml.data_collector.TeamGameLogs")
def test_fetch_team_game_logs_calls_api_with_correct_args(mock_cls):
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = [make_raw_team_df()]
    mock_cls.return_value = mock_instance

    result = fetch_team_game_logs("2022-23")

    mock_cls.assert_called_once_with(season_nullable="2022-23", timeout=30)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


@patch("ml.data_collector.TeamGameLogs")
def test_fetch_team_game_logs_returns_cleaned_df(mock_cls):
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = [make_raw_team_df()]
    mock_cls.return_value = mock_instance

    result = fetch_team_game_logs("2022-23")

    assert "home_away" in result.columns
    assert "season" in result.columns
    for col in result.columns:
        assert col == col.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_data_collector.py -v
```

Expected: `ERROR` — "No module named 'ml.data_collector'"

- [ ] **Step 3: Create `backend/ml/data_collector.py` with team log functions**

```python
import time
import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs, PlayerGameLogs

SEASONS: list[str] = ["2023-24", "2024-25", "2025-26"]
REQUEST_DELAY: float = 0.6

_TEAM_COLUMNS: list[str] = [
    "season", "team_id", "team_abbreviation", "team_name",
    "game_id", "game_date", "matchup", "home_away", "wl",
    "pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
    "ftm", "fta", "ft_pct", "oreb", "dreb", "reb",
    "ast", "tov", "stl", "blk", "plus_minus",
]

_PLAYER_COLUMNS: list[str] = [
    "season", "player_id", "player_name", "team_id", "team_abbreviation",
    "game_id", "game_date", "matchup", "home_away", "wl",
    "min", "pts", "reb", "ast", "stl", "blk", "tov",
    "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
    "ftm", "fta", "ft_pct", "plus_minus",
]


def clean_team_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"season_year": "season"})
    df["home_away"] = df["matchup"].apply(
        lambda x: "HOME" if "vs." in str(x) else "AWAY"
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["game_id", "team_id", "game_date"])
    return df[[c for c in _TEAM_COLUMNS if c in df.columns]]


def fetch_team_game_logs(season: str) -> pd.DataFrame:
    logs = TeamGameLogs(season_nullable=season, timeout=30)
    df = logs.get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    return clean_team_game_logs(df)
```

- [ ] **Step 4: Run team log tests**

```powershell
pytest tests/test_data_collector.py -v
```

Expected: All 8 tests pass (player tests are not in this file yet).

- [ ] **Step 5: Commit**

```powershell
git add backend/ml/data_collector.py backend/tests/test_data_collector.py
git commit -m "feat: add team game log fetcher and cleaner"
```

---

## Task 4: Data Collector — Player Game Logs

**Files:**
- Modify: `backend/ml/data_collector.py`
- Modify: `backend/tests/test_data_collector.py`

- [ ] **Step 1: Add failing player tests to `backend/tests/test_data_collector.py`**

Append to the bottom of the existing test file:
```python
from ml.data_collector import clean_player_game_logs, fetch_player_game_logs


def make_raw_player_df() -> pd.DataFrame:
    return pd.DataFrame({
        "SEASON_YEAR": ["2022-23"],
        "PLAYER_ID": [1629029],
        "PLAYER_NAME": ["Jayson Tatum"],
        "NICKNAME": ["JT"],
        "TEAM_ID": [1610612738],
        "TEAM_ABBREVIATION": ["BOS"],
        "TEAM_NAME": ["Boston Celtics"],
        "GAME_ID": ["0022300001"],
        "GAME_DATE": ["OCT 18, 2022"],
        "MATCHUP": ["BOS vs. NYK"],
        "WL": ["W"],
        "MIN": [38.5],
        "FGM": [13], "FGA": [25], "FG_PCT": [0.52],
        "FG3M": [4], "FG3A": [10], "FG3_PCT": [0.4],
        "FTM": [2], "FTA": [2], "FT_PCT": [1.0],
        "OREB": [1], "DREB": [7], "REB": [8],
        "AST": [5], "TOV": [3], "STL": [2], "BLK": [1],
        "BLKA": [0], "PF": [2], "PFD": [3],
        "PTS": [32], "PLUS_MINUS": [15.0],
        "NBA_FANTASY_PTS": [55.0], "DD2": [1], "TD3": [0],
    })


def test_clean_player_game_logs_selects_correct_columns():
    result = clean_player_game_logs(make_raw_player_df())
    expected = {
        "season", "player_id", "player_name", "team_id", "team_abbreviation",
        "game_id", "game_date", "matchup", "home_away", "wl",
        "min", "pts", "reb", "ast", "stl", "blk", "tov",
        "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "ftm", "fta", "ft_pct", "plus_minus",
    }
    assert expected.issubset(set(result.columns))
    assert "nickname" not in result.columns
    assert "nba_fantasy_pts" not in result.columns


def test_clean_player_game_logs_adds_home_away():
    result = clean_player_game_logs(make_raw_player_df())
    assert result.iloc[0]["home_away"] == "HOME"


def test_clean_player_game_logs_handles_null_minutes():
    raw = make_raw_player_df()
    raw.loc[0, "MIN"] = None
    result = clean_player_game_logs(raw)
    assert pd.isna(result.iloc[0]["min"])


@patch("ml.data_collector.PlayerGameLogs")
def test_fetch_player_game_logs_calls_api_with_correct_args(mock_cls):
    mock_instance = MagicMock()
    mock_instance.get_data_frames.return_value = [make_raw_player_df()]
    mock_cls.return_value = mock_instance

    result = fetch_player_game_logs("2022-23")

    mock_cls.assert_called_once_with(season_nullable="2022-23", timeout=30)
    assert isinstance(result, pd.DataFrame)
    assert "home_away" in result.columns
```

- [ ] **Step 2: Run new tests to verify they fail**

```powershell
pytest tests/test_data_collector.py::test_clean_player_game_logs_selects_correct_columns -v
```

Expected: `ImportError` — "cannot import name 'clean_player_game_logs'"

- [ ] **Step 3: Add player log functions to `backend/ml/data_collector.py`**

Append after the existing `fetch_team_game_logs` function:
```python
def clean_player_game_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"season_year": "season"})
    df["home_away"] = df["matchup"].apply(
        lambda x: "HOME" if "vs." in str(x) else "AWAY"
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["game_id", "player_id", "game_date"])
    return df[[c for c in _PLAYER_COLUMNS if c in df.columns]]


def fetch_player_game_logs(season: str) -> pd.DataFrame:
    logs = PlayerGameLogs(season_nullable=season, timeout=30)
    df = logs.get_data_frames()[0]
    time.sleep(REQUEST_DELAY)
    return clean_player_game_logs(df)
```

- [ ] **Step 4: Run full data collector test suite**

```powershell
pytest tests/test_data_collector.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```powershell
git add backend/ml/data_collector.py backend/tests/test_data_collector.py
git commit -m "feat: add player game log fetcher and cleaner"
```

---

## Task 5: Database Seeder

**Files:**
- Create: `backend/scripts/seed_database.py`
- Create: `backend/tests/test_seed_database.py`

- [ ] **Step 1: Write failing tests**

`backend/tests/test_seed_database.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_seed_database.py -v
```

Expected: `ERROR` — "No module named 'scripts.seed_database'"

- [ ] **Step 3: Create `backend/scripts/seed_database.py`**

```python
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
```

- [ ] **Step 4: Run seed tests**

```powershell
pytest tests/test_seed_database.py -v
```

Expected:
```
tests/test_seed_database.py::test_insert_team_game_logs_inserts_rows PASSED
tests/test_seed_database.py::test_insert_team_game_logs_is_idempotent PASSED
tests/test_seed_database.py::test_insert_team_game_logs_returns_zero_on_empty_df PASSED
tests/test_seed_database.py::test_insert_player_game_logs_inserts_rows PASSED
tests/test_seed_database.py::test_insert_player_game_logs_is_idempotent PASSED
tests/test_seed_database.py::test_count_rows PASSED

6 passed
```

- [ ] **Step 5: Run the full test suite**

```powershell
pytest tests/ -v
```

Expected: All tests across all 3 test files pass.

- [ ] **Step 6: Commit**

```powershell
git add backend/scripts/seed_database.py backend/tests/test_seed_database.py
git commit -m "feat: add idempotent database seeder for historical game logs"
```

---

## Task 6: End-to-End Verification

**Files:** None new — verifying the full pipeline works against live data.

- [ ] **Step 1: Start PostgreSQL**

```powershell
docker-compose up -d postgres
```

Wait ~5 seconds, then check:
```powershell
docker-compose ps
```

Expected: `postgres` shows `Up (healthy)`.

- [ ] **Step 2: Verify tables are created**

```powershell
cd backend
.\venv\Scripts\activate
python -c "
from app.models.database import create_tables
create_tables()
print('Tables created successfully')
"
```

Expected: `Tables created successfully`

- [ ] **Step 3: Smoke test — fetch one season of team logs**

Before running the full seeder (~10 min), verify the API is reachable:

```powershell
python -c "
from ml.data_collector import fetch_team_game_logs
df = fetch_team_game_logs('2023-24')
print(f'Rows: {len(df)}')
print(df[['season','team_abbreviation','game_date','matchup','wl','pts']].head(3).to_string())
"
```

Expected: Prints ~2460 rows and a sample like:
```
Rows: 2460
  season team_abbreviation  game_date         matchup wl  pts
0  2023-24               ATL 2024-04-14  ATL @ MIA      L   99
...
```

If this times out or errors: the nba_api may be rate-limiting. Wait 30 seconds and retry.

- [ ] **Step 4: Run the full seeder**

This fetches 3 seasons × 2 endpoints = 6 API calls. Takes ~10–15 minutes due to `REQUEST_DELAY` and large response sizes.

```powershell
python scripts/seed_database.py
```

Expected log:
```
INFO Creating database tables...
INFO Fetching team game logs for 2023-24...
INFO   Inserted 2460 rows (total: 2460)
INFO Fetching player game logs for 2023-24...
INFO   Inserted ~62000 rows (total: ~62000)
... (2024-25 and 2025-26)
INFO Seed complete — TeamGameLog: ~7380, PlayerGameLog: ~186000
```

- [ ] **Step 5: Verify row counts in PostgreSQL**

```powershell
docker exec -it $(docker-compose ps -q postgres) psql -U nba_user -d nba_predictor -c "SELECT season, COUNT(*) FROM team_game_logs GROUP BY season ORDER BY season;"
```

Expected:
```
  season  | count
----------+-------
 2023-24  |  2460
 2024-25  |  2460
 2025-26  |  ~2460
```

```powershell
docker exec -it $(docker-compose ps -q postgres) psql -U nba_user -d nba_predictor -c "SELECT season, COUNT(*) FROM player_game_logs GROUP BY season ORDER BY season;"
```

Expected: ~60,000–65,000 per season.

- [ ] **Step 6: Verify idempotency — run seeder again**

```powershell
python scripts/seed_database.py
```

Expected: All `Inserted N rows` lines show `0`. Final totals are unchanged.

- [ ] **Step 7: Final commit**

```powershell
cd ..
git add .
git commit -m "chore: Phase 1 complete — data pipeline verified end-to-end"
```

---

## Self-Review

**Spec coverage:**
- Set up Python project with virtual environment → Task 1, Step 7
- Install nba_api and explore endpoints → Task 3, Step 3 + Task 6, Step 3
- Pull last 3 seasons of team game logs → `SEASONS` constant + `fetch_team_game_logs`
- Pull player game logs → `fetch_player_game_logs`
- Set up PostgreSQL with Docker → Task 1, Step 2
- Write seed_database.py to load historical data → Task 5
- Verify data quality and handle missing values → `dropna` in clean functions; nullable columns in `PlayerGameLog`; idempotency tests

**Placeholder scan:** All implementation steps contain complete, runnable code. No TBDs.

**Type consistency:**
- `clean_team_game_logs(df: pd.DataFrame) -> pd.DataFrame` — consistent across data_collector.py and tests
- `fetch_team_game_logs(season: str) -> pd.DataFrame` — consistent
- `clean_player_game_logs` / `fetch_player_game_logs` — same signature pattern
- `insert_team_game_logs(session: Session, df: pd.DataFrame) -> int` — consistent between seed_database.py and test_seed_database.py
- `count_rows(session: Session, model: Type) -> int` — consistent
- `TeamGameLog` / `PlayerGameLog` — imported from `app.models` consistently in all files
