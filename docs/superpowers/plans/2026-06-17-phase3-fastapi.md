# Phase 3: FastAPI Prediction API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI REST API that serves NBA win probability, best player, and player stat predictions from the trained Phase 2 models.

**Architecture:** Thin FastAPI router layer → service layer (DB queries + feature building) → `ml/predict.py` (model inference). Models are loaded lazily from `.joblib` files on first request and cached in memory. Redis caching is optional — the API works without it and falls back gracefully if `REDIS_URL` is not set.

**Tech Stack:** FastAPI 0.115.6, Uvicorn 0.34.3, Pydantic v2 (bundled with FastAPI), SQLAlchemy (existing), XGBoost + LightGBM models from Phase 2, httpx 0.28.1 (tests), redis 5.2.1 (optional caching)

---

## Context: What Already Exists

All commands run from `backend/` with venv active: `.\venv\Scripts\activate`

**Already built (Phases 1–2):**
- `app/models/` — `TeamGameLog`, `PlayerGameLog` ORM models, `database.py` (engine, `SessionLocal`, `create_tables`)
- `ml/feature_engineering.py` — `build_team_features()`, `build_player_features()`, `TEAM_FEATURE_COLS`, `PLAYER_FEATURE_COLS`
- `ml/train_win_model.py` — `WIN_MODEL_PATH` = `models/win_model.joblib`
- `ml/train_player_model.py` — `BEST_PLAYER_MODEL_PATH`, `STAT_MODEL_PATHS`, `STAT_TARGETS = ["pts","reb","ast"]`
- `app/main.py`, `app/schemas/__init__.py`, `app/routers/__init__.py`, `app/services/__init__.py` — all empty placeholders
- `backend/requirements.txt` — includes pandas, numpy, xgboost, lightgbm, mlflow, sqlalchemy, etc.
- 51 passing tests in `tests/`

**Not yet installed:** fastapi, uvicorn, httpx, redis

---

## File Map

**Create:**
- `backend/ml/predict.py` — loads `.joblib` models, builds feature rows, returns prediction dicts
- `backend/app/schemas/predictions.py` — Pydantic response models
- `backend/app/routers/health.py` — `GET /health`
- `backend/app/routers/predictions.py` — `GET /predictions/win-probability`, `/best-player`, `/player-stats`
- `backend/app/services/prediction_service.py` — queries DB, calls `ml/predict` functions, adds confidence
- `backend/tests/test_predict.py` — unit tests for `ml/predict.py`
- `backend/tests/test_health.py` — tests for `GET /health`
- `backend/tests/test_prediction_service.py` — tests for service layer (SQLite in-memory)
- `backend/tests/test_predictions_api.py` — endpoint tests (mocked service, no DB needed)

**Modify:**
- `backend/requirements.txt` — add fastapi, uvicorn, httpx, redis
- `backend/app/models/database.py` — add `get_db()` yield dependency
- `backend/app/schemas/__init__.py` — re-export prediction schemas
- `backend/app/main.py` — full FastAPI app (currently a placeholder comment)

---

### Task 1: Add Phase 3 dependencies and `get_db()` yield

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/models/database.py`

- [ ] **Step 1: Add new packages to requirements.txt**

Append these four lines to `backend/requirements.txt`:

```
fastapi==0.115.6
uvicorn==0.34.3
httpx==0.28.1
redis==5.2.1
```

Full file after edit:
```
nba_api==1.11.4
pandas==2.3.3
numpy==2.4.6
sqlalchemy==2.0.51
psycopg2-binary==2.9.12
python-dotenv==1.2.2
pytest==9.1.0
pytest-mock==3.15.1
scikit-learn==1.9.0
xgboost==3.2.0
lightgbm==4.6.0
mlflow==3.13.0
fastapi==0.115.6
uvicorn==0.34.3
httpx==0.28.1
redis==5.2.1
```

- [ ] **Step 2: Install the new packages**

```powershell
pip install fastapi==0.115.6 uvicorn==0.34.3 httpx==0.28.1 redis==5.2.1
```

Expected: Packages install without error. Pydantic v2 is pulled in automatically as a FastAPI dependency.

- [ ] **Step 3: Add `get_db()` yield dependency to database.py**

`get_db()` is the FastAPI-idiomatic way to provide a database session per request and ensure it closes after the response. Replace the full contents of `backend/app/models/database.py` with:

```python
import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

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


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 4: Verify existing tests still pass**

```powershell
pytest tests/ -v
```

Expected: 51 passed, 0 failed. The `get_db()` addition is additive and cannot break anything.

- [ ] **Step 5: Commit**

```powershell
git add backend/requirements.txt backend/app/models/database.py
git commit -m "chore: add Phase 3 dependencies (fastapi, uvicorn, httpx, redis) and get_db() yield"
```

---

### Task 2: ML inference layer (`ml/predict.py`)

**Files:**
- Create: `backend/ml/predict.py`
- Create: `backend/tests/test_predict.py`

**Design notes:**
- Each predict function appends a synthetic "upcoming game" dummy row to the historical DataFrame. After `shift(1)` rolling in `build_team_features()`, that dummy row's features include ALL historical games — which is exactly what we want for prediction.
- Models are loaded from `.joblib` files lazily and cached in `_model_cache` dict. Call `clear_model_cache()` in tests to reset state.
- If a model file doesn't exist (not yet trained), raises `FileNotFoundError` — the API layer turns this into HTTP 503.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_predict.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from pathlib import Path

import joblib

from ml.feature_engineering import TEAM_FEATURE_COLS, PLAYER_FEATURE_COLS
from ml.train_player_model import STAT_TARGETS


# ---------------------------------------------------------------------------
# Fixtures: train tiny models and save to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_win_model_path(tmp_path):
    from xgboost import XGBClassifier
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (40, len(TEAM_FEATURE_COLS)))
    y = rng.integers(0, 2, 40)
    m = XGBClassifier(n_estimators=5, verbosity=0)
    m.fit(X, y)
    path = tmp_path / "win_model.joblib"
    joblib.dump(m, path)
    return path


@pytest.fixture()
def tiny_best_player_model_path(tmp_path):
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (40, len(PLAYER_FEATURE_COLS)))
    y = rng.integers(0, 2, 40)
    m = LGBMClassifier(n_estimators=5, verbose=-1)
    m.fit(X, y)
    path = tmp_path / "best_player_model.joblib"
    joblib.dump(m, path)
    return path


@pytest.fixture()
def tiny_stat_model_paths(tmp_path):
    from xgboost import XGBRegressor
    paths = {}
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (40, len(PLAYER_FEATURE_COLS)))
    for stat in STAT_TARGETS:
        y = rng.uniform(5, 30, 40)
        m = XGBRegressor(n_estimators=5, verbosity=0)
        m.fit(X, y)
        p = tmp_path / f"player_stat_{stat}.joblib"
        joblib.dump(m, p)
        paths[stat] = p
    return paths


def _make_team_df(team_id: int, n: int = 15) -> pd.DataFrame:
    base = date(2024, 10, 1)
    rng = np.random.default_rng(team_id)
    rows = []
    for i in range(n):
        rows.append({
            "team_id": team_id,
            "season": "2024-25",
            "game_id": f"T{team_id}G{i:04d}",
            "game_date": pd.Timestamp(base + timedelta(days=i * 3)),
            "home_away": "HOME" if i % 2 == 0 else "AWAY",
            "wl": "W" if rng.integers(0, 2) else "L",
            "pts": int(rng.integers(90, 130)),
        })
    return pd.DataFrame(rows)


def _make_player_df(n_players: int = 8, n_games: int = 15) -> pd.DataFrame:
    base = date(2024, 10, 1)
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_games):
        for p in range(1, n_players + 1):
            rows.append({
                "player_id": p,
                "player_name": f"Player {p}",
                "game_id": f"G{g:04d}",
                "game_date": pd.Timestamp(base + timedelta(days=g * 3)),
                "pts": int(rng.integers(5, 40)),
                "reb": int(rng.integers(2, 15)),
                "ast": int(rng.integers(0, 12)),
                "stl": int(rng.integers(0, 4)),
                "blk": int(rng.integers(0, 3)),
                "min": float(rng.uniform(15, 38)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_predict_win_probability_returns_probs_between_0_and_1(
    tiny_win_model_path, monkeypatch
):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "WIN_MODEL_PATH", tiny_win_model_path)

    from ml.predict import predict_win_probability
    result = predict_win_probability(_make_team_df(1), _make_team_df(2))
    assert 0.0 <= result["home_win_prob"] <= 1.0
    assert 0.0 <= result["away_win_prob"] <= 1.0


def test_predict_win_probability_returns_dict_with_expected_keys(
    tiny_win_model_path, monkeypatch
):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "WIN_MODEL_PATH", tiny_win_model_path)

    from ml.predict import predict_win_probability
    result = predict_win_probability(_make_team_df(1), _make_team_df(2))
    assert set(result.keys()) == {"home_win_prob", "away_win_prob"}


def test_predict_win_probability_raises_if_model_missing(monkeypatch, tmp_path):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "WIN_MODEL_PATH", tmp_path / "nonexistent.joblib")

    from ml.predict import predict_win_probability
    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict_win_probability(_make_team_df(1), _make_team_df(2))


def test_predict_best_player_returns_sorted_list(
    tiny_best_player_model_path, monkeypatch
):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "BEST_PLAYER_MODEL_PATH", tiny_best_player_model_path)

    from ml.predict import predict_best_player
    results = predict_best_player(_make_player_df(n_players=8, n_games=15))
    assert isinstance(results, list)
    assert len(results) > 0
    probs = [r["star_probability"] for r in results]
    assert probs == sorted(probs, reverse=True), "Results not sorted by star_probability"


def test_predict_best_player_result_has_expected_keys(
    tiny_best_player_model_path, monkeypatch
):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "BEST_PLAYER_MODEL_PATH", tiny_best_player_model_path)

    from ml.predict import predict_best_player
    results = predict_best_player(_make_player_df())
    for r in results:
        assert "player_id" in r
        assert "player_name" in r
        assert "star_probability" in r
        assert 0.0 <= r["star_probability"] <= 1.0


def test_predict_player_stats_returns_all_stat_targets(
    tiny_stat_model_paths, monkeypatch
):
    import ml.predict as mod
    mod.clear_model_cache()
    for stat, path in tiny_stat_model_paths.items():
        mod.STAT_MODEL_PATHS[stat] = path

    from ml.predict import predict_player_stats
    player_df = _make_player_df(n_players=1, n_games=15)
    result = predict_player_stats(player_df)
    for stat in STAT_TARGETS:
        assert stat in result
        assert isinstance(result[stat], float)


def test_predict_player_stats_raises_if_model_missing(tmp_path, monkeypatch):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "STAT_MODEL_PATHS", {"pts": tmp_path / "missing.joblib", "reb": tmp_path / "missing2.joblib", "ast": tmp_path / "missing3.joblib"})

    from ml.predict import predict_player_stats
    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict_player_stats(_make_player_df(n_players=1, n_games=15))
```

- [ ] **Step 2: Run tests to confirm they all fail**

```powershell
pytest tests/test_predict.py -v
```

Expected: 6 FAILED with `ModuleNotFoundError: No module named 'ml.predict'` or `ImportError`.

- [ ] **Step 3: Implement `ml/predict.py`**

Create `backend/ml/predict.py`:

```python
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    PLAYER_FEATURE_COLS,
    TEAM_FEATURE_COLS,
    build_player_features,
    build_team_features,
)
from ml.train_player_model import BEST_PLAYER_MODEL_PATH, STAT_MODEL_PATHS, STAT_TARGETS
from ml.train_win_model import WIN_MODEL_PATH

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


def clear_model_cache() -> None:
    _model_cache.clear()


def _load(path: Path) -> object:
    key = str(path)
    if key not in _model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run training scripts first."
            )
        _model_cache[key] = joblib.load(path)
    return _model_cache[key]


def _add_team_dummy_row(df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
    """Append a synthetic upcoming-game row so shift(1) rolling includes all history."""
    last_date = df["game_date"].max()
    dummy = {
        "team_id": int(df["team_id"].iloc[0]),
        "season": df["season"].iloc[0],
        "game_id": "PREDICT",
        "game_date": last_date + pd.Timedelta(days=1),
        "home_away": "HOME" if is_home else "AWAY",
        "wl": "W",
        "pts": 0,
    }
    return pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)


def _add_player_dummy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append one synthetic upcoming-game row per player."""
    last_dates = df.groupby("player_id")["game_date"].max()
    name_map = (
        df[["player_id", "player_name"]].drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
        .to_dict()
    ) if "player_name" in df.columns else {}
    dummies = []
    for player_id, last_date in last_dates.items():
        dummies.append({
            "player_id": player_id,
            "player_name": name_map.get(player_id, ""),
            "game_id": "PREDICT",
            "game_date": last_date + pd.Timedelta(days=1),
            "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0, "min": 0.0,
        })
    return pd.concat([df, pd.DataFrame(dummies)], ignore_index=True)


def predict_win_probability(
    home_team_df: pd.DataFrame,
    away_team_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Args:
        home_team_df: Historical game logs for home team (needs team_id, season,
            game_id, game_date, home_away, wl, pts columns).
        away_team_df: Same for away team.
    Returns:
        {"home_win_prob": 0.62, "away_win_prob": 0.38}
    """
    model = _load(WIN_MODEL_PATH)

    home_with_dummy = _add_team_dummy_row(home_team_df, is_home=True)
    away_with_dummy = _add_team_dummy_row(away_team_df, is_home=False)

    home_feat = build_team_features(home_with_dummy)
    away_feat = build_team_features(away_with_dummy)

    home_row = (
        home_feat[home_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS]
        .iloc[0].values.reshape(1, -1)
    )
    away_row = (
        away_feat[away_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS]
        .iloc[0].values.reshape(1, -1)
    )

    home_prob = float(model.predict_proba(home_row)[0][1])
    away_prob = float(model.predict_proba(away_row)[0][1])

    return {
        "home_win_prob": round(home_prob, 4),
        "away_win_prob": round(away_prob, 4),
    }


def predict_best_player(players_df: pd.DataFrame) -> list[dict]:
    """
    Args:
        players_df: Historical game logs for all players who will play
            (needs player_id, player_name, game_id, game_date, pts, reb, ast, stl, blk, min).
    Returns:
        List of {"player_id": int, "player_name": str, "star_probability": float}
        sorted by star_probability descending.
    """
    model = _load(BEST_PLAYER_MODEL_PATH)

    df_with_dummy = _add_player_dummy_rows(players_df)
    feat = build_player_features(df_with_dummy)

    latest = feat[feat["game_id"] == "PREDICT"].dropna(subset=PLAYER_FEATURE_COLS).copy()
    if latest.empty:
        return []

    X = latest[PLAYER_FEATURE_COLS].values
    probs = model.predict_proba(X)[:, 1]

    results = [
        {
            "player_id": int(row["player_id"]),
            "player_name": str(row.get("player_name", "")),
            "star_probability": round(float(probs[idx]), 4),
        }
        for idx, (_, row) in enumerate(latest.iterrows())
    ]
    return sorted(results, key=lambda x: x["star_probability"], reverse=True)


def predict_player_stats(player_df: pd.DataFrame) -> dict[str, float]:
    """
    Args:
        player_df: Historical game logs for one player
            (needs player_id, game_id, game_date, pts, reb, ast, stl, blk, min).
    Returns:
        {"pts": 22.3, "reb": 5.1, "ast": 3.8}
    """
    df_with_dummy = _add_player_dummy_rows(player_df)
    feat = build_player_features(df_with_dummy)

    predict_rows = feat[feat["game_id"] == "PREDICT"]
    if predict_rows.empty or predict_rows[PLAYER_FEATURE_COLS].isna().any().any():
        raise ValueError("Not enough historical data to build features for this player.")

    X = predict_rows[PLAYER_FEATURE_COLS].iloc[0].values.reshape(1, -1)
    result: dict[str, float] = {}
    for stat in STAT_TARGETS:
        model = _load(STAT_MODEL_PATHS[stat])
        result[stat] = round(float(model.predict(X)[0]), 1)
    return result
```

- [ ] **Step 4: Run tests and confirm all pass**

```powershell
pytest tests/test_predict.py -v
```

Expected: 6 passed, 0 failed.

- [ ] **Step 5: Commit**

```powershell
git add backend/ml/predict.py backend/tests/test_predict.py
git commit -m "feat: add ML inference layer with win probability, best player, and stat prediction"
```

---

### Task 3: Pydantic response schemas

**Files:**
- Create: `backend/app/schemas/predictions.py`
- Modify: `backend/app/schemas/__init__.py`

No test file for schemas — they're pure data definitions with no logic to test.

- [ ] **Step 1: Create `backend/app/schemas/predictions.py`**

```python
from pydantic import BaseModel


class WinProbabilityResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    home_win_prob: float
    away_win_prob: float
    confidence: str


class PlayerStarPrediction(BaseModel):
    player_id: int
    player_name: str
    star_probability: float


class BestPlayerResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    players: list[PlayerStarPrediction]


class StatPrediction(BaseModel):
    pts: float
    reb: float
    ast: float


class PlayerStatsResponse(BaseModel):
    player_id: int
    predicted_stats: StatPrediction
```

- [ ] **Step 2: Update `backend/app/schemas/__init__.py`**

```python
from app.schemas.predictions import (
    BestPlayerResponse,
    PlayerStarPrediction,
    PlayerStatsResponse,
    StatPrediction,
    WinProbabilityResponse,
)
```

- [ ] **Step 3: Verify import works**

```powershell
python -c "from app.schemas import WinProbabilityResponse, BestPlayerResponse, PlayerStatsResponse; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```powershell
git add backend/app/schemas/predictions.py backend/app/schemas/__init__.py
git commit -m "feat: add Pydantic response schemas for prediction API"
```

---

### Task 4: Health router + FastAPI skeleton

**Files:**
- Create: `backend/app/routers/health.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_health.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_health.py`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_returns_200():
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_check_returns_ok_status():
    resp = client.get("/health")
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to confirm it fails**

```powershell
pytest tests/test_health.py -v
```

Expected: 2 FAILED — `app.main` is a placeholder comment, no FastAPI app exists.

- [ ] **Step 3: Create `backend/app/routers/health.py`**

```python
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 4: Create `backend/app/main.py`** (skeleton, more routers added in Task 6)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health

app = FastAPI(title="NBA AI Predictor", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
```

- [ ] **Step 5: Run tests and confirm pass**

```powershell
pytest tests/test_health.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Run full test suite to confirm no regressions**

```powershell
pytest tests/ -v
```

Expected: 59 passed, 0 failed.

- [ ] **Step 7: Commit**

```powershell
git add backend/app/routers/health.py backend/app/main.py backend/tests/test_health.py
git commit -m "feat: add FastAPI app skeleton with health endpoint"
```

---

### Task 5: Prediction service

**Files:**
- Create: `backend/app/services/prediction_service.py`
- Create: `backend/tests/test_prediction_service.py`

The service layer bridges the API routers and the ML inference layer. It:
1. Queries the DB for recent game logs
2. Builds a DataFrame and calls the appropriate `ml/predict` function
3. Adds `confidence` level to win probability results
4. Raises `ValueError` (→ HTTP 422) if not enough data, propagates `FileNotFoundError` (→ HTTP 503) from model loading

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_prediction_service.py`:

```python
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
    # team 99 has no data
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
```

- [ ] **Step 2: Run tests to confirm they all fail**

```powershell
pytest tests/test_prediction_service.py -v
```

Expected: 8 FAILED with `ImportError` from `app.services.prediction_service`.

- [ ] **Step 3: Implement `backend/app/services/prediction_service.py`**

```python
from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from app.models.player_game_log import PlayerGameLog
from app.models.team_game_log import TeamGameLog
from ml.predict import predict_best_player, predict_player_stats, predict_win_probability

_N_GAMES = 20  # recent games to fetch per team/player for feature building


def _confidence(home_prob: float) -> str:
    spread = abs(home_prob - 0.5)
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
    """Query recent team game logs and predict win probability.

    Returns:
        {"home_win_prob": float, "away_win_prob": float, "confidence": str}
    Raises:
        ValueError: if a team has fewer than 2 games in the DB.
        FileNotFoundError: if win model .joblib not found (propagated from predict).
    """
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
                f"Not enough game data for team {team_id} (found {len(rows)} games, need at least 2)."
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
    return {**probs, "confidence": _confidence(probs["home_win_prob"])}


def get_best_player(
    db: Session,
    home_team_id: int,
    away_team_id: int,
) -> list[dict]:
    """Predict which player will be the star performer.

    Fetches recent player game logs for both teams and ranks all players.

    Returns:
        List of {"player_id": int, "player_name": str, "star_probability": float}
        sorted descending by star_probability.
    Raises:
        ValueError: if combined player data is insufficient.
        FileNotFoundError: if best player model .joblib not found.
    """
    def _fetch_team_players(team_id: int) -> list:
        recent_game_ids = (
            db.query(TeamGameLog.game_id)
            .filter(TeamGameLog.team_id == team_id)
            .order_by(TeamGameLog.game_date.desc())
            .limit(_N_GAMES)
            .subquery()
        )
        return (
            db.query(PlayerGameLog)
            .filter(PlayerGameLog.game_id.in_(recent_game_ids))
            .filter(PlayerGameLog.team_id == team_id)
            .filter(PlayerGameLog.pts.isnot(None))
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
        ValueError: if player has no game data in DB.
        FileNotFoundError: if stat model .joblib not found.
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
```

- [ ] **Step 4: Run tests and confirm all pass**

```powershell
pytest tests/test_prediction_service.py -v
```

Expected: 8 passed, 0 failed.

- [ ] **Step 5: Run full suite to confirm no regressions**

```powershell
pytest tests/ -v
```

Expected: 67 passed, 0 failed.

- [ ] **Step 6: Commit**

```powershell
git add backend/app/services/prediction_service.py backend/tests/test_prediction_service.py
git commit -m "feat: add prediction service layer with DB queries and confidence scoring"
```

---

### Task 6: Prediction router + API endpoint tests

**Files:**
- Create: `backend/app/routers/predictions.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_predictions_api.py`

The router tests mock the service functions entirely — they verify routing, response shape, and error-to-status-code mapping. No DB or model files needed.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_predictions_api.py`:

```python
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.database import get_db


def _fake_db():
    yield None


app.dependency_overrides[get_db] = _fake_db
client = TestClient(app)


# --- /predictions/win-probability ---

def test_win_probability_returns_200():
    with patch("app.routers.predictions.get_win_probability") as mock_svc:
        mock_svc.return_value = {
            "home_win_prob": 0.62,
            "away_win_prob": 0.38,
            "confidence": "high",
        }
        resp = client.get("/predictions/win-probability?home_team_id=1&away_team_id=2")
    assert resp.status_code == 200


def test_win_probability_response_shape():
    with patch("app.routers.predictions.get_win_probability") as mock_svc:
        mock_svc.return_value = {
            "home_win_prob": 0.55,
            "away_win_prob": 0.45,
            "confidence": "low",
        }
        resp = client.get("/predictions/win-probability?home_team_id=10&away_team_id=20")
    body = resp.json()
    assert body["home_team_id"] == 10
    assert body["away_team_id"] == 20
    assert body["home_win_prob"] == 0.55
    assert body["confidence"] == "low"


def test_win_probability_missing_param_returns_422():
    resp = client.get("/predictions/win-probability?home_team_id=1")
    assert resp.status_code == 422


def test_win_probability_value_error_returns_422():
    with patch("app.routers.predictions.get_win_probability") as mock_svc:
        mock_svc.side_effect = ValueError("Not enough game data for team 99")
        resp = client.get("/predictions/win-probability?home_team_id=1&away_team_id=99")
    assert resp.status_code == 422
    assert "Not enough game data" in resp.json()["detail"]


def test_win_probability_model_not_found_returns_503():
    with patch("app.routers.predictions.get_win_probability") as mock_svc:
        mock_svc.side_effect = FileNotFoundError("Model not found at models/win_model.joblib")
        resp = client.get("/predictions/win-probability?home_team_id=1&away_team_id=2")
    assert resp.status_code == 503


# --- /predictions/best-player ---

def test_best_player_returns_200():
    with patch("app.routers.predictions.get_best_player") as mock_svc:
        mock_svc.return_value = [
            {"player_id": 1, "player_name": "LeBron James", "star_probability": 0.87},
            {"player_id": 2, "player_name": "Stephen Curry", "star_probability": 0.75},
        ]
        resp = client.get("/predictions/best-player?home_team_id=1&away_team_id=2")
    assert resp.status_code == 200


def test_best_player_response_shape():
    with patch("app.routers.predictions.get_best_player") as mock_svc:
        mock_svc.return_value = [
            {"player_id": 5, "player_name": "Giannis", "star_probability": 0.90},
        ]
        resp = client.get("/predictions/best-player?home_team_id=3&away_team_id=4")
    body = resp.json()
    assert body["home_team_id"] == 3
    assert body["away_team_id"] == 4
    assert len(body["players"]) == 1
    assert body["players"][0]["player_name"] == "Giannis"
    assert body["players"][0]["star_probability"] == 0.90


def test_best_player_value_error_returns_422():
    with patch("app.routers.predictions.get_best_player") as mock_svc:
        mock_svc.side_effect = ValueError("Not enough player data")
        resp = client.get("/predictions/best-player?home_team_id=1&away_team_id=2")
    assert resp.status_code == 422


# --- /predictions/player-stats ---

def test_player_stats_returns_200():
    with patch("app.routers.predictions.get_player_stats") as mock_svc:
        mock_svc.return_value = {"pts": 27.5, "reb": 7.2, "ast": 8.1}
        resp = client.get("/predictions/player-stats?player_id=2544")
    assert resp.status_code == 200


def test_player_stats_response_shape():
    with patch("app.routers.predictions.get_player_stats") as mock_svc:
        mock_svc.return_value = {"pts": 22.3, "reb": 5.1, "ast": 3.8}
        resp = client.get("/predictions/player-stats?player_id=201939")
    body = resp.json()
    assert body["player_id"] == 201939
    assert body["predicted_stats"]["pts"] == 22.3
    assert body["predicted_stats"]["reb"] == 5.1
    assert body["predicted_stats"]["ast"] == 3.8


def test_player_stats_missing_param_returns_422():
    resp = client.get("/predictions/player-stats")
    assert resp.status_code == 422


def test_player_stats_not_found_returns_422():
    with patch("app.routers.predictions.get_player_stats") as mock_svc:
        mock_svc.side_effect = ValueError("No game data found for player 999")
        resp = client.get("/predictions/player-stats?player_id=999")
    assert resp.status_code == 422
    assert "No game data found" in resp.json()["detail"]
```

- [ ] **Step 2: Run tests to confirm they all fail**

```powershell
pytest tests/test_predictions_api.py -v
```

Expected: 12 FAILED — router doesn't exist yet and isn't registered in main.py.

- [ ] **Step 3: Create `backend/app/routers/predictions.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.schemas.predictions import (
    BestPlayerResponse,
    PlayerStarPrediction,
    PlayerStatsResponse,
    StatPrediction,
    WinProbabilityResponse,
)
from app.services.prediction_service import get_best_player, get_player_stats, get_win_probability

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/win-probability", response_model=WinProbabilityResponse)
def win_probability(
    home_team_id: int = Query(..., description="NBA team ID of the home team"),
    away_team_id: int = Query(..., description="NBA team ID of the away team"),
    db: Session = Depends(get_db),
) -> WinProbabilityResponse:
    try:
        result = get_win_probability(db, home_team_id, away_team_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return WinProbabilityResponse(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_win_prob=result["home_win_prob"],
        away_win_prob=result["away_win_prob"],
        confidence=result["confidence"],
    )


@router.get("/best-player", response_model=BestPlayerResponse)
def best_player(
    home_team_id: int = Query(..., description="NBA team ID of the home team"),
    away_team_id: int = Query(..., description="NBA team ID of the away team"),
    db: Session = Depends(get_db),
) -> BestPlayerResponse:
    try:
        players = get_best_player(db, home_team_id, away_team_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return BestPlayerResponse(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        players=[PlayerStarPrediction(**p) for p in players],
    )


@router.get("/player-stats", response_model=PlayerStatsResponse)
def player_stats(
    player_id: int = Query(..., description="NBA player ID"),
    db: Session = Depends(get_db),
) -> PlayerStatsResponse:
    try:
        stats = get_player_stats(db, player_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PlayerStatsResponse(
        player_id=player_id,
        predicted_stats=StatPrediction(**stats),
    )
```

- [ ] **Step 4: Register predictions router in `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, predictions

app = FastAPI(title="NBA AI Predictor", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)
```

- [ ] **Step 5: Run prediction API tests**

```powershell
pytest tests/test_predictions_api.py -v
```

Expected: 12 passed, 0 failed.

- [ ] **Step 6: Run full test suite**

```powershell
pytest tests/ -v
```

Expected: 79 passed, 0 failed.

- [ ] **Step 7: Commit**

```powershell
git add backend/app/routers/predictions.py backend/app/main.py backend/tests/test_predictions_api.py
git commit -m "feat: add prediction API endpoints (win-probability, best-player, player-stats)"
```

---

### Task 7: Optional Redis caching

**Files:**
- Modify: `backend/app/services/prediction_service.py`
- Create: `backend/tests/test_redis_cache.py`

Redis is optional: if `REDIS_URL` is not set (or Redis is down), all requests compute fresh predictions. The cache uses `PREDICT_CACHE_TTL` seconds TTL (default 3600 = 1 hour).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_redis_cache.py`:

```python
import json
from unittest.mock import MagicMock, patch

from app.services.prediction_service import _cache_get, _cache_set, _get_redis


def test_get_redis_returns_none_when_redis_url_not_set(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert _get_redis() is None


def test_cache_get_returns_none_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    result = _cache_get("any_key")
    assert result is None


def test_cache_set_is_noop_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    _cache_set("any_key", {"home_win_prob": 0.6})  # must not raise


def test_cache_get_returns_value_when_redis_has_key(monkeypatch):
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps({"home_win_prob": 0.72, "away_win_prob": 0.28}).encode()
    with patch("app.services.prediction_service._get_redis", return_value=mock_redis):
        result = _cache_get("win_prob:1:2")
    assert result["home_win_prob"] == 0.72


def test_cache_get_returns_none_when_key_missing(monkeypatch):
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    with patch("app.services.prediction_service._get_redis", return_value=mock_redis):
        result = _cache_get("win_prob:1:2")
    assert result is None


def test_cache_set_calls_setex_with_ttl(monkeypatch):
    mock_redis = MagicMock()
    with patch("app.services.prediction_service._get_redis", return_value=mock_redis):
        _cache_set("win_prob:1:2", {"home_win_prob": 0.6})
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert args[0] == "win_prob:1:2"
    assert args[1] == 3600  # default TTL


def test_get_win_probability_uses_cache_on_second_call(db=None):
    from unittest.mock import patch as p
    cached_value = {"home_win_prob": 0.99, "away_win_prob": 0.01, "confidence": "high"}
    with p("app.services.prediction_service._cache_get", return_value=cached_value):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from app.models.base import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            from app.services.prediction_service import get_win_probability
            result = get_win_probability(session, home_team_id=1, away_team_id=2)
    assert result["home_win_prob"] == 0.99
```

- [ ] **Step 2: Run tests to confirm they fail**

```powershell
pytest tests/test_redis_cache.py -v
```

Expected: 7 FAILED — `_cache_get`, `_cache_set`, `_get_redis` don't exist yet.

- [ ] **Step 3: Add Redis cache helpers to `backend/app/services/prediction_service.py`**

Add these imports at the top of the file (after existing imports):

```python
import json
import os
```

Add these three helper functions after the imports block and before `_confidence()`:

```python
PREDICT_CACHE_TTL: int = 3600


def _get_redis():
    """Return a Redis client or None if REDIS_URL not set / connection fails."""
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
```

Then update `get_win_probability()` to use the cache. Replace the function body with:

```python
def get_win_probability(
    db: Session,
    home_team_id: int,
    away_team_id: int,
) -> dict:
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
                f"Not enough game data for team {team_id} (found {len(rows)} games, need at least 2)."
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
```

- [ ] **Step 4: Run Redis cache tests**

```powershell
pytest tests/test_redis_cache.py -v
```

Expected: 7 passed, 0 failed.

- [ ] **Step 5: Run full test suite**

```powershell
pytest tests/ -v
```

Expected: 86 passed, 0 failed.

- [ ] **Step 6: Commit**

```powershell
git add backend/app/services/prediction_service.py backend/tests/test_redis_cache.py
git commit -m "feat: add optional Redis caching for win probability predictions"
```

---

### Task 8: Update HANDOFF.md and verify the server starts

**Files:**
- Modify: `backend/HANDOFF.md` (or `nba-ai-predictor/HANDOFF.md`)

- [ ] **Step 1: Smoke-test the API server starts without error**

Run uvicorn from the `backend/` directory (Ctrl+C to stop after confirming it starts):

```powershell
uvicorn app.main:app --reload --port 8000
```

Expected output (first 3 lines):
```
INFO:     Will watch for changes in these directories: [...]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...] using WatchFiles
```

Then in another terminal, verify the health endpoint:
```powershell
curl http://127.0.0.1:8000/health
```
Expected: `{"status":"ok"}`

And the OpenAPI docs are at: `http://127.0.0.1:8000/docs`

- [ ] **Step 2: Update HANDOFF.md**

Add a new section "Phase 3 — FastAPI REST API" to `nba-ai-predictor/HANDOFF.md` after the Phase 2 notes. Add the following content:

```markdown
---

## What Has Been Built (Phase 3)

### Git log additions
```
feat: add optional Redis caching for win probability predictions
feat: add prediction API endpoints (win-probability, best-player, player-stats)
feat: add prediction service layer with DB queries and confidence scoring
feat: add FastAPI app skeleton with health endpoint
feat: add Pydantic response schemas for prediction API
feat: add ML inference layer with win probability, best player, and stat prediction
chore: add Phase 3 dependencies (fastapi, uvicorn, httpx, redis) and get_db() yield
```

### New packages installed
| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.115.6 | Web framework |
| uvicorn | 0.34.3 | ASGI server |
| httpx | 0.28.1 | TestClient support |
| redis | 5.2.1 | Optional response caching |

### API Endpoints

**Start the server:**
```powershell
cd backend
.\venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — always returns `{"status": "ok"}` |
| GET | `/predictions/win-probability` | Win probability for a matchup |
| GET | `/predictions/best-player` | Star player prediction for a matchup |
| GET | `/predictions/player-stats` | Predicted pts/reb/ast for one player |

**Example requests:**
```
GET /predictions/win-probability?home_team_id=1610612744&away_team_id=1610612747
GET /predictions/best-player?home_team_id=1610612744&away_team_id=1610612747
GET /predictions/player-stats?player_id=2544
```

OpenAPI docs: `http://localhost:8000/docs`

### Architecture

```
Request → Router (app/routers/predictions.py)
        → Service (app/services/prediction_service.py)
            ├── Redis cache check (optional, graceful fallback)
            ├── DB query (SQLAlchemy, app/models/)
            └── ML inference (ml/predict.py)
                    └── .joblib model files (backend/models/)
```

### Error responses
| Situation | HTTP Status |
|---|---|
| Missing query param | 422 (FastAPI auto-validates) |
| Team/player not in DB | 422 |
| Model .joblib not found | 503 |

### Redis caching
Set `REDIS_URL=redis://localhost:6379` in `backend/.env` to enable.
If not set (or Redis is unreachable), predictions are computed fresh every request.
Only win probability is cached (TTL 1 hour). Cache key: `win_prob:{home_id}:{away_id}`.

### Test suite
86 tests, 0 failures.
```

- [ ] **Step 3: Run full test suite one final time**

```powershell
pytest tests/ -v
```

Expected: 86 passed, 0 failed.

- [ ] **Step 4: Commit**

```powershell
git add nba-ai-predictor/HANDOFF.md
git commit -m "docs: update HANDOFF.md with Phase 3 API architecture and endpoints"
```

---

## Self-Review

### Spec coverage
- [x] FastAPI REST API — Tasks 4, 6
- [x] Win probability endpoint — Task 6
- [x] Best player endpoint — Task 6
- [x] Player stat predictions — Task 6
- [x] Confidence score on predictions — Task 5 (`_confidence()`)
- [x] Redis caching — Task 7
- [x] TDD throughout — every task writes failing tests first
- [x] Service layer bridges routers and ML — Task 5
- [x] `get_db()` FastAPI yield dependency — Task 1
- [x] 503 when model not trained yet — Task 6 router exception handling
- [x] CORS — Task 4 (main.py)

### Placeholder scan — none found

### Type consistency
- `get_win_probability()` returns `dict` with keys `home_win_prob`, `away_win_prob`, `confidence` — router unpacks all three ✓
- `get_best_player()` returns `list[dict]` with keys `player_id`, `player_name`, `star_probability` — `PlayerStarPrediction(**p)` uses all three ✓
- `get_player_stats()` returns `dict[str, float]` with keys `pts`, `reb`, `ast` — `StatPrediction(**stats)` uses all three ✓
- `STAT_TARGETS = ["pts", "reb", "ast"]` (from Phase 2) — matches `StatPrediction` field names ✓
