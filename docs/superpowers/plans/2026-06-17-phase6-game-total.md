# Phase 6: Game Total (Over/Under) Prediction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the final unbuilt prediction — total game score (over/under) — across the full stack: feature engineering, XGBoost regressor, FastAPI endpoint, and a GameTotalCard frontend component.

**Architecture:** Add `TOTAL_FEATURE_COLS` and `build_total_model_dataset()` to feature_engineering.py (joins home+away team rows by game_id, target = home_pts + away_pts). A new `train_total_model.py` trains an XGBRegressor (device=cuda) and saves to `backend/models/total_model.joblib`. `predict_game_total()` in predict.py pairs live team feature vectors into the combined input shape. The service, router, and frontend follow the exact same patterns as `win-probability`.

**Tech Stack:** XGBoost 3.x (device=cuda), SQLAlchemy, FastAPI, Pydantic v2, Next.js 14 (App Router), TanStack React Query v5, Recharts-free plain card, Jest + Testing Library.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `backend/ml/feature_engineering.py` | Modify | Add `TOTAL_FEATURE_COLS`, `build_total_model_dataset()` |
| `backend/ml/train_total_model.py` | Create | `train_total_model()`, `TOTAL_MODEL_PATH`, `main()` |
| `backend/ml/predict.py` | Modify | Add `TOTAL_MODEL_PATH` import, `predict_game_total()` |
| `backend/app/schemas/predictions.py` | Modify | Add `GameTotalResponse` |
| `backend/app/services/prediction_service.py` | Modify | Add `get_game_total()`, `_total_confidence()` |
| `backend/app/routers/predictions.py` | Modify | Add `GET /predictions/game-total` |
| `backend/tests/test_feature_engineering.py` | Modify | Add 2 tests for new functions |
| `backend/tests/test_train_total_model.py` | Create | 3 tests for training script |
| `backend/tests/test_predict.py` | Modify | Add 2 tests for `predict_game_total` |
| `backend/tests/test_prediction_service.py` | Modify | Add 1 test for `get_game_total` |
| `backend/tests/test_predictions_api.py` | Modify | Add 2 tests for `/predictions/game-total` |
| `frontend/src/lib/types.ts` | Modify | Add `GameTotalResponse` interface |
| `frontend/src/lib/api.ts` | Modify | Add `fetchGameTotal()` |
| `frontend/src/hooks/useGameTotal.ts` | Create | `useGameTotal()` hook |
| `frontend/src/components/GameTotalCard.tsx` | Create | Display card |
| `frontend/src/app/page.tsx` | Modify | Wire in `GameTotalCard` |
| `frontend/__tests__/GameTotalCard.test.tsx` | Create | 3 component tests |

---

## Task 1: Feature Engineering — TOTAL_FEATURE_COLS + build_total_model_dataset()

**Files:**
- Modify: `backend/ml/feature_engineering.py`
- Modify: `backend/tests/test_feature_engineering.py`

### Context

`build_team_features()` already produces rolling stats per team per game. Each real game has **two rows** (one per team) that share the same `game_id`. `build_total_model_dataset` joins them on `game_id` — home team row becomes `home_*` features, away team row becomes `away_*` features — and sets `target = home_pts + away_pts`.

`TOTAL_FEATURE_COLS` is just the 6 existing `TEAM_FEATURE_COLS` prefixed twice (`home_` and `away_`), giving a 12-feature input vector.

- [ ] **Step 1: Write the failing tests**

Add at the bottom of `backend/tests/test_feature_engineering.py` (imports already present — `build_team_features`, `TEAM_FEATURE_COLS` etc. — just add these two functions to the import line and add the new test helper + tests):

```python
from ml.feature_engineering import (
    build_team_features,
    build_player_features,
    build_win_model_dataset,
    build_player_star_dataset,
    build_player_stats_dataset,
    build_total_model_dataset,   # NEW
    train_test_split_by_date,
    TEAM_FEATURE_COLS,
    PLAYER_FEATURE_COLS,
    TOTAL_FEATURE_COLS,          # NEW
)


def make_paired_team_df(n_games: int = 15) -> pd.DataFrame:
    """Home team (id=1) + away team (id=2) sharing the same game_id per game."""
    base = date(2024, 10, 1)
    rows = []
    for i in range(n_games):
        game_id = f"GAME{i:04d}"
        rows.append({
            "team_id": 1, "season": "2024-25", "game_id": game_id,
            "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
            "home_away": "HOME", "wl": "W", "pts": 110 + i,
        })
        rows.append({
            "team_id": 2, "season": "2024-25", "game_id": game_id,
            "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
            "home_away": "AWAY", "wl": "L", "pts": 100 + i,
        })
    return pd.DataFrame(rows)


def test_total_feature_cols_has_home_and_away_prefixes():
    for col in TEAM_FEATURE_COLS:
        assert f"home_{col}" in TOTAL_FEATURE_COLS
        assert f"away_{col}" in TOTAL_FEATURE_COLS


def test_build_total_model_dataset_has_total_target():
    df = make_paired_team_df(n_games=15)
    team_feat = build_team_features(df)
    result = build_total_model_dataset(team_feat)
    assert "total" in result.columns
    assert (result["total"] > 0).all()
    assert set(TOTAL_FEATURE_COLS).issubset(result.columns)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd backend
.\venv\Scripts\activate
pytest tests/test_feature_engineering.py::test_total_feature_cols_has_home_and_away_prefixes tests/test_feature_engineering.py::test_build_total_model_dataset_has_total_target -v
```

Expected: FAIL — `ImportError: cannot import name 'build_total_model_dataset'`

- [ ] **Step 3: Implement in feature_engineering.py**

Add immediately after `TEAM_FEATURE_COLS` and `PLAYER_FEATURE_COLS` definitions (around line 19), and after `build_win_model_dataset`:

```python
TOTAL_FEATURE_COLS: list[str] = (
    [f"home_{c}" for c in TEAM_FEATURE_COLS] +
    [f"away_{c}" for c in TEAM_FEATURE_COLS]
)
```

Then add this function after `build_win_model_dataset` (around line 98):

```python
def build_total_model_dataset(team_features_df: pd.DataFrame) -> pd.DataFrame:
    """Build game-total prediction dataset by joining home and away team rows.

    Requires that home and away rows for the same game share the same game_id.
    Target: home_pts + away_pts (total points scored in the game).
    """
    df = team_features_df.copy()
    df = df.dropna(subset=TEAM_FEATURE_COLS)

    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()

    home_cols = {"game_id": "game_id", "game_date": "game_date", "season": "season", "pts": "home_pts"}
    home_cols.update({c: f"home_{c}" for c in TEAM_FEATURE_COLS})
    home_r = home.rename(columns=home_cols)[
        ["game_id", "game_date", "season", "home_pts"] + [f"home_{c}" for c in TEAM_FEATURE_COLS]
    ]

    away_cols = {"game_id": "game_id", "pts": "away_pts"}
    away_cols.update({c: f"away_{c}" for c in TEAM_FEATURE_COLS})
    away_r = away.rename(columns=away_cols)[
        ["game_id", "away_pts"] + [f"away_{c}" for c in TEAM_FEATURE_COLS]
    ]

    merged = home_r.merge(away_r, on="game_id", how="inner")
    merged["total"] = merged["home_pts"] + merged["away_pts"]

    keep = ["game_id", "game_date", "season"] + TOTAL_FEATURE_COLS + ["total"]
    return merged[keep].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_feature_engineering.py::test_total_feature_cols_has_home_and_away_prefixes tests/test_feature_engineering.py::test_build_total_model_dataset_has_total_target -v
```

Expected: 2 passed

- [ ] **Step 5: Run full test suite to check no regressions**

```powershell
pytest tests/ -v --tb=short
```

Expected: 89 passed (87 existing + 2 new)

- [ ] **Step 6: Commit**

```powershell
git add backend/ml/feature_engineering.py backend/tests/test_feature_engineering.py
git commit -m "feat: add TOTAL_FEATURE_COLS and build_total_model_dataset to feature engineering"
```

---

## Task 2: Training Script — train_total_model.py

**Files:**
- Create: `backend/ml/train_total_model.py`
- Create: `backend/tests/test_train_total_model.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_train_total_model.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

import joblib

from ml.feature_engineering import TOTAL_FEATURE_COLS


def _make_total_df(n: int = 60) -> pd.DataFrame:
    """Minimal dataset matching build_total_model_dataset output."""
    rng = np.random.default_rng(0)
    data = {col: rng.uniform(0, 1, n) for col in TOTAL_FEATURE_COLS}
    data["game_id"] = [f"G{i:04d}" for i in range(n)]
    data["game_date"] = pd.date_range("2024-10-01", periods=n, freq="2D")
    data["season"] = ["2024-25"] * 40 + ["2025-26"] * 20
    data["total"] = rng.uniform(200, 250, n)
    return pd.DataFrame(data)


def test_train_total_model_returns_model():
    from ml.train_total_model import train_total_model
    df = _make_total_df()
    train = df[df["season"] == "2024-25"]
    test = df[df["season"] == "2025-26"]
    model = train_total_model(train, test, mlflow_tracking=False)
    from xgboost import XGBRegressor
    assert isinstance(model, XGBRegressor)


def test_train_total_model_can_predict():
    from ml.train_total_model import train_total_model
    import numpy as np
    df = _make_total_df()
    train = df[df["season"] == "2024-25"]
    test = df[df["season"] == "2025-26"]
    model = train_total_model(train, test, mlflow_tracking=False)
    X = np.random.rand(3, len(TOTAL_FEATURE_COLS))
    preds = model.predict(X)
    assert preds.shape == (3,)


def test_train_total_model_saves_to_path(tmp_path):
    from ml.train_total_model import train_total_model, TOTAL_MODEL_PATH
    import ml.train_total_model as mod
    import monkeypatch
    df = _make_total_df()
    train = df[df["season"] == "2024-25"]
    test = df[df["season"] == "2025-26"]
    target = tmp_path / "total_model.joblib"

    orig = mod.TOTAL_MODEL_PATH
    mod.TOTAL_MODEL_PATH = target
    try:
        train_total_model(train, test, mlflow_tracking=False)
    finally:
        mod.TOTAL_MODEL_PATH = orig

    assert target.exists()
    loaded = joblib.load(target)
    assert loaded is not None
```

Wait — `monkeypatch` is a pytest fixture, not an import. Fix the test to use `monkeypatch` as a fixture parameter:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

import joblib

from ml.feature_engineering import TOTAL_FEATURE_COLS


def _make_total_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.uniform(0, 1, n) for col in TOTAL_FEATURE_COLS}
    data["game_id"] = [f"G{i:04d}" for i in range(n)]
    data["game_date"] = pd.date_range("2024-10-01", periods=n, freq="2D")
    data["season"] = ["2024-25"] * 40 + ["2025-26"] * 20
    data["total"] = rng.uniform(200, 250, n)
    return pd.DataFrame(data)


def test_train_total_model_returns_model():
    from ml.train_total_model import train_total_model
    from xgboost import XGBRegressor
    df = _make_total_df()
    model = train_total_model(
        df[df["season"] == "2024-25"],
        df[df["season"] == "2025-26"],
        mlflow_tracking=False,
    )
    assert isinstance(model, XGBRegressor)


def test_train_total_model_can_predict():
    from ml.train_total_model import train_total_model
    df = _make_total_df()
    model = train_total_model(
        df[df["season"] == "2024-25"],
        df[df["season"] == "2025-26"],
        mlflow_tracking=False,
    )
    X = np.random.default_rng(1).uniform(0, 1, (3, len(TOTAL_FEATURE_COLS)))
    assert model.predict(X).shape == (3,)


def test_train_total_model_saves_to_path(tmp_path, monkeypatch):
    import ml.train_total_model as mod
    monkeypatch.setattr(mod, "TOTAL_MODEL_PATH", tmp_path / "total_model.joblib")

    from ml.train_total_model import train_total_model
    df = _make_total_df()
    train_total_model(
        df[df["season"] == "2024-25"],
        df[df["season"] == "2025-26"],
        mlflow_tracking=False,
    )
    assert (tmp_path / "total_model.joblib").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_train_total_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'ml.train_total_model'`

- [ ] **Step 3: Implement train_total_model.py**

Create `backend/ml/train_total_model.py`:

```python
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    TOTAL_FEATURE_COLS,
    build_team_features,
    build_total_model_dataset,
    train_test_split_by_date,
)

logger = logging.getLogger(__name__)

TOTAL_MODEL_PATH: Path = Path(__file__).parent.parent / "models" / "total_model.joblib"

_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
    "device": "cuda",
}


def train_total_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlflow_tracking: bool = True,
) -> XGBRegressor:
    X_train = train_df[TOTAL_FEATURE_COLS].values
    y_train = train_df["total"].values
    X_test = test_df[TOTAL_FEATURE_COLS].values
    y_test = test_df["total"].values

    model = XGBRegressor(**_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    logger.info(f"Total Model — MAE: {mae:.2f}")

    TOTAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TOTAL_MODEL_PATH)

    if mlflow_tracking:
        import mlflow
        mlflow.set_experiment("game_total")
        with mlflow.start_run():
            mlflow.log_params(_PARAMS)
            mlflow.log_metric("mae", mae)
            mlflow.log_artifact(str(TOTAL_MODEL_PATH))

    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv()

    from app.models.database import get_session
    from app.models.team_game_log import TeamGameLog

    session = get_session()
    try:
        rows = session.query(TeamGameLog).all()
        if not rows:
            raise RuntimeError("No data in DB. Run seed_database.py first.")
        df = pd.DataFrame([{
            "team_id": r.team_id,
            "season": r.season,
            "game_id": r.game_id,
            "game_date": pd.Timestamp(r.game_date),
            "home_away": r.home_away,
            "wl": r.wl,
            "pts": r.pts,
        } for r in rows])
    finally:
        session.close()

    team_features = build_team_features(df)
    dataset = build_total_model_dataset(team_features)
    train_df, test_df = train_test_split_by_date(dataset)
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    train_total_model(train_df, test_df, mlflow_tracking=True)
    logger.info(f"Model saved to {TOTAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_train_total_model.py -v
```

Expected: 3 passed

- [ ] **Step 5: Run full suite — no regressions**

```powershell
pytest tests/ -v --tb=short
```

Expected: 92 passed

- [ ] **Step 6: Commit**

```powershell
git add backend/ml/train_total_model.py backend/tests/test_train_total_model.py
git commit -m "feat: add XGBoost game total (over/under) training script"
```

---

## Task 3: Inference — predict_game_total()

**Files:**
- Modify: `backend/ml/predict.py`
- Modify: `backend/tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

Add at the bottom of `backend/tests/test_predict.py`:

```python
@pytest.fixture()
def tiny_total_model_path(tmp_path):
    from xgboost import XGBRegressor
    from ml.feature_engineering import TOTAL_FEATURE_COLS
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (40, len(TOTAL_FEATURE_COLS)))
    y = rng.uniform(200, 250, 40)
    m = XGBRegressor(n_estimators=5, verbosity=0)
    m.fit(X, y)
    path = tmp_path / "total_model.joblib"
    joblib.dump(m, path)
    return path


def test_predict_game_total_returns_positive_float(tiny_total_model_path, monkeypatch):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "TOTAL_MODEL_PATH", tiny_total_model_path)

    from ml.predict import predict_game_total
    result = predict_game_total(_make_team_df(1), _make_team_df(2))
    assert "predicted_total" in result
    assert result["predicted_total"] > 0


def test_predict_game_total_raises_if_model_missing(tmp_path, monkeypatch):
    import ml.predict as mod
    mod.clear_model_cache()
    monkeypatch.setattr(mod, "TOTAL_MODEL_PATH", tmp_path / "nonexistent.joblib")

    from ml.predict import predict_game_total
    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict_game_total(_make_team_df(1), _make_team_df(2))
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_predict.py::test_predict_game_total_returns_positive_float tests/test_predict.py::test_predict_game_total_raises_if_model_missing -v
```

Expected: FAIL — `cannot import name 'predict_game_total'`

- [ ] **Step 3: Implement predict_game_total() in predict.py**

Add these two lines to the imports at the top of `backend/ml/predict.py` (after the existing imports):

```python
from ml.feature_engineering import (
    PLAYER_FEATURE_COLS,
    TEAM_FEATURE_COLS,
    TOTAL_FEATURE_COLS,        # ADD THIS
    build_player_features,
    build_team_features,
)
from ml.train_player_model import BEST_PLAYER_MODEL_PATH, STAT_MODEL_PATHS, STAT_TARGETS
from ml.train_total_model import TOTAL_MODEL_PATH       # ADD THIS LINE
from ml.train_win_model import WIN_MODEL_PATH
```

Then add this function at the end of `backend/ml/predict.py`:

```python
def predict_game_total(
    home_team_df: pd.DataFrame,
    away_team_df: pd.DataFrame,
) -> dict[str, float]:
    """Predict total points scored in a game (home + away).

    Args:
        home_team_df: Historical game logs for the home team.
            Required columns: team_id, season, game_id, game_date, home_away, wl, pts.
        away_team_df: Same for the away team.

    Returns:
        {"predicted_total": float}

    Raises:
        FileNotFoundError: if the total model .joblib has not been trained yet.
    """
    model = _load(TOTAL_MODEL_PATH)

    home_with_dummy = _add_team_dummy_row(home_team_df, is_home=True)
    away_with_dummy = _add_team_dummy_row(away_team_df, is_home=False)

    home_feat = build_team_features(home_with_dummy)
    away_feat = build_team_features(away_with_dummy)

    home_row = (
        home_feat[home_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS].iloc[0]
    )
    away_row = (
        away_feat[away_feat["game_id"] == "PREDICT"][TEAM_FEATURE_COLS].iloc[0]
    )

    combined = {f"home_{c}": home_row[c] for c in TEAM_FEATURE_COLS}
    combined.update({f"away_{c}": away_row[c] for c in TEAM_FEATURE_COLS})

    X = pd.DataFrame([combined])[TOTAL_FEATURE_COLS].values
    predicted_total = round(float(model.predict(X)[0]), 1)

    return {"predicted_total": predicted_total}
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_predict.py::test_predict_game_total_returns_positive_float tests/test_predict.py::test_predict_game_total_raises_if_model_missing -v
```

Expected: 2 passed

- [ ] **Step 5: Run full suite — no regressions**

```powershell
pytest tests/ -v --tb=short
```

Expected: 94 passed

- [ ] **Step 6: Commit**

```powershell
git add backend/ml/predict.py
git commit -m "feat: add predict_game_total inference function"
```

---

## Task 4: Backend API — Schema, Service, Router

**Files:**
- Modify: `backend/app/schemas/predictions.py`
- Modify: `backend/app/services/prediction_service.py`
- Modify: `backend/app/routers/predictions.py`
- Modify: `backend/tests/test_prediction_service.py`
- Modify: `backend/tests/test_predictions_api.py`

- [ ] **Step 1: Write failing tests**

**In `backend/tests/test_prediction_service.py`**, read the existing `_add_team_games` helper — it creates team rows with unique-per-team game_ids like `T1G0000`. For the total model service, the service uses the same `_fetch_team` pattern as win probability (it just passes two separate team DataFrames to `predict_game_total`). Add at the bottom:

```python
from app.services.prediction_service import (
    get_win_probability,
    get_best_player,
    get_player_stats,
    get_game_total,             # ADD
)


def test_get_game_total_raises_503_on_missing_model(db, monkeypatch):
    _add_team_games(db, team_id=1, n=15)
    _add_team_games(db, team_id=2, n=15)
    db.commit()

    monkeypatch.setattr(
        "ml.predict.TOTAL_MODEL_PATH",
        __import__("pathlib").Path("/nonexistent/total_model.joblib"),
    )
    import ml.predict as mod
    mod.clear_model_cache()

    with pytest.raises(FileNotFoundError):
        get_game_total(db, home_team_id=1, away_team_id=2)
```

**In `backend/tests/test_predictions_api.py`**, find the existing pattern (uses `TestClient` and `monkeypatch` to mock service calls). Add two tests at the bottom — read the file first to find the existing import line and client fixture, then add:

```python
def test_game_total_endpoint_returns_200(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.predictions.get_game_total",
        lambda db, home_team_id, away_team_id: {"predicted_total": 221.5, "confidence": "medium"},
    )
    response = client.get(
        "/predictions/game-total?home_team_id=1610612744&away_team_id=1610612747"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_total"] == 221.5
    assert data["confidence"] == "medium"


def test_game_total_endpoint_returns_503_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.predictions.get_game_total",
        lambda db, home_team_id, away_team_id: (_ for _ in ()).throw(
            FileNotFoundError("Model not found at /path/to/total_model.joblib")
        ),
    )
    response = client.get(
        "/predictions/game-total?home_team_id=1610612744&away_team_id=1610612747"
    )
    assert response.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_prediction_service.py::test_get_game_total_raises_503_on_missing_model tests/test_predictions_api.py::test_game_total_endpoint_returns_200 tests/test_predictions_api.py::test_game_total_endpoint_returns_503_when_model_missing -v
```

Expected: FAIL — import errors

- [ ] **Step 3: Add GameTotalResponse to schemas**

Add at the end of `backend/app/schemas/predictions.py`:

```python
class GameTotalResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    predicted_total: float
    confidence: str
```

- [ ] **Step 4: Add get_game_total() to prediction_service.py**

Add the confidence helper and service function. In `backend/app/services/prediction_service.py`, add after `_confidence()`:

```python
_NBA_AVG_TOTAL: float = 224.0


def _total_confidence(predicted: float) -> str:
    deviation = abs(predicted - _NBA_AVG_TOTAL)
    if deviation >= 12:
        return "high"
    if deviation >= 6:
        return "medium"
    return "low"
```

Then add the import of `predict_game_total` at the top (modify the existing ml.predict import line):

```python
from ml.predict import predict_best_player, predict_game_total, predict_player_stats, predict_win_probability
```

Then add the function at the end of `prediction_service.py`:

```python
def get_game_total(
    db: Session,
    home_team_id: int,
    away_team_id: int,
) -> dict:
    """Predict total game score (home + away points).

    Returns:
        {"predicted_total": float, "confidence": str}
    Raises:
        ValueError: if a team has fewer than 2 games in the DB.
        FileNotFoundError: propagated from predict if model not trained.
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
                f"Not enough game data for team {team_id} "
                f"(found {len(rows)} games, need at least 2)."
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

    result = predict_game_total(home_df, away_df)
    return {**result, "confidence": _total_confidence(result["predicted_total"])}
```

- [ ] **Step 5: Add route to router**

In `backend/app/routers/predictions.py`, update the import lines:

```python
from app.schemas.predictions import (
    BestPlayerResponse,
    GameTotalResponse,           # ADD
    PlayerStarPrediction,
    PlayerStatsResponse,
    StatPrediction,
    WinProbabilityResponse,
)
from app.services.prediction_service import get_best_player, get_game_total, get_player_stats, get_win_probability
```

Then add the route at the end of `predictions.py`:

```python
@router.get("/game-total", response_model=GameTotalResponse)
def game_total(
    home_team_id: int = Query(..., description="NBA team ID of the home team"),
    away_team_id: int = Query(..., description="NBA team ID of the away team"),
    db: Session = Depends(get_db),
) -> GameTotalResponse:
    try:
        result = get_game_total(db, home_team_id, away_team_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return GameTotalResponse(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        predicted_total=result["predicted_total"],
        confidence=result["confidence"],
    )
```

- [ ] **Step 6: Run all three new tests to verify they pass**

```powershell
pytest tests/test_prediction_service.py::test_get_game_total_raises_503_on_missing_model tests/test_predictions_api.py::test_game_total_endpoint_returns_200 tests/test_predictions_api.py::test_game_total_endpoint_returns_503_when_model_missing -v
```

Expected: 3 passed

- [ ] **Step 7: Run full suite — no regressions**

```powershell
pytest tests/ -v --tb=short
```

Expected: 97 passed

- [ ] **Step 8: Commit**

```powershell
git add backend/app/schemas/predictions.py backend/app/services/prediction_service.py backend/app/routers/predictions.py backend/tests/test_prediction_service.py backend/tests/test_predictions_api.py
git commit -m "feat: add game-total prediction endpoint (schema, service, router)"
```

---

## Task 5: Frontend — Types, API Fetch, Hook

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useGameTotal.ts`

- [ ] **Step 1: Add GameTotalResponse to types.ts**

Read `frontend/src/lib/types.ts` first. It has 5 interfaces. Add at the end:

```typescript
export interface GameTotalResponse {
  home_team_id: number
  away_team_id: number
  predicted_total: number
  confidence: string
}
```

- [ ] **Step 2: Add fetchGameTotal to api.ts**

Read `frontend/src/lib/api.ts`. It exports `fetchWinProbability`, `fetchBestPlayer`, `fetchPlayerStats`. Add:

```typescript
export function fetchGameTotal(
  homeTeamId: number,
  awayTeamId: number,
): Promise<GameTotalResponse> {
  return apiFetch<GameTotalResponse>(
    `/predictions/game-total?home_team_id=${homeTeamId}&away_team_id=${awayTeamId}`,
  )
}
```

Also add `GameTotalResponse` to the import from `./types` at the top of api.ts.

- [ ] **Step 3: Add test for fetchGameTotal in api.test.ts**

In `frontend/__tests__/api.test.ts`, add one test following the same pattern as the existing fetch tests:

```typescript
it("fetchGameTotal calls correct URL", async () => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      home_team_id: 1,
      away_team_id: 2,
      predicted_total: 221.5,
      confidence: "medium",
    }),
  } as Response)

  await fetchGameTotal(1, 2)
  expect(global.fetch).toHaveBeenCalledWith(
    expect.stringContaining("/predictions/game-total?home_team_id=1&away_team_id=2"),
  )
})
```

- [ ] **Step 4: Create useGameTotal.ts**

Create `frontend/src/hooks/useGameTotal.ts`:

```typescript
import { useQuery } from "@tanstack/react-query"

import { fetchGameTotal } from "@/lib/api"
import type { GameTotalResponse } from "@/lib/types"

export function useGameTotal(
  homeTeamId: number | null,
  awayTeamId: number | null,
) {
  return useQuery<GameTotalResponse, Error>({
    queryKey: ["game-total", homeTeamId, awayTeamId],
    queryFn: () => fetchGameTotal(homeTeamId!, awayTeamId!),
    enabled: homeTeamId !== null && awayTeamId !== null,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })
}
```

- [ ] **Step 5: Run frontend tests**

```powershell
cd frontend
npm test -- --testPathPattern="api.test"
```

Expected: 5 passed (4 existing + 1 new)

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/hooks/useGameTotal.ts frontend/__tests__/api.test.ts
git commit -m "feat: add GameTotalResponse type, fetchGameTotal API call, and useGameTotal hook"
```

---

## Task 6: Frontend — GameTotalCard Component + Wire into Page

**Files:**
- Create: `frontend/src/components/GameTotalCard.tsx`
- Modify: `frontend/src/app/page.tsx`
- Create: `frontend/__tests__/GameTotalCard.test.tsx`

- [ ] **Step 1: Write failing component tests**

Create `frontend/__tests__/GameTotalCard.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { GameTotalCard } from "@/components/GameTotalCard"
import { useGameTotal } from "@/hooks/useGameTotal"

jest.mock("@/hooks/useGameTotal")
const mockUseGameTotal = useGameTotal as jest.MockedFunction<typeof useGameTotal>

describe("GameTotalCard", () => {
  it("shows loading state", () => {
    mockUseGameTotal.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as any)
    render(<GameTotalCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })

  it("shows predicted total and confidence badge", () => {
    mockUseGameTotal.mockReturnValue({
      data: {
        home_team_id: 1,
        away_team_id: 2,
        predicted_total: 221.5,
        confidence: "high",
      },
      isLoading: false,
      error: null,
    } as any)
    render(<GameTotalCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText("221.5")).toBeInTheDocument()
    expect(screen.getByText("high")).toBeInTheDocument()
  })

  it("shows error message", () => {
    mockUseGameTotal.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Model not trained"),
    } as any)
    render(<GameTotalCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText(/model not trained/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
npm test -- --testPathPattern="GameTotalCard"
```

Expected: FAIL — `Cannot find module '@/components/GameTotalCard'`

- [ ] **Step 3: Implement GameTotalCard.tsx**

Create `frontend/src/components/GameTotalCard.tsx`:

```tsx
"use client"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useGameTotal } from "@/hooks/useGameTotal"

const CONFIDENCE_VARIANT: Record<string, "default" | "secondary" | "outline"> = {
  high: "default",
  medium: "secondary",
  low: "outline",
}

interface Props {
  homeTeamId: number
  awayTeamId: number
}

export function GameTotalCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useGameTotal(homeTeamId, awayTeamId)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center justify-between">
          Game Total (O/U)
          {data && (
            <Badge variant={CONFIDENCE_VARIANT[data.confidence] ?? "outline"}>
              {data.confidence}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <div className="text-center">
            <div className="text-4xl font-bold">{data.predicted_total}</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">
              Predicted Total Points
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
npm test -- --testPathPattern="GameTotalCard"
```

Expected: 3 passed

- [ ] **Step 5: Wire into page.tsx**

In `frontend/src/app/page.tsx`, add the import:

```tsx
import { GameTotalCard } from "@/components/GameTotalCard"
```

Then update the matchup results section. Currently:

```tsx
{matchup && (
  <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
    <WinProbabilityCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
    <BestPlayerCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
  </section>
)}
```

Replace with:

```tsx
{matchup && (
  <>
    <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
      <WinProbabilityCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
      <BestPlayerCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
    </section>
    <section className="mb-8">
      <GameTotalCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
    </section>
  </>
)}
```

- [ ] **Step 6: Run full frontend test suite**

```powershell
npm test
```

Expected: 17 passed (13 existing + 1 api + 3 GameTotalCard)

- [ ] **Step 7: Production build check**

```powershell
npm run build
```

Expected: ✓ Compiled successfully

- [ ] **Step 8: Commit**

```powershell
git add frontend/src/components/GameTotalCard.tsx frontend/src/app/page.tsx frontend/__tests__/GameTotalCard.test.tsx
git commit -m "feat: add GameTotalCard component and wire into home page"
```

---

## Task 7: Update HANDOFF.md

**Files:**
- Modify: `nba-ai-predictor/HANDOFF.md`

- [ ] **Step 1: Update HANDOFF.md**

Add a Phase 6 section to `HANDOFF.md` with the git log, new API endpoint, new component, and updated test counts (backend 97, frontend 17).

- [ ] **Step 2: Commit**

```powershell
git add HANDOFF.md
git commit -m "docs: update HANDOFF.md with Phase 6 game total prediction"
```

---

## Self-Review

**Spec coverage:**
1. ✅ Predicts game total score — `build_total_model_dataset` + `train_total_model.py` + `predict_game_total`
2. ✅ Confidence score — `_total_confidence()` in service
3. ✅ API endpoint — `GET /predictions/game-total`
4. ✅ Frontend display — `GameTotalCard.tsx` with total + badge
5. ✅ Wired into page — added below WinProbabilityCard/BestPlayerCard grid

**Placeholder scan:** No TBDs, all code blocks are complete.

**Type consistency:**
- `TOTAL_FEATURE_COLS` defined in Task 1, used in Tasks 2, 3
- `TOTAL_MODEL_PATH` defined in `train_total_model.py` (Task 2), imported in `predict.py` (Task 3)
- `predict_game_total` defined in Task 3, imported in `prediction_service.py` (Task 4)
- `get_game_total` defined in Task 4, imported in router (Task 4) and tests (Task 4)
- `GameTotalResponse` defined in Task 4 (schema), re-used in Task 5 (types.ts), Task 6 (component test)
- `useGameTotal` created in Task 5, used in Task 6 — signatures match
