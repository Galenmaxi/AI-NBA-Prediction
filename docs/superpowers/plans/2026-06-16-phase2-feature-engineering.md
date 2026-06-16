# Phase 2: Feature Engineering + Model Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build rolling-window features from game log data and train XGBoost/LightGBM models that predict win probability, best player (star), and individual player stat lines, all with zero data leakage.

**Architecture:** Feature engineering computes rolling stats per team/player using a shift-before-roll pattern so no current-game data can leak into features. Training scripts accept DataFrames (fully testable with synthetic data) and have a `main()` that loads from PostgreSQL. MLflow tracks experiments locally in `backend/mlruns/`.

**Tech Stack:** scikit-learn ≥1.5, xgboost ≥2.1, lightgbm ≥4.5, mlflow ≥2.0, joblib (ships with scikit-learn)

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `backend/requirements.txt` | Modify | Add scikit-learn, xgboost, lightgbm, mlflow |
| `backend/models/.gitkeep` | Create | Keeps `models/` dir in git; `.joblib` files are already git-ignored |
| `backend/ml/feature_engineering.py` | Replace placeholder | Rolling features, dataset builders, train/test split |
| `backend/ml/train_win_model.py` | Replace placeholder | XGBoost win probability training |
| `backend/ml/train_player_model.py` | Replace placeholder | LightGBM best player + XGBoost stat regressors |
| `backend/tests/test_feature_engineering.py` | Create | Unit tests for all feature functions (TDD) |
| `backend/tests/test_train_win_model.py` | Create | Smoke tests for win model pipeline |
| `backend/tests/test_train_player_model.py` | Create | Smoke tests for player model pipeline |

---

### Task 1: Install Phase 2 Dependencies

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/models/.gitkeep`

- [ ] **Step 1: Install new packages**

With venv active (`.\venv\Scripts\activate` from `backend/`):

```powershell
pip install scikit-learn xgboost lightgbm mlflow
```

- [ ] **Step 2: Verify the install**

```powershell
python -c "import sklearn, xgboost, lightgbm, mlflow; print(sklearn.__version__, xgboost.__version__, lightgbm.__version__, mlflow.__version__)"
```

Expected: four version strings printed with no errors.

- [ ] **Step 3: Pin exact versions in requirements.txt**

Run `pip freeze` and add the four new packages. Example output (your versions may differ slightly — use what pip installed):

```
scikit-learn==1.6.1
xgboost==2.1.4
lightgbm==4.6.0
mlflow==2.22.0
```

Add those exact lines to `backend/requirements.txt`.

- [ ] **Step 4: Verify all 21 existing tests still pass**

```powershell
pytest tests/ -v
```

Expected: `21 passed`

- [ ] **Step 5: Create the models directory**

```powershell
New-Item -ItemType Directory -Force backend\models
New-Item -ItemType File -Path backend\models\.gitkeep
```

- [ ] **Step 6: Commit**

```powershell
git add backend/requirements.txt backend/models/.gitkeep
git commit -m "chore: add Phase 2 ML dependencies (scikit-learn, xgboost, lightgbm, mlflow)"
```

---

### Task 2: Team Feature Engineering

**Files:**
- Replace: `backend/ml/feature_engineering.py`
- Create: `backend/tests/test_feature_engineering.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_feature_engineering.py`:

```python
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from ml.feature_engineering import build_team_features


def make_team_df(
    wl_sequence: list[str],
    team_id: int = 1,
    pts_sequence: list[int] | None = None,
) -> pd.DataFrame:
    n = len(wl_sequence)
    if pts_sequence is None:
        pts_sequence = [100 + i for i in range(n)]
    base_date = date(2024, 10, 1)
    return pd.DataFrame({
        "team_id": [team_id] * n,
        "season": ["2024-25"] * n,
        "game_id": [f"G{team_id}{i:03d}" for i in range(n)],
        "game_date": pd.to_datetime([base_date + timedelta(days=i * 2) for i in range(n)]),
        "home_away": ["HOME" if i % 2 == 0 else "AWAY" for i in range(n)],
        "wl": wl_sequence,
        "pts": pts_sequence,
    })


def test_build_team_features_adds_is_home():
    df = make_team_df(["W", "L"])
    result = build_team_features(df)
    assert result.iloc[0]["is_home"] == 1
    assert result.iloc[1]["is_home"] == 0


def test_build_team_features_first_game_win_pct_is_nan():
    df = make_team_df(["W", "W", "W"])
    result = build_team_features(df)
    assert pd.isna(result.iloc[0]["win_pct_last10"])


def test_build_team_features_win_pct_uses_only_past_games():
    """Game 5 (a loss) must have win_pct = 1.0 from its prior 4 wins."""
    df = make_team_df(["W", "W", "W", "W", "L"])
    result = build_team_features(df)
    assert result.iloc[4]["win_pct_last10"] == pytest.approx(1.0)
    assert result.iloc[4]["wl"] == "L"


def test_build_team_features_pts_avg_last5_uses_only_past():
    pts = [100, 110, 120, 130, 140, 999]
    df = make_team_df(["W"] * 6, pts_sequence=pts)
    result = build_team_features(df)
    expected = (100 + 110 + 120 + 130 + 140) / 5
    assert result.iloc[5]["pts_avg_last5"] == pytest.approx(expected)


def test_build_team_features_rest_days_first_game():
    df = make_team_df(["W", "L"])
    result = build_team_features(df)
    assert result.iloc[0]["rest_days"] == pytest.approx(7.0)


def test_build_team_features_rest_days_subsequent():
    df = make_team_df(["W", "L"])  # games 2 days apart (timedelta days=2)
    result = build_team_features(df)
    assert result.iloc[1]["rest_days"] == pytest.approx(2.0)


def test_build_team_features_season_win_pct_uses_only_past():
    df = make_team_df(["W", "W", "L", "W"])
    result = build_team_features(df)
    # game 4 sees history: W, W, L → 2/3 ≈ 0.667
    assert result.iloc[3]["season_win_pct"] == pytest.approx(2 / 3, abs=0.01)


def test_build_team_features_preserves_original_columns():
    df = make_team_df(["W", "L", "W"])
    result = build_team_features(df)
    for col in ["team_id", "season", "game_id", "game_date", "wl", "pts"]:
        assert col in result.columns


def test_build_team_features_two_teams_independent():
    team1 = make_team_df(["W", "W", "W"], team_id=1)
    team2 = make_team_df(["L", "L", "L"], team_id=2)
    result = build_team_features(pd.concat([team1, team2], ignore_index=True))
    t1 = result[result["team_id"] == 1]
    t2 = result[result["team_id"] == 2]
    assert t1.iloc[2]["win_pct_last10"] == pytest.approx(1.0)
    assert t2.iloc[2]["win_pct_last10"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_feature_engineering.py -v
```

Expected: all 9 tests FAIL with `ImportError: cannot import name 'build_team_features'`

- [ ] **Step 3: Implement `build_team_features`**

Replace the placeholder in `backend/ml/feature_engineering.py` with:

```python
import pandas as pd

TEAM_FEATURE_COLS: list[str] = [
    "is_home",
    "rest_days",
    "win_pct_last10",
    "pts_avg_last5",
    "pts_avg_last10",
    "season_win_pct",
]

PLAYER_FEATURE_COLS: list[str] = [
    "pts_avg_last5", "pts_avg_last10",
    "reb_avg_last5", "reb_avg_last10",
    "ast_avg_last5", "ast_avg_last10",
    "stl_avg_last5",
    "blk_avg_last5",
    "min_avg_last5",
]


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling team features to a team game log DataFrame.

    Uses shift(1)-before-roll so no current-game data leaks into features.
    All rolling stats reflect only games played strictly before the current game.
    """
    df = df.copy()
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    df["is_home"] = (df["home_away"] == "HOME").astype(int)
    df["win"] = (df["wl"] == "W").astype(int)

    df["rest_days"] = (
        df.groupby("team_id")["game_date"]
        .transform(lambda x: x.diff().dt.days.fillna(7.0))
    )
    df["win_pct_last10"] = (
        df.groupby("team_id")["win"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    df["pts_avg_last5"] = (
        df.groupby("team_id")["pts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["pts_avg_last10"] = (
        df.groupby("team_id")["pts"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    df["season_win_pct"] = (
        df.groupby(["team_id", "season"])["win"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean().fillna(0.5))
    )

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_feature_engineering.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```powershell
git add backend/ml/feature_engineering.py backend/tests/test_feature_engineering.py
git commit -m "feat: add team rolling feature engineering with no-leakage guarantee"
```

---

### Task 3: Player Feature Engineering

**Files:**
- Modify: `backend/ml/feature_engineering.py`
- Modify: `backend/tests/test_feature_engineering.py`

- [ ] **Step 1: Write the failing tests**

Update the import at the top of `backend/tests/test_feature_engineering.py` to:

```python
from ml.feature_engineering import build_team_features, build_player_features
```

Then append these tests at the bottom of the file:

```python
def make_player_df(pts_sequence: list[int], player_id: int = 100) -> pd.DataFrame:
    n = len(pts_sequence)
    base_date = date(2024, 10, 1)
    return pd.DataFrame({
        "player_id": [player_id] * n,
        "season": ["2024-25"] * n,
        "game_id": [f"PG{player_id}{i:03d}" for i in range(n)],
        "game_date": pd.to_datetime([base_date + timedelta(days=i * 2) for i in range(n)]),
        "wl": ["W"] * n,
        "pts": pts_sequence,
        "reb": [5] * n,
        "ast": [3] * n,
        "stl": [1] * n,
        "blk": [0] * n,
        "min": [30.0] * n,
    })


def test_build_player_features_first_game_pts_avg_is_nan():
    df = make_player_df([30, 25, 20])
    result = build_player_features(df)
    assert pd.isna(result.iloc[0]["pts_avg_last5"])


def test_build_player_features_pts_avg_uses_only_past():
    pts = [10, 20, 30, 40, 50, 999]
    df = make_player_df(pts)
    result = build_player_features(df)
    # game 6 (index 5): shift(1) then rolling(5) sees games 1-5: [10,20,30,40,50]
    expected = (10 + 20 + 30 + 40 + 50) / 5
    assert result.iloc[5]["pts_avg_last5"] == pytest.approx(expected)


def test_build_player_features_adds_all_expected_columns():
    df = make_player_df([20, 25, 30])
    result = build_player_features(df)
    expected_cols = {
        "pts_avg_last5", "pts_avg_last10",
        "reb_avg_last5", "reb_avg_last10",
        "ast_avg_last5", "ast_avg_last10",
        "stl_avg_last5", "blk_avg_last5", "min_avg_last5",
    }
    assert expected_cols.issubset(set(result.columns))


def test_build_player_features_two_players_independent():
    p1 = make_player_df([100, 100, 100], player_id=1)
    p2 = make_player_df([10, 10, 10], player_id=2)
    result = build_player_features(pd.concat([p1, p2], ignore_index=True))
    r1 = result[result["player_id"] == 1]
    r2 = result[result["player_id"] == 2]
    assert r1.iloc[2]["pts_avg_last5"] == pytest.approx(100.0)
    assert r2.iloc[2]["pts_avg_last5"] == pytest.approx(10.0)
```

- [ ] **Step 2: Run new tests to verify they fail**

```powershell
pytest tests/test_feature_engineering.py -k "player" -v
```

Expected: FAIL with `ImportError: cannot import name 'build_player_features'`

- [ ] **Step 3: Implement `build_player_features`**

Add this function to `backend/ml/feature_engineering.py` (after `build_team_features`):

```python
def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling player features to a player game log DataFrame.

    Uses shift(1)-before-roll for zero data leakage.
    NaN stats (DNP games) are preserved; callers should filter as needed.
    """
    df = df.copy()
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    _roll_stats: dict[str, list[int]] = {
        "pts": [5, 10],
        "reb": [5, 10],
        "ast": [5, 10],
        "stl": [5],
        "blk": [5],
        "min": [5],
    }

    for stat, windows in _roll_stats.items():
        if stat not in df.columns:
            continue
        for w in windows:
            df[f"{stat}_avg_last{w}"] = (
                df.groupby("player_id")[stat]
                .transform(lambda x, window=w: x.shift(1).rolling(window, min_periods=1).mean())
            )

    return df
```

> **Note:** The `window=w` default argument captures the loop variable correctly — without it, all lambdas would reference the last value of `w`.

- [ ] **Step 4: Run all feature engineering tests**

```powershell
pytest tests/test_feature_engineering.py -v
```

Expected: `13 passed`

- [ ] **Step 5: Commit**

```powershell
git add backend/ml/feature_engineering.py backend/tests/test_feature_engineering.py
git commit -m "feat: add player rolling feature engineering"
```

---

### Task 4: Dataset Builders + Train/Test Split

**Files:**
- Modify: `backend/ml/feature_engineering.py`
- Modify: `backend/tests/test_feature_engineering.py`

- [ ] **Step 1: Write the failing tests**

Update the import at the top of `backend/tests/test_feature_engineering.py` to:

```python
from ml.feature_engineering import (
    build_team_features,
    build_player_features,
    build_win_model_dataset,
    build_player_star_dataset,
    build_player_stats_dataset,
    train_test_split_by_date,
    TEAM_FEATURE_COLS,
    PLAYER_FEATURE_COLS,
)
```

Then append these tests at the bottom of the file:

```python
def make_full_team_df(n_games: int = 20) -> pd.DataFrame:
    rows = []
    base = date(2024, 10, 1)
    for team_id in [1, 2]:
        for i in range(n_games):
            rows.append({
                "team_id": team_id,
                "season": "2024-25",
                "game_id": f"G{team_id}{i:03d}",
                "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
                "home_away": "HOME" if team_id == 1 else "AWAY",
                "wl": "W" if i % 2 == 0 else "L",
                "pts": 100 + i,
            })
    return pd.DataFrame(rows)


def make_full_player_df(n_games: int = 20) -> pd.DataFrame:
    rows = []
    base = date(2024, 10, 1)
    for player_id, base_pts in [(1, 30), (2, 20), (3, 15)]:
        for i in range(n_games):
            rows.append({
                "player_id": player_id,
                "season": "2024-25",
                "game_id": f"PG{player_id}{i:03d}",
                "game_date": pd.Timestamp(base + timedelta(days=i * 2)),
                "wl": "W" if i % 2 == 0 else "L",
                "pts": base_pts + i % 5,
                "reb": 5 + i % 3,
                "ast": 3 + i % 4,
                "stl": 1,
                "blk": 0,
                "min": 30.0,
                "team_id": 1,
            })
    return pd.DataFrame(rows)


def test_build_win_model_dataset_returns_expected_columns():
    features = build_team_features(make_full_team_df(15))
    result = build_win_model_dataset(features)
    for col in TEAM_FEATURE_COLS + ["target", "game_id", "team_id", "game_date", "season"]:
        assert col in result.columns, f"Missing column: {col}"


def test_build_win_model_dataset_target_is_binary():
    features = build_team_features(make_full_team_df(15))
    result = build_win_model_dataset(features)
    assert set(result["target"].unique()).issubset({0, 1})


def test_build_win_model_dataset_drops_nan_rows():
    """First game per team has NaN rolling features and must be excluded."""
    features = build_team_features(make_full_team_df(5))
    result = build_win_model_dataset(features)
    assert result[TEAM_FEATURE_COLS].isna().sum().sum() == 0


def test_build_player_star_dataset_has_is_star_column():
    features = build_player_features(make_full_player_df(15))
    result = build_player_star_dataset(features)
    assert "is_star" in result.columns
    assert set(result["is_star"].unique()).issubset({0, 1})


def test_build_player_stats_dataset_returns_expected_columns():
    features = build_player_features(make_full_player_df(15))
    result = build_player_stats_dataset(features)
    for col in PLAYER_FEATURE_COLS + ["pts", "reb", "ast"]:
        assert col in result.columns, f"Missing column: {col}"


def test_train_test_split_by_date_correct_seasons():
    rows = []
    for season, n in [("2023-24", 10), ("2024-25", 10), ("2025-26", 5)]:
        for i in range(n):
            rows.append({"season": season, "game_id": f"{season}-{i}"})
    df = pd.DataFrame(rows)
    train, test = train_test_split_by_date(df, test_season="2025-26")
    assert set(train["season"].unique()) == {"2023-24", "2024-25"}
    assert set(test["season"].unique()) == {"2025-26"}


def test_train_test_split_no_overlap():
    rows = [{"season": "2023-24", "game_id": f"A{i}"} for i in range(5)]
    rows += [{"season": "2025-26", "game_id": f"B{i}"} for i in range(5)]
    df = pd.DataFrame(rows)
    train, test = train_test_split_by_date(df)
    assert len(set(train["game_id"]) & set(test["game_id"])) == 0
```

- [ ] **Step 2: Run new tests to verify they fail**

```powershell
pytest tests/test_feature_engineering.py -k "dataset or split" -v
```

Expected: FAIL with `ImportError` on `build_win_model_dataset`

- [ ] **Step 3: Implement dataset builders and split**

Add these four functions to `backend/ml/feature_engineering.py` (after `build_player_features`):

```python
def build_win_model_dataset(team_features_df: pd.DataFrame) -> pd.DataFrame:
    """Build win probability training dataset.

    Returns one row per team per game with rolling features and a binary target.
    Rows with NaN rolling features (first game of a team) are dropped.
    """
    df = team_features_df.copy()
    df["target"] = (df["wl"] == "W").astype(int)
    df = df.dropna(subset=TEAM_FEATURE_COLS)
    keep = ["game_id", "team_id", "game_date", "season"] + TEAM_FEATURE_COLS + ["target"]
    return df[keep].reset_index(drop=True)


def build_player_star_dataset(player_features_df: pd.DataFrame) -> pd.DataFrame:
    """Build best-player (star) prediction dataset.

    Target: within each game_id, which player had the highest fantasy score?
    Fantasy score = pts + 1.2*reb + 1.5*ast + 3*stl + 3*blk
    Returns one row per player per game with 'is_star' binary target.
    """
    df = player_features_df.copy()
    df = df.dropna(subset=PLAYER_FEATURE_COLS)
    df["fantasy_score"] = (
        df["pts"] + 1.2 * df["reb"] + 1.5 * df["ast"] + 3 * df["stl"] + 3 * df["blk"]
    )
    df["is_star"] = df.groupby("game_id")["fantasy_score"].transform(
        lambda x: (x == x.max()).astype(int)
    )
    keep = ["game_id", "player_id", "game_date", "season"] + PLAYER_FEATURE_COLS + ["is_star"]
    available = [c for c in keep if c in df.columns]
    return df[available].reset_index(drop=True)


def build_player_stats_dataset(player_features_df: pd.DataFrame) -> pd.DataFrame:
    """Build player stat prediction dataset.

    Returns one row per player per game with rolling features and target stat columns.
    Rows missing rolling features (first games) are dropped.
    """
    df = player_features_df.copy()
    df = df.dropna(subset=PLAYER_FEATURE_COLS)
    keep = (
        ["game_id", "player_id", "game_date", "season"]
        + PLAYER_FEATURE_COLS
        + ["pts", "reb", "ast", "stl", "blk"]
    )
    available = [c for c in keep if c in df.columns]
    return df[available].reset_index(drop=True)


def train_test_split_by_date(
    df: pd.DataFrame,
    test_season: str = "2025-26",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season: train = all seasons except test_season, test = test_season.

    Guarantees zero temporal leakage: the test set is always newer than the train set.
    """
    train = df[df["season"] != test_season].copy().reset_index(drop=True)
    test = df[df["season"] == test_season].copy().reset_index(drop=True)
    return train, test
```

- [ ] **Step 4: Run all feature engineering tests**

```powershell
pytest tests/test_feature_engineering.py -v
```

Expected: `20 passed`

- [ ] **Step 5: Commit**

```powershell
git add backend/ml/feature_engineering.py backend/tests/test_feature_engineering.py
git commit -m "feat: add dataset builders and date-based train/test split"
```

---

### Task 5: Win Probability Model

**Files:**
- Replace: `backend/ml/train_win_model.py`
- Create: `backend/tests/test_train_win_model.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_train_win_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from ml.feature_engineering import TEAM_FEATURE_COLS
from ml.train_win_model import FEATURE_COLS as WIN_FEATURE_COLS
from ml.train_win_model import WIN_MODEL_PATH, train_win_model


def make_win_dataset(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = date(2023, 10, 1)
    rows = []
    for i in range(n):
        rows.append({
            "game_id": f"G{i:04d}",
            "team_id": i % 30 + 1,
            "game_date": pd.Timestamp(base + timedelta(days=i)),
            "season": "2023-24",
            "is_home": int(rng.integers(0, 2)),
            "rest_days": float(rng.integers(1, 8)),
            "win_pct_last10": float(rng.uniform(0, 1)),
            "pts_avg_last5": float(rng.uniform(90, 130)),
            "pts_avg_last10": float(rng.uniform(90, 130)),
            "season_win_pct": float(rng.uniform(0, 1)),
            "target": int(rng.integers(0, 2)),
        })
    return pd.DataFrame(rows)


def test_train_win_model_returns_model():
    df = make_win_dataset(60)
    train, test = df.iloc[:45], df.iloc[45:]
    model = train_win_model(train, test, mlflow_tracking=False)
    assert model is not None


def test_train_win_model_model_can_predict_probabilities():
    df = make_win_dataset(60)
    train, test = df.iloc[:45], df.iloc[45:]
    model = train_win_model(train, test, mlflow_tracking=False)
    proba = model.predict_proba(test[WIN_FEATURE_COLS].values)
    assert proba.shape == (len(test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_win_feature_cols_match_team_feature_cols():
    assert set(WIN_FEATURE_COLS) == set(TEAM_FEATURE_COLS)


def test_train_win_model_saves_to_path(tmp_path):
    import ml.train_win_model as mod
    original = mod.WIN_MODEL_PATH
    mod.WIN_MODEL_PATH = tmp_path / "win_model.joblib"
    try:
        df = make_win_dataset(60)
        train_win_model(df.iloc[:45], df.iloc[45:], mlflow_tracking=False)
        assert (tmp_path / "win_model.joblib").exists()
    finally:
        mod.WIN_MODEL_PATH = original
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_train_win_model.py -v
```

Expected: FAIL with `ImportError: cannot import name 'train_win_model'`

- [ ] **Step 3: Implement `train_win_model.py`**

Replace the placeholder in `backend/ml/train_win_model.py` with:

```python
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    TEAM_FEATURE_COLS,
    build_team_features,
    build_win_model_dataset,
    train_test_split_by_date,
)

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = TEAM_FEATURE_COLS
WIN_MODEL_PATH: Path = Path(__file__).parent.parent / "models" / "win_model.joblib"

_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}


def train_win_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlflow_tracking: bool = True,
) -> XGBClassifier:
    """Train XGBoost win probability model.

    Args:
        train_df: DataFrame with FEATURE_COLS + 'target' column.
        test_df: DataFrame with FEATURE_COLS + 'target' column.
        mlflow_tracking: When False, skips MLflow (used in tests).

    Returns:
        Trained XGBClassifier. Model is also saved to WIN_MODEL_PATH.
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["target"].values

    model = XGBClassifier(**_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)
    ll = log_loss(y_test, preds_proba)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Win Model — log_loss: {ll:.4f}, accuracy: {acc:.4f}")

    WIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, WIN_MODEL_PATH)

    if mlflow_tracking:
        import mlflow
        mlflow.set_experiment("win_probability")
        with mlflow.start_run():
            mlflow.log_params(_PARAMS)
            mlflow.log_metric("log_loss", ll)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_artifact(str(WIN_MODEL_PATH))

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
    dataset = build_win_model_dataset(team_features)
    train_df, test_df = train_test_split_by_date(dataset)
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    train_win_model(train_df, test_df, mlflow_tracking=True)
    logger.info(f"Model saved to {WIN_MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_train_win_model.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Run full test suite**

```powershell
pytest tests/ -v
```

Expected: `45 passed` (21 original + 20 feature engineering + 4 win model)

- [ ] **Step 6: Commit**

```powershell
git add backend/ml/train_win_model.py backend/tests/test_train_win_model.py
git commit -m "feat: add XGBoost win probability model training pipeline"
```

---

### Task 6: Best Player + Player Stat Models

**Files:**
- Replace: `backend/ml/train_player_model.py`
- Create: `backend/tests/test_train_player_model.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_train_player_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from ml.feature_engineering import PLAYER_FEATURE_COLS
from ml.train_player_model import (
    BEST_PLAYER_MODEL_PATH,
    STAT_MODEL_PATHS,
    STAT_TARGETS,
    train_best_player_model,
    train_stat_models,
)


def make_player_dataset(n_games: int = 20, n_players: int = 5) -> pd.DataFrame:
    """Synthetic player dataset: n_games × n_players rows, one star per game."""
    rng = np.random.default_rng(42)
    base = date(2023, 10, 1)
    rows = []
    for game_i in range(n_games):
        game_id = f"G{game_i:04d}"
        for player_id in range(1, n_players + 1):
            rows.append({
                "game_id": game_id,
                "player_id": player_id,
                "game_date": pd.Timestamp(base + timedelta(days=game_i)),
                "season": "2023-24",
                "pts_avg_last5": float(rng.uniform(10, 35)),
                "pts_avg_last10": float(rng.uniform(10, 35)),
                "reb_avg_last5": float(rng.uniform(3, 12)),
                "reb_avg_last10": float(rng.uniform(3, 12)),
                "ast_avg_last5": float(rng.uniform(1, 10)),
                "ast_avg_last10": float(rng.uniform(1, 10)),
                "stl_avg_last5": float(rng.uniform(0, 3)),
                "blk_avg_last5": float(rng.uniform(0, 2)),
                "min_avg_last5": float(rng.uniform(20, 40)),
                "pts": int(rng.integers(5, 45)),
                "reb": int(rng.integers(2, 15)),
                "ast": int(rng.integers(0, 12)),
                "stl": int(rng.integers(0, 4)),
                "blk": int(rng.integers(0, 3)),
                "is_star": 0,
            })
    df = pd.DataFrame(rows)
    df["is_star"] = df.groupby("game_id")["pts"].transform(
        lambda x: (x == x.max()).astype(int)
    )
    return df


def test_train_best_player_model_returns_model():
    df = make_player_dataset(20, 5)
    model = train_best_player_model(df.iloc[:75], df.iloc[75:], mlflow_tracking=False)
    assert model is not None


def test_train_best_player_model_can_predict():
    df = make_player_dataset(20, 5)
    model = train_best_player_model(df.iloc[:75], df.iloc[75:], mlflow_tracking=False)
    proba = model.predict_proba(df.iloc[75:][PLAYER_FEATURE_COLS].values)
    assert proba.shape[1] == 2


def test_train_stat_models_returns_dict_for_all_targets():
    df = make_player_dataset(20, 5)
    models = train_stat_models(df.iloc[:75], df.iloc[75:], mlflow_tracking=False)
    assert isinstance(models, dict)
    for stat in STAT_TARGETS:
        assert stat in models, f"Missing model for stat: {stat}"


def test_train_stat_models_each_model_can_predict():
    df = make_player_dataset(20, 5)
    models = train_stat_models(df.iloc[:75], df.iloc[75:], mlflow_tracking=False)
    for stat, model in models.items():
        preds = model.predict(df.iloc[75:][PLAYER_FEATURE_COLS].values)
        assert len(preds) == len(df.iloc[75:])


def test_stat_model_paths_cover_all_targets():
    for stat in STAT_TARGETS:
        assert stat in STAT_MODEL_PATHS, f"No path defined for stat: {stat}"


def test_train_best_player_model_saves_to_path(tmp_path):
    import ml.train_player_model as mod
    original = mod.BEST_PLAYER_MODEL_PATH
    mod.BEST_PLAYER_MODEL_PATH = tmp_path / "best_player_model.joblib"
    try:
        df = make_player_dataset(20, 5)
        train_best_player_model(df.iloc[:75], df.iloc[75:], mlflow_tracking=False)
        assert (tmp_path / "best_player_model.joblib").exists()
    finally:
        mod.BEST_PLAYER_MODEL_PATH = original
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_train_player_model.py -v
```

Expected: FAIL with `ImportError: cannot import name 'train_best_player_model'`

- [ ] **Step 3: Implement `train_player_model.py`**

Replace the placeholder in `backend/ml/train_player_model.py` with:

```python
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    PLAYER_FEATURE_COLS,
    build_player_features,
    build_player_star_dataset,
    build_player_stats_dataset,
    train_test_split_by_date,
)

logger = logging.getLogger(__name__)

BEST_PLAYER_MODEL_PATH: Path = (
    Path(__file__).parent.parent / "models" / "best_player_model.joblib"
)
STAT_TARGETS: list[str] = ["pts", "reb", "ast"]
STAT_MODEL_PATHS: dict[str, Path] = {
    stat: Path(__file__).parent.parent / "models" / f"player_stat_{stat}.joblib"
    for stat in STAT_TARGETS
}

_BEST_PLAYER_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
    "verbose": -1,
}

_STAT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
    "verbosity": 0,
}


def train_best_player_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlflow_tracking: bool = True,
) -> LGBMClassifier:
    """Train LightGBM binary classifier to predict which player stars in a game.

    Args:
        train_df: DataFrame with PLAYER_FEATURE_COLS + 'is_star' column.
        test_df: DataFrame with PLAYER_FEATURE_COLS + 'is_star' column.
        mlflow_tracking: When False, skips MLflow (used in tests).

    Returns:
        Trained LGBMClassifier. Model is also saved to BEST_PLAYER_MODEL_PATH.
    """
    X_train = train_df[PLAYER_FEATURE_COLS].values
    y_train = train_df["is_star"].values
    X_test = test_df[PLAYER_FEATURE_COLS].values
    y_test = test_df["is_star"].values

    model = LGBMClassifier(**_BEST_PLAYER_PARAMS)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Best Player Model — accuracy: {acc:.4f}")

    BEST_PLAYER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BEST_PLAYER_MODEL_PATH)

    if mlflow_tracking:
        import mlflow
        mlflow.set_experiment("best_player")
        with mlflow.start_run():
            mlflow.log_params(_BEST_PLAYER_PARAMS)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_artifact(str(BEST_PLAYER_MODEL_PATH))

    return model


def train_stat_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlflow_tracking: bool = True,
) -> dict[str, XGBRegressor]:
    """Train one XGBoost regressor per target stat (pts, reb, ast).

    Args:
        train_df: DataFrame with PLAYER_FEATURE_COLS + stat target columns.
        test_df: DataFrame with PLAYER_FEATURE_COLS + stat target columns.
        mlflow_tracking: When False, skips MLflow (used in tests).

    Returns:
        Dict mapping stat name → trained XGBRegressor.
    """
    X_train = train_df[PLAYER_FEATURE_COLS].values
    X_test = test_df[PLAYER_FEATURE_COLS].values
    models: dict[str, XGBRegressor] = {}

    for stat in STAT_TARGETS:
        y_train = train_df[stat].values.astype(float)
        y_test = test_df[stat].values.astype(float)

        model = XGBRegressor(**_STAT_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        mae = mean_absolute_error(y_test, model.predict(X_test))
        logger.info(f"Stat Model ({stat}) — MAE: {mae:.4f}")

        path = STAT_MODEL_PATHS[stat]
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

        if mlflow_tracking:
            import mlflow
            mlflow.set_experiment(f"player_stat_{stat}")
            with mlflow.start_run():
                mlflow.log_params(_STAT_PARAMS)
                mlflow.log_metric("mae", mae)
                mlflow.log_artifact(str(path))

        models[stat] = model

    return models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv()

    from app.models.database import get_session
    from app.models.player_game_log import PlayerGameLog

    session = get_session()
    try:
        rows = session.query(PlayerGameLog).all()
        if not rows:
            raise RuntimeError("No data in DB. Run seed_database.py first.")
        df = pd.DataFrame([{
            "player_id": r.player_id,
            "season": r.season,
            "game_id": r.game_id,
            "game_date": pd.Timestamp(r.game_date),
            "wl": r.wl,
            "pts": r.pts,
            "reb": r.reb,
            "ast": r.ast,
            "stl": r.stl,
            "blk": r.blk,
            "min": r.min,
        } for r in rows if r.pts is not None])
    finally:
        session.close()

    player_features = build_player_features(df)
    star_ds = build_player_star_dataset(player_features)
    stats_ds = build_player_stats_dataset(player_features)

    train_star, test_star = train_test_split_by_date(star_ds)
    train_stats, test_stats = train_test_split_by_date(stats_ds)
    logger.info(f"Star dataset — train: {len(train_star)}, test: {len(test_star)}")
    logger.info(f"Stats dataset — train: {len(train_stats)}, test: {len(test_stats)}")

    train_best_player_model(train_star, test_star, mlflow_tracking=True)
    train_stat_models(train_stats, test_stats, mlflow_tracking=True)
    logger.info("All player models saved.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run player model tests**

```powershell
pytest tests/test_train_player_model.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Run the full test suite**

```powershell
pytest tests/ -v
```

Expected: `51 passed`

Breakdown: test_models(3) + test_data_collector(12) + test_seed_database(6) + test_feature_engineering(20) + test_train_win_model(4) + test_train_player_model(6) = 51

- [ ] **Step 6: Commit**

```powershell
git add backend/ml/train_player_model.py backend/tests/test_train_player_model.py
git commit -m "feat: add LightGBM best player classifier and XGBoost player stat regressors"
```

---

## Post-Phase 2: Running the Full Training Pipeline

Once Docker is running and the database is seeded (see HANDOFF.md Task 6), you can train all models:

```powershell
cd backend
.\venv\Scripts\activate

# Train win probability model (~2 min)
python ml/train_win_model.py

# Train player models (~3-5 min)
python ml/train_player_model.py

# Verify models were saved
Get-ChildItem models/
# Expected: win_model.joblib, best_player_model.joblib,
#           player_stat_pts.joblib, player_stat_reb.joblib, player_stat_ast.joblib

# View MLflow experiments in browser
mlflow ui
# Open http://localhost:5000
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `feature_engineering.py` — rolling averages (win%, pts avg 5/10), rest days, season win%, player rolling stats
- ✅ Train/test split by date with no leakage (`train_test_split_by_date`)
- ✅ Win Probability model (XGBoost binary classifier, log loss + accuracy)
- ✅ Best Player model (LightGBM binary classifier, accuracy)
- ✅ Player Stat model (XGBoost regressor per stat, MAE)
- ✅ MLflow experiment logging in all three trainers
- ✅ Models saved as `.joblib` files

**No-leakage guarantee:** Every rolling stat uses `shift(1)` before `.rolling()`, meaning the current game's data is never included in its own features. Verified by tests (`test_build_team_features_win_pct_uses_only_past_games`, `test_build_team_features_pts_avg_last5_uses_only_past`, `test_build_player_features_pts_avg_uses_only_past`).

**Type consistency:**
- `TEAM_FEATURE_COLS` defined in `feature_engineering.py`, imported as `FEATURE_COLS` in `train_win_model.py` — same list, verified by `test_win_feature_cols_match_team_feature_cols`
- `PLAYER_FEATURE_COLS` defined in `feature_engineering.py`, imported directly in `train_player_model.py`
- All `mlflow_tracking=False` signatures match across trainers and tests
