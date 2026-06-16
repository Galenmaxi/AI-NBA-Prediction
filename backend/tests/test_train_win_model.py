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
    model = train_win_model(df.iloc[:45], df.iloc[45:], mlflow_tracking=False)
    assert model is not None


def test_train_win_model_model_can_predict_probabilities():
    df = make_win_dataset(60)
    model = train_win_model(df.iloc[:45], df.iloc[45:], mlflow_tracking=False)
    proba = model.predict_proba(df.iloc[45:][WIN_FEATURE_COLS].values)
    assert proba.shape == (15, 2)
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
