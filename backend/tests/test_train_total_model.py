from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
