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
