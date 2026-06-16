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
