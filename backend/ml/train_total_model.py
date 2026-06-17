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
