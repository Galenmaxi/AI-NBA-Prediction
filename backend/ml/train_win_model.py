from __future__ import annotations

import logging
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
