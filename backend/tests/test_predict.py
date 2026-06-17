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
    monkeypatch.setattr(mod, "STAT_MODEL_PATHS", {
        "pts": tmp_path / "missing.joblib",
        "reb": tmp_path / "missing2.joblib",
        "ast": tmp_path / "missing3.joblib",
    })

    from ml.predict import predict_player_stats
    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict_player_stats(_make_player_df(n_players=1, n_games=15))


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
