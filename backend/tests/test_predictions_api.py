from unittest.mock import patch

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


# --- /predictions/game-total ---

def test_game_total_returns_200():
    with patch("app.routers.predictions.get_game_total") as mock_svc:
        mock_svc.return_value = {"predicted_total": 221.5, "confidence": "medium"}
        resp = client.get("/predictions/game-total?home_team_id=1610612744&away_team_id=1610612747")
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_total"] == 221.5
    assert body["confidence"] == "medium"


def test_game_total_returns_503_when_model_missing():
    with patch("app.routers.predictions.get_game_total") as mock_svc:
        mock_svc.side_effect = FileNotFoundError("Model not found at /path/total_model.joblib")
        resp = client.get("/predictions/game-total?home_team_id=1610612744&away_team_id=1610612747")
    assert resp.status_code == 503
    assert "Model not found" in resp.json()["detail"]
