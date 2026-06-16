import json
from unittest.mock import MagicMock, patch

from app.services.prediction_service import _cache_get, _cache_set, _get_redis


def test_get_redis_returns_none_when_redis_url_not_set(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert _get_redis() is None


def test_cache_get_returns_none_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    result = _cache_get("any_key")
    assert result is None


def test_cache_set_is_noop_when_no_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    _cache_set("any_key", {"home_win_prob": 0.6})  # must not raise


def test_cache_get_returns_value_when_redis_has_key():
    mock_redis = MagicMock()
    mock_redis.get.return_value = json.dumps(
        {"home_win_prob": 0.72, "away_win_prob": 0.28}
    ).encode()
    with patch("app.services.prediction_service._get_redis", return_value=mock_redis):
        result = _cache_get("win_prob:1:2")
    assert result["home_win_prob"] == 0.72


def test_cache_get_returns_none_when_key_missing():
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    with patch("app.services.prediction_service._get_redis", return_value=mock_redis):
        result = _cache_get("win_prob:1:2")
    assert result is None


def test_cache_set_calls_setex_with_ttl():
    mock_redis = MagicMock()
    with patch("app.services.prediction_service._get_redis", return_value=mock_redis):
        _cache_set("win_prob:1:2", {"home_win_prob": 0.6})
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert args[0] == "win_prob:1:2"
    assert args[1] == 3600  # default TTL


def test_get_win_probability_uses_cache_on_cache_hit():
    cached_value = {"home_win_prob": 0.99, "away_win_prob": 0.01, "confidence": "high"}
    with patch("app.services.prediction_service._cache_get", return_value=cached_value):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from app.models.base import Base
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            from app.services.prediction_service import get_win_probability
            result = get_win_probability(session, home_team_id=1, away_team_id=2)
    assert result["home_win_prob"] == 0.99
