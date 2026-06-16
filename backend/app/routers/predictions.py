from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.models.database import get_db
from app.schemas.predictions import (
    BestPlayerResponse,
    PlayerStarPrediction,
    PlayerStatsResponse,
    StatPrediction,
    WinProbabilityResponse,
)
from app.services.prediction_service import get_best_player, get_player_stats, get_win_probability

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/win-probability", response_model=WinProbabilityResponse)
def win_probability(
    home_team_id: int = Query(..., description="NBA team ID of the home team"),
    away_team_id: int = Query(..., description="NBA team ID of the away team"),
    db: Session = Depends(get_db),
) -> WinProbabilityResponse:
    try:
        result = get_win_probability(db, home_team_id, away_team_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return WinProbabilityResponse(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_win_prob=result["home_win_prob"],
        away_win_prob=result["away_win_prob"],
        confidence=result["confidence"],
    )


@router.get("/best-player", response_model=BestPlayerResponse)
def best_player(
    home_team_id: int = Query(..., description="NBA team ID of the home team"),
    away_team_id: int = Query(..., description="NBA team ID of the away team"),
    db: Session = Depends(get_db),
) -> BestPlayerResponse:
    try:
        players = get_best_player(db, home_team_id, away_team_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return BestPlayerResponse(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        players=[PlayerStarPrediction(**p) for p in players],
    )


@router.get("/player-stats", response_model=PlayerStatsResponse)
def player_stats(
    player_id: int = Query(..., description="NBA player ID"),
    db: Session = Depends(get_db),
) -> PlayerStatsResponse:
    try:
        stats = get_player_stats(db, player_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PlayerStatsResponse(
        player_id=player_id,
        predicted_stats=StatPrediction(**stats),
    )
