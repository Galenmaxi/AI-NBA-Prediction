from pydantic import BaseModel


class WinProbabilityResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    home_win_prob: float
    away_win_prob: float
    confidence: str


class PlayerStarPrediction(BaseModel):
    player_id: int
    player_name: str
    star_probability: float


class BestPlayerResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    players: list[PlayerStarPrediction]


class StatPrediction(BaseModel):
    pts: float
    reb: float
    ast: float


class PlayerStatsResponse(BaseModel):
    player_id: int
    predicted_stats: StatPrediction


class GameTotalResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    predicted_total: float
    confidence: str
