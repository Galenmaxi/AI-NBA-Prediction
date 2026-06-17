export interface WinProbabilityResponse {
  home_team_id: number
  away_team_id: number
  home_win_prob: number
  away_win_prob: number
  confidence: "low" | "medium" | "high"
}

export interface PlayerStarPrediction {
  player_id: number
  player_name: string
  star_probability: number
}

export interface BestPlayerResponse {
  home_team_id: number
  away_team_id: number
  players: PlayerStarPrediction[]
}

export interface StatPrediction {
  pts: number
  reb: number
  ast: number
}

export interface PlayerStatsResponse {
  player_id: number
  predicted_stats: StatPrediction
}

export interface GameTotalResponse {
  home_team_id: number
  away_team_id: number
  predicted_total: number
  confidence: "low" | "medium" | "high"
}
