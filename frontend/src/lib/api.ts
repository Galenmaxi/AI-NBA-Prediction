import type { BestPlayerResponse, PlayerStatsResponse, WinProbabilityResponse } from "./types"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail ?? "API error")
  }
  return res.json() as Promise<T>
}

export function fetchWinProbability(
  homeTeamId: number,
  awayTeamId: number,
): Promise<WinProbabilityResponse> {
  return apiFetch(
    `/predictions/win-probability?home_team_id=${homeTeamId}&away_team_id=${awayTeamId}`,
  )
}

export function fetchBestPlayer(
  homeTeamId: number,
  awayTeamId: number,
): Promise<BestPlayerResponse> {
  return apiFetch(
    `/predictions/best-player?home_team_id=${homeTeamId}&away_team_id=${awayTeamId}`,
  )
}

export function fetchPlayerStats(playerId: number): Promise<PlayerStatsResponse> {
  return apiFetch(`/predictions/player-stats?player_id=${playerId}`)
}
