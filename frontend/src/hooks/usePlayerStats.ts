import { useQuery } from "@tanstack/react-query"
import { fetchPlayerStats } from "@/lib/api"
import type { PlayerStatsResponse } from "@/lib/types"

export function usePlayerStats(playerId: number | null) {
  return useQuery<PlayerStatsResponse, Error>({
    queryKey: ["playerStats", playerId],
    queryFn: () => fetchPlayerStats(playerId!),
    enabled: playerId !== null && playerId > 0,
  })
}
