import { useQuery } from "@tanstack/react-query"

import { fetchGameTotal } from "@/lib/api"
import type { GameTotalResponse } from "@/lib/types"

export function useGameTotal(
  homeTeamId: number | null,
  awayTeamId: number | null,
) {
  return useQuery<GameTotalResponse, Error>({
    queryKey: ["game-total", homeTeamId, awayTeamId],
    queryFn: () => fetchGameTotal(homeTeamId!, awayTeamId!),
    enabled: homeTeamId !== null && awayTeamId !== null,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })
}
