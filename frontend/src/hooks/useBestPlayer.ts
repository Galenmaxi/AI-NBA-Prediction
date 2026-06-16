import { useQuery } from "@tanstack/react-query"
import { fetchBestPlayer } from "@/lib/api"
import type { BestPlayerResponse } from "@/lib/types"

export function useBestPlayer(homeTeamId: number | null, awayTeamId: number | null) {
  return useQuery<BestPlayerResponse, Error>({
    queryKey: ["bestPlayer", homeTeamId, awayTeamId],
    queryFn: () => fetchBestPlayer(homeTeamId!, awayTeamId!),
    enabled: homeTeamId !== null && awayTeamId !== null,
  })
}
