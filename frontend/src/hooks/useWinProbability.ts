import { useQuery } from "@tanstack/react-query"
import { fetchWinProbability } from "@/lib/api"
import type { WinProbabilityResponse } from "@/lib/types"

export function useWinProbability(homeTeamId: number | null, awayTeamId: number | null) {
  return useQuery<WinProbabilityResponse, Error>({
    queryKey: ["winProbability", homeTeamId, awayTeamId],
    queryFn: () => fetchWinProbability(homeTeamId!, awayTeamId!),
    enabled: homeTeamId !== null && awayTeamId !== null,
  })
}
