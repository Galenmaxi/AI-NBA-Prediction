"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useBestPlayer } from "@/hooks/useBestPlayer"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

export function BestPlayerCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useBestPlayer(homeTeamId, awayTeamId)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Star Player Predictions</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <ol className="space-y-3">
            {data.players.map((player, i) => (
              <li key={player.player_id} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-muted-foreground text-sm w-5">{i + 1}.</span>
                  <span className="font-medium text-sm">{player.player_name}</span>
                </div>
                <span className="text-sm font-semibold text-blue-600">
                  {Math.round(player.star_probability * 100)}%
                </span>
              </li>
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  )
}
