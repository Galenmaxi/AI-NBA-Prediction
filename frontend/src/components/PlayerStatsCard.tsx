"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { usePlayerStats } from "@/hooks/usePlayerStats"

interface Props {
  playerId: number
}

export function PlayerStatsCard({ playerId }: Props) {
  const { data, isLoading, error } = usePlayerStats(playerId)

  return (
    <Card className="w-full max-w-xs">
      <CardHeader>
        <CardTitle className="text-base">Player #{playerId}</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold">{data.predicted_stats.pts.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">PTS</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{data.predicted_stats.reb.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">REB</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{data.predicted_stats.ast.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">AST</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
