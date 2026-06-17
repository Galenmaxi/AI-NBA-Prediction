"use client"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useGameTotal } from "@/hooks/useGameTotal"

const CONFIDENCE_VARIANT: Record<string, "default" | "secondary" | "outline"> = {
  high: "default",
  medium: "secondary",
  low: "outline",
}

interface Props {
  homeTeamId: number
  awayTeamId: number
}

export function GameTotalCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useGameTotal(homeTeamId, awayTeamId)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center justify-between">
          Game Total (O/U)
          {data && (
            <Badge variant={CONFIDENCE_VARIANT[data.confidence] ?? "outline"}>
              {data.confidence}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <div className="text-center">
            <div className="text-4xl font-bold">{data.predicted_total}</div>
            <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">
              Predicted Total Points
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
