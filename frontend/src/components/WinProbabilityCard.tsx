"use client"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useWinProbability } from "@/hooks/useWinProbability"
import { NBA_TEAMS } from "@/lib/teams"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

function teamAbbrev(id: number): string {
  return NBA_TEAMS.find((t) => t.id === id)?.abbreviation ?? String(id)
}

export function WinProbabilityCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useWinProbability(homeTeamId, awayTeamId)

  const chartData = data
    ? [
        { team: `${teamAbbrev(homeTeamId)} (H)`, prob: Math.round(data.home_win_prob * 100) },
        { team: `${teamAbbrev(awayTeamId)} (A)`, prob: Math.round(data.away_win_prob * 100) },
      ]
    : []

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Win Probability
          {data && (
            <Badge variant={data.confidence === "high" ? "default" : "secondary"}>
              {data.confidence}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <>
            <div className="flex justify-around mb-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {Math.round(data.home_win_prob * 100)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {teamAbbrev(homeTeamId)} · Home
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-600">
                  {Math.round(data.away_win_prob * 100)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {teamAbbrev(awayTeamId)} · Away
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={100}>
              <BarChart layout="vertical" data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="team" width={72} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="prob" maxBarSize={28}>
                  <Cell fill="#2563eb" />
                  <Cell fill="#dc2626" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </CardContent>
    </Card>
  )
}
