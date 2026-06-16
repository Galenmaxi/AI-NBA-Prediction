"use client"
import { useState } from "react"
import { NBA_TEAMS } from "@/lib/teams"

interface Props {
  onSubmit: (homeTeamId: number, awayTeamId: number) => void
}

export function MatchupSelector({ onSubmit }: Props) {
  const [homeTeamId, setHomeTeamId] = useState<number>(NBA_TEAMS[9].id) // GSW default
  const [awayTeamId, setAwayTeamId] = useState<number>(NBA_TEAMS[13].id) // LAL default

  return (
    <div className="flex flex-wrap gap-6 items-end">
      <div>
        <label className="text-sm font-medium block mb-1">Home Team</label>
        <select
          value={homeTeamId}
          onChange={(e) => setHomeTeamId(Number(e.target.value))}
          className="border rounded-md px-3 py-2 text-sm bg-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {NBA_TEAMS.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </div>
      <div className="text-xl font-bold text-muted-foreground self-end pb-2">vs.</div>
      <div>
        <label className="text-sm font-medium block mb-1">Away Team</label>
        <select
          value={awayTeamId}
          onChange={(e) => setAwayTeamId(Number(e.target.value))}
          className="border rounded-md px-3 py-2 text-sm bg-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {NBA_TEAMS.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </div>
      <button
        onClick={() => onSubmit(homeTeamId, awayTeamId)}
        className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 font-medium text-sm transition-colors"
      >
        Predict
      </button>
    </div>
  )
}
