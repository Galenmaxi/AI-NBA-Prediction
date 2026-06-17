"use client"
import { useState } from "react"
import { BestPlayerCard } from "@/components/BestPlayerCard"
import { GameTotalCard } from "@/components/GameTotalCard"
import { MatchupSelector } from "@/components/MatchupSelector"
import { PlayerStatsCard } from "@/components/PlayerStatsCard"
import { WinProbabilityCard } from "@/components/WinProbabilityCard"

interface Matchup {
  homeTeamId: number
  awayTeamId: number
}

export default function HomePage() {
  const [matchup, setMatchup] = useState<Matchup | null>(null)
  const [playerIdInput, setPlayerIdInput] = useState("")
  const [activePlayerId, setActivePlayerId] = useState<number | null>(null)

  return (
    <main className="container mx-auto p-6 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">NBA AI Predictor</h1>
        <p className="text-muted-foreground mt-1">
          Select a matchup to see AI-powered win probability and player predictions.
        </p>
      </div>

      <section className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Select Matchup</h2>
        <MatchupSelector onSubmit={(h, a) => setMatchup({ homeTeamId: h, awayTeamId: a })} />
      </section>

      {matchup && (
        <>
          <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <WinProbabilityCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
            <BestPlayerCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
          </section>
          <section className="mb-8">
            <GameTotalCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
          </section>
        </>
      )}

      <section>
        <h2 className="text-lg font-semibold mb-4">Player Stats Predictor</h2>
        <p className="text-sm text-muted-foreground mb-3">
          Enter an NBA player ID (e.g. 2544 = LeBron James, 201939 = Stephen Curry).
        </p>
        <div className="flex gap-3 items-end">
          <div>
            <label className="text-sm font-medium block mb-1">Player ID</label>
            <input
              type="number"
              value={playerIdInput}
              onChange={(e) => setPlayerIdInput(e.target.value)}
              placeholder="e.g. 2544"
              className="border rounded-md px-3 py-2 text-sm w-40 focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <button
            onClick={() => setActivePlayerId(Number(playerIdInput))}
            disabled={!playerIdInput || Number(playerIdInput) <= 0}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 font-medium text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Predict
          </button>
        </div>
        {activePlayerId && (
          <div className="mt-4">
            <PlayerStatsCard playerId={activePlayerId} />
          </div>
        )}
      </section>
    </main>
  )
}
