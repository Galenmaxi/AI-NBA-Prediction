"use client"
import { useState } from "react"
import { BestPlayerCard } from "@/components/BestPlayerCard"
import { GameTotalCard } from "@/components/GameTotalCard"
import { MatchupSelector } from "@/components/MatchupSelector"
import { StarStatCard } from "@/components/StarStatCard"
import { WinProbabilityCard } from "@/components/WinProbabilityCard"

interface Matchup {
  homeTeamId: number
  awayTeamId: number
}

export default function HomePage() {
  const [matchup, setMatchup] = useState<Matchup | null>(null)

  return (
    <main style={{ minHeight: "100vh", padding: "28px 28px 60px" }}>
      <div style={{ maxWidth: 1120, margin: "0 auto" }}>

        <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, paddingBottom: 24, borderBottom: "1px solid rgba(255,255,255,0.07)", marginBottom: 26 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ width: 42, height: 42, borderRadius: 11, background: "linear-gradient(135deg,#3a7ff7,#1d4ed8)", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 24, color: "#fff", boxShadow: "0 6px 20px rgba(58,127,247,0.35)" }}>
              AI
            </div>
            <div>
              <div style={{ fontFamily: "'Barlow Semi Condensed', sans-serif", fontWeight: 700, fontSize: 23, letterSpacing: "0.2px", lineHeight: 1 }}>NBA AI Predictor</div>
              <div style={{ color: "#7d8aa0", fontSize: 13.5, fontWeight: 500, marginTop: 3 }}>ML-powered game predictions</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 13px", border: "1px solid rgba(74,222,128,0.25)", background: "rgba(74,222,128,0.08)", borderRadius: 999 }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#4ade80", boxShadow: "0 0 8px #4ade80", display: "inline-block" }} />
            <span style={{ fontSize: 12.5, fontWeight: 600, color: "#86efac", letterSpacing: "0.3px" }}>MODELS LIVE</span>
          </div>
        </header>

        <section style={{ marginBottom: 22 }}>
          <MatchupSelector onSubmit={(h, a) => setMatchup({ homeTeamId: h, awayTeamId: a })} />
        </section>

        {matchup && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 20 }}>
            <WinProbabilityCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
            <BestPlayerCard     homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
            <StarStatCard       homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
            <GameTotalCard      homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
          </div>
        )}

      </div>
    </main>
  )
}
