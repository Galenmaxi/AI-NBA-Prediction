"use client"
import { useWinProbability } from "@/hooks/useWinProbability"
import { NBA_TEAMS } from "@/lib/teams"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

function getTeam(id: number) {
  return NBA_TEAMS.find((t) => t.id === id) ?? NBA_TEAMS[0]
}

function confidenceBadge(c: string) {
  if (c === "high")   return { bg: "rgba(74,222,128,0.10)",  fg: "#86efac", bd: "rgba(74,222,128,0.30)" }
  if (c === "medium") return { bg: "rgba(240,176,43,0.10)",  fg: "#fcd34d", bd: "rgba(240,176,43,0.30)" }
  return                       { bg: "rgba(148,163,184,0.10)", fg: "#cbd5e1", bd: "rgba(148,163,184,0.25)" }
}

const CARD: React.CSSProperties = {
  background: "linear-gradient(180deg,#131b28,#0f1620)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 18,
  padding: 22,
}

export function WinProbabilityCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useWinProbability(homeTeamId, awayTeamId)
  const home = getTeam(homeTeamId)
  const away = getTeam(awayTeamId)

  const homeP = data ? Math.round(data.home_win_prob * 100) : null
  const awayP = data ? Math.round(data.away_win_prob * 100) : null
  const badge = data ? confidenceBadge(data.confidence) : null

  return (
    <div style={CARD}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span style={{ fontSize: 16 }}>🏆</span>
          <span style={{ fontFamily: "'Barlow Semi Condensed', sans-serif", fontWeight: 700, fontSize: 16.5, letterSpacing: "0.3px" }}>Win Probability</span>
        </div>
        {badge && (
          <span style={{ padding: "4px 11px", borderRadius: 999, fontSize: 11.5, fontWeight: 700, letterSpacing: "0.4px", background: badge.bg, color: badge.fg, border: `1px solid ${badge.bd}` }}>
            {data!.confidence}
          </span>
        )}
      </div>

      {isLoading && <p style={{ color: "#6f7d94", fontSize: 14 }}>Loading...</p>}
      {error && <p style={{ color: "#f87171", fontSize: 14 }}>{error.message}</p>}

      {data && homeP !== null && awayP !== null && (
        <>
          <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: 14 }}>
            <div>
              <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 44, lineHeight: 0.9, color: home.color }}>
                <span>{homeP}%</span>
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "#9aa6ba", marginTop: 5 }}>{home.abbreviation} · Home</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 44, lineHeight: 0.9, color: away.color }}>
                <span>{awayP}%</span>
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "#9aa6ba", marginTop: 5 }}>{away.abbreviation} · Away</div>
            </div>
          </div>

          <div style={{ display: "flex", height: 14, borderRadius: 8, overflow: "hidden", background: "#0c121c" }}>
            <div style={{ width: `${homeP}%`, background: home.color, transition: "width 0.5s ease" }} />
            <div style={{ width: `${awayP}%`, background: away.color, transition: "width 0.5s ease" }} />
          </div>

          <div style={{ fontSize: 12, color: "#6f7d94", marginTop: 10, fontWeight: 500 }}>
            Model: XGBoost · trained on 3 seasons of game logs
          </div>
        </>
      )}
    </div>
  )
}
