"use client"
import { useGameTotal } from "@/hooks/useGameTotal"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

function confidenceBadge(c: string) {
  if (c === "high")   return { bg: "rgba(74,222,128,0.10)",  fg: "#86efac", bd: "rgba(74,222,128,0.30)" }
  if (c === "medium") return { bg: "rgba(240,176,43,0.10)",  fg: "#fcd34d", bd: "rgba(240,176,43,0.30)" }
  return                       { bg: "rgba(148,163,184,0.10)", fg: "#cbd5e1", bd: "rgba(148,163,184,0.25)" }
}

const NBA_AVG = 224.0
const CARD: React.CSSProperties = {
  background: "linear-gradient(180deg,#131b28,#0f1620)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 18,
  padding: 22,
}

export function GameTotalCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useGameTotal(homeTeamId, awayTeamId)

  const badge = data ? confidenceBadge(data.confidence) : null
  const total = data ? data.predicted_total : null
  const isOver = total !== null && total >= NBA_AVG
  const dev = total !== null ? Math.abs(total - NBA_AVG) : 0
  const barW = Math.min(48, (dev / 30) * 50)

  return (
    <div style={CARD}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span style={{ fontSize: 16 }}>🎯</span>
          <span style={{ fontFamily: "'Barlow Semi Condensed', sans-serif", fontWeight: 700, fontSize: 16.5, letterSpacing: "0.3px" }}>Game Total (O/U)</span>
        </div>
        {badge && (
          <span style={{ padding: "4px 11px", borderRadius: 999, fontSize: 11.5, fontWeight: 700, letterSpacing: "0.4px", background: badge.bg, color: badge.fg, border: `1px solid ${badge.bd}` }}>
            {data!.confidence}
          </span>
        )}
      </div>

      {isLoading && <p style={{ color: "#6f7d94", fontSize: 14 }}>Loading...</p>}
      {error && <p style={{ color: "#f87171", fontSize: 14 }}>{error.message}</p>}

      {data && total !== null && (
        <>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 54, lineHeight: 0.85, color: "#f4f6fa" }}>
                {total}
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "#9aa6ba", marginTop: 6 }}>predicted combined points</div>
            </div>
            <div style={{ textAlign: "right" }}>
              {isOver ? (
                <div style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "7px 13px", borderRadius: 10, background: "rgba(74,222,128,0.10)", border: "1px solid rgba(74,222,128,0.25)" }}>
                  <span style={{ color: "#86efac" }}>▲</span>
                  <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 17, color: "#86efac" }}>OVER</span>
                </div>
              ) : (
                <div style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "7px 13px", borderRadius: 10, background: "rgba(248,113,113,0.10)", border: "1px solid rgba(248,113,113,0.25)" }}>
                  <span style={{ color: "#fca5a5" }}>▼</span>
                  <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 17, color: "#fca5a5" }}>UNDER</span>
                </div>
              )}
              <div style={{ fontSize: 12, color: "#6f7d94", marginTop: 9, fontWeight: 500 }}>
                league avg <span style={{ fontFamily: "'Barlow Condensed', sans-serif", color: "#9aa6ba" }}>{NBA_AVG}</span>
              </div>
            </div>
          </div>

          <div style={{ position: "relative", height: 8, borderRadius: 5, background: "#0c121c", marginTop: 20 }}>
            <div style={{ position: "absolute", left: "50%", top: -4, bottom: -4, width: 2, background: "#56657d" }} />
            <div style={{
              position: "absolute",
              height: "100%",
              borderRadius: 5,
              width: `${barW}%`,
              left: isOver ? "50%" : `${50 - barW}%`,
              background: isOver
                ? "linear-gradient(90deg,#4ade80,#22c55e)"
                : "linear-gradient(90deg,#f87171,#ef4444)",
              transition: "width 0.5s ease",
            }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#56657d", fontWeight: 600, marginTop: 7 }}>
            <span>UNDER</span>
            <span style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>224 avg</span>
            <span>OVER</span>
          </div>
        </>
      )}
    </div>
  )
}
