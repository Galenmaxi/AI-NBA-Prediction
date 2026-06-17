"use client"
import { useBestPlayer } from "@/hooks/useBestPlayer"
import { usePlayerStats } from "@/hooks/usePlayerStats"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

const CARD: React.CSSProperties = {
  background: "linear-gradient(180deg,#131b28,#0f1620)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 18,
  padding: 22,
}

const TILE: React.CSSProperties = {
  background: "#0c121c",
  border: "1px solid rgba(255,255,255,0.06)",
  borderRadius: 11,
  padding: "13px 8px",
  textAlign: "center",
}

export function StarStatCard({ homeTeamId, awayTeamId }: Props) {
  const { data: bestData, isLoading: bestLoading, error: bestError } = useBestPlayer(homeTeamId, awayTeamId)
  const topPlayerId = bestData?.players[0]?.player_id ?? null
  const topPlayerName = bestData?.players[0]?.player_name ?? null

  const { data: statsData, isLoading: statsLoading, error: statsError } = usePlayerStats(topPlayerId)

  const isLoading = bestLoading || (topPlayerId !== null && statsLoading)
  const error = bestError ?? statsError

  const stats = statsData
    ? [
        { label: "PTS", val: statsData.predicted_stats.pts.toFixed(1) },
        { label: "REB", val: statsData.predicted_stats.reb.toFixed(1) },
        { label: "AST", val: statsData.predicted_stats.ast.toFixed(1) },
      ]
    : []

  return (
    <div style={CARD}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <span style={{ fontSize: 16 }}>📊</span>
          <span style={{ fontFamily: "'Barlow Semi Condensed', sans-serif", fontWeight: 700, fontSize: 16.5, letterSpacing: "0.3px" }}>Predicted Stat Line</span>
        </div>
        {topPlayerName && (
          <span style={{ fontSize: 13, fontWeight: 600, color: "#9aa6ba" }}>{topPlayerName}</span>
        )}
      </div>

      {isLoading && <p style={{ color: "#6f7d94", fontSize: 14 }}>Loading...</p>}
      {error && <p style={{ color: "#f87171", fontSize: 14 }}>{error.message}</p>}

      {statsData && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 9 }}>
            {stats.map((s) => (
              <div key={s.label} style={TILE}>
                <div style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 27, lineHeight: 1, color: "#f4f6fa" }}>{s.val}</div>
                <div style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.6px", color: "#6f7d94", marginTop: 6 }}>{s.label}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: "#6f7d94", marginTop: 14, fontWeight: 500 }}>
            Per-stat XGBoost regressors · MAE-evaluated
          </div>
        </>
      )}
    </div>
  )
}
