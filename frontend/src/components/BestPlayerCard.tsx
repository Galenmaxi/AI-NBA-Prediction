"use client"
import { useBestPlayer } from "@/hooks/useBestPlayer"

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

export function BestPlayerCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useBestPlayer(homeTeamId, awayTeamId)

  const maxProb = data ? Math.max(...data.players.map((p) => p.star_probability)) : 1

  return (
    <div style={CARD}>
      <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 18 }}>
        <span style={{ fontSize: 16 }}>⭐</span>
        <span style={{ fontFamily: "'Barlow Semi Condensed', sans-serif", fontWeight: 700, fontSize: 16.5, letterSpacing: "0.3px" }}>Predicted Star Performer</span>
      </div>

      {isLoading && <p style={{ color: "#6f7d94", fontSize: 14 }}>Loading...</p>}
      {error && <p style={{ color: "#f87171", fontSize: 14 }}>{error.message}</p>}

      {data && (
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          {data.players.map((player, i) => {
            const pct = Math.round(player.star_probability * 100)
            const barW = Math.round((player.star_probability / maxProb) * 100)
            return (
              <div key={player.player_id} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 22, fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 16, color: "#56657d", textAlign: "center" }}>{i + 1}</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 5 }}>
                    <span style={{ fontWeight: 600, fontSize: 14, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{player.player_name}</span>
                    <span style={{ fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 15, color: "#f0792b", marginLeft: 8 }}>{pct}%</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 4, background: "#0c121c", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${barW}%`, borderRadius: 4, background: "linear-gradient(90deg,#f0792b,#f6a96b)", transition: "width 0.4s ease" }} />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
