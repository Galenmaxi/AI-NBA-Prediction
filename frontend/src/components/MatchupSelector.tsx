"use client"
import { useState } from "react"
import { NBA_TEAMS } from "@/lib/teams"

interface Props {
  onSubmit: (homeTeamId: number, awayTeamId: number) => void
}

function getTeam(id: number) {
  return NBA_TEAMS.find((t) => t.id === id) ?? NBA_TEAMS[0]
}

const PANEL: React.CSSProperties = {
  background: "linear-gradient(180deg,#131b28,#0f1620)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 18,
  padding: 22,
}

const LBL: React.CSSProperties = {
  fontSize: 11.5,
  fontWeight: 600,
  letterSpacing: "0.8px",
  color: "#6f7d94",
  marginBottom: 7,
  textTransform: "uppercase" as const,
}

export function MatchupSelector({ onSubmit }: Props) {
  const [homeTeamId, setHomeTeamId] = useState<number>(NBA_TEAMS[9].id)
  const [awayTeamId, setAwayTeamId] = useState<number>(NBA_TEAMS[13].id)

  const home = getTeam(homeTeamId)
  const away = getTeam(awayTeamId)

  return (
    <div style={PANEL}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 18 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 18, flex: 1, minWidth: 260 }}>
          <div style={{ flex: 1 }}>
            <div style={LBL}>Home Team</div>
            <div style={{ position: "relative" }}>
              <span style={{ position: "absolute", left: 13, top: "50%", transform: "translateY(-50%)", width: 10, height: 18, borderRadius: 3, background: home.color, zIndex: 1 }} />
              <select
                value={homeTeamId}
                onChange={(e) => setHomeTeamId(Number(e.target.value))}
                style={{ width: "100%", padding: "11px 34px 11px 32px", background: "#0c121c", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 11, color: "#f4f6fa", fontSize: 14, fontWeight: 600, cursor: "pointer", outline: "none", appearance: "none" as const }}
              >
                {NBA_TEAMS.map((t) => (
                  <option key={t.id} value={t.id}>{t.abbreviation} — {t.name}</option>
                ))}
              </select>
              <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "#6f7d94", fontSize: 10, pointerEvents: "none" }}>▼</span>
            </div>
          </div>

          <div style={{ fontWeight: 700, fontSize: 17, color: "#56657d", letterSpacing: 1, paddingTop: 20 }}>VS</div>

          <div style={{ flex: 1 }}>
            <div style={LBL}>Away Team</div>
            <div style={{ position: "relative" }}>
              <span style={{ position: "absolute", left: 13, top: "50%", transform: "translateY(-50%)", width: 10, height: 18, borderRadius: 3, background: away.color, zIndex: 1 }} />
              <select
                value={awayTeamId}
                onChange={(e) => setAwayTeamId(Number(e.target.value))}
                style={{ width: "100%", padding: "11px 34px 11px 32px", background: "#0c121c", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 11, color: "#f4f6fa", fontSize: 14, fontWeight: 600, cursor: "pointer", outline: "none", appearance: "none" as const }}
              >
                {NBA_TEAMS.map((t) => (
                  <option key={t.id} value={t.id}>{t.abbreviation} — {t.name}</option>
                ))}
              </select>
              <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "#6f7d94", fontSize: 10, pointerEvents: "none" }}>▼</span>
            </div>
          </div>
        </div>

        <button
          onClick={() => onSubmit(homeTeamId, awayTeamId)}
          style={{ display: "flex", alignItems: "center", gap: 9, padding: "13px 26px", background: "linear-gradient(135deg,#f0792b,#e25f12)", border: "none", borderRadius: 11, color: "#fff", fontFamily: "'Barlow Semi Condensed', sans-serif", fontWeight: 700, fontSize: 15, letterSpacing: "0.4px", cursor: "pointer", boxShadow: "0 8px 22px rgba(240,121,43,0.32)", whiteSpace: "nowrap" }}
        >
          ⚡ PREDICT
        </button>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 18, marginTop: 20, paddingTop: 18, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <div style={{ width: 46, height: 46, borderRadius: 12, background: home.color, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 13, color: "#fff", boxShadow: "0 4px 14px rgba(0,0,0,0.4)" }}>
            {home.abbreviation}
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{home.name}</div>
            <div style={{ fontSize: 11, color: "#6f7d94", fontWeight: 600, letterSpacing: "0.5px" }}>HOME</div>
          </div>
        </div>
        <div style={{ color: "#56657d", fontSize: 14, fontWeight: 600 }}>@</div>
        <div style={{ display: "flex", alignItems: "center", gap: 11, flexDirection: "row-reverse" }}>
          <div style={{ width: 46, height: 46, borderRadius: 12, background: away.color, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Barlow Condensed', sans-serif", fontWeight: 700, fontSize: 13, color: "#fff", boxShadow: "0 4px 14px rgba(0,0,0,0.4)" }}>
            {away.abbreviation}
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{away.name}</div>
            <div style={{ fontSize: 11, color: "#6f7d94", fontWeight: 600, letterSpacing: "0.5px" }}>AWAY</div>
          </div>
        </div>
      </div>
    </div>
  )
}
