import { fetchWinProbability, fetchBestPlayer, fetchPlayerStats, fetchGameTotal } from "@/lib/api"

const mockFetch = jest.fn()
global.fetch = mockFetch

beforeEach(() => mockFetch.mockReset())

describe("fetchWinProbability", () => {
  it("calls the correct URL with team IDs", async () => {
    const mockData = {
      home_team_id: 1,
      away_team_id: 2,
      home_win_prob: 0.6,
      away_win_prob: 0.4,
      confidence: "high",
    }
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => mockData })
    const result = await fetchWinProbability(1, 2)
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/predictions/win-probability?home_team_id=1&away_team_id=2"
    )
    expect(result).toEqual(mockData)
  })

  it("throws the API error detail on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: "Team not found" }),
    })
    await expect(fetchWinProbability(999, 888)).rejects.toThrow("Team not found")
  })
})

describe("fetchBestPlayer", () => {
  it("calls the correct URL", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({ home_team_id: 1, away_team_id: 2, players: [] }) })
    await fetchBestPlayer(1, 2)
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/predictions/best-player?home_team_id=1&away_team_id=2"
    )
  })
})

describe("fetchPlayerStats", () => {
  it("calls the correct URL", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ player_id: 2544, predicted_stats: { pts: 25.0, reb: 7.0, ast: 8.0 } }),
    })
    await fetchPlayerStats(2544)
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/predictions/player-stats?player_id=2544"
    )
  })
})

describe("fetchGameTotal", () => {
  it("calls the correct URL with team IDs", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ home_team_id: 1, away_team_id: 2, predicted_total: 221.5, confidence: "medium" }),
    })
    await fetchGameTotal(1, 2)
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/predictions/game-total?home_team_id=1&away_team_id=2"
    )
  })
})
