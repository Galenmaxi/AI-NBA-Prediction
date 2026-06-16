import { render, screen } from "@testing-library/react"
import { PlayerStatsCard } from "@/components/PlayerStatsCard"
import * as playerStatsHook from "@/hooks/usePlayerStats"

jest.mock("@/hooks/usePlayerStats")

const mockUsePlayerStats = playerStatsHook.usePlayerStats as jest.Mock

describe("PlayerStatsCard", () => {
  it("shows loading state while fetching", () => {
    mockUsePlayerStats.mockReturnValue({ isLoading: true, data: undefined, error: null })
    render(<PlayerStatsCard playerId={2544} />)
    expect(screen.getByText("Loading...")).toBeInTheDocument()
  })

  it("shows predicted pts, reb, and ast", () => {
    mockUsePlayerStats.mockReturnValue({
      isLoading: false,
      data: {
        player_id: 2544,
        predicted_stats: { pts: 27.3, reb: 7.8, ast: 8.2 },
      },
      error: null,
    })
    render(<PlayerStatsCard playerId={2544} />)
    expect(screen.getByText("27.3")).toBeInTheDocument()
    expect(screen.getByText("7.8")).toBeInTheDocument()
    expect(screen.getByText("8.2")).toBeInTheDocument()
    expect(screen.getByText("PTS")).toBeInTheDocument()
    expect(screen.getByText("REB")).toBeInTheDocument()
    expect(screen.getByText("AST")).toBeInTheDocument()
  })

  it("shows error message on fetch failure", () => {
    mockUsePlayerStats.mockReturnValue({
      isLoading: false,
      data: undefined,
      error: new Error("No game data found for player 9999"),
    })
    render(<PlayerStatsCard playerId={9999} />)
    expect(screen.getByText(/No game data found/)).toBeInTheDocument()
  })
})
