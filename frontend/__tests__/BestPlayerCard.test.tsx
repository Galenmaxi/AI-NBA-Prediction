import { render, screen } from "@testing-library/react"
import { BestPlayerCard } from "@/components/BestPlayerCard"
import * as bestPlayerHook from "@/hooks/useBestPlayer"

jest.mock("@/hooks/useBestPlayer")

const mockUseBestPlayer = bestPlayerHook.useBestPlayer as jest.Mock

describe("BestPlayerCard", () => {
  it("shows loading state while fetching", () => {
    mockUseBestPlayer.mockReturnValue({ isLoading: true, data: undefined, error: null })
    render(<BestPlayerCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText("Loading...")).toBeInTheDocument()
  })

  it("shows ranked player list with star probabilities", () => {
    mockUseBestPlayer.mockReturnValue({
      isLoading: false,
      data: {
        home_team_id: 1,
        away_team_id: 2,
        players: [
          { player_id: 2544, player_name: "LeBron James", star_probability: 0.85 },
          { player_id: 201939, player_name: "Stephen Curry", star_probability: 0.72 },
          { player_id: 203954, player_name: "Joel Embiid", star_probability: 0.61 },
        ],
      },
      error: null,
    })
    render(<BestPlayerCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText("LeBron James")).toBeInTheDocument()
    expect(screen.getByText("85%")).toBeInTheDocument()
    expect(screen.getByText("Stephen Curry")).toBeInTheDocument()
    expect(screen.getByText("72%")).toBeInTheDocument()
  })

  it("shows error message on fetch failure", () => {
    mockUseBestPlayer.mockReturnValue({
      isLoading: false,
      data: undefined,
      error: new Error("Not enough player data"),
    })
    render(<BestPlayerCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText(/Not enough player data/)).toBeInTheDocument()
  })
})
