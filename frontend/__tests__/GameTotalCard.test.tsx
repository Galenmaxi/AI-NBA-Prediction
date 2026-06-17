import { render, screen } from "@testing-library/react"
import { GameTotalCard } from "@/components/GameTotalCard"
import { useGameTotal } from "@/hooks/useGameTotal"

jest.mock("@/hooks/useGameTotal")
const mockUseGameTotal = useGameTotal as jest.MockedFunction<typeof useGameTotal>

describe("GameTotalCard", () => {
  it("shows loading state", () => {
    mockUseGameTotal.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as any)
    render(<GameTotalCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })

  it("shows predicted total and confidence badge", () => {
    mockUseGameTotal.mockReturnValue({
      data: {
        home_team_id: 1,
        away_team_id: 2,
        predicted_total: 221.5,
        confidence: "high",
      },
      isLoading: false,
      error: null,
    } as any)
    render(<GameTotalCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText("221.5")).toBeInTheDocument()
    expect(screen.getByText("high")).toBeInTheDocument()
  })

  it("shows error message", () => {
    mockUseGameTotal.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Model not trained"),
    } as any)
    render(<GameTotalCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText(/model not trained/i)).toBeInTheDocument()
  })
})
