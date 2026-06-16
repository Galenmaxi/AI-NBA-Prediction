import { render, screen } from "@testing-library/react"
import { WinProbabilityCard } from "@/components/WinProbabilityCard"
import * as winProbHook from "@/hooks/useWinProbability"

jest.mock("@/hooks/useWinProbability")

jest.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Cell: () => null,
}))

const mockUseWinProbability = winProbHook.useWinProbability as jest.Mock

describe("WinProbabilityCard", () => {
  it("shows loading state while fetching", () => {
    mockUseWinProbability.mockReturnValue({ isLoading: true, data: undefined, error: null })
    render(<WinProbabilityCard homeTeamId={1610612744} awayTeamId={1610612747} />)
    expect(screen.getByText("Loading...")).toBeInTheDocument()
  })

  it("shows win percentages and confidence badge when data is loaded", () => {
    mockUseWinProbability.mockReturnValue({
      isLoading: false,
      data: {
        home_team_id: 1610612744,
        away_team_id: 1610612747,
        home_win_prob: 0.65,
        away_win_prob: 0.35,
        confidence: "high",
      },
      error: null,
    })
    render(<WinProbabilityCard homeTeamId={1610612744} awayTeamId={1610612747} />)
    expect(screen.getByText("65%")).toBeInTheDocument()
    expect(screen.getByText("35%")).toBeInTheDocument()
    expect(screen.getByText("high")).toBeInTheDocument()
  })

  it("shows error message on fetch failure", () => {
    mockUseWinProbability.mockReturnValue({
      isLoading: false,
      data: undefined,
      error: new Error("Model not trained"),
    })
    render(<WinProbabilityCard homeTeamId={1} awayTeamId={2} />)
    expect(screen.getByText(/Model not trained/)).toBeInTheDocument()
  })
})
