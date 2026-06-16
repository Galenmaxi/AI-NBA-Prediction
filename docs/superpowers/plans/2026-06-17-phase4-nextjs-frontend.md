# Next.js 14 Frontend Dashboard — Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Next.js 14 App Router frontend that displays NBA win probability, best player, and player stat predictions from the Phase 3 FastAPI backend (running at `http://localhost:8000`).

**Architecture:** Single-page dashboard with a matchup team selector; React Query (TanStack v5) fetches from the FastAPI backend; shadcn/ui Card + Badge components for layout; Recharts horizontal BarChart for win probability visualization; Tailwind CSS v3 for styling. All API responses are typed end-to-end with TypeScript interfaces matching the Phase 3 Pydantic schemas.

**Tech Stack:** Next.js 14.2, React 18, TypeScript 5, Tailwind CSS 3, shadcn/ui (Card, Badge), Recharts 2, TanStack React Query 5, Jest 29, @testing-library/react 14.

---

## File Map

```
frontend/
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── postcss.config.mjs
├── jest.config.ts
├── jest.setup.ts
├── .env.local                            ← NEXT_PUBLIC_API_URL=http://localhost:8000
├── src/
│   ├── app/
│   │   ├── layout.tsx                    ← root layout, wraps with <Providers>
│   │   ├── providers.tsx                 ← React Query QueryClientProvider
│   │   ├── page.tsx                      ← home page (matchup selector + result cards)
│   │   └── globals.css                   ← Tailwind base + shadcn CSS variables
│   ├── components/
│   │   ├── ui/
│   │   │   ├── card.tsx                  ← shadcn Card (inline, no CLI required)
│   │   │   └── badge.tsx                 ← shadcn Badge (inline, no CLI required)
│   │   ├── MatchupSelector.tsx           ← home/away team <select> dropdowns + Predict button
│   │   ├── WinProbabilityCard.tsx        ← win prob % + confidence badge + Recharts bar
│   │   ├── BestPlayerCard.tsx            ← ranked list of star player predictions
│   │   └── PlayerStatsCard.tsx           ← pts/reb/ast stat display
│   ├── lib/
│   │   ├── types.ts                      ← TypeScript interfaces matching backend schemas
│   │   ├── api.ts                        ← fetch wrappers for 3 prediction endpoints
│   │   ├── teams.ts                      ← NBA_TEAMS: all 30 teams with id + name + abbrev
│   │   └── utils.ts                      ← cn() helper (clsx + tailwind-merge)
│   └── hooks/
│       ├── useWinProbability.ts
│       ├── useBestPlayer.ts
│       └── usePlayerStats.ts
└── __tests__/
    ├── api.test.ts
    ├── WinProbabilityCard.test.tsx
    ├── BestPlayerCard.test.tsx
    └── PlayerStatsCard.test.tsx
```

---

## Task 1: Bootstrap Next.js 14 project

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.ts`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.mjs`
- Create: `frontend/jest.config.ts`
- Create: `frontend/jest.setup.ts`
- Create: `frontend/.env.local`
- Create: `frontend/src/app/globals.css`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "nba-predictor-frontend",
  "version": "0.4.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "test": "jest --watchAll=false",
    "test:watch": "jest"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.62.3",
    "class-variance-authority": "^0.7.1",
    "clsx": "^2.1.1",
    "lucide-react": "^0.462.0",
    "next": "14.2.20",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.14.1",
    "tailwind-merge": "^2.5.5",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^14.3.1",
    "@testing-library/user-event": "^14.5.2",
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.4.20",
    "jest": "^29.7.0",
    "jest-environment-jsdom": "^29.7.0",
    "postcss": "^8",
    "tailwindcss": "^3.4.16",
    "typescript": "^5"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Create `frontend/next.config.ts`**

```ts
import type { NextConfig } from "next"

const nextConfig: NextConfig = {}

export default nextConfig
```

- [ ] **Step 4: Create `frontend/tailwind.config.ts`**

```ts
import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}

export default config
```

- [ ] **Step 5: Create `frontend/postcss.config.mjs`**

```mjs
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}

export default config
```

- [ ] **Step 6: Create `frontend/jest.config.ts`**

```ts
import type { Config } from "jest"
import nextJest from "next/jest.js"

const createJestConfig = nextJest({ dir: "./" })

const config: Config = {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
}

export default createJestConfig(config)
```

- [ ] **Step 7: Create `frontend/jest.setup.ts`**

```ts
import "@testing-library/jest-dom"
```

- [ ] **Step 8: Create `frontend/.env.local`**

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 9: Create `frontend/src/app/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
    --radius: 0.5rem;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
}
```

- [ ] **Step 10: Install dependencies**

From `frontend/`:
```powershell
cd c:\Users\ASUS\OneDrive\Documents\AI_NBA_Prediction\nba-ai-predictor\frontend
npm install
```

Expected: Packages installed with no errors (warnings OK).

- [ ] **Step 11: Verify build works**

```powershell
# Still in frontend/
npx next build 2>&1 | head -20
```

Expected: Build fails with "No pages/app" error — that's correct because we haven't created `src/app/layout.tsx` yet. What we must NOT see: TypeScript errors or module-not-found errors about the listed packages.

- [ ] **Step 12: Commit**

```powershell
git add frontend/package.json frontend/tsconfig.json frontend/next.config.ts frontend/tailwind.config.ts frontend/postcss.config.mjs frontend/jest.config.ts frontend/jest.setup.ts frontend/.env.local frontend/src/app/globals.css
git commit -m "chore: bootstrap Next.js 14 frontend with Tailwind, React Query, Jest"
```

---

## Task 2: TypeScript types + API fetch functions + tests

**Files:**
- Create: `frontend/src/lib/types.ts`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/__tests__/api.test.ts`

- [ ] **Step 1: Create `frontend/src/lib/types.ts`**

```ts
export interface WinProbabilityResponse {
  home_team_id: number
  away_team_id: number
  home_win_prob: number
  away_win_prob: number
  confidence: "low" | "medium" | "high"
}

export interface PlayerStarPrediction {
  player_id: number
  player_name: string
  star_probability: number
}

export interface BestPlayerResponse {
  home_team_id: number
  away_team_id: number
  players: PlayerStarPrediction[]
}

export interface StatPrediction {
  pts: number
  reb: number
  ast: number
}

export interface PlayerStatsResponse {
  player_id: number
  predicted_stats: StatPrediction
}
```

- [ ] **Step 2: Write the failing test first — `frontend/__tests__/api.test.ts`**

```ts
import { fetchWinProbability, fetchBestPlayer, fetchPlayerStats } from "@/lib/api"

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
```

- [ ] **Step 3: Run test to verify it fails**

```powershell
cd c:\Users\ASUS\OneDrive\Documents\AI_NBA_Prediction\nba-ai-predictor\frontend
npx jest __tests__/api.test.ts --no-coverage 2>&1 | tail -10
```

Expected: FAIL — `Cannot find module '@/lib/api'`

- [ ] **Step 4: Create `frontend/src/lib/api.ts`**

```ts
import type { BestPlayerResponse, PlayerStatsResponse, WinProbabilityResponse } from "./types"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((err as { detail?: string }).detail ?? "API error")
  }
  return res.json() as Promise<T>
}

export function fetchWinProbability(
  homeTeamId: number,
  awayTeamId: number,
): Promise<WinProbabilityResponse> {
  return apiFetch(
    `/predictions/win-probability?home_team_id=${homeTeamId}&away_team_id=${awayTeamId}`,
  )
}

export function fetchBestPlayer(
  homeTeamId: number,
  awayTeamId: number,
): Promise<BestPlayerResponse> {
  return apiFetch(
    `/predictions/best-player?home_team_id=${homeTeamId}&away_team_id=${awayTeamId}`,
  )
}

export function fetchPlayerStats(playerId: number): Promise<PlayerStatsResponse> {
  return apiFetch(`/predictions/player-stats?player_id=${playerId}`)
}
```

- [ ] **Step 5: Run test to verify it passes**

```powershell
npx jest __tests__/api.test.ts --no-coverage 2>&1 | tail -10
```

Expected: PASS — 4 tests, 0 failures

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/__tests__/api.test.ts
git commit -m "feat: add TypeScript types and API fetch functions with tests"
```

---

## Task 3: Shared utilities + shadcn UI components + NBA teams data

**Files:**
- Create: `frontend/src/lib/utils.ts`
- Create: `frontend/src/lib/teams.ts`
- Create: `frontend/src/components/ui/card.tsx`
- Create: `frontend/src/components/ui/badge.tsx`

- [ ] **Step 1: Create `frontend/src/lib/utils.ts`**

```ts
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

- [ ] **Step 2: Create `frontend/src/lib/teams.ts`**

```ts
export const NBA_TEAMS = [
  { id: 1610612737, name: "Atlanta Hawks", abbreviation: "ATL" },
  { id: 1610612738, name: "Boston Celtics", abbreviation: "BOS" },
  { id: 1610612751, name: "Brooklyn Nets", abbreviation: "BKN" },
  { id: 1610612766, name: "Charlotte Hornets", abbreviation: "CHA" },
  { id: 1610612741, name: "Chicago Bulls", abbreviation: "CHI" },
  { id: 1610612739, name: "Cleveland Cavaliers", abbreviation: "CLE" },
  { id: 1610612742, name: "Dallas Mavericks", abbreviation: "DAL" },
  { id: 1610612743, name: "Denver Nuggets", abbreviation: "DEN" },
  { id: 1610612765, name: "Detroit Pistons", abbreviation: "DET" },
  { id: 1610612744, name: "Golden State Warriors", abbreviation: "GSW" },
  { id: 1610612745, name: "Houston Rockets", abbreviation: "HOU" },
  { id: 1610612754, name: "Indiana Pacers", abbreviation: "IND" },
  { id: 1610612746, name: "LA Clippers", abbreviation: "LAC" },
  { id: 1610612747, name: "Los Angeles Lakers", abbreviation: "LAL" },
  { id: 1610612763, name: "Memphis Grizzlies", abbreviation: "MEM" },
  { id: 1610612748, name: "Miami Heat", abbreviation: "MIA" },
  { id: 1610612749, name: "Milwaukee Bucks", abbreviation: "MIL" },
  { id: 1610612750, name: "Minnesota Timberwolves", abbreviation: "MIN" },
  { id: 1610612740, name: "New Orleans Pelicans", abbreviation: "NOP" },
  { id: 1610612752, name: "New York Knicks", abbreviation: "NYK" },
  { id: 1610612760, name: "Oklahoma City Thunder", abbreviation: "OKC" },
  { id: 1610612753, name: "Orlando Magic", abbreviation: "ORL" },
  { id: 1610612755, name: "Philadelphia 76ers", abbreviation: "PHI" },
  { id: 1610612756, name: "Phoenix Suns", abbreviation: "PHX" },
  { id: 1610612757, name: "Portland Trail Blazers", abbreviation: "POR" },
  { id: 1610612758, name: "Sacramento Kings", abbreviation: "SAC" },
  { id: 1610612759, name: "San Antonio Spurs", abbreviation: "SAS" },
  { id: 1610612761, name: "Toronto Raptors", abbreviation: "TOR" },
  { id: 1610612762, name: "Utah Jazz", abbreviation: "UTA" },
  { id: 1610612764, name: "Washington Wizards", abbreviation: "WAS" },
] as const

export type NBATeam = (typeof NBA_TEAMS)[number]
```

- [ ] **Step 3: Create `frontend/src/components/ui/card.tsx`**

```tsx
import * as React from "react"
import { cn } from "@/lib/utils"

const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("rounded-xl border bg-card text-card-foreground shadow", className)}
      {...props}
    />
  ),
)
Card.displayName = "Card"

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
  ),
)
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn("font-semibold leading-none tracking-tight", className)}
      {...props}
    />
  ),
)
CardTitle.displayName = "CardTitle"

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
  ),
)
CardContent.displayName = "CardContent"

export { Card, CardContent, CardHeader, CardTitle }
```

- [ ] **Step 4: Create `frontend/src/components/ui/badge.tsx`**

```tsx
import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        destructive: "border-transparent bg-destructive text-destructive-foreground",
        outline: "text-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
```

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/lib/utils.ts frontend/src/lib/teams.ts frontend/src/components/ui/card.tsx frontend/src/components/ui/badge.tsx
git commit -m "feat: add shared utilities, shadcn Card/Badge, and NBA teams data"
```

---

## Task 4: React Query setup + data-fetching hooks

**Files:**
- Create: `frontend/src/app/providers.tsx`
- Create: `frontend/src/hooks/useWinProbability.ts`
- Create: `frontend/src/hooks/useBestPlayer.ts`
- Create: `frontend/src/hooks/usePlayerStats.ts`

- [ ] **Step 1: Create `frontend/src/app/providers.tsx`**

```tsx
"use client"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState } from "react"

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000,
            retry: 1,
          },
        },
      }),
  )
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}
```

- [ ] **Step 2: Create `frontend/src/hooks/useWinProbability.ts`**

```ts
import { useQuery } from "@tanstack/react-query"
import { fetchWinProbability } from "@/lib/api"
import type { WinProbabilityResponse } from "@/lib/types"

export function useWinProbability(homeTeamId: number | null, awayTeamId: number | null) {
  return useQuery<WinProbabilityResponse, Error>({
    queryKey: ["winProbability", homeTeamId, awayTeamId],
    queryFn: () => fetchWinProbability(homeTeamId!, awayTeamId!),
    enabled: homeTeamId !== null && awayTeamId !== null,
  })
}
```

- [ ] **Step 3: Create `frontend/src/hooks/useBestPlayer.ts`**

```ts
import { useQuery } from "@tanstack/react-query"
import { fetchBestPlayer } from "@/lib/api"
import type { BestPlayerResponse } from "@/lib/types"

export function useBestPlayer(homeTeamId: number | null, awayTeamId: number | null) {
  return useQuery<BestPlayerResponse, Error>({
    queryKey: ["bestPlayer", homeTeamId, awayTeamId],
    queryFn: () => fetchBestPlayer(homeTeamId!, awayTeamId!),
    enabled: homeTeamId !== null && awayTeamId !== null,
  })
}
```

- [ ] **Step 4: Create `frontend/src/hooks/usePlayerStats.ts`**

```ts
import { useQuery } from "@tanstack/react-query"
import { fetchPlayerStats } from "@/lib/api"
import type { PlayerStatsResponse } from "@/lib/types"

export function usePlayerStats(playerId: number | null) {
  return useQuery<PlayerStatsResponse, Error>({
    queryKey: ["playerStats", playerId],
    queryFn: () => fetchPlayerStats(playerId!),
    enabled: playerId !== null && playerId > 0,
  })
}
```

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/app/providers.tsx frontend/src/hooks/
git commit -m "feat: add React Query provider and win-probability, best-player, player-stats hooks"
```

---

## Task 5: WinProbabilityCard with Recharts bar

**Files:**
- Create: `frontend/src/components/WinProbabilityCard.tsx`
- Create: `frontend/__tests__/WinProbabilityCard.test.tsx`

- [ ] **Step 1: Write the failing test — `frontend/__tests__/WinProbabilityCard.test.tsx`**

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
npx jest __tests__/WinProbabilityCard.test.tsx --no-coverage 2>&1 | tail -10
```

Expected: FAIL — `Cannot find module '@/components/WinProbabilityCard'`

- [ ] **Step 3: Create `frontend/src/components/WinProbabilityCard.tsx`**

```tsx
"use client"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useWinProbability } from "@/hooks/useWinProbability"
import { NBA_TEAMS } from "@/lib/teams"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

function teamAbbrev(id: number): string {
  return NBA_TEAMS.find((t) => t.id === id)?.abbreviation ?? String(id)
}

export function WinProbabilityCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useWinProbability(homeTeamId, awayTeamId)

  const chartData = data
    ? [
        { team: `${teamAbbrev(homeTeamId)} (H)`, prob: Math.round(data.home_win_prob * 100) },
        { team: `${teamAbbrev(awayTeamId)} (A)`, prob: Math.round(data.away_win_prob * 100) },
      ]
    : []

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Win Probability
          {data && (
            <Badge variant={data.confidence === "high" ? "default" : "secondary"}>
              {data.confidence}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <>
            <div className="flex justify-around mb-4">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {Math.round(data.home_win_prob * 100)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {teamAbbrev(homeTeamId)} · Home
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-600">
                  {Math.round(data.away_win_prob * 100)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {teamAbbrev(awayTeamId)} · Away
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={100}>
              <BarChart layout="vertical" data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="team"
                  width={72}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="prob" maxBarSize={28}>
                  <Cell fill="#2563eb" />
                  <Cell fill="#dc2626" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
npx jest __tests__/WinProbabilityCard.test.tsx --no-coverage 2>&1 | tail -10
```

Expected: PASS — 3 tests, 0 failures

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/components/WinProbabilityCard.tsx frontend/__tests__/WinProbabilityCard.test.tsx
git commit -m "feat: add WinProbabilityCard with Recharts horizontal bar and confidence badge"
```

---

## Task 6: BestPlayerCard

**Files:**
- Create: `frontend/src/components/BestPlayerCard.tsx`
- Create: `frontend/__tests__/BestPlayerCard.test.tsx`

- [ ] **Step 1: Write the failing test — `frontend/__tests__/BestPlayerCard.test.tsx`**

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
npx jest __tests__/BestPlayerCard.test.tsx --no-coverage 2>&1 | tail -10
```

Expected: FAIL — `Cannot find module '@/components/BestPlayerCard'`

- [ ] **Step 3: Create `frontend/src/components/BestPlayerCard.tsx`**

```tsx
"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useBestPlayer } from "@/hooks/useBestPlayer"

interface Props {
  homeTeamId: number
  awayTeamId: number
}

export function BestPlayerCard({ homeTeamId, awayTeamId }: Props) {
  const { data, isLoading, error } = useBestPlayer(homeTeamId, awayTeamId)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Star Player Predictions</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <ol className="space-y-3">
            {data.players.map((player, i) => (
              <li key={player.player_id} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-muted-foreground text-sm w-5">{i + 1}.</span>
                  <span className="font-medium text-sm">{player.player_name}</span>
                </div>
                <span className="text-sm font-semibold text-blue-600">
                  {Math.round(player.star_probability * 100)}%
                </span>
              </li>
            ))}
          </ol>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
npx jest __tests__/BestPlayerCard.test.tsx --no-coverage 2>&1 | tail -10
```

Expected: PASS — 3 tests, 0 failures

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/components/BestPlayerCard.tsx frontend/__tests__/BestPlayerCard.test.tsx
git commit -m "feat: add BestPlayerCard with ranked star-probability list"
```

---

## Task 7: PlayerStatsCard

**Files:**
- Create: `frontend/src/components/PlayerStatsCard.tsx`
- Create: `frontend/__tests__/PlayerStatsCard.test.tsx`

- [ ] **Step 1: Write the failing test — `frontend/__tests__/PlayerStatsCard.test.tsx`**

```tsx
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
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
npx jest __tests__/PlayerStatsCard.test.tsx --no-coverage 2>&1 | tail -10
```

Expected: FAIL — `Cannot find module '@/components/PlayerStatsCard'`

- [ ] **Step 3: Create `frontend/src/components/PlayerStatsCard.tsx`**

```tsx
"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { usePlayerStats } from "@/hooks/usePlayerStats"

interface Props {
  playerId: number
}

export function PlayerStatsCard({ playerId }: Props) {
  const { data, isLoading, error } = usePlayerStats(playerId)

  return (
    <Card className="w-full max-w-xs">
      <CardHeader>
        <CardTitle className="text-base">Player #{playerId}</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && <p className="text-muted-foreground text-sm">Loading...</p>}
        {error && <p className="text-destructive text-sm">{error.message}</p>}
        {data && (
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold">{data.predicted_stats.pts.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">PTS</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{data.predicted_stats.reb.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">REB</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{data.predicted_stats.ast.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground uppercase tracking-wide mt-1">AST</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
npx jest __tests__/PlayerStatsCard.test.tsx --no-coverage 2>&1 | tail -10
```

Expected: PASS — 3 tests, 0 failures

- [ ] **Step 5: Commit**

```powershell
git add frontend/src/components/PlayerStatsCard.tsx frontend/__tests__/PlayerStatsCard.test.tsx
git commit -m "feat: add PlayerStatsCard with pts/reb/ast predicted stat display"
```

---

## Task 8: MatchupSelector + Home page + layout + build verify

**Files:**
- Create: `frontend/src/components/MatchupSelector.tsx`
- Create: `frontend/src/app/layout.tsx`
- Create: `frontend/src/app/page.tsx`

- [ ] **Step 1: Create `frontend/src/components/MatchupSelector.tsx`**

```tsx
"use client"
import { useState } from "react"
import { NBA_TEAMS } from "@/lib/teams"

interface Props {
  onSubmit: (homeTeamId: number, awayTeamId: number) => void
}

export function MatchupSelector({ onSubmit }: Props) {
  const [homeTeamId, setHomeTeamId] = useState<number>(NBA_TEAMS[9].id) // GSW default
  const [awayTeamId, setAwayTeamId] = useState<number>(NBA_TEAMS[13].id) // LAL default

  return (
    <div className="flex flex-wrap gap-6 items-end">
      <div>
        <label className="text-sm font-medium block mb-1">Home Team</label>
        <select
          value={homeTeamId}
          onChange={(e) => setHomeTeamId(Number(e.target.value))}
          className="border rounded-md px-3 py-2 text-sm bg-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {NBA_TEAMS.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </div>
      <div className="text-xl font-bold text-muted-foreground self-end pb-2">vs.</div>
      <div>
        <label className="text-sm font-medium block mb-1">Away Team</label>
        <select
          value={awayTeamId}
          onChange={(e) => setAwayTeamId(Number(e.target.value))}
          className="border rounded-md px-3 py-2 text-sm bg-background focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {NBA_TEAMS.map((t) => (
            <option key={t.id} value={t.id}>
              {t.name}
            </option>
          ))}
        </select>
      </div>
      <button
        onClick={() => onSubmit(homeTeamId, awayTeamId)}
        className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 font-medium text-sm transition-colors"
      >
        Predict
      </button>
    </div>
  )
}
```

- [ ] **Step 2: Create `frontend/src/app/layout.tsx`**

```tsx
import type { Metadata } from "next"
import "./globals.css"
import { Providers } from "./providers"

export const metadata: Metadata = {
  title: "NBA AI Predictor",
  description: "AI-powered NBA game predictions",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
```

- [ ] **Step 3: Create `frontend/src/app/page.tsx`**

```tsx
"use client"
import { useState } from "react"
import { BestPlayerCard } from "@/components/BestPlayerCard"
import { MatchupSelector } from "@/components/MatchupSelector"
import { PlayerStatsCard } from "@/components/PlayerStatsCard"
import { WinProbabilityCard } from "@/components/WinProbabilityCard"

interface Matchup {
  homeTeamId: number
  awayTeamId: number
}

export default function HomePage() {
  const [matchup, setMatchup] = useState<Matchup | null>(null)
  const [playerIdInput, setPlayerIdInput] = useState("")
  const [activePlayerId, setActivePlayerId] = useState<number | null>(null)

  return (
    <main className="container mx-auto p-6 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">NBA AI Predictor</h1>
        <p className="text-muted-foreground mt-1">
          Select a matchup to see AI-powered win probability and player predictions.
        </p>
      </div>

      <section className="mb-8">
        <h2 className="text-lg font-semibold mb-4">Select Matchup</h2>
        <MatchupSelector onSubmit={(h, a) => setMatchup({ homeTeamId: h, awayTeamId: a })} />
      </section>

      {matchup && (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <WinProbabilityCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
          <BestPlayerCard homeTeamId={matchup.homeTeamId} awayTeamId={matchup.awayTeamId} />
        </section>
      )}

      <section>
        <h2 className="text-lg font-semibold mb-4">Player Stats Predictor</h2>
        <p className="text-sm text-muted-foreground mb-3">
          Enter an NBA player ID (e.g. 2544 = LeBron James, 201939 = Stephen Curry).
        </p>
        <div className="flex gap-3 items-end">
          <div>
            <label className="text-sm font-medium block mb-1">Player ID</label>
            <input
              type="number"
              value={playerIdInput}
              onChange={(e) => setPlayerIdInput(e.target.value)}
              placeholder="e.g. 2544"
              className="border rounded-md px-3 py-2 text-sm w-40 focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <button
            onClick={() => setActivePlayerId(Number(playerIdInput))}
            disabled={!playerIdInput || Number(playerIdInput) <= 0}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 font-medium text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Predict
          </button>
        </div>
        {activePlayerId && (
          <div className="mt-4">
            <PlayerStatsCard playerId={activePlayerId} />
          </div>
        )}
      </section>
    </main>
  )
}
```

- [ ] **Step 4: Run all tests to verify nothing broken**

```powershell
cd c:\Users\ASUS\OneDrive\Documents\AI_NBA_Prediction\nba-ai-predictor\frontend
npx jest --no-coverage 2>&1 | tail -15
```

Expected: All 13 tests pass (4 api + 3 WinProbabilityCard + 3 BestPlayerCard + 3 PlayerStatsCard), 0 failures.

- [ ] **Step 5: Run production build to verify TypeScript and no broken imports**

```powershell
npx next build 2>&1 | tail -20
```

Expected: Build succeeds with output like `✓ Compiled successfully` or `Route (app) / ...`. No TypeScript errors, no missing module errors.

- [ ] **Step 6: Commit**

```powershell
git add frontend/src/components/MatchupSelector.tsx frontend/src/app/layout.tsx frontend/src/app/page.tsx
git commit -m "feat: add home page with matchup selector and player stats predictor"
```

---

## Task 9: Update HANDOFF.md with Phase 4

**Files:**
- Modify: `nba-ai-predictor/HANDOFF.md`

- [ ] **Step 1: Add Phase 4 section to HANDOFF.md**

Append the following section after the Phase 3 section (before "Important Decisions Made"):

```markdown
---

## What Has Been Built (Phase 4)

### Git log additions
```
feat: add home page with matchup selector and player stats predictor
feat: add PlayerStatsCard with pts/reb/ast predicted stat display
feat: add BestPlayerCard with ranked star-probability list
feat: add WinProbabilityCard with Recharts horizontal bar and confidence badge
feat: add React Query provider and prediction hooks
feat: add shared utilities, shadcn Card/Badge, and NBA teams data
feat: add TypeScript types and API fetch functions with tests
chore: bootstrap Next.js 14 frontend with Tailwind, React Query, Jest
```

### Frontend tech stack installed
| Package | Version | Purpose |
|---|---|---|
| next | 14.2.20 | Framework (App Router) |
| react + react-dom | 18.3.1 | UI runtime |
| @tanstack/react-query | 5.x | Server state + caching |
| recharts | 2.x | Win probability bar chart |
| tailwindcss | 3.x | Utility-first CSS |
| class-variance-authority + clsx + tailwind-merge | latest | shadcn/ui utilities |

### Running the frontend

```powershell
cd frontend
npm install        # first time only
npm run dev        # http://localhost:3000
```

The backend must also be running:
```powershell
cd backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

### Pages and components
| Component | Purpose |
|---|---|
| `src/app/page.tsx` | Home: matchup selector, win prob, best player, player stats |
| `src/components/MatchupSelector.tsx` | Dropdown for all 30 NBA teams + Predict button |
| `src/components/WinProbabilityCard.tsx` | Win % display + confidence badge + Recharts bar |
| `src/components/BestPlayerCard.tsx` | Ranked list of star-probability predictions |
| `src/components/PlayerStatsCard.tsx` | Predicted pts/reb/ast for a given player ID |

### Test suite
13 frontend tests, 0 failures.

| File | Tests |
|---|---|
| `__tests__/api.test.ts` | 4 |
| `__tests__/WinProbabilityCard.test.tsx` | 3 |
| `__tests__/BestPlayerCard.test.tsx` | 3 |
| `__tests__/PlayerStatsCard.test.tsx` | 3 |

Run from `frontend/`:
```powershell
npm test
```

### Key file paths
| What | Path |
|---|---|
| Frontend root | `frontend/` |
| API fetch functions | `frontend/src/lib/api.ts` |
| TypeScript types | `frontend/src/lib/types.ts` |
| NBA team IDs | `frontend/src/lib/teams.ts` |
| Backend API URL | `frontend/.env.local` → `NEXT_PUBLIC_API_URL` |
| Phase 4 plan | `docs/superpowers/plans/2026-06-17-phase4-nextjs-frontend.md` |
```

- [ ] **Step 2: Commit**

```powershell
git add nba-ai-predictor/HANDOFF.md
git commit -m "docs: update HANDOFF.md with Phase 4 Next.js frontend"
```

---

## Self-Review

**Spec coverage:**
- [x] Next.js 14 App Router — Task 1 (bootstrap) + Task 8 (layout, page)
- [x] Tailwind CSS + shadcn/ui — Task 1 (tailwind config, globals.css) + Task 3 (card, badge)
- [x] Recharts — Task 5 (WinProbabilityCard bar chart)
- [x] React Query — Task 4 (providers + 3 hooks)
- [x] Win probability prediction display — Task 5
- [x] Best player display — Task 6
- [x] Player stats display — Task 7
- [x] HANDOFF.md update — Task 9
- [x] Tests for all components — Tasks 2, 5, 6, 7
- [x] TypeScript end-to-end — types match backend Pydantic schemas (Task 2)

**Placeholder scan:** No TBDs, every step has code.

**Type consistency:**
- `WinProbabilityResponse.confidence` typed as `"low" | "medium" | "high"` in `types.ts`; used as string in Badge variant — consistent.
- `useWinProbability(homeTeamId: number | null, awayTeamId: number | null)` — called with `number` from components — consistent.
- `PlayerStatsCard({ playerId: number })` — hook signature `usePlayerStats(playerId: number | null)` — consistent.
- `NBA_TEAMS[9].id` is `1610612744` (GSW), `NBA_TEAMS[13].id` is `1610612747` (LAL) — verified against teams array index.
