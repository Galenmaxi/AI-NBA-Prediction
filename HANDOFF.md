# NBA AI Predictor — Agent Handoff Document

> Last updated: 2026-06-16
> Use this document to onboard a new Claude Code agent to the current state of the project.

---

## What We Are Building

An **NBA AI Prediction System** — a full-stack application that:

1. Predicts **win probability** for each team before/during a game
2. Predicts the **best player (star performer)** of a given game
3. Predicts **individual player stat lines** (points, assists, rebounds, steals, blocks)
4. Predicts **game total score** (over/under)
5. Provides a **confidence score** for each prediction

---

## Tech Stack

### Python Backend / ML
| Package | Version | Purpose |
|---|---|---|
| Python | 3.13.5 (64-bit) | Runtime — **NOT 3.11** as originally planned |
| nba_api | 1.11.4 | NBA stats data (stats.nba.com) |
| pandas | 3.0.3 | Data wrangling |
| numpy | 2.4.6 | Numerical ops |
| sqlalchemy | 2.0.51 | ORM |
| psycopg2-binary | 2.9.12 | PostgreSQL driver |
| python-dotenv | 1.2.2 | Env var loading |
| pytest | 9.1.0 | Testing |
| pytest-mock | 3.15.1 | Mocking in tests |

> **Important:** The originally specified versions (pandas 2.1.4, numpy 1.26.2) had no pre-built wheels for Python 3.13 and failed to build from source (requires GCC ≥ 8.4, system has 6.3.0). All versions above are the actual installed versions that work.

### Planned (not yet installed — future phases)
- `scikit-learn` — preprocessing pipelines
- `xgboost` + `lightgbm` — primary ML models
- `mlflow` — experiment tracking
- `fastapi` + `uvicorn` — REST API
- `celery` + `redis` — background jobs

### Infrastructure
| Tool | Purpose |
|---|---|
| Docker + docker-compose | Local PostgreSQL + Redis |
| PostgreSQL 15 | Stores game logs and predictions |
| Redis 7 | Cache / task queue (Phase 3+) |
| Railway or Render | Backend hosting (Phase 5) |
| Vercel | Frontend hosting (Phase 5) |

### Frontend (Phase 4 — not started)
- Next.js 14 (App Router)
- Tailwind CSS + shadcn/ui
- Recharts
- React Query

---

## Project Structure

```
nba-ai-predictor/
├── HANDOFF.md                      ← this file
├── docker-compose.yml              ← PostgreSQL 15 + Redis 7
├── .env                            ← local secrets (git-ignored)
├── .env.example                    ← template (committed)
├── .gitignore
├── docs/
│   └── superpowers/plans/
│       └── 2026-06-16-phase1-data-pipeline.md   ← full Phase 1 plan
├── backend/
│   ├── pytest.ini                  ← pythonpath = . (required for imports to work)
│   ├── requirements.txt            ← pinned to Python 3.13 compatible versions
│   ├── venv/                       ← virtual environment (git-ignored)
│   ├── app/
│   │   ├── models/
│   │   │   ├── base.py             ← DeclarativeBase
│   │   │   ├── team_game_log.py    ← TeamGameLog ORM model
│   │   │   ├── player_game_log.py  ← PlayerGameLog ORM model
│   │   │   ├── database.py         ← engine, SessionLocal, create_tables()
│   │   │   └── __init__.py         ← re-exports Base, TeamGameLog, PlayerGameLog
│   │   ├── main.py                 ← placeholder (Phase 3)
│   │   ├── schemas/                ← placeholder (Phase 3)
│   │   ├── routers/                ← placeholder (Phase 3)
│   │   └── services/               ← placeholder (Phase 3)
│   ├── ml/
│   │   ├── data_collector.py       ← COMPLETE: fetch + clean team/player logs
│   │   ├── feature_engineering.py  ← placeholder (Phase 2)
│   │   ├── train_win_model.py      ← placeholder (Phase 2)
│   │   ├── train_player_model.py   ← placeholder (Phase 2)
│   │   └── predict.py              ← placeholder (Phase 2/3)
│   ├── scripts/
│   │   └── seed_database.py        ← COMPLETE: idempotent historical data loader
│   └── tests/
│       ├── test_models.py          ← 3 tests (SQLite in-memory)
│       ├── test_data_collector.py  ← 12 tests (mocked nba_api)
│       └── test_seed_database.py   ← 6 tests (SQLite in-memory)
└── frontend/                       ← empty, Phase 4
```

---

## What Has Been Built (Phase 1)

### Git log
```
73ef105  feat: add idempotent database seeder for historical game logs
78fa484  feat: add team and player game log fetcher with cleaning logic
bbb7c90  feat: add SQLAlchemy models for team and player game logs
fdf0bb2  chore: initialize project structure with docker-compose and requirements
```

### Database Models (`backend/app/models/`)

**`TeamGameLog`** — table `team_game_logs`
- One row per team per game
- Unique constraint on `(game_id, team_id)`
- Key columns: `season`, `team_id`, `team_abbreviation`, `game_date`, `matchup`, `home_away` (HOME/AWAY), `wl`, `pts`, `fgm/fga/fg_pct`, `fg3m/fg3a/fg3_pct`, `ftm/fta/ft_pct`, `oreb/dreb/reb`, `ast`, `tov`, `stl`, `blk`, `plus_minus`

**`PlayerGameLog`** — table `player_game_logs`
- One row per player per game
- Unique constraint on `(game_id, player_id)`
- Same stat columns as above, all nullable (DNP players have no stats)
- Extra: `player_id`, `player_name`

### Data Collector (`backend/ml/data_collector.py`)

```python
SEASONS: list[str] = ["2023-24", "2024-25", "2025-26"]

fetch_team_game_logs(season: str) -> pd.DataFrame
fetch_player_game_logs(season: str) -> pd.DataFrame
clean_team_game_logs(df: pd.DataFrame) -> pd.DataFrame
clean_player_game_logs(df: pd.DataFrame) -> pd.DataFrame
```

- Uses `nba_api.stats.endpoints.TeamGameLogs` and `PlayerGameLogs`
- `timeout=30`, `REQUEST_DELAY=0.6s` between calls
- All columns lowercased, `SEASON_YEAR` renamed to `season`
- `home_away` derived from `MATCHUP`: `"vs."` → HOME, `"@"` → AWAY
- Dates parsed with `format="mixed"` (handles both ISO and "OCT 18, 2022")

### Database Seeder (`backend/scripts/seed_database.py`)

```python
insert_team_game_logs(session, df) -> int   # returns rows inserted
insert_player_game_logs(session, df) -> int
count_rows(session, model) -> int
main()  # fetches all SEASONS and seeds the DB
```

- **Idempotent** — checks for existing `(game_id, team_id)` or `(game_id, player_id)` before inserting
- Safe to re-run; second run inserts 0 rows
- `main()` calls `create_tables()` first, then iterates over all `SEASONS`

### Test Suite

```
21 tests, 0 failures, 0 warnings
```

| File | Tests | Notes |
|---|---|---|
| `test_models.py` | 3 | SQLite in-memory, no Docker needed |
| `test_data_collector.py` | 12 | Fully mocked nba_api calls |
| `test_seed_database.py` | 6 | SQLite in-memory, no Docker needed |

Run from `backend/` with venv active:
```powershell
.\venv\Scripts\activate
pytest tests/ -v
```

---

## Environment Setup (for a new machine)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v    # should show 21 passed
```

**IDE note:** VS Code will show `Cannot find module 'nba_api.stats.endpoints'` because the IDE Python interpreter points to the system Python, not the venv. This is a false positive — the venv has nba_api installed and all tests pass. Point VS Code to `backend/venv/Scripts/python.exe` to resolve it.

---

## Outstanding Step (Task 6 — needs Docker)

Docker Desktop is **not installed** on this machine. Once installed, complete Phase 1 by running:

```powershell
# 1. Start PostgreSQL
cd c:\Users\ASUS\OneDrive\Documents\AI_NBA_Prediction\nba-ai-predictor
docker compose up -d postgres

# 2. Create tables
cd backend
.\venv\Scripts\activate
python -c "from app.models.database import create_tables; create_tables(); print('ok')"

# 3. Smoke test (verify nba_api live — already confirmed working, ~2786 rows for 2023-24)
python -c "
from ml.data_collector import fetch_team_game_logs
df = fetch_team_game_logs('2023-24')
print(f'Rows: {len(df)}')
print(df[['season','team_abbreviation','game_date','matchup','wl','pts']].head(3).to_string())
"

# 4. Full seed (~10-15 min due to rate limiting)
python scripts/seed_database.py

# 5. Verify in Postgres
docker exec -it $(docker compose ps -q postgres) psql -U nba_user -d nba_predictor -c "SELECT season, COUNT(*) FROM team_game_logs GROUP BY season ORDER BY season;"

# 6. Re-run seeder — all Inserted N rows should be 0
python scripts/seed_database.py
```

Expected final counts: ~2700–2800 team rows per season, ~60,000–65,000 player rows per season.

---

## What's Next — Phase 2: Feature Engineering + Model Training

Plan file to create: `docs/superpowers/plans/2026-06-16-phase2-feature-engineering.md`

### Phase 2 tasks (from original project brief)
1. Write `feature_engineering.py` — rolling averages, rest days, matchup stats
2. Build train/test split by date (no data leakage — always train on past, test on future)
3. Train **Win Probability model** (XGBoost binary classifier) — evaluate with log loss + accuracy
4. Train **Best Player model** (LightGBM multi-class/ranking) — evaluate with top-1 accuracy
5. Train **Player Stat model** (XGBoost regressor per stat) — evaluate with MAE
6. Set up MLflow to log experiments
7. Save best models as `.joblib` files

### Key ML constraints to enforce
- **No data leakage:** features for game on date D must only use data from before date D
- Rolling windows (last 5, last 10 games) must be computed per-team/player, sorted by `game_date`
- Train/test split = split by date (e.g., train on seasons before 2025-26, test on 2025-26)

### Feature ideas for Win Probability model
- Team win % last 10 games
- Avg pts scored / allowed last 5 games
- Home/away record
- Rest days since last game
- Head-to-head record last 2 seasons
- Offensive/Defensive/Net rating rolling averages
- Key player availability flag (injury — manual input initially)

### Data already available from Phase 1
All features above can be computed from `team_game_logs` and `player_game_logs` tables. No additional data collection needed to start Phase 2.

---

## What Has Been Built (Phase 3)

### Git log additions
```
6876502  feat: add optional Redis caching for win probability predictions
22a0c1b  feat: add prediction API endpoints (win-probability, best-player, player-stats)
2008530  feat: add prediction service layer with DB queries and confidence scoring
d82d521  feat: add FastAPI app skeleton with health endpoint
43faedd  feat: add Pydantic response schemas for prediction API
2af6bfd  feat: add ML inference layer with win probability, best player, and stat prediction
224cc93  chore: add Phase 3 dependencies (fastapi, uvicorn, httpx, redis) and get_db() yield
```

### New packages installed
| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.115.6 | Web framework |
| uvicorn | 0.34.3 | ASGI server |
| httpx | 0.28.1 | TestClient support in tests |
| redis | 5.2.1 | Optional response caching (graceful fallback if not running) |

### API Endpoints

**Start the server:**
```powershell
cd backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Always returns `{"status": "ok"}` |
| GET | `/predictions/win-probability` | Win probability for a matchup |
| GET | `/predictions/best-player` | Star player prediction for a matchup |
| GET | `/predictions/player-stats` | Predicted pts/reb/ast for one player |

**Example requests:**
```
GET /predictions/win-probability?home_team_id=1610612744&away_team_id=1610612747
GET /predictions/best-player?home_team_id=1610612744&away_team_id=1610612747
GET /predictions/player-stats?player_id=2544
```

OpenAPI docs (auto-generated): `http://localhost:8000/docs`

### Architecture
```
Request → Router (app/routers/predictions.py)
        → Service (app/services/prediction_service.py)
            ├── Redis cache check (optional, graceful fallback)
            ├── DB query via SQLAlchemy (app/models/)
            └── ML inference (ml/predict.py)
                    └── .joblib model files (backend/models/)
```

### Error responses
| Situation | HTTP status |
|---|---|
| Missing query param | 422 (FastAPI auto-validates) |
| Team/player not in DB | 422 |
| Model .joblib not found (not yet trained) | 503 |

### Redis caching
Set `REDIS_URL=redis://localhost:6379` in `backend/.env` to enable.
If not set (or Redis is unreachable), predictions are computed fresh every request.
Only win probability is cached. TTL = 1 hour. Cache key: `win_prob:{home_id}:{away_id}`.

### Test suite
87 tests, 0 failures.

| File | Tests |
|---|---|
| `test_models.py` | 3 |
| `test_data_collector.py` | 12 |
| `test_seed_database.py` | 6 |
| `test_feature_engineering.py` | 20 |
| `test_train_win_model.py` | 4 |
| `test_train_player_model.py` | 6 |
| `test_predict.py` | 7 |
| `test_health.py` | 2 |
| `test_prediction_service.py` | 8 |
| `test_predictions_api.py` | 12 |
| `test_redis_cache.py` | 7 |

---

## What Has Been Built (Phase 4)

### Git log additions
```
62b461b  feat: add home page with matchup selector and player stats predictor
b295d20  feat: add PlayerStatsCard with pts/reb/ast predicted stat display
6acd6df  feat: add BestPlayerCard with ranked star-probability list
37a0c83  feat: add WinProbabilityCard with Recharts horizontal bar and confidence badge
8dafa4f  feat: add React Query provider and win-probability, best-player, player-stats hooks
52504a4  feat: add shared utilities, shadcn Card/Badge, and NBA teams data
695811a  feat: add TypeScript types and API fetch functions with tests
4c27d78  chore: bootstrap Next.js 14 frontend with Tailwind, React Query, Jest
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

### Test suite (frontend)
13 tests, 0 failures.

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

### Key file paths (Phase 4)
| What | Path |
|---|---|
| Frontend root | `frontend/` |
| API fetch functions | `frontend/src/lib/api.ts` |
| TypeScript types | `frontend/src/lib/types.ts` |
| NBA team IDs | `frontend/src/lib/teams.ts` |
| Backend API URL | `frontend/.env.local` → `NEXT_PUBLIC_API_URL` |
| Phase 4 plan | `docs/superpowers/plans/2026-06-17-phase4-nextjs-frontend.md` |

---

## Important Decisions Made

| Decision | Reason |
|---|---|
| Upgraded to Python 3.13-compatible package versions | System Python is 3.13.5; original pinned versions had no pre-built wheels |
| `pytest.ini` with `pythonpath = .` added | Without it, `app`, `ml`, `scripts` are not on the Python path when running pytest from `backend/` |
| Tests use SQLite in-memory | No Docker needed to run the test suite; all 21 tests pass without any running services |
| Seeder is row-by-row with existence check | Simple and backend-agnostic (works with SQLite for tests, PostgreSQL for prod) |
| `format="mixed"` for date parsing | nba_api can return dates as ISO ("2022-10-18") or text ("OCT 18, 2022") depending on endpoint |
| `home_away` derived from `MATCHUP` string | "vs." = home game, "@" = away game — this is the NBA stats API convention |

---

## Key File Paths

| What | Path |
|---|---|
| Virtual environment Python | `backend/venv/Scripts/python.exe` |
| Run tests | `cd backend && .\venv\Scripts\activate && pytest tests/ -v` |
| Run seeder | `cd backend && .\venv\Scripts\activate && python scripts/seed_database.py` |
| DB connection string | `backend/.env` → `DATABASE_URL` |
| Docker Compose | `docker-compose.yml` (project root) |
| Phase 1 detailed plan | `docs/superpowers/plans/2026-06-16-phase1-data-pipeline.md` |
| Phase 2 detailed plan | `docs/superpowers/plans/2026-06-16-phase2-feature-engineering.md` |
| Phase 3 detailed plan | `docs/superpowers/plans/2026-06-17-phase3-fastapi.md` |
| Phase 4 detailed plan | `docs/superpowers/plans/2026-06-17-phase4-nextjs-frontend.md` |

---

## What Has Been Built (Phase 5)

### Git log additions
```
633def0  chore: add Railway/Vercel deployment config and clean up stale TS config files
```

### New files added
| File | Purpose |
|---|---|
| `backend/Procfile` | Railway start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| `backend/.python-version` | Pins Python 3.13 for Railway nixpacks auto-detection |
| `frontend/.env.example` | Template showing `NEXT_PUBLIC_API_URL` env var for Vercel |

### Deploying the Backend (Railway)

Railway auto-detects Python via `requirements.txt` and reads `Procfile` for the start command.

**Steps:**
1. Go to [railway.app](https://railway.app) → log in with GitHub
2. Click **New Project** → **Deploy from GitHub repo** → select `Galenmaxi/AI-NBA-Prediction`
3. In project settings → **Root Directory** → set to `backend`
4. Railway will auto-detect Python and install `requirements.txt`
5. Add a **PostgreSQL** plugin: click **+ New** → **Database** → **Add PostgreSQL**
6. Railway auto-sets `DATABASE_URL` from the plugin — no manual config needed
7. Optional: add `REDIS_URL` from a Redis plugin if you want caching
8. Click **Deploy** — first build takes ~10 minutes (heavy ML packages)
9. After deploy, copy your Railway URL: `https://your-app.up.railway.app`
10. Verify: visit `https://your-app.up.railway.app/health` → should return `{"status":"ok"}`

> **Note:** Prediction endpoints return HTTP 503 until the database is seeded and ML models are trained. The health endpoint always works. See the seeding section in Phase 1 for how to run the seeder against the Railway DATABASE_URL once it's deployed.

### Deploying the Frontend (Vercel)

**Steps:**
1. Go to [vercel.com](https://vercel.com) → log in with GitHub
2. Click **Add New** → **Project** → import `Galenmaxi/AI-NBA-Prediction`
3. In project settings → **Root Directory** → set to `frontend`
4. Vercel auto-detects Next.js — leave Build Command and Output Directory as defaults
5. Under **Environment Variables** → add:
   - Key: `NEXT_PUBLIC_API_URL`
   - Value: `https://your-app.up.railway.app` (your Railway URL from above)
6. Click **Deploy** — build takes ~2 minutes
7. Vercel gives you a URL like `https://ai-nba-prediction.vercel.app`
8. Open it and click **Predict** on a matchup — the health check passes, predictions return 503 until DB is seeded

### Seeding the Railway Database (after backend deploy)

Once Railway is deployed, seed the database from your local machine:

```powershell
# Set DATABASE_URL to your Railway Postgres connection string
# (copy it from Railway project → PostgreSQL plugin → Connect tab)
$env:DATABASE_URL = "postgresql://postgres:xxx@monorail.proxy.rlwy.net:PORT/railway"

cd backend
.\venv\Scripts\activate
python scripts/seed_database.py
```

This takes ~10-15 minutes due to NBA API rate limiting. After seeding, the `/predictions/*` endpoints will work (still return 503 until models are trained).

### Training ML Models (after DB is seeded)

```powershell
cd backend
.\venv\Scripts\activate
python ml/train_win_model.py
python ml/train_player_model.py
```

Trained `.joblib` files are saved to `backend/models/` (gitignored — upload to Railway manually or add a training step to the deploy pipeline).
