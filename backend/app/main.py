from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health, predictions

app = FastAPI(title="NBA AI Predictor", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)
