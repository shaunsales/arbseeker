"""
Antlyr - Web Application

FastAPI app for browsing data, running backtests, and visualizing results.

Run with: uvicorn app.main:app --reload
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import data, backtest, basis, strategy, coinroutes

# App setup
app = FastAPI(
    title="Antlyr",
    description="Trading strategy backtesting and data management",
    version="0.1.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(basis.router, prefix="/basis", tags=["basis"])
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])
app.include_router(strategy.router, prefix="/strategy", tags=["strategy"])
app.include_router(coinroutes.router, prefix="/coinroutes", tags=["coinroutes"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
