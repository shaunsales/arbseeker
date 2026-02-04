"""
Strategy Lab - Web Application

FastAPI app for browsing data, running backtests, and visualizing results.

Run with: uvicorn app.main:app --reload
"""

from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.routes import data, backtest, basis

# App setup
app = FastAPI(
    title="Strategy Lab",
    description="Trading strategy backtesting and data management",
    version="0.1.0",
)

# Static files and templates
APP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - redirect to data browser."""
    return templates.TemplateResponse("index.html", {"request": request})


# Include routers
app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(basis.router, prefix="/basis", tags=["basis"])
app.include_router(backtest.router, prefix="/backtest", tags=["backtest"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
