"""
Backtest routes.

Handles strategy configuration, execution, and results.
"""

from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=HTMLResponse)
async def backtest_page(request: Request):
    """Backtest configuration page."""
    return templates.TemplateResponse("pages/backtest.html", {
        "request": request,
    })


@router.get("/strategies")
async def list_strategies():
    """List available strategies."""
    # TODO: Implement strategy discovery
    return {
        "strategies": [
            {"name": "basis_arb", "description": "Basis Arbitrage (CME vs DeFi)"},
            {"name": "trend_following", "description": "Trend Following (coming soon)"},
        ]
    }
