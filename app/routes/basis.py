"""
Basis Builder Routes

Web UI for creating and managing basis files.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json

from core.data.storage import list_all_data
from core.data.basis import (
    BasisSpec, 
    create_basis_file, 
    load_basis, 
    list_basis_files,
    find_overlapping_periods,
)

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=HTMLResponse)
async def basis_builder(request: Request):
    """Basis builder page."""
    data_tree = list_all_data()
    basis_files = list_basis_files()
    
    return templates.TemplateResponse("basis/builder.html", {
        "request": request,
        "data_tree": data_tree,
        "basis_files": basis_files,
    })


@router.get("/check-overlap")
async def check_overlap(
    base_venue: str,
    base_market: str,
    base_ticker: str,
    quote_venue: str,
    quote_market: str,
    quote_ticker: str,
    interval: str,
):
    """Check for overlapping periods between base and quote data."""
    overlap = find_overlapping_periods(
        base_venue, base_market, base_ticker,
        quote_venue, quote_market, quote_ticker,
        interval,
    )
    return overlap


@router.post("/create")
async def create_basis(request: Request):
    """Create a new basis file."""
    data = await request.json()
    
    spec = BasisSpec(
        base_venue=data["base_venue"],
        base_market=data["base_market"],
        base_ticker=data["base_ticker"],
        quote_venues=[{
            "venue": data["quote_venue"],
            "market": data["quote_market"],
            "ticker": data["quote_ticker"],
            "name": data.get("quote_name", data["quote_ticker"]),
        }],
        interval=data["interval"],
        periods=data["periods"],
        price_column=data.get("price_column", "close"),
    )
    
    try:
        df, result = create_basis_file(spec, save=True)
        return {
            "success": True,
            "path": str(result.path),
            "bars": result.bars,
            "coverage_pct": result.coverage_pct,
            "start": result.start,
            "end": result.end,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/list")
async def list_basis():
    """List all available basis files."""
    return list_basis_files()


@router.get("/preview/{ticker}/{interval}", response_class=HTMLResponse)
async def preview_basis(request: Request, ticker: str, interval: str, period: Optional[str] = None):
    """Preview a basis file."""
    periods = [period] if period else None
    df = load_basis(ticker, interval, periods)
    
    if df is None:
        return templates.TemplateResponse("basis/partials/no_data.html", {
            "request": request,
            "ticker": ticker,
            "interval": interval,
        })
    
    # Prepare chart data
    chart_data = {
        "timestamps": [t.isoformat() for t in df.index],
        "base_price": df["base_price"].fillna(0).tolist(),
    }
    
    # Find quote venues from columns
    quote_venues = []
    for col in df.columns:
        if col.endswith("_basis_bps"):
            venue = col.replace("_basis_bps", "")
            quote_venues.append(venue)
            chart_data[f"{venue}_price"] = df[f"{venue}_price"].fillna(0).tolist()
            chart_data[f"{venue}_basis_bps"] = df[f"{venue}_basis_bps"].fillna(0).tolist()
    
    # Stats
    stats = {
        "bars": len(df),
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "quality_ok": (df["data_quality"] == "ok").sum(),
        "quality_pct": (df["data_quality"] == "ok").mean() * 100,
    }
    
    # Basis stats for each quote venue
    for venue in quote_venues:
        col = f"{venue}_basis_bps"
        stats[f"{venue}_mean_bps"] = df[col].mean()
        stats[f"{venue}_std_bps"] = df[col].std()
        stats[f"{venue}_min_bps"] = df[col].min()
        stats[f"{venue}_max_bps"] = df[col].max()
    
    return templates.TemplateResponse("basis/partials/preview.html", {
        "request": request,
        "ticker": ticker,
        "interval": interval,
        "quote_venues": quote_venues,
        "stats": stats,
        "chart_data": json.dumps(chart_data),
    })
