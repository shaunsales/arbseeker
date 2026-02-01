"""
Data management routes.

Handles browsing, downloading, and previewing market data.
"""

import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.data.storage import list_all_data, load_ohlcv, list_available_years, delete_year, clear_all_data
from core.data.binance import download_binance_year, list_binance_symbols, INTERVALS
from core.data.validator import validate_ohlcv, get_data_summary

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Track download jobs
download_jobs: dict[str, dict] = {}


class DownloadRequest(BaseModel):
    """Request to download data."""
    venue: str = "binance"
    market: str = "futures"
    ticker: str
    interval: str
    year: int


@router.get("/", response_class=HTMLResponse)
async def data_browser(request: Request):
    """Data browser page."""
    data_tree = list_all_data()
    return templates.TemplateResponse("data/browser.html", {
        "request": request,
        "data_tree": data_tree,
        "intervals": INTERVALS,
    })


@router.get("/tree", response_class=HTMLResponse)
async def data_tree(request: Request):
    """Get data tree as HTML fragment (for HTMX)."""
    data_tree = list_all_data()
    return templates.TemplateResponse("data/partials/tree.html", {
        "request": request,
        "data_tree": data_tree,
    })


@router.get("/preview/{venue}/{market}/{ticker}/{interval}", response_class=HTMLResponse)
async def data_preview(
    request: Request,
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    years: Optional[str] = None,
):
    """Preview data with chart and stats."""
    try:
        available_years = list_available_years(venue, market, ticker, interval)
        
        if not available_years:
            return templates.TemplateResponse("data/partials/no_data.html", {
                "request": request,
                "ticker": ticker,
                "interval": interval,
            })
        
        # Parse years parameter or use all
        if years:
            selected_years = [int(y) for y in years.split(",")]
        else:
            selected_years = available_years
        
        # Load data
        df = load_ohlcv(venue, market, ticker, interval, years=selected_years)
        
        # Get stats
        summary = get_data_summary(df)
        report = validate_ohlcv(df, interval)
        
        # Generate chart data (sampled for performance)
        chart_data = _prepare_chart_data(df, max_points=2000)
        
        return templates.TemplateResponse("data/partials/preview.html", {
            "request": request,
            "venue": venue,
            "market": market,
            "ticker": ticker,
            "interval": interval,
            "available_years": available_years,
            "selected_years": selected_years,
            "summary": summary,
            "report": report,
            "chart_data": json.dumps(chart_data),
        })
    except Exception as e:
        return templates.TemplateResponse("data/partials/error.html", {
            "request": request,
            "error": str(e),
        })


@router.post("/download")
async def download_data(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start a data download job."""
    job_id = f"{request.ticker}_{request.interval}_{request.year}"
    
    # Check if already downloading
    if job_id in download_jobs and download_jobs[job_id]["status"] == "running":
        return {"job_id": job_id, "status": "already_running"}
    
    # Initialize job
    download_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting download...",
    }
    
    # Start background download
    background_tasks.add_task(
        _download_task,
        job_id,
        request.ticker,
        request.interval,
        request.year,
        request.market,
    )
    
    return {"job_id": job_id, "status": "started"}


@router.get("/download/status/{job_id}")
async def download_status(job_id: str):
    """Get download job status."""
    if job_id not in download_jobs:
        return {"status": "not_found"}
    return download_jobs[job_id]


@router.delete("/{venue}/{market}/{ticker}/{interval}/{year}")
async def delete_data(venue: str, market: str, ticker: str, interval: str, year: int):
    """Delete a specific year's data."""
    deleted = delete_year(venue, market, ticker, interval, year)
    return {"deleted": deleted}


@router.get("/symbols/{market}")
async def get_symbols(market: str = "futures"):
    """Get available trading symbols."""
    symbols = list_binance_symbols(market)
    return {"symbols": symbols}


async def _download_task(job_id: str, ticker: str, interval: str, year: int, market: str):
    """Background task for downloading data."""
    def progress_callback(month: int, total: int, message: str):
        download_jobs[job_id] = {
            "status": "running",
            "progress": int((month / total) * 100),
            "message": message,
        }
    
    try:
        path = download_binance_year(
            symbol=ticker,
            interval=interval,
            year=year,
            market=market,
            progress_callback=progress_callback,
        )
        
        download_jobs[job_id] = {
            "status": "complete",
            "progress": 100,
            "message": f"Saved to {path}" if path else "No data available",
            "path": str(path) if path else None,
        }
    except Exception as e:
        download_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e),
        }


def _prepare_chart_data(df, max_points: int = 2000) -> dict:
    """Prepare OHLCV data for Plotly chart."""
    # Sample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]
    
    return {
        "timestamps": [t.isoformat() for t in df.index],
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["volume"].tolist(),
    }
