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
import numpy as np

from core.data.storage import list_all_data
from core.data.basis import (
    BasisSpec, 
    create_basis_file, 
    load_basis, 
    list_basis_files,
    find_overlapping_periods,
)
from core.analysis.basis_stats import compute_basis_stats

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


@router.get("/", response_class=HTMLResponse)
async def basis_builder(request: Request):
    """Basis builder page."""
    data_tree = list_all_data()
    basis_files = list_basis_files()
    
    return templates.TemplateResponse("pages/basis.html", {
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
        return templates.TemplateResponse("partials/basis/no_data.html", {
            "request": request,
            "ticker": ticker,
            "interval": interval,
        })
    
    # Find quote venues from columns
    quote_venues = []
    for col in df.columns:
        if col.endswith("_basis_bps"):
            quote_venues.append(col.replace("_basis_bps", ""))

    # Downsample for charting if too many points
    MAX_CHART_POINTS = 5000
    chart_df = df
    chart_interval = interval
    if len(df) > MAX_CHART_POINTS:
        step = len(df) // MAX_CHART_POINTS + 1
        # Map step size to a readable resample label
        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(interval, 1)
        chart_mins = interval_minutes * step
        if chart_mins >= 1440:
            chart_interval = f"{chart_mins // 1440}d"
        elif chart_mins >= 60:
            chart_interval = f"{chart_mins // 60}h"
        else:
            chart_interval = f"{chart_mins}m"
        chart_df = df.iloc[::step]

    # Build chart data — skip NaN values per series
    timestamps_epoch = [int(t.timestamp()) for t in chart_df.index]

    def _series_data(values):
        """Return list of {time, value} dicts, skipping NaN."""
        return [
            {"time": timestamps_epoch[i], "value": float(v)}
            for i, v in enumerate(values)
            if not (v is None or (isinstance(v, float) and np.isnan(v)))
        ]

    chart_data = {
        "base_price": _series_data(chart_df["base_price"].values),
    }
    for venue in quote_venues:
        chart_data[f"{venue}_price"] = _series_data(chart_df[f"{venue}_price"].values)
        chart_data[f"{venue}_basis_bps"] = _series_data(chart_df[f"{venue}_basis_bps"].values)

    # Build quality timeline chart data
    # Color-coded: green=ok, yellow=ffill, red=stale/gap
    quality_color_map = {"ok": None}  # skip ok bars
    quality_chart = []
    for i, (ts, q) in enumerate(zip(timestamps_epoch, chart_df["data_quality"].values)):
        if q == "ok":
            continue
        color = "#ef4444" if "stale" in str(q) or "gap" in str(q) else "#f59e0b"
        quality_chart.append({"time": ts, "value": 1, "color": color})
    chart_data["quality"] = quality_chart
    
    # Stats
    quality_counts = df["data_quality"].value_counts()
    stats = {
        "bars": len(df),
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "quality_ok": int(quality_counts.get("ok", 0)),
        "quality_pct": float(quality_counts.get("ok", 0)) / len(df) * 100 if len(df) > 0 else 0,
        "quality_breakdown": {str(k): int(v) for k, v in quality_counts.items()},
    }
    
    # Basis stats for each quote venue
    venue_stats = {}
    for venue in quote_venues:
        col = f"{venue}_basis_bps"
        venue_stats[venue] = compute_basis_stats(df[col], interval)
    
    return templates.TemplateResponse("partials/basis/preview.html", {
        "request": request,
        "ticker": ticker,
        "interval": interval,
        "chart_interval": chart_interval,
        "quote_venues": quote_venues,
        "stats": stats,
        "venue_stats": venue_stats,
        "chart_data": json.dumps(chart_data),
    })
