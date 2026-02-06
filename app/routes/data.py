"""
Data management routes.

Handles browsing, downloading, and previewing market data.
"""

import asyncio
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import io

from core.data.storage import list_all_data, load_ohlcv, list_available_years, list_available_periods, delete_period, clear_all_data, get_data_path
from core.data.binance import download_binance_year, list_binance_symbols, INTERVALS
from core.data.hyperliquid import download_hyperliquid_range
from core.data.hyperliquid_s3 import download_s3_range, validate_aws_credentials
from core.data.validator import validate_ohlcv, get_data_summary

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Track download jobs
download_jobs: dict[str, dict] = {}
# Track log lines per job (for SSE streaming)
download_logs: dict[str, list[str]] = {}
# Track cancel events per job
download_cancel_events: dict[str, threading.Event] = {}


class DownloadRequest(BaseModel):
    """Request to download data."""
    venue: str = "binance"
    market: str = "futures"
    ticker: str
    interval: str
    year: Optional[int] = None  # For Binance
    start_month: Optional[str] = None  # For Hyperliquid (YYYY-MM)
    end_month: Optional[str] = None  # For Hyperliquid (YYYY-MM)
    source: Optional[str] = None  # "api" or "s3" for Hyperliquid
    start_date: Optional[str] = None  # For S3 (YYYY-MM-DD)
    end_date: Optional[str] = None  # For S3 (YYYY-MM-DD)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None


class AWSValidateRequest(BaseModel):
    """Request to validate AWS credentials."""
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: Optional[str] = None


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
    periods: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
):
    """Preview data with chart, stats, pagination, and date filters."""
    try:
        available_periods = list_available_periods(venue, market, ticker, interval)
        
        if not available_periods:
            return templates.TemplateResponse("data/partials/no_data.html", {
                "request": request,
                "ticker": ticker,
                "interval": interval,
            })
        
        # Parse periods parameter or use all
        if periods:
            selected_periods = periods.split(",")
        else:
            selected_periods = available_periods
        
        # Load data
        df = load_ohlcv(venue, market, ticker, interval, periods=selected_periods)
        
        # Apply date filters if provided
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize("UTC")
            df = df[df.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize("UTC")
            df = df[df.index <= end_dt]
        
        # Get stats (on filtered data)
        summary = get_data_summary(df)
        report = validate_ohlcv(df, interval)
        
        # Pagination for table
        total_rows = len(df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        table_df = df.iloc[start_idx:end_idx]
        
        # Prepare table data
        table_data = _prepare_table_data(table_df)
        
        # Generate chart data (all points - Plotly handles it well)
        chart_data = _prepare_chart_data(df, max_points=50000)
        
        # Date range for filters
        date_range = {
            "min": df.index.min().strftime("%Y-%m-%d") if len(df) > 0 else "",
            "max": df.index.max().strftime("%Y-%m-%d") if len(df) > 0 else "",
            "start": start_date or "",
            "end": end_date or "",
        }
        
        return templates.TemplateResponse("data/partials/preview.html", {
            "request": request,
            "venue": venue,
            "market": market,
            "ticker": ticker,
            "interval": interval,
            "available_periods": available_periods,
            "selected_periods": selected_periods,
            "summary": summary,
            "report": report,
            "chart_data": json.dumps(chart_data),
            "table_data": table_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "total_pages": total_pages,
            },
            "date_range": date_range,
        })
    except Exception as e:
        return templates.TemplateResponse("data/partials/error.html", {
            "request": request,
            "error": str(e),
        })


@router.post("/download")
async def download_data(request: DownloadRequest, background_tasks: BackgroundTasks):
    """Start a data download job."""
    # Generate job ID based on venue
    if request.venue == "binance":
        job_id = f"binance_{request.ticker}_{request.interval}_{request.year}"
    elif request.venue == "hyperliquid" and request.source == "s3":
        job_id = f"hl_s3_{request.ticker}_{request.interval}_{request.start_date}_{request.end_date}"
    else:
        job_id = f"hyperliquid_{request.ticker}_{request.interval}_{request.start_month}_{request.end_month}"
    
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
    if request.venue == "binance":
        background_tasks.add_task(
            _download_binance_task,
            job_id,
            request.ticker,
            request.interval,
            request.year,
            request.market,
        )
    elif request.venue == "hyperliquid" and request.source == "s3":
        download_logs[job_id] = []
        background_tasks.add_task(
            _download_hyperliquid_s3_task,
            job_id,
            request.ticker,
            request.interval,
            request.start_date,
            request.end_date,
            request.aws_access_key_id,
            request.aws_secret_access_key,
            request.aws_session_token,
        )
    else:
        background_tasks.add_task(
            _download_hyperliquid_task,
            job_id,
            request.ticker,
            request.interval,
            request.start_month,
            request.end_month,
        )
    
    return {"job_id": job_id, "status": "started"}


@router.get("/download/status/{job_id}")
async def download_status(job_id: str):
    """Get download job status."""
    if job_id not in download_jobs:
        return {"status": "not_found"}
    return download_jobs[job_id]


@router.post("/download/cancel/{job_id}")
async def cancel_download(job_id: str):
    """Cancel a running download job."""
    if job_id not in download_jobs:
        return {"cancelled": False, "message": "Job not found"}
    if download_jobs[job_id].get("status") != "running":
        return {"cancelled": False, "message": "Job is not running"}
    if job_id in download_cancel_events:
        download_cancel_events[job_id].set()
        download_jobs[job_id]["status"] = "cancelled"
        download_jobs[job_id]["message"] = "Cancelling..."
        return {"cancelled": True, "message": "Cancel requested"}
    return {"cancelled": False, "message": "Job does not support cancellation"}


@router.delete("/{venue}/{market}/{ticker}/{interval}/{period}")
async def delete_data(venue: str, market: str, ticker: str, interval: str, period: str):
    """Delete a specific period's data."""
    deleted = delete_period(venue, market, ticker, interval, period)
    return {"deleted": deleted}


@router.get("/symbols/{market}")
async def get_symbols(market: str = "futures"):
    """Get available trading symbols."""
    symbols = list_binance_symbols(market)
    return {"symbols": symbols}


async def _download_binance_task(job_id: str, ticker: str, interval: str, year: int, market: str):
    """Background task for downloading Binance data."""
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


async def _download_hyperliquid_task(job_id: str, ticker: str, interval: str, start_month: str, end_month: str):
    """Background task for downloading Hyperliquid data."""
    def progress_callback(current: int, total: int, message: str):
        download_jobs[job_id] = {
            "status": "running",
            "progress": int((current / total) * 100),
            "message": message,
        }
    
    try:
        paths = download_hyperliquid_range(
            symbol=ticker,
            interval=interval,
            start_period=start_month,
            end_period=end_month,
            progress_callback=progress_callback,
        )
        
        download_jobs[job_id] = {
            "status": "complete",
            "progress": 100,
            "message": f"Downloaded {len(paths)} months" if paths else "No data available",
            "paths": [str(p) for p in paths] if paths else None,
        }
    except Exception as e:
        download_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e),
        }


@router.post("/aws/validate")
async def validate_aws(request: AWSValidateRequest):
    """Validate AWS credentials against the Hyperliquid S3 bucket."""
    loop = asyncio.get_event_loop()
    success, message = await loop.run_in_executor(
        None,
        lambda: validate_aws_credentials(
            aws_access_key_id=request.aws_access_key_id,
            aws_secret_access_key=request.aws_secret_access_key,
            aws_session_token=request.aws_session_token or None,
        ),
    )
    return {"valid": success, "message": message}


@router.get("/download/logs/{job_id}")
async def download_logs_stream(job_id: str):
    """SSE endpoint that streams log lines for an S3 download job."""
    async def event_generator():
        sent = 0
        while True:
            logs = download_logs.get(job_id, [])
            # Send any new lines
            while sent < len(logs):
                line = logs[sent]
                yield f"data: {json.dumps({'line': line})}\n\n"
                sent += 1

            # Check if job is done
            job = download_jobs.get(job_id, {})
            status = job.get("status", "")
            if status in ("complete", "error", "cancelled"):
                # Send final status event
                yield f"event: done\ndata: {json.dumps({'status': status, 'message': job.get('message', '')})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _download_hyperliquid_s3_task(
    job_id: str,
    ticker: str,
    interval: str,
    start_date_str: str,
    end_date_str: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
):
    """Background task for downloading Hyperliquid data from S3."""
    cancel_event = threading.Event()
    download_cancel_events[job_id] = cancel_event

    def log_callback(msg: str):
        if job_id in download_logs:
            download_logs[job_id].append(msg)

    def progress_callback(current: int, total: int, message: str):
        download_jobs[job_id] = {
            "status": "running",
            "progress": int((current / total) * 100),
            "message": message,
        }

    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        loop = asyncio.get_event_loop()
        paths = await loop.run_in_executor(
            None,
            lambda: download_s3_range(
                symbol=ticker,
                start_date=start_dt,
                end_date=end_dt,
                interval=interval,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                log_callback=log_callback,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            ),
        )

        if cancel_event.is_set():
            saved = len(paths) if paths else 0
            download_jobs[job_id] = {
                "status": "cancelled",
                "progress": download_jobs.get(job_id, {}).get("progress", 0),
                "message": f"Cancelled. Saved {saved} monthly file(s) from partial download.",
                "paths": [str(p) for p in paths] if paths else None,
            }
        else:
            download_jobs[job_id] = {
                "status": "complete",
                "progress": 100,
                "message": f"Downloaded {len(paths)} monthly file(s)" if paths else "No data available",
                "paths": [str(p) for p in paths] if paths else None,
            }
    except Exception as e:
        log_callback(f"\nERROR: {e}")
        download_jobs[job_id] = {
            "status": "error",
            "progress": 0,
            "message": str(e),
        }
    finally:
        download_cancel_events.pop(job_id, None)


@router.get("/export/{venue}/{market}/{ticker}/{interval}")
async def export_data(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    format: str = "csv",
    periods: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Export data as CSV or Parquet."""
    try:
        available_periods = list_available_periods(venue, market, ticker, interval)
        
        if not available_periods:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Parse periods or use all
        if periods:
            selected_periods = periods.split(",")
        else:
            selected_periods = available_periods
        
        # Load data
        df = load_ohlcv(venue, market, ticker, interval, periods=selected_periods)
        
        # Apply date filters
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize("UTC")
            df = df[df.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize("UTC")
            df = df[df.index <= end_dt]
        
        filename = f"{ticker}_{interval}_{datetime.now().strftime('%Y%m%d')}"
        
        if format == "parquet":
            buffer = io.BytesIO()
            df.to_parquet(buffer, engine="pyarrow")
            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}.parquet"}
            )
        else:  # CSV
            buffer = io.StringIO()
            df.to_csv(buffer)
            buffer.seek(0)
            return StreamingResponse(
                iter([buffer.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _prepare_chart_data(df, max_points: int = 2000) -> dict:
    """Prepare OHLCV data for Plotly chart."""
    if len(df) == 0:
        return {"timestamps": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
    
    # Sample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]
    
    # Fill NaN values to avoid JSON issues
    df = df.fillna(0)
    
    return {
        "timestamps": [t.isoformat() for t in df.index],
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["volume"].tolist(),
    }


def _prepare_table_data(df) -> list[dict]:
    """Prepare OHLCV data for table display."""
    if len(df) == 0:
        return []
    
    records = []
    for ts, row in df.iterrows():
        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
            "open": f"{row['open']:.2f}",
            "high": f"{row['high']:.2f}",
            "low": f"{row['low']:.2f}",
            "close": f"{row['close']:.2f}",
            "volume": f"{row['volume']:,.0f}",
        })
    return records
