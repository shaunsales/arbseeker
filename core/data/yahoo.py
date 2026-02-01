"""
Yahoo Finance data downloader.

Downloads historical klines data from Yahoo Finance.
Stores as monthly or yearly Parquet files.

Supports gold futures (GC=F) and other Yahoo symbols.
"""

import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import yfinance as yf

from core.data.storage import save_monthly, save_ohlcv, get_data_path, list_available_periods
from core.data.validator import validate_ohlcv, fill_gaps
from core.data.market_hours import add_market_open_from_volume, detect_near_close, get_interval_minutes


# Yahoo Finance limitations by interval
INTERVAL_LIMITS = {
    "1m": 7,      # 7 days max
    "2m": 60,     # 60 days
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,   # ~2 years
    "1h": 730,
    "1d": 10000,  # Effectively unlimited
}


def download_yahoo_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> Optional[pd.DataFrame]:
    """
    Download a single month of OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Yahoo symbol (e.g., 'GC=F' for gold futures)
        interval: Bar interval (e.g., '15m', '1h')
        year: Year
        month: Month (1-12)
        
    Returns:
        DataFrame with OHLCV data + market_open column, or None if not available
        
    Note:
        market_open is determined by volume > 0 (reliable for TradFi)
    """
    # Calculate date range for the month
    start_date = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    
    # Check if we need to paginate (for short intervals)
    max_days = INTERVAL_LIMITS.get(interval, 60)
    total_days = (end_date - start_date).days
    
    if total_days <= max_days:
        # Single request
        df = _fetch_yahoo_chunk(symbol, start_date, end_date, interval)
    else:
        # Paginate
        dfs = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_days), end_date)
            chunk_df = _fetch_yahoo_chunk(symbol, current_start, current_end, interval)
            if chunk_df is not None and len(chunk_df) > 0:
                dfs.append(chunk_df)
            current_start = current_end
            time_module.sleep(0.5)  # Rate limiting
        
        if not dfs:
            return None
        df = pd.concat(dfs, axis=0)
    
    if df is None or df.empty:
        return None
    
    # Clean up
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    # Add market_open based on volume (volume > 0 means market was open)
    df = add_market_open_from_volume(df, "volume", "market_open")
    
    # Add near_close flag (30 min before market closes)
    interval_mins = get_interval_minutes(interval)
    df = detect_near_close(df, "market_open", buffer_minutes=30, interval_minutes=interval_mins)
    
    return df


def _fetch_yahoo_chunk(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Fetch a single chunk of data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
        )
        
        if df.empty:
            return None
        
        # Normalize timezone to UTC
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")
        else:
            df.index = df.index.tz_localize("UTC")
        
        # Standardize column names
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # Keep only OHLCV columns
        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        
        return df
        
    except Exception as e:
        print(f"Error fetching {symbol} {start_date.date()}-{end_date.date()}: {e}")
        return None


def download_yahoo_range(
    symbol: str,
    interval: str,
    start_period: str,
    end_period: str,
    market: str = "futures",
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[Path]:
    """
    Download a range of monthly data from Yahoo Finance.
    
    Args:
        symbol: Yahoo symbol (e.g., 'GC=F')
        interval: Bar interval (e.g., '15m')
        start_period: Start month as "YYYY-MM" (e.g., "2025-10")
        end_period: End month as "YYYY-MM" (e.g., "2025-12")
        market: Market type for storage path (e.g., 'futures')
        force: Re-download even if file exists
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        List of paths to saved parquet files
        
    Note:
        market_open column is determined by volume > 0 (reliable for TradFi)
    """
    # Parse periods
    start_year, start_month = map(int, start_period.split("-"))
    end_year, end_month = map(int, end_period.split("-"))
    
    # Generate list of months to download
    months_to_download = []
    current_year, current_month = start_year, start_month
    while (current_year, current_month) <= (end_year, end_month):
        months_to_download.append((current_year, current_month))
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    
    # Check existing data
    existing = list_available_periods("yahoo", market, symbol, interval)
    
    paths = []
    total = len(months_to_download)
    
    for i, (year, month) in enumerate(months_to_download):
        period = f"{year}-{month:02d}"
        
        if progress_callback:
            progress_callback(i + 1, total, f"Downloading {symbol} {period}...")
        
        # Skip if exists and not forcing
        if period in existing and not force:
            print(f"  {period} already exists, skipping")
            path = get_data_path("yahoo", market, symbol, interval, period)
            paths.append(path)
            continue
        
        print(f"  Downloading {symbol} {period}...")
        df = download_yahoo_month(symbol, interval, year, month)
        
        if df is not None and len(df) > 0:
            # Validate
            report = validate_ohlcv(df, interval)
            print(f"    {len(df):,} bars, {report.coverage_pct:.1f}% coverage")
            
            # Fill gaps if needed
            if report.gap_count > 0:
                df = fill_gaps(df, interval)
                print(f"    Filled {report.gap_count} gaps")
            
            # Save
            path = save_monthly(df, "yahoo", market, symbol, interval, year, month)
            paths.append(path)
            print(f"    Saved: {path}")
        else:
            print(f"    No data available for {period}")
        
        time_module.sleep(0.5)  # Rate limiting
    
    if progress_callback:
        progress_callback(total, total, "Complete")
    
    return paths


def download_gold_futures(
    interval: str = "15m",
    start_period: str = "2025-10",
    end_period: str = "2025-12",
    force: bool = False,
) -> list[Path]:
    """
    Convenience function to download CME Gold futures data.
    
    Uses GC=F (continuous front-month contract).
    """
    print(f"Downloading CME Gold Futures (GC=F) @ {interval}")
    print(f"Period: {start_period} to {end_period}")
    print()
    
    return download_yahoo_range(
        symbol="GC=F",
        interval=interval,
        start_period=start_period,
        end_period=end_period,
        market="futures",
        force=force,
    )
