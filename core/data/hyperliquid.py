"""
Hyperliquid data downloader.

Downloads historical klines data from Hyperliquid API.
Stores as monthly or yearly Parquet files.

Supports perpetual futures like PAXG-USD.
"""

import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import requests

from core.data.storage import save_monthly, get_data_path, list_available_periods
from core.data.validator import validate_ohlcv, fill_gaps
from core.data.market_hours import add_market_open_always


# Hyperliquid API endpoint
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"

# Interval mapping (Hyperliquid uses different format)
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Max candles per request
MAX_CANDLES = 5000


def download_hyperliquid_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> Optional[pd.DataFrame]:
    """
    Download a single month of OHLCV data from Hyperliquid.
    
    Args:
        symbol: Hyperliquid symbol (e.g., 'PAXG' for PAXG-USD perp)
        interval: Bar interval (e.g., '15m', '1h')
        year: Year
        month: Month (1-12)
        
    Returns:
        DataFrame with OHLCV data + market_open column, or None if not available
    """
    # Calculate date range for the month
    start_date = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    
    # Convert to milliseconds
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    # Get interval in Hyperliquid format
    hl_interval = INTERVAL_MAP.get(interval, interval)
    
    # Fetch data (may need pagination)
    all_candles = []
    current_start = start_ms
    
    while current_start < end_ms:
        candles = _fetch_hyperliquid_candles(
            symbol, hl_interval, current_start, end_ms
        )
        
        if not candles:
            break
        
        all_candles.extend(candles)
        
        # Get the last candle's timestamp for pagination
        last_ts = candles[-1]["t"]
        if last_ts <= current_start:
            break  # No progress, stop
        
        current_start = last_ts + 1  # Next batch starts after last candle
        
        if len(candles) < MAX_CANDLES:
            break  # Got all available data
        
        time_module.sleep(0.2)  # Rate limiting
    
    if not all_candles:
        return None
    
    # Convert to DataFrame
    df = _candles_to_dataframe(all_candles)
    
    # Filter to requested month
    df = df[(df.index >= start_date) & (df.index < end_date)]
    
    if df.empty:
        return None
    
    # Add market_open column (DeFi is 24/7, always open)
    df = add_market_open_always(df, "market_open")
    
    return df


def _fetch_hyperliquid_candles(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Fetch candles from Hyperliquid API."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        }
    }
    
    try:
        response = requests.post(
            HYPERLIQUID_API,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        # API returns list of candles directly
        if isinstance(data, list):
            return data
        return []
        
    except Exception as e:
        print(f"Error fetching {symbol} candles: {e}")
        return []


def _candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Convert Hyperliquid candle data to DataFrame."""
    records = []
    for c in candles:
        records.append({
            "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"]),
        })
    
    df = pd.DataFrame(records)
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    return df


def download_hyperliquid_range(
    symbol: str,
    interval: str,
    start_period: str,
    end_period: str,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[Path]:
    """
    Download a range of monthly data from Hyperliquid.
    
    Args:
        symbol: Hyperliquid symbol (e.g., 'PAXG')
        interval: Bar interval (e.g., '15m')
        start_period: Start month as "YYYY-MM" (e.g., "2025-10")
        end_period: End month as "YYYY-MM" (e.g., "2025-12")
        force: Re-download even if file exists
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        List of paths to saved parquet files
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
    existing = list_available_periods("hyperliquid", "perp", symbol, interval)
    
    paths = []
    total = len(months_to_download)
    
    for i, (year, month) in enumerate(months_to_download):
        period = f"{year}-{month:02d}"
        
        if progress_callback:
            progress_callback(i + 1, total, f"Downloading {symbol} {period}...")
        
        # Skip if exists and not forcing
        if period in existing and not force:
            print(f"  {period} already exists, skipping")
            path = get_data_path("hyperliquid", "perp", symbol, interval, period)
            paths.append(path)
            continue
        
        print(f"  Downloading {symbol} {period}...")
        df = download_hyperliquid_month(symbol, interval, year, month)
        
        if df is not None and len(df) > 0:
            # Validate
            report = validate_ohlcv(df, interval)
            print(f"    {len(df):,} bars, {report.coverage_pct:.1f}% coverage")
            
            # Fill gaps if needed (for 24/7 market, gaps are unusual)
            if report.gap_count > 0:
                df = fill_gaps(df, interval)
                print(f"    Filled {report.gap_count} gaps")
            
            # Save
            path = save_monthly(df, "hyperliquid", "perp", symbol, interval, year, month)
            paths.append(path)
            print(f"    Saved: {path}")
        else:
            print(f"    No data available for {period}")
        
        time_module.sleep(0.3)  # Rate limiting
    
    if progress_callback:
        progress_callback(total, total, "Complete")
    
    return paths


def download_paxg_perp(
    interval: str = "15m",
    start_period: str = "2025-10",
    end_period: str = "2025-12",
    force: bool = False,
) -> list[Path]:
    """
    Convenience function to download Hyperliquid PAXG perpetual data.
    """
    print(f"Downloading Hyperliquid PAXG Perp @ {interval}")
    print(f"Period: {start_period} to {end_period}")
    print()
    
    return download_hyperliquid_range(
        symbol="PAXG",
        interval=interval,
        start_period=start_period,
        end_period=end_period,
        force=force,
    )


def list_hyperliquid_symbols() -> list[str]:
    """Fetch list of available perpetual symbols from Hyperliquid."""
    try:
        response = requests.post(
            HYPERLIQUID_API,
            json={"type": "meta"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        if "universe" in data:
            return [asset["name"] for asset in data["universe"]]
        return []
        
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []
