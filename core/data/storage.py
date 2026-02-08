"""
Parquet storage utilities for OHLCV data.

Storage structure:
    data/{venue}/{market}/{ticker}/{interval}/{period}.parquet

Supported period formats:
    - Yearly:  2024.parquet
    - Monthly: 2024-10.parquet

Examples:
    data/binance/futures/BTCUSDT/1h/2024.parquet
    data/hyperliquid/perp/PAXG/15m/2025-10.parquet

Standard OHLCV Schema:
    Required columns:
        - timestamp (DatetimeIndex, UTC) - bar open time, aligned to within 1 second
        - open, high, low, close (float64) - OHLC prices
        - volume (float64) - trading volume in base currency

    Optional columns:
        - market_open (bool) - True if market is open at this bar
        - quote_volume (float64) - volume in quote currency
        - count (int) - number of trades in bar
        - taker_buy_volume (float64) - taker buy volume

    Computed at runtime (NOT stored):
        - near_close - compute from market_open transitions
        - mid - compute as (high + low) / 2 or (bid + ask) / 2
"""

import re
from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Base data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_data_path(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    period: Optional[str] = None,
) -> Path:
    """
    Get the path to a data file or directory.
    
    Args:
        venue: Data source (e.g., 'binance', 'hyperliquid')
        market: Market type (e.g., 'futures', 'spot', 'perp')
        ticker: Trading pair (e.g., 'BTCUSDT', 'GC=F')
        interval: Bar interval (e.g., '1h', '15m')
        period: Optional period for specific file:
                - Year: "2024" or 2024
                - Month: "2024-10"
        
    Returns:
        Path to directory (if period is None) or specific parquet file
    """
    base_path = DATA_DIR / venue / market / ticker / interval
    if period is not None:
        return base_path / f"{period}.parquet"
    return base_path


def save_ohlcv(
    df: pd.DataFrame,
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    period: str,
) -> Path:
    """
    Save OHLCV data for a specific period (year or month).
    
    Args:
        df: DataFrame with OHLCV data (must have datetime index)
        venue: Data source
        market: Market type
        ticker: Trading pair
        interval: Bar interval
        period: Period string - "2024" for year, "2024-10" for month
        
    Returns:
        Path to saved file
    """
    file_path = get_data_path(venue, market, ticker, interval, period)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Sort by time
    df = df.sort_index()
    
    # Save to parquet
    df.to_parquet(file_path, engine="pyarrow", compression="snappy")
    
    return file_path


def save_yearly(
    df: pd.DataFrame,
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    year: int,
) -> Path:
    """Save OHLCV data for a specific year. Wrapper for save_ohlcv."""
    return save_ohlcv(df, venue, market, ticker, interval, str(year))


def save_monthly(
    df: pd.DataFrame,
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    year: int,
    month: int,
) -> Path:
    """Save OHLCV data for a specific month."""
    period = f"{year}-{month:02d}"
    return save_ohlcv(df, venue, market, ticker, interval, period)


def load_ohlcv(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    periods: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for specified periods.
    
    Args:
        venue: Data source
        market: Market type
        ticker: Trading pair
        interval: Bar interval
        periods: List of periods to load (None = all available)
                 Supports: ["2024", "2025"] for years
                          ["2025-10", "2025-11", "2025-12"] for months
                          Mixed: ["2024", "2025-10", "2025-11"]
        
    Returns:
        Concatenated DataFrame with OHLCV data
    """
    base_path = get_data_path(venue, market, ticker, interval)
    
    if not base_path.exists():
        raise FileNotFoundError(f"No data found at {base_path}")
    
    # Get available periods if not specified
    if periods is None:
        periods = list_available_periods(venue, market, ticker, interval)
    
    if not periods:
        raise FileNotFoundError(f"No data files found at {base_path}")
    
    # Load and concatenate
    dfs = []
    for period in _sort_periods(periods):
        file_path = base_path / f"{period}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found, skipping")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found for periods {periods}")
    
    # Concatenate and sort
    result = pd.concat(dfs, axis=0)
    result = result.sort_index()
    
    # Remove duplicates (in case of overlapping data)
    result = result[~result.index.duplicated(keep="first")]
    
    return result


def _sort_periods(periods: list[str]) -> list[str]:
    """Sort periods chronologically (years before months of same year)."""
    def sort_key(p: str):
        if "-" in p:  # Monthly: 2024-10
            year, month = p.split("-")
            return (int(year), int(month))
        else:  # Yearly: 2024
            return (int(p), 0)
    return sorted(periods, key=sort_key)


def _parse_period(filename: str) -> Optional[str]:
    """Parse period from filename. Returns None if not a valid period."""
    stem = Path(filename).stem
    # Match yearly (2024) or monthly (2024-10)
    if re.match(r"^\d{4}$", stem):  # Year only
        return stem
    elif re.match(r"^\d{4}-\d{2}$", stem):  # Year-month
        return stem
    return None


def list_available_periods(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
) -> list[str]:
    """
    List all available periods (years and months) for a given data path.
    
    Returns:
        Sorted list of periods (e.g., ["2024", "2025-10", "2025-11"])
    """
    base_path = get_data_path(venue, market, ticker, interval)
    
    if not base_path.exists():
        return []
    
    periods = []
    for file_path in base_path.glob("*.parquet"):
        period = _parse_period(file_path.name)
        if period:
            periods.append(period)
    
    return _sort_periods(periods)


def list_available_years(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
) -> list[int]:
    """
    List all available full years for a given data path.
    Backward compatible - only returns year files, not monthly.
    
    Returns:
        Sorted list of years with data
    """
    periods = list_available_periods(venue, market, ticker, interval)
    years = [int(p) for p in periods if re.match(r"^\d{4}$", p)]
    return sorted(years)


def list_all_data() -> dict:
    """
    List all available data in the data directory.
    
    Returns:
        Nested dict: {venue: {market: {ticker: {interval: [periods]}}}}
    """
    result = {}
    
    if not DATA_DIR.exists():
        return result
    
    for venue_path in DATA_DIR.iterdir():
        if not venue_path.is_dir() or venue_path.name.startswith("."):
            continue
        # Skip raw source files — not browsable OHLCV data
        if venue_path.name == "sources":
            continue
        venue = venue_path.name
        result[venue] = {}
        
        for market_path in venue_path.iterdir():
            if not market_path.is_dir():
                continue
            market = market_path.name
            result[venue][market] = {}
            
            for ticker_path in market_path.iterdir():
                if not ticker_path.is_dir():
                    continue
                ticker = ticker_path.name
                result[venue][market][ticker] = {}
                
                for interval_path in ticker_path.iterdir():
                    if not interval_path.is_dir():
                        continue
                    interval = interval_path.name
                    
                    periods = []
                    for file_path in interval_path.glob("*.parquet"):
                        periods.append(file_path.stem)
                    
                    if periods:
                        # Sort: years (4 digits) first, then year-month (YYYY-MM)
                        result[venue][market][ticker][interval] = sorted(periods)
    
    return result


def delete_period(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    period: str,
) -> bool:
    """
    Delete a specific period's data file.
    
    Returns:
        True if file was deleted, False if it didn't exist
    """
    file_path = get_data_path(venue, market, ticker, interval, period)
    
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def clear_all_data(venue: str, market: str, ticker: str, interval: str) -> int:
    """
    Delete all data for a specific ticker/interval.
    
    Returns:
        Number of files deleted
    """
    base_path = get_data_path(venue, market, ticker, interval)
    
    if not base_path.exists():
        return 0
    
    count = 0
    for file_path in base_path.glob("*.parquet"):
        file_path.unlink()
        count += 1
    
    return count
