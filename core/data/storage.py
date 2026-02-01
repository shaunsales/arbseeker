"""
Parquet storage utilities for OHLCV data.

Storage structure:
    data/{venue}/{market}/{ticker}/{interval}/{year}.parquet

Example:
    data/binance/futures/BTCUSDT/1h/2024.parquet
"""

from pathlib import Path
from typing import Optional
import pandas as pd

# Base data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_data_path(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    year: Optional[int] = None,
) -> Path:
    """
    Get the path to a data file or directory.
    
    Args:
        venue: Data source (e.g., 'binance', 'hyperliquid')
        market: Market type (e.g., 'futures', 'spot')
        ticker: Trading pair (e.g., 'BTCUSDT')
        interval: Bar interval (e.g., '1h', '15m')
        year: Optional year for specific file
        
    Returns:
        Path to directory (if year is None) or specific parquet file
    """
    base_path = DATA_DIR / venue / market / ticker / interval
    if year is not None:
        return base_path / f"{year}.parquet"
    return base_path


def save_yearly(
    df: pd.DataFrame,
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    year: int,
) -> Path:
    """
    Save OHLCV data for a specific year.
    
    Args:
        df: DataFrame with OHLCV data (must have datetime index)
        venue: Data source
        market: Market type
        ticker: Trading pair
        interval: Bar interval
        year: Year to save
        
    Returns:
        Path to saved file
    """
    file_path = get_data_path(venue, market, ticker, interval, year)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Sort by time
    df = df.sort_index()
    
    # Save to parquet
    df.to_parquet(file_path, engine="pyarrow", compression="snappy")
    
    return file_path


def load_ohlcv(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for specified years.
    
    Args:
        venue: Data source
        market: Market type
        ticker: Trading pair
        interval: Bar interval
        years: List of years to load (None = all available)
        
    Returns:
        Concatenated DataFrame with OHLCV data
    """
    base_path = get_data_path(venue, market, ticker, interval)
    
    if not base_path.exists():
        raise FileNotFoundError(f"No data found at {base_path}")
    
    # Get available years if not specified
    if years is None:
        years = list_available_years(venue, market, ticker, interval)
    
    if not years:
        raise FileNotFoundError(f"No yearly data files found at {base_path}")
    
    # Load and concatenate
    dfs = []
    for year in sorted(years):
        file_path = base_path / f"{year}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found, skipping")
    
    if not dfs:
        raise FileNotFoundError(f"No data files found for years {years}")
    
    # Concatenate and sort
    result = pd.concat(dfs, axis=0)
    result = result.sort_index()
    
    # Remove duplicates (in case of overlapping data)
    result = result[~result.index.duplicated(keep="first")]
    
    return result


def list_available_years(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
) -> list[int]:
    """
    List all available years for a given data path.
    
    Returns:
        Sorted list of years with data
    """
    base_path = get_data_path(venue, market, ticker, interval)
    
    if not base_path.exists():
        return []
    
    years = []
    for file_path in base_path.glob("*.parquet"):
        try:
            year = int(file_path.stem)
            years.append(year)
        except ValueError:
            continue
    
    return sorted(years)


def list_all_data() -> dict:
    """
    List all available data in the data directory.
    
    Returns:
        Nested dict: {venue: {market: {ticker: {interval: [years]}}}}
    """
    result = {}
    
    if not DATA_DIR.exists():
        return result
    
    for venue_path in DATA_DIR.iterdir():
        if not venue_path.is_dir() or venue_path.name.startswith("."):
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
                    
                    years = []
                    for file_path in interval_path.glob("*.parquet"):
                        try:
                            year = int(file_path.stem)
                            years.append(year)
                        except ValueError:
                            continue
                    
                    if years:
                        result[venue][market][ticker][interval] = sorted(years)
    
    return result


def delete_year(
    venue: str,
    market: str,
    ticker: str,
    interval: str,
    year: int,
) -> bool:
    """
    Delete a specific year's data file.
    
    Returns:
        True if file was deleted, False if it didn't exist
    """
    file_path = get_data_path(venue, market, ticker, interval, year)
    
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
