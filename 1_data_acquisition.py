#!/usr/bin/env python3
"""
Stage 1: Data Acquisition

Fetches raw price data from TradFi (Yahoo Finance) and DeFi (Aster),
cleans/normalizes it, and saves in a common parquet format.

Output: data/cleaned/{asset}_{source}.parquet
"""

import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from aster.rest_api import Client as AsterClient


# ============================================
# Configuration
# ============================================

BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_CLEANED_DIR = BASE_DIR / "data" / "cleaned"

# Target date range (must be within last 30 days for Yahoo 1m data)
# Using 14 days of recent data for ~10 trading days
END_DATE = datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc)

# Assets to fetch: (name, yahoo_symbol, aster_symbol, market_open_hour_utc)
# market_open_hour_utc: Hour when the market opens in UTC
#   - Stocks (TSLA): NYSE opens 9:30 AM ET = 14:30 UTC
#   - Gold futures: Trade nearly 24h, use 0:00 UTC
ASSETS = [
    ("TSLA", "TSLA", "TSLAUSDT", 14),  # NYSE opens 14:30 UTC, we'll use 14:30
    ("GOLD", "GC=F", "XAUUSDT", 0),     # Futures trade 24h, start at 00:00
]

# Minutes per day
MINUTES_PER_DAY = 24 * 60  # 1440

# Standard column schema for cleaned data
SCHEMA_COLUMNS = ["open", "high", "low", "close", "mid", "volume", "source", "symbol"]

# API limits
ASTER_MAX_LIMIT = 500  # Aster API max records per request
YAHOO_MAX_DAYS = 7     # Yahoo 1m data max days per request


# ============================================
# Data Fetching Functions
# ============================================

def fetch_yahoo_data_paginated(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical 1m data from Yahoo Finance with pagination.
    Yahoo limits 1m data to 7 days per request, so we paginate.
    
    Args:
        symbol: Yahoo Finance symbol (e.g., 'TSLA', 'GC=F')
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    all_data = []
    current_start = start_date
    
    ticker = yf.Ticker(symbol)
    
    while current_start < end_date:
        current_end = min(current_start + pd.Timedelta(days=YAHOO_MAX_DAYS), end_date)
        
        print(f"    Fetching {current_start.date()} to {current_end.date()}...")
        
        try:
            df = ticker.history(
                start=current_start,
                end=current_end,
                interval="1m"
            )
            
            if not df.empty:
                # Normalize timezone to UTC
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("UTC")
                else:
                    df.index = df.index.tz_localize("UTC")
                
                all_data.append(df)
                print(f"      Got {len(df)} records")
            else:
                print(f"      No data returned")
                
        except Exception as e:
            print(f"      Error: {e}")
        
        current_start = current_end
        time.sleep(0.5)  # Rate limiting
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all chunks
    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    
    return combined


def fetch_aster_klines_paginated(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1m"
) -> pd.DataFrame:
    """
    Fetch historical klines from Aster with pagination.
    Aster limits to 500 records per request, so we paginate backwards.
    
    Args:
        symbol: Aster symbol (e.g., 'TSLAUSDT')
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        interval: Kline interval (default '1m')
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    client = AsterClient()
    all_data = []
    
    # Convert to milliseconds
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    
    current_end_ms = end_ms
    
    while current_end_ms > start_ms:
        print(f"    Fetching batch ending at {datetime.fromtimestamp(current_end_ms/1000, tz=timezone.utc)}...")
        
        try:
            data = client.klines(
                symbol=symbol,
                interval=interval,
                limit=ASTER_MAX_LIMIT,
                endTime=current_end_ms
            )
            
            if not data:
                print(f"      No more data")
                break
            
            print(f"      Got {len(data)} records")
            all_data.extend(data)
            
            # Move to before the earliest record
            earliest_time = min(int(d[0]) for d in data)
            if earliest_time >= current_end_ms:
                break  # No progress, exit
            current_end_ms = earliest_time - 1
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"      Error: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    # Parse timestamp and set index
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    
    # Convert price columns to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    
    # Filter to requested date range
    df = df[(df.index >= start_date) & (df.index < end_date)]
    
    return df


def fetch_aster_klines(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical klines from Aster.
    
    Args:
        symbol: Aster symbol (e.g., 'TSLAUSDT')
        interval: Kline interval (default '1m')
        limit: Number of candles to fetch (max 5000)
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    client = AsterClient()
    
    try:
        data = client.klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        print(f"  Error fetching Aster {symbol}: {e}")
        return pd.DataFrame()
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    
    # Convert types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


# ============================================
# Data Cleaning Functions
# ============================================

def clean_ohlcv_data(df: pd.DataFrame, source: str, symbol: str) -> pd.DataFrame:
    """
    Clean and normalize OHLCV data to standard schema.
    
    Args:
        df: Raw DataFrame with OHLCV columns
        source: Data source name ('yahoo', 'aster', 'hyperliquid')
        symbol: Original symbol
    
    Returns:
        Cleaned DataFrame with standard schema
    """
    if df.empty:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    
    # Standardize column names to lowercase
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=SCHEMA_COLUMNS)
    
    # Convert price columns to numeric (handles string values from some APIs)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Calculate mid price
    df["mid"] = (df["high"] + df["low"]) / 2
    
    # Handle volume (may not exist in all sources)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    
    # Add metadata
    df["source"] = source
    df["symbol"] = symbol
    
    # Select and order columns
    result = df[SCHEMA_COLUMNS].copy()
    
    # Remove any rows with NaN in price columns
    result = result.dropna(subset=["open", "high", "low", "close"])
    
    # Ensure index is named 'timestamp' and remove freq attribute for consistency
    result.index.name = "timestamp"
    result.index.freq = None
    
    return result


# ============================================
# LOCF (Last Observation Carried Forward)
# ============================================

def apply_locf_to_full_timeline(
    tradfi_df: pd.DataFrame,
    defi_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Expand TradFi data to match DeFi's 24/7 timeline using LOCF.
    
    When traditional markets are closed, we carry forward the last known price.
    This simulates TradFi being "open" 24/7 for easier comparison.
    
    Args:
        tradfi_df: Yahoo/TradFi data (market hours only)
        defi_df: Aster/DeFi data (24/7)
    
    Returns:
        TradFi DataFrame reindexed to DeFi timeline with LOCF applied
    """
    if tradfi_df.empty or defi_df.empty:
        return tradfi_df
    
    # Get the full timeline from DeFi data
    full_timeline = defi_df.index
    
    # Reindex TradFi data to full timeline
    expanded = tradfi_df.reindex(full_timeline)
    
    # Apply LOCF (forward fill) for price columns
    price_cols = ["open", "high", "low", "close", "mid"]
    expanded[price_cols] = expanded[price_cols].ffill()
    
    # Fill volume with 0 during closed hours (no trading occurred)
    expanded["volume"] = expanded["volume"].fillna(0.0)
    
    # Forward fill metadata columns
    expanded["source"] = expanded["source"].ffill()
    expanded["symbol"] = expanded["symbol"].ffill()
    
    # Drop any remaining NaN rows at the start (before first TradFi data point)
    expanded = expanded.dropna(subset=["close"])
    
    return expanded


# ============================================
# Data Persistence Functions
# ============================================

def clean_data_directory():
    """Remove all existing parquet files from the cleaned data directory."""
    if DATA_CLEANED_DIR.exists():
        for f in DATA_CLEANED_DIR.glob("*.parquet"):
            f.unlink()
            print(f"  Removed {f.name}")


def save_to_parquet(df: pd.DataFrame, asset: str, source: str) -> Optional[Path]:
    """
    Save DataFrame to parquet file.
    
    Args:
        df: Cleaned DataFrame
        asset: Asset name (e.g., 'TSLA')
        source: Source name (e.g., 'yahoo', 'aster')
    
    Returns:
        Path to saved file, or None if empty
    """
    if df.empty:
        return None
    
    DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_{source}.parquet"
    
    df.to_parquet(filepath, engine="pyarrow")
    return filepath


def load_from_parquet(asset: str, source: str) -> pd.DataFrame:
    """
    Load DataFrame from parquet file.
    
    Args:
        asset: Asset name (e.g., 'TSLA')
        source: Source name (e.g., 'yahoo', 'aster')
    
    Returns:
        DataFrame or empty DataFrame if file doesn't exist
    """
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_{source}.parquet"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(filepath, engine="pyarrow")


# ============================================
# Main Acquisition Pipeline
# ============================================

def acquire_asset_data(
    asset_name: str,
    yahoo_symbol: str,
    aster_symbol: str,
    start_date: datetime,
    end_date: datetime
) -> dict:
    """
    Acquire data for a single asset from Yahoo and Aster.
    
    Returns:
        Dict with 'yahoo' and 'aster' keys containing saved file paths
    """
    results = {"yahoo": None, "aster": None}
    
    # Yahoo Finance (TradFi)
    print(f"  [Yahoo] Fetching {yahoo_symbol}...")
    raw_df = fetch_yahoo_data_paginated(yahoo_symbol, start_date, end_date)
    if not raw_df.empty:
        cleaned = clean_ohlcv_data(raw_df, "yahoo", yahoo_symbol)
        results["yahoo"] = save_to_parquet(cleaned, asset_name, "yahoo")
        print(f"    ✓ {len(cleaned)} records saved")
    else:
        print(f"    ✗ No data")
    
    # Aster (DeFi)
    print(f"  [Aster] Fetching {aster_symbol}...")
    raw_df = fetch_aster_klines_paginated(aster_symbol, start_date, end_date)
    if not raw_df.empty:
        cleaned = clean_ohlcv_data(raw_df, "aster", aster_symbol)
        results["aster"] = save_to_parquet(cleaned, asset_name, "aster")
        print(f"    ✓ {len(cleaned)} records saved")
    else:
        print(f"    ✗ No data")
    
    return results


def align_timestamps(yahoo_df: pd.DataFrame, aster_df: pd.DataFrame) -> tuple:
    """
    Align Yahoo and Aster DataFrames to have matching timestamps.
    
    Returns:
        Tuple of (aligned_yahoo_df, aligned_aster_df)
    """
    if yahoo_df.empty or aster_df.empty:
        return yahoo_df, aster_df
    
    # Find common timestamps
    common_idx = yahoo_df.index.intersection(aster_df.index)
    
    return yahoo_df.loc[common_idx], aster_df.loc[common_idx]


def get_start_date_for_asset(market_open_hour: int, days: int = 14) -> datetime:
    """
    Calculate start date aligned to market open hour.
    
    Args:
        market_open_hour: Hour (UTC) when market opens (e.g., 14 for NYSE)
        days: Number of days to go back
    
    Returns:
        Start datetime aligned to market open
    """
    start = END_DATE - timedelta(days=days)
    # Set to market open hour, 30 minutes past (for NYSE 9:30 AM = 14:30 UTC)
    if market_open_hour == 14:  # NYSE stocks
        return start.replace(hour=14, minute=30, second=0, microsecond=0)
    else:  # Futures/24h markets
        return start.replace(hour=0, minute=0, second=0, microsecond=0)


def run_acquisition():
    """Run the full data acquisition pipeline."""
    print("=" * 60)
    print("Stage 1: Data Acquisition")
    print("=" * 60)
    
    # Clean existing data
    print("\n[1] Cleaning existing data files...")
    clean_data_directory()
    
    # Acquire data for each asset with market-aligned start times
    all_results = {}
    step = 2
    
    for name, yahoo_sym, aster_sym, market_open_hour in ASSETS:
        # Calculate start date aligned to market open
        start_date = get_start_date_for_asset(market_open_hour)
        
        print(f"\n[{step}] Acquiring {name} data...")
        print(f"  Period: {start_date} to {END_DATE}")
        
        all_results[name] = acquire_asset_data(
            name, yahoo_sym, aster_sym, start_date, END_DATE
        )
        step += 1
    
    # Apply LOCF to Yahoo data to match Aster's 24/7 timeline
    print(f"\n[{step}] Applying LOCF to TradFi data...")
    for name, _, _, _ in ASSETS:
        yahoo_df = load_from_parquet(name, "yahoo")
        aster_df = load_from_parquet(name, "aster")
        
        if not yahoo_df.empty and not aster_df.empty:
            yahoo_original_len = len(yahoo_df)
            
            # Apply LOCF to expand Yahoo to Aster's timeline
            yahoo_expanded = apply_locf_to_full_timeline(yahoo_df, aster_df)
            
            # Save expanded data
            save_to_parquet(yahoo_expanded, name, "yahoo")
            
            print(f"  {name}:")
            print(f"    Yahoo: {yahoo_original_len} → {len(yahoo_expanded)} bars (LOCF applied)")
            print(f"    Aster: {len(aster_df)} bars (24/7)")
        else:
            print(f"  {name}: Missing data, skipping LOCF")
    
    # Summary
    print("\n" + "=" * 60)
    print("Acquisition Summary")
    print("=" * 60)
    print(f"\nData saved to: {DATA_CLEANED_DIR}/")
    
    for name, _, _, _ in ASSETS:
        yahoo_df = load_from_parquet(name, "yahoo")
        aster_df = load_from_parquet(name, "aster")
        yahoo_ok = f"✓ {len(yahoo_df)}" if not yahoo_df.empty else "✗"
        aster_ok = f"✓ {len(aster_df)}" if not aster_df.empty else "✗"
        print(f"  {name:8} | Yahoo: {yahoo_ok:>8} | Aster: {aster_ok:>8}")
    
    print("\n" + "=" * 60)
    print("Stage 1 complete. Run Stage 2 for basis analysis.")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_acquisition()
