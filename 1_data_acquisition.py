#!/usr/bin/env python3
"""
Stage 1: Data Acquisition

Fetches raw price data from TradFi (Yahoo, TradingView) and DeFi (Aster),
cleans/normalizes it, and saves in a common parquet format.

Supports configurable intervals: 1m, 5m, 15m, 1h, etc.

Output: data/cleaned/{asset}_{source}_{interval}.parquet
"""

import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from aster.rest_api import Client as AsterClient
from tvDatafeed import TvDatafeed, Interval as TvInterval


# ============================================
# Configuration
# ============================================

BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_CLEANED_DIR = BASE_DIR / "data" / "cleaned"

# Default end date (yesterday end of day to exclude today's incomplete data)
END_DATE = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=23, minute=59, second=0, microsecond=0)

# Supported intervals mapping
INTERVAL_MAPPING = {
    # interval_key: (yahoo_interval, aster_interval, tv_interval, minutes_per_bar)
    "1m":  ("1m",  "1m",  TvInterval.in_1_minute,  1),
    "5m":  ("5m",  "5m",  TvInterval.in_5_minute,  5),
    "15m": ("15m", "15m", TvInterval.in_15_minute, 15),
    "30m": ("30m", "30m", TvInterval.in_30_minute, 30),
    "1h":  ("1h",  "1h",  TvInterval.in_1_hour,    60),
    "4h":  ("1h",  "4h",  TvInterval.in_4_hour,    240),  # Yahoo doesn't have 4h
    "1d":  ("1d",  "1d",  TvInterval.in_daily,     1440),
}

# Assets configuration
# (name, tradfi_symbol, defi_symbol, tradfi_source, tradfi_exchange, defi_source)
# tradfi_source: "yahoo" or "tradingview"
# tradfi_exchange: TradingView exchange (e.g., "COMEX") - only used for TV
# defi_source: "aster" or "hyperliquid"
ASSETS = [
    ("GOLD", "GC1!",  "XAUUSDT",  "tradingview", "COMEX", "aster"),
    ("GOLD_HL", "GC1!",  "PAXG",  "tradingview", "COMEX", "hyperliquid"),  # Hyperliquid PAXG
]

# Standard column schema for cleaned data
SCHEMA_COLUMNS = ["open", "high", "low", "close", "mid", "volume", "source", "symbol"]

# API limits
ASTER_MAX_LIMIT = 1500  # Aster API max records per request
YAHOO_MAX_DAYS = 7      # Yahoo 1m data max days per request
TV_MAX_BARS = 5000      # TradingView max bars per request


# ============================================
# Data Fetching Functions
# ============================================

def fetch_yahoo_data_paginated(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1m"
) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance with pagination.
    Yahoo limits 1m data to 7 days per request, so we paginate.
    
    Args:
        symbol: Yahoo Finance symbol (e.g., 'TSLA', 'GC=F')
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        interval: Data interval (e.g., '1m', '5m', '15m', '1h')
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    all_data = []
    current_start = start_date
    
    # Determine chunk size based on interval
    if interval == "1m":
        chunk_days = YAHOO_MAX_DAYS
    elif interval in ["5m", "15m", "30m"]:
        chunk_days = 60  # Yahoo allows ~60 days for 5m-30m
    else:
        chunk_days = 365  # Longer intervals can fetch more
    
    ticker = yf.Ticker(symbol)
    
    while current_start < end_date:
        current_end = min(current_start + pd.Timedelta(days=chunk_days), end_date)
        
        print(f"    Fetching {current_start.date()} to {current_end.date()}...")
        
        try:
            df = ticker.history(
                start=current_start,
                end=current_end,
                interval=interval
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


def fetch_tradingview_data(
    symbol: str,
    exchange: str,
    interval: TvInterval,
    n_bars: int = 5000
) -> pd.DataFrame:
    """
    Fetch historical data from TradingView.
    
    Args:
        symbol: TradingView symbol (e.g., 'GC1!' for gold continuous)
        exchange: Exchange name (e.g., 'COMEX', 'CME')
        interval: TvInterval enum (e.g., TvInterval.in_1_minute)
        n_bars: Number of bars to fetch (max 5000)
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    try:
        tv = TvDatafeed()
        
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=min(n_bars, TV_MAX_BARS)
        )
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # TradingView returns timezone-naive index, localize to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Standardize column names
        df = df.rename(columns={
            "open": "Open",
            "high": "High", 
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        
        # Remove the 'symbol' column if present (we'll add our own)
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        
        return df
        
    except Exception as e:
        print(f"    Error fetching TradingView {symbol}@{exchange}: {e}")
        return pd.DataFrame()


def fetch_tradingview_data_paginated(
    symbol: str,
    exchange: str,
    start_date: datetime,
    end_date: datetime,
    interval: TvInterval,
    minutes_per_bar: int
) -> pd.DataFrame:
    """
    Fetch historical data from TradingView with pagination.
    
    TradingView limits to 5000 bars per request, so we may need multiple requests
    for longer time periods.
    
    Args:
        symbol: TradingView symbol (e.g., 'GC1!')
        exchange: Exchange name (e.g., 'COMEX')
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        interval: TvInterval enum
        minutes_per_bar: Minutes per bar for calculating pagination
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    all_data = []
    
    # Calculate how many bars we need
    total_minutes = (end_date - start_date).total_seconds() / 60
    total_bars_needed = int(total_minutes / minutes_per_bar)
    
    # TradingView returns most recent data first, so we fetch in chunks
    bars_fetched = 0
    
    while bars_fetched < total_bars_needed:
        bars_to_fetch = min(TV_MAX_BARS, total_bars_needed - bars_fetched)
        
        print(f"    Fetching {bars_to_fetch} bars from TradingView...")
        
        df = fetch_tradingview_data(symbol, exchange, interval, bars_to_fetch)
        
        if df.empty:
            break
        
        # Filter to our date range
        df = df[(df.index >= start_date) & (df.index < end_date)]
        
        if df.empty:
            break
            
        all_data.append(df)
        bars_fetched += len(df)
        
        # TradingView doesn't support true pagination by date, so we get what we can
        # in one request. Break after first successful fetch.
        break
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all chunks
    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    
    return combined


def fetch_hyperliquid_candles(
    coin: str,
    interval: str = "15m",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetch historical candle data from Hyperliquid DEX.
    
    Uses the official Public Info API (no API key required).
    Endpoint: https://api.hyperliquid.xyz/info
    
    Args:
        coin: Coin symbol (e.g., 'GOLD', 'BTC', 'ETH')
        interval: Candle interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
        start_time: Start datetime (UTC). If None, fetches last available data.
        end_time: End datetime (UTC). If None, fetches up to now.
    
    Returns:
        DataFrame with OHLCV data, UTC-normalized index
    """
    import requests
    
    url = "https://api.hyperliquid.xyz/info"
    
    # Build request payload
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
        }
    }
    
    # Add start time if specified (milliseconds)
    if start_time:
        payload["req"]["startTime"] = int(start_time.timestamp() * 1000)
    
    # Add end time if specified
    if end_time:
        payload["req"]["endTime"] = int(end_time.timestamp() * 1000)
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"  Error fetching Hyperliquid {coin}: {e}")
        return pd.DataFrame()
    
    if not data:
        return pd.DataFrame()
    
    # Parse response - Hyperliquid returns list of candles
    # Each candle: {"t": timestamp_ms, "o": open, "h": high, "l": low, "c": close, "v": volume}
    records = []
    for candle in data:
        records.append({
            "timestamp": pd.to_datetime(candle["t"], unit="ms", utc=True),
            "Open": float(candle["o"]),
            "High": float(candle["h"]),
            "Low": float(candle["l"]),
            "Close": float(candle["c"]),
            "Volume": float(candle["v"]),
        })
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df = df.set_index("timestamp")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    return df


def fetch_hyperliquid_paginated(
    coin: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "15m"
) -> pd.DataFrame:
    """
    Fetch Hyperliquid candle data with pagination for longer time periods.
    
    Args:
        coin: Coin symbol (e.g., 'GOLD')
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        interval: Candle interval
    
    Returns:
        DataFrame with OHLCV data
    """
    all_data = []
    current_start = start_date
    
    # Interval to minutes mapping
    interval_mins = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440
    }
    mins = interval_mins.get(interval, 15)
    
    # Fetch in chunks (Hyperliquid may limit response size)
    chunk_size = timedelta(days=7)  # 7 days per request
    
    while current_start < end_date:
        chunk_end = min(current_start + chunk_size, end_date)
        
        print(f"    Fetching Hyperliquid {coin} {interval} from {current_start.date()} to {chunk_end.date()}...")
        
        df = fetch_hyperliquid_candles(coin, interval, current_start, chunk_end)
        
        if not df.empty:
            all_data.append(df)
        
        current_start = chunk_end
        time.sleep(0.2)  # Rate limiting
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    
    # Filter to exact date range
    combined = combined[(combined.index >= start_date) & (combined.index < end_date)]
    
    return combined


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
# Data Validation Functions
# ============================================

def validate_data(
    df: pd.DataFrame,
    source_name: str,
    interval_minutes: int,
    price_col: str = "close",
    max_gap_multiplier: float = 3.0,
    max_price_change_pct: float = 10.0,
    print_details: bool = True
) -> dict:
    """
    Validate DataFrame for common data quality issues.
    
    Checks:
        1. Empty data
        2. Out of order timestamps
        3. Duplicate timestamps
        4. Data gaps (missing bars)
        5. Price outliers (extreme single-bar moves)
        6. Zero/negative prices
        7. NaN values
    
    Args:
        df: DataFrame with datetime index
        source_name: Name for logging (e.g., 'TradFi', 'DeFi')
        interval_minutes: Expected bar interval in minutes
        price_col: Column to check for price outliers
        max_gap_multiplier: Max allowed gap as multiple of interval
        max_price_change_pct: Max allowed single-bar % change
        print_details: Whether to print validation details
    
    Returns:
        Dict with validation results and issues found
    """
    results = {
        "source": source_name,
        "valid": True,
        "total_bars": len(df),
        "issues": [],
        "warnings": [],
        "stats": {}
    }
    
    if df.empty:
        results["valid"] = False
        results["issues"].append("Empty DataFrame")
        return results
    
    # 1. Check index is sorted (out of order)
    if not df.index.is_monotonic_increasing:
        out_of_order = (~df.index.to_series().diff().dropna().ge(pd.Timedelta(0))).sum()
        results["issues"].append(f"Out of order timestamps: {out_of_order}")
        results["valid"] = False
    
    # 2. Check for duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        results["issues"].append(f"Duplicate timestamps: {duplicates}")
        results["valid"] = False
    
    # 3. Check for gaps
    expected_delta = pd.Timedelta(minutes=interval_minutes)
    max_allowed_gap = expected_delta * max_gap_multiplier
    
    time_diffs = df.index.to_series().diff().dropna()
    gaps = time_diffs[time_diffs > max_allowed_gap]
    
    if len(gaps) > 0:
        results["warnings"].append(f"Data gaps found: {len(gaps)} gaps > {max_gap_multiplier}x interval")
        results["stats"]["gap_count"] = len(gaps)
        results["stats"]["max_gap_hours"] = gaps.max().total_seconds() / 3600
        results["stats"]["total_gap_hours"] = (gaps.sum() - expected_delta * len(gaps)).total_seconds() / 3600
    
    # 4. Check for price outliers (extreme moves)
    if price_col in df.columns:
        pct_changes = df[price_col].pct_change().abs() * 100
        outliers = pct_changes[pct_changes > max_price_change_pct]
        
        if len(outliers) > 0:
            results["warnings"].append(f"Price outliers: {len(outliers)} bars with >{max_price_change_pct}% move")
            results["stats"]["outlier_count"] = len(outliers)
            results["stats"]["max_pct_change"] = pct_changes.max()
    
    # 5. Check for zero/negative prices
    if price_col in df.columns:
        bad_prices = (df[price_col] <= 0).sum()
        if bad_prices > 0:
            results["issues"].append(f"Zero/negative prices: {bad_prices}")
            results["valid"] = False
    
    # 6. Check for NaN values in key columns
    key_cols = ["open", "high", "low", "close"] if "open" in df.columns else [price_col]
    for col in key_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                results["warnings"].append(f"NaN values in {col}: {nan_count}")
    
    # Summary stats
    results["stats"]["date_range"] = f"{df.index.min()} to {df.index.max()}"
    results["stats"]["days_covered"] = (df.index.max() - df.index.min()).days
    
    # Print results
    if print_details:
        status = "✓ VALID" if results["valid"] else "✗ INVALID"
        print(f"  [{source_name}] {status} - {len(df):,} bars")
        
        for issue in results["issues"]:
            print(f"    ✗ {issue}")
        for warning in results["warnings"]:
            print(f"    ⚠ {warning}")
        
        if results["stats"].get("gap_count"):
            print(f"    → Largest gap: {results['stats']['max_gap_hours']:.1f} hours")
    
    return results


def validate_merged_data(
    df: pd.DataFrame,
    interval_minutes: int,
    max_basis_bps: float = 500.0,
    print_details: bool = True
) -> dict:
    """
    Validate merged DataFrame with both TradFi and DeFi data.
    
    Additional checks specific to merged data:
        1. Price correlation between venues
        2. Basis distribution sanity
        3. Volume sanity
    
    Args:
        df: Merged DataFrame
        interval_minutes: Expected bar interval in minutes
        max_basis_bps: Max expected basis in bps
        print_details: Whether to print validation details
    
    Returns:
        Dict with validation results
    """
    results = {
        "valid": True,
        "total_bars": len(df),
        "issues": [],
        "warnings": [],
        "stats": {}
    }
    
    if df.empty:
        results["valid"] = False
        results["issues"].append("Empty DataFrame")
        return results
    
    # 1. Basic validation
    base_results = validate_data(
        df, "Merged", interval_minutes, 
        price_col="tradfi_close", 
        print_details=False
    )
    results["issues"].extend(base_results["issues"])
    results["warnings"].extend(base_results["warnings"])
    results["stats"].update(base_results["stats"])
    
    if not base_results["valid"]:
        results["valid"] = False
    
    # 2. Check price correlation
    if "tradfi_close" in df.columns and "defi_close" in df.columns:
        correlation = df["tradfi_close"].corr(df["defi_close"])
        results["stats"]["price_correlation"] = correlation
        
        if correlation < 0.95:
            results["warnings"].append(f"Low price correlation: {correlation:.4f}")
        if correlation < 0.80:
            results["issues"].append(f"Very low price correlation: {correlation:.4f}")
            results["valid"] = False
    
    # 3. Check basis distribution
    if "basis_bps" in df.columns:
        bps = df["basis_bps"]
        results["stats"]["basis_mean"] = bps.mean()
        results["stats"]["basis_std"] = bps.std()
        results["stats"]["basis_min"] = bps.min()
        results["stats"]["basis_max"] = bps.max()
        
        extreme_basis = (abs(bps) > max_basis_bps).sum()
        if extreme_basis > 0:
            results["warnings"].append(f"Extreme basis values: {extreme_basis} bars > {max_basis_bps} bps")
    
    # 4. Check for LOCF artifacts (long runs of identical prices)
    if "tradfi_close" in df.columns:
        tradfi_unchanged = (df["tradfi_close"].diff() == 0).sum()
        pct_unchanged = tradfi_unchanged / len(df) * 100
        results["stats"]["tradfi_unchanged_pct"] = pct_unchanged
        
        if pct_unchanged > 50:
            results["warnings"].append(f"High LOCF ratio: {pct_unchanged:.1f}% bars unchanged (market closed)")
    
    # Print results
    if print_details:
        status = "✓ VALID" if results["valid"] else "✗ INVALID"
        print(f"  [Merged] {status} - {len(df):,} bars")
        
        for issue in results["issues"]:
            print(f"    ✗ {issue}")
        for warning in results["warnings"]:
            print(f"    ⚠ {warning}")
        
        if "price_correlation" in results["stats"]:
            print(f"    → Price correlation: {results['stats']['price_correlation']:.4f}")
        if "basis_mean" in results["stats"]:
            print(f"    → Basis: mean={results['stats']['basis_mean']:+.1f} bps, std={results['stats']['basis_std']:.1f} bps")
    
    return results


def print_validation_report(
    tradfi_results: dict,
    defi_results: dict,
    merged_results: dict
):
    """Print a comprehensive validation report."""
    print("\n" + "=" * 60)
    print("Data Validation Report")
    print("=" * 60)
    
    all_valid = (
        tradfi_results.get("valid", False) and
        defi_results.get("valid", False) and
        merged_results.get("valid", False)
    )
    
    total_issues = (
        len(tradfi_results.get("issues", [])) +
        len(defi_results.get("issues", [])) +
        len(merged_results.get("issues", []))
    )
    
    total_warnings = (
        len(tradfi_results.get("warnings", [])) +
        len(defi_results.get("warnings", [])) +
        len(merged_results.get("warnings", []))
    )
    
    print(f"\nOverall Status: {'✓ ALL VALID' if all_valid else '✗ ISSUES FOUND'}")
    print(f"  Issues: {total_issues}")
    print(f"  Warnings: {total_warnings}")
    
    # Summary stats
    if merged_results.get("stats"):
        stats = merged_results["stats"]
        print(f"\nData Summary:")
        print(f"  Date range: {stats.get('date_range', 'N/A')}")
        print(f"  Days covered: {stats.get('days_covered', 'N/A')}")
        print(f"  Total bars: {merged_results.get('total_bars', 0):,}")
        
        if "gap_count" in stats:
            print(f"  Data gaps: {stats['gap_count']} (total {stats.get('total_gap_hours', 0):.1f} hours)")
        
        if "price_correlation" in stats:
            print(f"  Price correlation: {stats['price_correlation']:.4f}")
        
        if "tradfi_unchanged_pct" in stats:
            print(f"  LOCF bars (market closed): {stats['tradfi_unchanged_pct']:.1f}%")


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


def save_to_parquet(df: pd.DataFrame, asset: str, source: str, interval: str = "1m") -> Optional[Path]:
    """
    Save DataFrame to parquet file.
    
    Args:
        df: Cleaned DataFrame
        asset: Asset name (e.g., 'TSLA')
        source: Source name (e.g., 'yahoo', 'aster', 'tradingview')
        interval: Data interval (e.g., '1m', '5m', '15m')
    
    Returns:
        Path to saved file, or None if empty
    """
    if df.empty:
        return None
    
    DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_{source}_{interval}.parquet"
    
    df.to_parquet(filepath, engine="pyarrow")
    return filepath


def load_from_parquet(asset: str, source: str, interval: str = "1m") -> pd.DataFrame:
    """
    Load DataFrame from parquet file.
    
    Args:
        asset: Asset name (e.g., 'TSLA')
        source: Source name (e.g., 'yahoo', 'aster', 'tradingview')
        interval: Data interval (e.g., '1m', '5m', '15m')
    
    Returns:
        DataFrame or empty DataFrame if file doesn't exist
    """
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_{source}_{interval}.parquet"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(filepath, engine="pyarrow")


def _is_cme_market_open(timestamp: pd.Timestamp) -> bool:
    """
    Check if CME Globex is open at given UTC timestamp.
    
    CME Gold futures: Sunday 6 PM - Friday 5 PM ET (23 hrs/day)
    Daily break: 5-6 PM ET (22:00-23:00 UTC)
    
    Args:
        timestamp: UTC timestamp to check
    
    Returns:
        True if CME is open, False otherwise
    """
    from datetime import time
    
    weekday = timestamp.weekday()  # 0=Mon, 6=Sun
    t = timestamp.time()
    
    CME_BREAK_START = time(22, 0)   # 5 PM ET
    CME_BREAK_END = time(23, 0)     # 6 PM ET
    
    # Saturday - fully closed
    if weekday == 5:
        return False
    
    # Sunday - only open after 23:00 UTC (6 PM ET)
    if weekday == 6:
        return t >= CME_BREAK_END
    
    # Friday - closed after 22:00 UTC (5 PM ET)
    if weekday == 4:
        return t < CME_BREAK_START
    
    # Mon-Thu: Open except during daily break (22:00-23:00 UTC)
    return not (CME_BREAK_START <= t < CME_BREAK_END)


def create_merged_data(
    tradfi_df: pd.DataFrame,
    defi_df: pd.DataFrame,
    asset: str,
    tradfi_source: str,
    interval: str,
    max_basis_bps: float = 500.0
) -> pd.DataFrame:
    """
    Create merged DataFrame with both TradFi and DeFi data aligned by timestamp.
    
    Output schema:
        timestamp (index), tradfi_close, defi_close, tradfi_volume, defi_volume,
        tradfi_open, tradfi_high, tradfi_low, defi_open, defi_high, defi_low
    
    Args:
        tradfi_df: Cleaned TradFi DataFrame
        defi_df: Cleaned DeFi DataFrame  
        asset: Asset name (e.g., 'GOLD')
        tradfi_source: TradFi source name (e.g., 'tradingview')
        interval: Data interval (e.g., '15m')
        max_basis_bps: Maximum allowed basis in bps (outliers filtered)
    
    Returns:
        Merged DataFrame with both venues' data
    """
    if tradfi_df.empty or defi_df.empty:
        return pd.DataFrame()
    
    # Find common timestamps (inner join)
    common_idx = tradfi_df.index.intersection(defi_df.index)
    
    if len(common_idx) == 0:
        return pd.DataFrame()
    
    # Create merged DataFrame
    merged = pd.DataFrame(index=common_idx)
    merged.index.name = "timestamp"
    
    # TradFi columns
    merged["tradfi_open"] = tradfi_df.loc[common_idx, "open"]
    merged["tradfi_high"] = tradfi_df.loc[common_idx, "high"]
    merged["tradfi_low"] = tradfi_df.loc[common_idx, "low"]
    merged["tradfi_close"] = tradfi_df.loc[common_idx, "close"]
    merged["tradfi_volume"] = tradfi_df.loc[common_idx, "volume"]
    
    # DeFi columns
    merged["defi_open"] = defi_df.loc[common_idx, "open"]
    merged["defi_high"] = defi_df.loc[common_idx, "high"]
    merged["defi_low"] = defi_df.loc[common_idx, "low"]
    merged["defi_close"] = defi_df.loc[common_idx, "close"]
    merged["defi_volume"] = defi_df.loc[common_idx, "volume"]
    
    # Calculate derived fields
    merged["tradfi_mid"] = (merged["tradfi_high"] + merged["tradfi_low"]) / 2
    merged["defi_mid"] = (merged["defi_high"] + merged["defi_low"]) / 2
    
    # Basis calculations
    merged["basis_absolute"] = merged["defi_close"] - merged["tradfi_close"]
    merged["basis_bps"] = (merged["basis_absolute"] / merged["tradfi_close"]) * 10000
    
    # Market hours flags
    # TradFi (CME Gold): Open Sun 6PM - Fri 5PM ET, daily break 5-6PM ET
    # In UTC: Sun 23:00 - Fri 22:00, daily break 22:00-23:00
    merged["tradfi_market_open"] = merged.index.to_series().apply(
        lambda ts: _is_cme_market_open(ts)
    )
    
    # DeFi: Always open (24/7)
    merged["defi_market_open"] = True
    
    # Combined: Both markets tradeable
    merged["both_markets_open"] = merged["tradfi_market_open"] & merged["defi_market_open"]
    
    # Detect LOCF bars (price unchanged from previous bar during closed hours)
    merged["tradfi_is_locf"] = (
        (merged["tradfi_close"].diff() == 0) & 
        (~merged["tradfi_market_open"])
    )
    
    # Volume/Liquidity metrics
    # Dollar volume (notional traded)
    merged["tradfi_dollar_volume"] = merged["tradfi_close"] * merged["tradfi_volume"]
    merged["defi_dollar_volume"] = merged["defi_close"] * merged["defi_volume"]
    merged["combined_dollar_volume"] = merged["tradfi_dollar_volume"] + merged["defi_dollar_volume"]
    
    # Volume flags for liquidity analysis
    merged["tradfi_has_volume"] = merged["tradfi_volume"] > 0
    merged["defi_has_volume"] = merged["defi_volume"] > 0
    merged["both_have_volume"] = merged["tradfi_has_volume"] & merged["defi_has_volume"]
    
    # Rolling volume metrics (for liquidity trends)
    # 4-bar = 1 hour at 15m intervals
    merged["tradfi_volume_1h"] = merged["tradfi_volume"].rolling(4, min_periods=1).sum()
    merged["defi_volume_1h"] = merged["defi_volume"].rolling(4, min_periods=1).sum()
    
    # Metadata
    merged["asset"] = asset
    merged["tradfi_source"] = tradfi_source
    merged["interval"] = interval
    
    # Filter outliers (bad data)
    pre_filter_len = len(merged)
    merged = merged[abs(merged["basis_bps"]) <= max_basis_bps]
    filtered_count = pre_filter_len - len(merged)
    
    if filtered_count > 0:
        print(f"    Filtered {filtered_count} outlier bars (|basis| > {max_basis_bps} bps)")
    
    return merged.sort_index()


def save_merged_data(merged_df: pd.DataFrame, asset: str, interval: str) -> Optional[Path]:
    """Save merged DataFrame to parquet file."""
    if merged_df.empty:
        return None
    
    DATA_CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_merged_{interval}.parquet"
    
    merged_df.to_parquet(filepath, engine="pyarrow")
    return filepath


def load_merged_data(asset: str, interval: str) -> pd.DataFrame:
    """Load merged DataFrame from parquet file."""
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_merged_{interval}.parquet"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(filepath, engine="pyarrow")


# ============================================
# Main Acquisition Pipeline
# ============================================

def acquire_asset_data(
    asset_name: str,
    tradfi_symbol: str,
    defi_symbol: str,
    tradfi_source: str,
    tradfi_exchange: Optional[str],
    start_date: datetime,
    end_date: datetime,
    interval: str = "1m",
    defi_source: str = "aster"
) -> dict:
    """
    Acquire data for a single asset from TradFi and DeFi sources.
    
    Args:
        asset_name: Asset name (e.g., 'GOLD')
        tradfi_symbol: TradFi symbol (e.g., 'GC1!' or 'TSLA')
        defi_symbol: DeFi symbol (e.g., 'XAUUSDT' for Aster, 'PAXG' for Hyperliquid)
        tradfi_source: TradFi data source ('yahoo' or 'tradingview')
        tradfi_exchange: TradingView exchange (e.g., 'COMEX') - only for TV
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        interval: Data interval (e.g., '1m', '5m', '15m')
        defi_source: DeFi data source ('aster' or 'hyperliquid')
    
    Returns:
        Dict with 'tradfi' and 'defi' keys containing saved file paths
    """
    results = {"tradfi": None, "defi": None}
    
    # Get interval mappings
    yahoo_interval, aster_interval, tv_interval, minutes_per_bar = INTERVAL_MAPPING[interval]
    
    # TradFi data (Yahoo or TradingView)
    if tradfi_source == "yahoo":
        print(f"  [Yahoo] Fetching {tradfi_symbol} @ {interval}...")
        raw_df = fetch_yahoo_data_paginated(tradfi_symbol, start_date, end_date, yahoo_interval)
        source_name = "yahoo"
    elif tradfi_source == "tradingview":
        print(f"  [TradingView] Fetching {tradfi_symbol}@{tradfi_exchange} @ {interval}...")
        raw_df = fetch_tradingview_data_paginated(
            tradfi_symbol, tradfi_exchange, start_date, end_date, tv_interval, minutes_per_bar
        )
        source_name = "tradingview"
    else:
        print(f"  Unknown TradFi source: {tradfi_source}")
        raw_df = pd.DataFrame()
        source_name = tradfi_source
    
    if not raw_df.empty:
        cleaned = clean_ohlcv_data(raw_df, source_name, tradfi_symbol)
        results["tradfi"] = save_to_parquet(cleaned, asset_name, source_name, interval)
        print(f"    ✓ {len(cleaned)} records saved")
    else:
        print(f"    ✗ No data")
    
    # DeFi data (Aster or Hyperliquid)
    if defi_source == "aster":
        print(f"  [Aster] Fetching {defi_symbol} @ {interval}...")
        raw_df = fetch_aster_klines_paginated(defi_symbol, start_date, end_date, aster_interval)
        source_label = "aster"
    elif defi_source == "hyperliquid":
        print(f"  [Hyperliquid] Fetching {defi_symbol} @ {interval}...")
        raw_df = fetch_hyperliquid_paginated(defi_symbol, start_date, end_date, interval)
        source_label = "hyperliquid"
    else:
        print(f"  Unknown DeFi source: {defi_source}")
        raw_df = pd.DataFrame()
        source_label = defi_source
    
    if not raw_df.empty:
        cleaned = clean_ohlcv_data(raw_df, source_label, defi_symbol)
        results["defi"] = save_to_parquet(cleaned, asset_name, source_label, interval)
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


def get_start_date(days: int, end_date: datetime = None) -> datetime:
    """
    Calculate start date for data acquisition.
    
    Args:
        days: Number of days to go back
        end_date: End date (defaults to END_DATE)
    
    Returns:
        Start datetime
    """
    if end_date is None:
        end_date = END_DATE
    start = end_date - timedelta(days=days)
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


def run_acquisition(
    interval: str = "15m",
    days: int = 60,
    assets: list = None,
    apply_locf: bool = True
):
    """
    Run the full data acquisition pipeline.
    
    Args:
        interval: Data interval (e.g., '1m', '5m', '15m', '1h')
        days: Number of days of historical data to fetch
        assets: List of assets to fetch (defaults to ASSETS config)
        apply_locf: Whether to apply LOCF to TradFi data
    
    Returns:
        Dict of results for each asset
    """
    if assets is None:
        assets = ASSETS
    
    if interval not in INTERVAL_MAPPING:
        raise ValueError(f"Unsupported interval: {interval}. Supported: {list(INTERVAL_MAPPING.keys())}")
    
    print("=" * 60)
    print("Stage 1: Data Acquisition")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Interval: {interval}")
    print(f"  Days: {days}")
    print(f"  Assets: {[a[0] for a in assets]}")
    
    # Calculate date range
    end_date = END_DATE
    start_date = get_start_date(days, end_date)
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    
    # Clean existing data
    print("\n[1] Cleaning existing data files...")
    clean_data_directory()
    
    # Acquire data for each asset
    all_results = {}
    step = 2
    
    for asset_config in assets:
        # Support both 5-element (legacy) and 6-element (with defi_source) tuples
        if len(asset_config) == 6:
            name, tradfi_sym, defi_sym, tradfi_source, tradfi_exchange, defi_source = asset_config
        else:
            name, tradfi_sym, defi_sym, tradfi_source, tradfi_exchange = asset_config
            defi_source = "aster"  # Default
        
        print(f"\n[{step}] Acquiring {name} data...")
        
        all_results[name] = acquire_asset_data(
            asset_name=name,
            tradfi_symbol=tradfi_sym,
            defi_symbol=defi_sym,
            tradfi_source=tradfi_source,
            tradfi_exchange=tradfi_exchange,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            defi_source=defi_source
        )
        # Store defi_source for later use
        all_results[name]["defi_source"] = defi_source
        step += 1
    
    # Apply LOCF to TradFi data to match DeFi's 24/7 timeline
    if apply_locf:
        print(f"\n[{step}] Applying LOCF to TradFi data...")
        for asset_config in assets:
            if len(asset_config) == 6:
                name, _, _, tradfi_source, _, defi_source = asset_config
            else:
                name, _, _, tradfi_source, _ = asset_config
                defi_source = "aster"
            
            tradfi_df = load_from_parquet(name, tradfi_source, interval)
            defi_df = load_from_parquet(name, defi_source, interval)
            
            if not tradfi_df.empty and not defi_df.empty:
                tradfi_original_len = len(tradfi_df)
                
                # Apply LOCF to expand TradFi to DeFi's timeline
                tradfi_expanded = apply_locf_to_full_timeline(tradfi_df, defi_df)
                
                # Save expanded data
                save_to_parquet(tradfi_expanded, name, tradfi_source, interval)
                
                print(f"  {name}:")
                print(f"    TradFi: {tradfi_original_len} → {len(tradfi_expanded)} bars (LOCF applied)")
                print(f"    DeFi: {len(defi_df)} bars (24/7)")
            else:
                print(f"  {name}: Missing data, skipping LOCF")
        step += 1
    
    # Create merged data files
    print(f"\n[{step}] Creating merged data files...")
    for asset_config in assets:
        if len(asset_config) == 6:
            name, _, _, tradfi_source, _, defi_source = asset_config
        else:
            name, _, _, tradfi_source, _ = asset_config
            defi_source = "aster"
        
        tradfi_df = load_from_parquet(name, tradfi_source, interval)
        defi_df = load_from_parquet(name, defi_source, interval)
        
        if not tradfi_df.empty and not defi_df.empty:
            merged_df = create_merged_data(tradfi_df, defi_df, name, tradfi_source, interval)
            
            if not merged_df.empty:
                save_merged_data(merged_df, name, interval)
                print(f"  {name}: ✓ {len(merged_df)} merged bars")
                print(f"    Date range: {merged_df.index.min()} to {merged_df.index.max()}")
                print(f"    Basis stats: mean={merged_df['basis_bps'].mean():.1f} bps, std={merged_df['basis_bps'].std():.1f} bps")
            else:
                print(f"  {name}: ✗ No overlapping data to merge")
        else:
            print(f"  {name}: ✗ Missing data, cannot merge")
    step += 1
    
    # Validate data
    _, _, _, minutes_per_bar = INTERVAL_MAPPING[interval]
    print(f"\n[{step}] Validating data quality...")
    
    validation_results = {}
    for asset_config in assets:
        if len(asset_config) == 6:
            name, _, _, tradfi_source, _, defi_source = asset_config
        else:
            name, _, _, tradfi_source, _ = asset_config
            defi_source = "aster"
        
        tradfi_df = load_from_parquet(name, tradfi_source, interval)
        defi_df = load_from_parquet(name, defi_source, interval)
        merged_df = load_merged_data(name, interval)
        
        tradfi_results = validate_data(tradfi_df, "TradFi", minutes_per_bar)
        defi_results = validate_data(defi_df, "DeFi", minutes_per_bar)
        merged_results = validate_merged_data(merged_df, minutes_per_bar)
        
        validation_results[name] = {
            "tradfi": tradfi_results,
            "defi": defi_results,
            "merged": merged_results
        }
        
        print_validation_report(tradfi_results, defi_results, merged_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("Acquisition Summary")
    print("=" * 60)
    print(f"\nData saved to: {DATA_CLEANED_DIR}/")
    
    for asset_config in assets:
        if len(asset_config) == 6:
            name, _, _, tradfi_source, _, defi_source = asset_config
        else:
            name, _, _, tradfi_source, _ = asset_config
            defi_source = "aster"
        
        tradfi_df = load_from_parquet(name, tradfi_source, interval)
        defi_df = load_from_parquet(name, defi_source, interval)
        merged_df = load_merged_data(name, interval)
        
        tradfi_ok = f"✓ {len(tradfi_df)}" if not tradfi_df.empty else "✗"
        defi_ok = f"✓ {len(defi_df)}" if not defi_df.empty else "✗"
        merged_ok = f"✓ {len(merged_df)}" if not merged_df.empty else "✗"
        
        print(f"  {name:8} | TradFi: {tradfi_ok:>8} | DeFi: {defi_ok:>8} | Merged: {merged_ok:>8}")
    
    print("\n" + "=" * 60)
    print("Stage 1 complete. Merged data ready for analysis.")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Acquisition Pipeline")
    parser.add_argument("--interval", "-i", default="15m", 
                        choices=list(INTERVAL_MAPPING.keys()),
                        help="Data interval (default: 15m)")
    parser.add_argument("--days", "-d", type=int, default=60,
                        help="Number of days to fetch (default: 60)")
    parser.add_argument("--no-locf", action="store_true",
                        help="Skip LOCF application")
    
    args = parser.parse_args()
    
    run_acquisition(
        interval=args.interval,
        days=args.days,
        apply_locf=not args.no_locf
    )
