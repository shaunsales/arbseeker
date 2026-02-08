"""
Basis Data Builder

Creates pre-computed basis files with one base venue and multiple quote venues.
Basis files are stored at: data/basis/{base_ticker}/{interval}/{period}.parquet

Schema:
- timestamp (DatetimeIndex, UTC) - aligned to bar open
- base_price (float64) - price from base venue
- {quote_venue}_price (float64) - price from each quote venue
- {quote_venue}_basis_abs (float64) - quote_price - base_price
- {quote_venue}_basis_bps (float64) - (quote - base) / base * 10000
- data_quality (str) - "ok", "base_gap", "quote_gap", "{venue}_stale"
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from core.data.storage import load_ohlcv, list_available_periods, DATA_DIR


BASIS_DIR = DATA_DIR / "basis"


@dataclass
class BasisSpec:
    """Specification for a basis file."""
    base_venue: str
    base_market: str
    base_ticker: str
    quote_venues: list[dict]  # [{venue, market, ticker, name}, ...]
    interval: str
    periods: list[str]  # ["2025-01", "2025-02", ...] or ["2025"]
    price_column: str = "close"  # Which price to use for basis calculation
    ffill_limit: Optional[int] = None  # Max bars to forward-fill (None = auto from interval)


@dataclass 
class BasisResult:
    """Result of basis file creation."""
    path: Path
    bars: int
    base_ticker: str
    quote_venues: list[str]
    coverage_pct: float
    start: str
    end: str


def create_basis_file(spec: BasisSpec, save: bool = True) -> tuple[pd.DataFrame, Optional[BasisResult]]:
    """
    Create a basis file from base and quote venue data.
    
    Args:
        spec: BasisSpec defining the base/quote venues and periods
        save: Whether to save the result to disk
        
    Returns:
        Tuple of (DataFrame, BasisResult) or (DataFrame, None) if not saved
    """
    # Load base data - try specified periods first, fall back to all available
    base_df = _load_data_flexible(
        spec.base_venue, spec.base_market, spec.base_ticker, 
        spec.interval, spec.periods
    )
    
    if base_df is None or len(base_df) == 0:
        raise ValueError(f"No base data found for {spec.base_venue}/{spec.base_market}/{spec.base_ticker}")
    
    # Start building result DataFrame
    result = pd.DataFrame(index=base_df.index)
    result["base_price"] = base_df[spec.price_column]
    
    # Track data quality
    quality_flags = pd.Series("ok", index=result.index)
    
    # Detect gaps in base data
    expected_freq = _interval_to_freq(spec.interval)
    if expected_freq:
        base_gaps = base_df.index.to_series().diff() > pd.Timedelta(expected_freq) * 1.5
        quality_flags = quality_flags.where(~base_gaps, "base_gap")
    
    # Determine forward-fill limit (bars)
    ffill_limit = spec.ffill_limit
    if ffill_limit is None:
        # Default: 1 hour worth of bars
        ffill_limit = _ffill_limit_for_interval(spec.interval)

    # Load and align each quote venue
    for quote in spec.quote_venues:
        venue_name = quote.get("name", quote["venue"])
        
        quote_df = _load_data_flexible(
            quote["venue"], quote["market"], quote["ticker"],
            spec.interval, spec.periods
        )
        
        if quote_df is None or len(quote_df) == 0:
            print(f"Warning: No data for quote venue {venue_name}")
            continue
        
        # Align quote data to base timestamps
        raw_aligned = quote_df[spec.price_column].reindex(result.index)
        
        # Forward-fill sparse quote data (last traded price is still valid)
        aligned_quote = raw_aligned.ffill(limit=ffill_limit)
        
        # Track data quality per bar
        has_real_data = raw_aligned.notna()
        has_filled_data = aligned_quote.notna() & ~has_real_data
        has_no_data = aligned_quote.isna()
        
        # Compute staleness: consecutive bars since last real data point
        # Groups of consecutive non-real bars get incrementing counts
        not_real = ~has_real_data
        groups = has_real_data.cumsum()
        staleness = not_real.groupby(groups).cumsum().astype(int)
        result[f"{venue_name}_staleness"] = staleness
        
        # Add price column (with ffill applied)
        result[f"{venue_name}_price"] = aligned_quote
        
        # Calculate basis
        result[f"{venue_name}_basis_abs"] = aligned_quote - result["base_price"]
        result[f"{venue_name}_basis_bps"] = (
            (aligned_quote - result["base_price"]) / result["base_price"] * 10000
        )
        
        # Update quality flags
        quality_flags = quality_flags.where(~has_filled_data, f"{venue_name}_ffill")
        quality_flags = quality_flags.where(~has_no_data, f"{venue_name}_stale")
    
    result["data_quality"] = quality_flags
    
    # Drop rows where base price is NaN
    result = result.dropna(subset=["base_price"])
    
    if len(result) == 0:
        raise ValueError("No overlapping data between base and quote venues")
    
    # Calculate coverage
    ok_count = (result["data_quality"] == "ok").sum()
    coverage_pct = ok_count / len(result) * 100
    
    # Save if requested
    basis_result = None
    if save:
        path = _get_basis_path(spec.base_ticker, spec.interval, spec.periods)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path)
        
        quote_names = [q.get("name", q["venue"]) for q in spec.quote_venues]
        basis_result = BasisResult(
            path=path,
            bars=len(result),
            base_ticker=spec.base_ticker,
            quote_venues=quote_names,
            coverage_pct=coverage_pct,
            start=result.index.min().isoformat(),
            end=result.index.max().isoformat(),
        )
    
    return result, basis_result


def load_basis(
    base_ticker: str,
    interval: str,
    periods: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load a basis file.
    
    Args:
        base_ticker: The base ticker name
        interval: Data interval (e.g., "1h", "15m")
        periods: Optional list of periods to load
        
    Returns:
        DataFrame with basis data or None if not found
    """
    basis_path = BASIS_DIR / base_ticker / interval
    
    if not basis_path.exists():
        return None
    
    # Find available periods
    available = [f.stem for f in basis_path.glob("*.parquet")]
    
    if not available:
        return None
    
    # Filter to requested periods
    if periods:
        to_load = [p for p in periods if p in available]
    else:
        to_load = available
    
    if not to_load:
        return None
    
    # Load and concatenate
    dfs = []
    for period in sorted(to_load):
        path = basis_path / f"{period}.parquet"
        df = pd.read_parquet(path)
        dfs.append(df)
    
    if len(dfs) == 1:
        return dfs[0]
    
    return pd.concat(dfs).sort_index()


def find_overlapping_periods(
    base_venue: str, base_market: str, base_ticker: str,
    quote_venue: str, quote_market: str, quote_ticker: str,
    interval: str,
) -> dict:
    """
    Find overlapping time periods between base and quote data.
    
    Returns:
        Dict with 'base_periods', 'quote_periods', 'overlap' info
    """
    base_periods = list_available_periods(base_venue, base_market, base_ticker, interval)
    quote_periods = list_available_periods(quote_venue, quote_market, quote_ticker, interval)
    
    if not base_periods or not quote_periods:
        return {
            "base_periods": base_periods or [],
            "quote_periods": quote_periods or [],
            "overlap_start": None,
            "overlap_end": None,
            "has_overlap": False,
        }
    
    # Get date ranges for each
    base_start, base_end = _periods_to_date_range(base_periods)
    quote_start, quote_end = _periods_to_date_range(quote_periods)
    
    # Find overlap
    overlap_start = max(base_start, quote_start) if base_start and quote_start else None
    overlap_end = min(base_end, quote_end) if base_end and quote_end else None
    
    has_overlap = overlap_start and overlap_end and overlap_start < overlap_end
    
    return {
        "base_periods": base_periods,
        "quote_periods": quote_periods,
        "base_range": (base_start.isoformat() if base_start else None, 
                       base_end.isoformat() if base_end else None),
        "quote_range": (quote_start.isoformat() if quote_start else None,
                        quote_end.isoformat() if quote_end else None),
        "overlap_start": overlap_start.isoformat() if overlap_start else None,
        "overlap_end": overlap_end.isoformat() if overlap_end else None,
        "has_overlap": has_overlap,
    }


def list_basis_files() -> dict:
    """
    List all available basis files.
    
    Returns:
        Nested dict: {base_ticker: {interval: [periods]}}
    """
    result = {}
    
    if not BASIS_DIR.exists():
        return result
    
    for ticker_path in BASIS_DIR.iterdir():
        if not ticker_path.is_dir():
            continue
        ticker = ticker_path.name
        result[ticker] = {}
        
        for interval_path in ticker_path.iterdir():
            if not interval_path.is_dir():
                continue
            interval = interval_path.name
            
            periods = [f.stem for f in interval_path.glob("*.parquet")]
            if periods:
                result[ticker][interval] = sorted(periods)
    
    return result


def _get_basis_path(base_ticker: str, interval: str, periods: list[str]) -> Path:
    """Generate path for basis file based on periods."""
    # If single period, use that as filename
    if len(periods) == 1:
        filename = f"{periods[0]}.parquet"
    else:
        # Multiple periods - use range
        sorted_periods = sorted(periods)
        filename = f"{sorted_periods[0]}_{sorted_periods[-1]}.parquet"
    
    return BASIS_DIR / base_ticker / interval / filename


def _load_data_flexible(
    venue: str, market: str, ticker: str, interval: str, periods: list[str]
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data with flexible period handling.
    
    Handles mismatches between requested periods (e.g., "2025-10") and 
    stored periods (e.g., "2025" yearly file). Loads all available data
    and filters by date range.
    """
    # First try loading with specified periods
    try:
        df = load_ohlcv(venue, market, ticker, interval, periods=periods)
        if df is not None and len(df) > 0:
            return df
    except FileNotFoundError:
        pass
    
    # Fall back to loading all available data
    available = list_available_periods(venue, market, ticker, interval)
    if not available:
        return None
    
    df = load_ohlcv(venue, market, ticker, interval, periods=available)
    if df is None or len(df) == 0:
        return None
    
    # Filter to requested date range based on periods
    start_date, end_date = _periods_to_date_range(periods)
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    return df if len(df) > 0 else None


def _periods_to_date_range(periods: list[str]) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Convert period strings to start/end date range."""
    if not periods:
        return None, None
    
    sorted_periods = sorted(periods)
    first, last = sorted_periods[0], sorted_periods[-1]
    
    # Parse start date
    if len(first) == 4:  # Year: "2025"
        start = pd.Timestamp(f"{first}-01-01", tz="UTC")
    elif len(first) == 7:  # Month: "2025-01"
        start = pd.Timestamp(f"{first}-01", tz="UTC")
    else:
        start = None
    
    # Parse end date (end of period)
    if len(last) == 4:  # Year: "2025"
        end = pd.Timestamp(f"{int(last)+1}-01-01", tz="UTC") - pd.Timedelta(seconds=1)
    elif len(last) == 7:  # Month: "2025-12"
        year, month = int(last[:4]), int(last[5:7])
        if month == 12:
            end = pd.Timestamp(f"{year+1}-01-01", tz="UTC") - pd.Timedelta(seconds=1)
        else:
            end = pd.Timestamp(f"{year}-{month+1:02d}-01", tz="UTC") - pd.Timedelta(seconds=1)
    else:
        end = None
    
    return start, end


def _interval_to_freq(interval: str) -> Optional[str]:
    """Convert interval string to pandas frequency."""
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    return mapping.get(interval)


def _ffill_limit_for_interval(interval: str) -> int:
    """Default forward-fill limit in bars (approx 1 hour of data)."""
    limits = {
        "1m": 60,     # 60 bars = 1 hour
        "5m": 12,     # 12 bars = 1 hour
        "15m": 4,     # 4 bars = 1 hour
        "30m": 2,     # 2 bars = 1 hour
        "1h": 6,      # 6 bars = 6 hours
        "4h": 3,      # 3 bars = 12 hours
        "1d": 1,      # 1 bar = 1 day
    }
    return limits.get(interval, 60)
