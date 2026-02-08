"""
OHLCV data validation and gap filling utilities.

Validates downloaded data for:
- Missing timestamps (gaps)
- Data quality (nulls, zeros, outliers)

Applies LOCF (Last Observation Carried Forward) for gaps.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional
import pandas as pd
import numpy as np

# Interval to timedelta mapping
INTERVAL_DELTAS = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
}


@dataclass
class Gap:
    """Represents a gap in the data."""
    start: pd.Timestamp
    end: pd.Timestamp
    missing_bars: int
    
    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass
class ValidationReport:
    """Report from data validation."""
    total_bars: int
    expected_bars: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    interval: str
    
    # Gap analysis
    gap_count: int = 0
    gaps: list[Gap] = field(default_factory=list)
    total_missing_bars: int = 0
    
    # Data quality
    null_count: int = 0
    zero_volume_count: int = 0
    
    # Outliers (optional)
    price_outliers: int = 0
    volume_outliers: int = 0
    
    @property
    def coverage_pct(self) -> float:
        """Percentage of expected bars present."""
        if self.expected_bars == 0:
            return 0.0
        return (self.total_bars / self.expected_bars) * 100
    
    @property
    def is_valid(self) -> bool:
        """Check if data passes basic validation."""
        return self.coverage_pct >= 95.0 and self.null_count == 0
    
    def __str__(self) -> str:
        return (
            f"ValidationReport(\n"
            f"  period: {self.start_time} to {self.end_time}\n"
            f"  interval: {self.interval}\n"
            f"  bars: {self.total_bars:,} / {self.expected_bars:,} ({self.coverage_pct:.1f}%)\n"
            f"  gaps: {self.gap_count} ({self.total_missing_bars:,} missing bars)\n"
            f"  nulls: {self.null_count}, zero_volume: {self.zero_volume_count}\n"
            f")"
        )


def validate_ohlcv(
    df: pd.DataFrame,
    interval: str,
    check_outliers: bool = False,
) -> ValidationReport:
    """
    Validate OHLCV DataFrame for gaps and data quality.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        interval: Expected bar interval (e.g., '1h')
        check_outliers: Whether to check for price/volume outliers
        
    Returns:
        ValidationReport with findings
    """
    if df.empty:
        return ValidationReport(
            total_bars=0,
            expected_bars=0,
            start_time=pd.NaT,
            end_time=pd.NaT,
            interval=interval,
        )
    
    # Ensure sorted
    df = df.sort_index()
    
    start_time = df.index.min()
    end_time = df.index.max()
    
    # Calculate expected bars
    delta = INTERVAL_DELTAS.get(interval)
    if delta is None:
        raise ValueError(f"Unknown interval: {interval}")
    
    expected_bars = int((end_time - start_time) / delta) + 1
    
    # Find gaps
    gaps = []
    total_missing = 0
    
    time_diffs = df.index.to_series().diff()
    gap_mask = time_diffs > delta
    
    for idx in df.index[gap_mask]:
        prev_idx = df.index[df.index.get_loc(idx) - 1]
        gap_duration = idx - prev_idx
        missing_bars = int(gap_duration / delta) - 1
        
        if missing_bars > 0:
            gaps.append(Gap(
                start=prev_idx,
                end=idx,
                missing_bars=missing_bars,
            ))
            total_missing += missing_bars
    
    # Check for nulls
    null_count = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
    
    # Check for zero volume
    zero_volume_count = (df["volume"] == 0).sum() if "volume" in df.columns else 0
    
    report = ValidationReport(
        total_bars=len(df),
        expected_bars=expected_bars,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
        gap_count=len(gaps),
        gaps=gaps,
        total_missing_bars=total_missing,
        null_count=null_count,
        zero_volume_count=zero_volume_count,
    )
    
    # Optional outlier detection
    if check_outliers:
        report.price_outliers = _count_price_outliers(df)
        report.volume_outliers = _count_volume_outliers(df)
    
    return report


def _count_price_outliers(df: pd.DataFrame, threshold: float = 5.0) -> int:
    """Count bars where price change exceeds threshold std devs."""
    returns = df["close"].pct_change()
    std = returns.std()
    if std == 0:
        return 0
    z_scores = (returns - returns.mean()) / std
    return (z_scores.abs() > threshold).sum()


def _count_volume_outliers(df: pd.DataFrame, threshold: float = 5.0) -> int:
    """Count bars where volume exceeds threshold std devs from mean."""
    if "volume" not in df.columns:
        return 0
    volume = df["volume"]
    std = volume.std()
    if std == 0:
        return 0
    z_scores = (volume - volume.mean()) / std
    return (z_scores.abs() > threshold).sum()


def fill_gaps(
    df: pd.DataFrame,
    interval: str,
    method: str = "locf",
) -> pd.DataFrame:
    """
    Fill gaps in OHLCV data using specified method.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        interval: Bar interval (e.g., '1h')
        method: Fill method - 'locf' (last observation carried forward)
        
    Returns:
        DataFrame with gaps filled
    """
    if df.empty:
        return df
    
    delta = INTERVAL_DELTAS.get(interval)
    if delta is None:
        raise ValueError(f"Unknown interval: {interval}")
    
    # Create complete date range
    start = df.index.min()
    end = df.index.max()
    
    # Generate expected timestamps
    freq_map = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1D",
    }
    freq = freq_map.get(interval, interval)
    
    full_index = pd.date_range(start=start, end=end, freq=freq)
    
    # Reindex to full range
    df_filled = df.reindex(full_index)
    
    if method == "locf":
        # Forward fill OHLC (use previous close for all)
        for col in ["open", "high", "low", "close"]:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].ffill()
        
        # For volume/trades, fill with 0 (no activity during gap)
        for col in ["volume", "quote_volume", "trades"]:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(0)
    
    return df_filled


def detect_anomalies(
    df: pd.DataFrame,
    price_threshold: float = 0.1,
    volume_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Detect anomalous bars in OHLCV data.
    
    Args:
        df: OHLCV DataFrame
        price_threshold: Flag if price changes > this fraction (e.g., 0.1 = 10%)
        volume_threshold: Flag if volume > this many std devs from mean
        
    Returns:
        DataFrame with anomaly flags added
    """
    df = df.copy()
    
    # Price spike detection
    df["price_change"] = df["close"].pct_change().abs()
    df["price_anomaly"] = df["price_change"] > price_threshold
    
    # Volume spike detection
    if "volume" in df.columns:
        volume_mean = df["volume"].mean()
        volume_std = df["volume"].std()
        if volume_std > 0:
            df["volume_zscore"] = (df["volume"] - volume_mean) / volume_std
            df["volume_anomaly"] = df["volume_zscore"].abs() > volume_threshold
        else:
            df["volume_zscore"] = 0
            df["volume_anomaly"] = False
    
    # OHLC consistency check (high >= low, etc.)
    df["ohlc_invalid"] = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for OHLCV data.
    
    Returns:
        Dict with summary stats
    """
    if df.empty:
        return {"empty": True}
    
    return {
        "bars": len(df),
        "start": df.index.min().isoformat(),
        "end": df.index.max().isoformat(),
        "days": (df.index.max() - df.index.min()).days + 1,
        "open_price": float(df["open"].iloc[0]),
        "close_price": float(df["close"].iloc[-1]),
        "high_price": float(df["high"].max()),
        "low_price": float(df["low"].min()),
        "total_volume": float(df["volume"].sum()) if "volume" in df.columns else 0,
        "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else 0,
        "price_change_pct": float((df["close"].iloc[-1] / df["open"].iloc[0] - 1) * 100),
    }
