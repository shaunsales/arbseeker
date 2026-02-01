"""
Market hours utilities.

For TradFi data (Yahoo), market_open is determined by volume:
- volume > 0 means market was open
- volume = 0 means market was closed (LOCF prices)

For crypto (Hyperliquid, Binance), market_open is always True (24/7).

This approach is more reliable than calculating schedules because:
- Automatically handles holidays and special closures
- No timezone conversion errors
- Ground truth from actual trading activity
"""

from typing import Optional
import pandas as pd


# Close buffer for position management (minutes before market close)
DEFAULT_CLOSE_BUFFER_MINUTES = 30


def add_market_open_from_volume(
    df: pd.DataFrame,
    volume_column: str = "volume",
    column_name: str = "market_open",
) -> pd.DataFrame:
    """
    Add market_open column based on volume.
    
    For TradFi markets: market_open = (volume > 0)
    This reliably detects closed markets, holidays, etc.
    
    Args:
        df: DataFrame with volume column
        volume_column: Name of volume column
        column_name: Name of market_open column to add
        
    Returns:
        DataFrame with market_open column added
    """
    df = df.copy()
    if volume_column in df.columns:
        df[column_name] = df[volume_column] > 0
    else:
        # No volume data, assume always open
        df[column_name] = True
    return df


def add_market_open_always(
    df: pd.DataFrame,
    column_name: str = "market_open",
) -> pd.DataFrame:
    """
    Add market_open column that's always True.
    
    For 24/7 crypto markets.
    """
    df = df.copy()
    df[column_name] = True
    return df


def detect_near_close(
    df: pd.DataFrame,
    market_open_column: str = "market_open",
    buffer_minutes: int = DEFAULT_CLOSE_BUFFER_MINUTES,
    interval_minutes: int = 15,
    column_name: str = "near_close",
) -> pd.DataFrame:
    """
    Add near_close column that's True when market is about to close.
    
    Detects transitions from market_open=True to market_open=False
    and marks the buffer_minutes before that transition.
    
    Args:
        df: DataFrame with market_open column
        market_open_column: Name of market_open column
        buffer_minutes: Minutes before close to flag
        interval_minutes: Bar interval in minutes
        column_name: Name of near_close column to add
        
    Returns:
        DataFrame with near_close column added
    """
    df = df.copy()
    
    if market_open_column not in df.columns:
        df[column_name] = False
        return df
    
    # Number of bars in buffer
    buffer_bars = max(1, buffer_minutes // interval_minutes)
    
    # Find where market transitions from open to closed
    market_open = df[market_open_column].astype(int)
    closes_next = (market_open.shift(-1) == 0) & (market_open == 1)
    
    # Mark buffer_bars before each close
    near_close = pd.Series(False, index=df.index)
    for i in range(buffer_bars):
        near_close = near_close | closes_next.shift(-i).fillna(False)
    
    df[column_name] = near_close
    return df


def get_interval_minutes(interval: str) -> int:
    """Convert interval string to minutes."""
    multipliers = {"m": 1, "h": 60, "d": 1440}
    
    for suffix, mult in multipliers.items():
        if interval.endswith(suffix):
            try:
                return int(interval[:-1]) * mult
            except ValueError:
                pass
    
    # Default fallback
    return 15
