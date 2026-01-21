#!/usr/bin/env python3
"""
Generate test fixtures for basis arbitrage tests.

Creates sample parquet and CSV files that mirror real data structure.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent


def generate_yahoo_sample():
    """Generate sample Yahoo Finance data (TSLA-like)."""
    np.random.seed(42)
    
    # 3 days of 1-minute data during market hours
    dates = []
    base_date = datetime(2024, 1, 15, 14, 30)  # Start at market open (UTC)
    
    for day in range(3):
        day_start = base_date + timedelta(days=day)
        # Market hours: 14:30 - 21:00 UTC (6.5 hours = 390 minutes)
        for minute in range(390):
            dates.append(day_start + timedelta(minutes=minute))
    
    dates = pd.DatetimeIndex(dates, tz="UTC")
    
    # Generate realistic TSLA-like prices with some trend and noise
    base_price = 250.0
    returns = np.random.normal(0, 0.001, len(dates))
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from prices
    df = pd.DataFrame({
        "open": prices + np.random.uniform(-0.5, 0.5, len(dates)),
        "high": prices + np.random.uniform(0, 1.0, len(dates)),
        "low": prices - np.random.uniform(0, 1.0, len(dates)),
        "close": prices + np.random.uniform(-0.3, 0.3, len(dates)),
        "mid": prices,
        "volume": np.random.randint(10000, 100000, len(dates)).astype(float),
        "source": "yahoo",
        "symbol": "TSLA",
    }, index=dates)
    df.index.name = "timestamp"
    
    return df


def generate_aster_sample():
    """Generate sample Aster data (TSLAUSDT-like)."""
    np.random.seed(43)
    
    # 3 days of 1-minute data (24/7)
    dates = pd.date_range(
        start="2024-01-15 14:30:00",
        periods=3 * 24 * 60,  # 3 days
        freq="1min",
        tz="UTC"
    )
    
    # Generate prices with slight premium over TradFi
    base_price = 250.5  # Slight premium
    returns = np.random.normal(0, 0.001, len(dates))
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "open": prices + np.random.uniform(-0.5, 0.5, len(dates)),
        "high": prices + np.random.uniform(0, 1.0, len(dates)),
        "low": prices - np.random.uniform(0, 1.0, len(dates)),
        "close": prices + np.random.uniform(-0.3, 0.3, len(dates)),
        "mid": prices,
        "volume": np.random.randint(1000, 50000, len(dates)).astype(float),
        "source": "aster",
        "symbol": "TSLAUSDT",
    }, index=dates)
    df.index.name = "timestamp"
    
    return df


def generate_basis_sample():
    """Generate sample basis analysis output."""
    np.random.seed(44)
    
    # 2 days of aligned minute data during market hours
    dates = []
    base_date = datetime(2024, 1, 15, 14, 30)
    
    for day in range(2):
        day_start = base_date + timedelta(days=day)
        for minute in range(390):
            dates.append(day_start + timedelta(minutes=minute))
    
    dates = pd.DatetimeIndex(dates, tz="UTC")
    
    # Generate correlated prices
    tradfi_base = 250.0
    tradfi_returns = np.random.normal(0, 0.001, len(dates))
    tradfi_prices = tradfi_base * np.cumprod(1 + tradfi_returns)
    
    # DeFi prices with basis spread
    basis_bps = np.random.normal(15, 30, len(dates))  # Mean 15 bps, std 30 bps
    defi_prices = tradfi_prices * (1 + basis_bps / 10000)
    
    df = pd.DataFrame({
        "tradfi_mid": tradfi_prices,
        "defi_mid": defi_prices,
        "basis_absolute": defi_prices - tradfi_prices,
        "basis_bps": basis_bps,
        "market_open": True,  # All during market hours
    }, index=dates)
    df.index.name = "timestamp"
    
    return df


def main():
    """Generate all fixtures."""
    print("Generating test fixtures...")
    
    # Yahoo sample
    yahoo_df = generate_yahoo_sample()
    yahoo_path = FIXTURES_DIR / "yahoo_sample.parquet"
    yahoo_df.to_parquet(yahoo_path)
    print(f"  ✓ {yahoo_path.name}: {len(yahoo_df)} records")
    
    # Aster sample
    aster_df = generate_aster_sample()
    aster_path = FIXTURES_DIR / "aster_sample.parquet"
    aster_df.to_parquet(aster_path)
    print(f"  ✓ {aster_path.name}: {len(aster_df)} records")
    
    # Basis sample
    basis_df = generate_basis_sample()
    basis_path = FIXTURES_DIR / "basis_sample.csv"
    basis_df.to_csv(basis_path)
    print(f"  ✓ {basis_path.name}: {len(basis_df)} records")
    
    print("\nFixtures generated successfully!")


if __name__ == "__main__":
    main()
