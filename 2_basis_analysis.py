#!/usr/bin/env python3
"""
Stage 2: Basis Analysis

Loads cleaned price data, aligns timestamps, calculates basis metrics,
and generates backtest output for arbitrage analysis.

Output: output/backtest/{asset}_basis.csv
"""

from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ============================================
# Configuration
# ============================================

BASE_DIR = Path(__file__).parent
DATA_CLEANED_DIR = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR = BASE_DIR / "output" / "backtest"

# Assets to analyze: (name, yahoo_label, aster_label)
ASSETS = [
    ("TSLA", "TSLA (Stock)", "TSLAUSDT (Aster)"),
    ("GOLD", "GC=F (Gold Futures)", "XAUUSDT (Aster)"),
]

# NYSE market hours in UTC
NYSE_OPEN_UTC = time(14, 30)   # 9:30 AM ET
NYSE_CLOSE_UTC = time(21, 0)  # 4:00 PM ET


# ============================================
# Data Loading
# ============================================

def load_price_data(asset: str, source: str) -> pd.DataFrame:
    """Load cleaned price data from parquet."""
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_{source}.parquet"
    
    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return pd.DataFrame()
    
    return pd.read_parquet(filepath)


# ============================================
# Basis Calculation Functions
# ============================================

def calculate_basis(tradfi_mid: float, defi_mid: float) -> tuple:
    """
    Calculate basis between TradFi and DeFi prices.
    
    Args:
        tradfi_mid: TradFi mid price
        defi_mid: DeFi mid price
    
    Returns:
        Tuple of (basis_absolute, basis_bps)
        - basis_absolute: DeFi - TradFi in $
        - basis_bps: Basis in basis points (1 bps = 0.01%)
    """
    basis_absolute = defi_mid - tradfi_mid
    
    # Calculate basis points: (defi - tradfi) / tradfi * 10000
    if tradfi_mid != 0:
        basis_bps = (basis_absolute / tradfi_mid) * 10000
    else:
        basis_bps = 0.0
    
    return basis_absolute, basis_bps


def is_market_open(timestamp: pd.Timestamp) -> bool:
    """
    Check if NYSE is open at given UTC timestamp.
    
    NYSE hours: 9:30 AM - 4:00 PM ET (14:30 - 21:00 UTC)
    Only on weekdays (Mon-Fri).
    """
    # Check if weekday (0=Mon, 4=Fri)
    if timestamp.weekday() > 4:
        return False
    
    # Check time
    t = timestamp.time()
    return NYSE_OPEN_UTC <= t < NYSE_CLOSE_UTC


def calculate_basis_for_asset(
    yahoo_df: pd.DataFrame,
    aster_df: pd.DataFrame,
    asset_name: str
) -> pd.DataFrame:
    """
    Calculate basis metrics for an asset.
    
    Args:
        yahoo_df: Yahoo (TradFi) price data
        aster_df: Aster (DeFi) price data
        asset_name: Asset name for logging
    
    Returns:
        DataFrame with basis metrics
    """
    if yahoo_df.empty or aster_df.empty:
        print(f"  Missing data for {asset_name}")
        return pd.DataFrame()
    
    # Align timestamps (inner join - only where both have data)
    # Since LOCF was applied, they should mostly align
    common_idx = yahoo_df.index.intersection(aster_df.index)
    
    if len(common_idx) == 0:
        print(f"  No overlapping timestamps for {asset_name}")
        return pd.DataFrame()
    
    # Create basis DataFrame
    basis_data = []
    
    for ts in common_idx:
        tradfi_mid = yahoo_df.loc[ts, "mid"]
        defi_mid = aster_df.loc[ts, "mid"]
        
        basis_abs, basis_bps = calculate_basis(tradfi_mid, defi_mid)
        market_open = is_market_open(ts)
        
        basis_data.append({
            "timestamp": ts,
            "tradfi_mid": tradfi_mid,
            "defi_mid": defi_mid,
            "basis_absolute": basis_abs,
            "basis_bps": basis_bps,
            "market_open": market_open,
        })
    
    df = pd.DataFrame(basis_data)
    df.set_index("timestamp", inplace=True)
    
    return df


# ============================================
# Statistics Functions
# ============================================

def calculate_basis_statistics(basis_df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for basis data.
    
    Args:
        basis_df: DataFrame with basis metrics
    
    Returns:
        Dict with statistics
    """
    if basis_df.empty:
        return {}
    
    bps = basis_df["basis_bps"]
    
    stats = {
        "count": len(basis_df),
        "mean_bps": bps.mean(),
        "std_bps": bps.std(),
        "min_bps": bps.min(),
        "max_bps": bps.max(),
        "median_bps": bps.median(),
        "p5_bps": bps.quantile(0.05),
        "p95_bps": bps.quantile(0.95),
        "pct_positive": (bps > 0).mean() * 100,
        "pct_gt_10bps": (bps.abs() > 10).mean() * 100,
        "pct_gt_20bps": (bps.abs() > 20).mean() * 100,
        "pct_gt_50bps": (bps.abs() > 50).mean() * 100,
    }
    
    # Market hours vs off-hours comparison
    if "market_open" in basis_df.columns:
        market_hours = basis_df[basis_df["market_open"]]
        off_hours = basis_df[~basis_df["market_open"]]
        
        if len(market_hours) > 0:
            stats["market_hours_mean_bps"] = market_hours["basis_bps"].mean()
            stats["market_hours_std_bps"] = market_hours["basis_bps"].std()
            stats["market_hours_count"] = len(market_hours)
        
        if len(off_hours) > 0:
            stats["off_hours_mean_bps"] = off_hours["basis_bps"].mean()
            stats["off_hours_std_bps"] = off_hours["basis_bps"].std()
            stats["off_hours_count"] = len(off_hours)
    
    return stats


# ============================================
# Mean Reversion Tests
# ============================================

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Tests if the series is stationary (mean-reverting).
    p-value < 0.05 suggests the series is stationary.
    
    Returns:
        Dict with ADF statistic, p-value, and interpretation
    """
    if len(series) < 20:
        return {"error": "Insufficient data"}
    
    # Run ADF test
    result = adfuller(series.dropna(), autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Interpret
    is_stationary = p_value < 0.05
    
    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "critical_1pct": critical_values['1%'],
        "critical_5pct": critical_values['5%'],
        "critical_10pct": critical_values['10%'],
        "is_stationary": is_stationary,
        "interpretation": "Mean-reverting" if is_stationary else "Not stationary"
    }


def calculate_half_life(series: pd.Series) -> dict:
    """
    Calculate half-life of mean reversion using OLS regression.
    
    Uses the Ornstein-Uhlenbeck model:
    y(t) - y(t-1) = lambda * (y(t-1) - mean) + noise
    
    Half-life = -ln(2) / lambda
    
    Returns:
        Dict with half-life in minutes and interpretation
    """
    if len(series) < 20:
        return {"error": "Insufficient data"}
    
    series = series.dropna()
    
    # Lagged series
    lag = series.shift(1)
    delta = series - lag
    
    # Remove NaN
    lag = lag.iloc[1:]
    delta = delta.iloc[1:]
    
    # OLS regression: delta = alpha + beta * lag
    X = add_constant(lag)
    model = OLS(delta, X).fit()
    
    # Lambda (mean reversion speed)
    lambda_coef = model.params.iloc[1]
    
    if lambda_coef >= 0:
        return {
            "half_life_minutes": float('inf'),
            "lambda": lambda_coef,
            "interpretation": "No mean reversion detected"
        }
    
    # Half-life in periods (minutes since we have 1-min data)
    half_life = -np.log(2) / lambda_coef
    
    # Interpret
    if half_life < 60:
        interpretation = f"Fast reversion (~{half_life:.0f} min)"
    elif half_life < 240:
        interpretation = f"Moderate reversion (~{half_life/60:.1f} hrs)"
    else:
        interpretation = f"Slow reversion (~{half_life/60:.1f} hrs)"
    
    return {
        "half_life_minutes": half_life,
        "lambda": lambda_coef,
        "interpretation": interpretation
    }


def calculate_hurst_exponent(series: pd.Series, max_lag: int = 100) -> dict:
    """
    Calculate Hurst exponent to determine mean-reverting vs trending behavior.
    
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    Returns:
        Dict with Hurst exponent and interpretation
    """
    if len(series) < max_lag * 2:
        return {"error": "Insufficient data"}
    
    series = series.dropna().values
    
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        # Calculate variance of lagged differences
        pp = series[lag:]
        pm = series[:-lag]
        tau.append(np.sqrt(np.std(pp - pm)))
    
    # Fit log-log regression
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    
    # Linear regression
    poly = np.polyfit(log_lags, log_tau, 1)
    hurst = poly[0]
    
    # Interpret
    if hurst < 0.4:
        interpretation = "Strongly mean-reverting"
    elif hurst < 0.5:
        interpretation = "Mean-reverting"
    elif hurst < 0.6:
        interpretation = "Random walk"
    else:
        interpretation = "Trending"
    
    return {
        "hurst_exponent": hurst,
        "interpretation": interpretation
    }


def calculate_mean_reversion_stats(basis_df: pd.DataFrame) -> dict:
    """
    Calculate all mean reversion statistics for basis data.
    
    Returns:
        Dict with ADF, half-life, and Hurst results
    """
    if basis_df.empty:
        return {}
    
    bps = basis_df["basis_bps"]
    
    results = {
        "adf": adf_test(bps),
        "half_life": calculate_half_life(bps),
        "hurst": calculate_hurst_exponent(bps),
    }
    
    return results


def print_mean_reversion_stats(mr_stats: dict, asset_name: str):
    """Print formatted mean reversion statistics."""
    if not mr_stats:
        return
    
    print(f"\n  {asset_name} Mean Reversion Analysis")
    print("  " + "-" * 40)
    
    # ADF Test
    adf = mr_stats.get("adf", {})
    if "error" not in adf:
        print(f"  ADF Test:")
        print(f"    Statistic:       {adf['adf_statistic']:.4f}")
        print(f"    p-value:         {adf['p_value']:.4f}")
        print(f"    Critical (5%):   {adf['critical_5pct']:.4f}")
        print(f"    Result:          {adf['interpretation']}")
    
    # Half-life
    hl = mr_stats.get("half_life", {})
    if "error" not in hl:
        print(f"  Half-Life:")
        if hl['half_life_minutes'] != float('inf'):
            print(f"    Value:           {hl['half_life_minutes']:.1f} minutes")
        print(f"    Interpretation:  {hl['interpretation']}")
    
    # Hurst
    hurst = mr_stats.get("hurst", {})
    if "error" not in hurst:
        print(f"  Hurst Exponent:")
        print(f"    Value:           {hurst['hurst_exponent']:.3f}")
        print(f"    Interpretation:  {hurst['interpretation']}")


def print_statistics(stats: dict, asset_name: str):
    """Print formatted statistics."""
    if not stats:
        print(f"  No statistics available for {asset_name}")
        return
    
    print(f"\n  {asset_name} Basis Statistics")
    print("  " + "-" * 40)
    print(f"  Records:           {stats['count']:,}")
    print(f"  Mean Basis:        {stats['mean_bps']:+.2f} bps")
    print(f"  Std Dev:           {stats['std_bps']:.2f} bps")
    print(f"  Min / Max:         {stats['min_bps']:.2f} / {stats['max_bps']:.2f} bps")
    print(f"  Median:            {stats['median_bps']:.2f} bps")
    print(f"  5th / 95th %ile:   {stats['p5_bps']:.2f} / {stats['p95_bps']:.2f} bps")
    print(f"  % Positive:        {stats['pct_positive']:.1f}%")
    print(f"  % > 10 bps:        {stats['pct_gt_10bps']:.1f}%")
    print(f"  % > 20 bps:        {stats['pct_gt_20bps']:.1f}%")
    print(f"  % > 50 bps:        {stats['pct_gt_50bps']:.1f}%")
    
    if "market_hours_mean_bps" in stats:
        print(f"\n  Market Hours ({stats.get('market_hours_count', 0):,} bars):")
        print(f"    Mean: {stats['market_hours_mean_bps']:+.2f} bps, Std: {stats['market_hours_std_bps']:.2f} bps")
    
    if "off_hours_mean_bps" in stats:
        print(f"  Off Hours ({stats.get('off_hours_count', 0):,} bars):")
        print(f"    Mean: {stats['off_hours_mean_bps']:+.2f} bps, Std: {stats['off_hours_std_bps']:.2f} bps")


# ============================================
# Output Functions
# ============================================

def save_basis_to_csv(basis_df: pd.DataFrame, asset: str) -> Path:
    """Save basis data to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath = OUTPUT_DIR / f"{asset.lower()}_basis.csv"
    basis_df.to_csv(filepath)
    
    return filepath


# ============================================
# Main Analysis Pipeline
# ============================================

def run_basis_analysis():
    """Run basis analysis for all configured assets."""
    print("=" * 60)
    print("Stage 2: Basis Analysis")
    print("=" * 60)
    
    all_stats = {}
    
    for asset, yahoo_label, aster_label in ASSETS:
        print(f"\n[{asset}] Analyzing {yahoo_label} vs {aster_label}...")
        
        # Load data
        yahoo_df = load_price_data(asset, "yahoo")
        aster_df = load_price_data(asset, "aster")
        
        if yahoo_df.empty or aster_df.empty:
            print(f"  Skipping {asset} - missing data")
            continue
        
        print(f"  Loaded {len(yahoo_df):,} Yahoo bars, {len(aster_df):,} Aster bars")
        
        # Calculate basis
        basis_df = calculate_basis_for_asset(yahoo_df, aster_df, asset)
        
        if basis_df.empty:
            continue
        
        print(f"  Calculated basis for {len(basis_df):,} aligned bars")
        
        # Calculate statistics
        stats = calculate_basis_statistics(basis_df)
        all_stats[asset] = stats
        
        # Print statistics
        print_statistics(stats, asset)
        
        # Calculate and print mean reversion statistics
        mr_stats = calculate_mean_reversion_stats(basis_df)
        all_stats[asset]["mean_reversion"] = mr_stats
        print_mean_reversion_stats(mr_stats, asset)
        
        # Save to CSV
        csv_path = save_basis_to_csv(basis_df, asset)
        print(f"\n  ✓ Saved: {csv_path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Basis Analysis Summary")
    print("=" * 60)
    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    
    for asset, _, _ in ASSETS:
        if asset in all_stats:
            stats = all_stats[asset]
            mr = stats.get("mean_reversion", {})
            adf_p = mr.get("adf", {}).get("p_value", None)
            hl = mr.get("half_life", {}).get("half_life_minutes", None)
            hurst = mr.get("hurst", {}).get("hurst_exponent", None)
            
            adf_str = f"p={adf_p:.4f}" if adf_p else "N/A"
            hl_str = f"{hl:.0f}m" if hl and hl != float('inf') else "N/A"
            hurst_str = f"H={hurst:.2f}" if hurst else "N/A"
            
            print(f"  {asset:8} | Mean: {stats['mean_bps']:+6.2f} bps | ADF: {adf_str} | Half-life: {hl_str} | {hurst_str}")
    
    print("\n" + "=" * 60)
    print("Stage 2 complete. Run Stage 3 for visualization.")
    print("=" * 60)
    
    return all_stats


if __name__ == "__main__":
    run_basis_analysis()
