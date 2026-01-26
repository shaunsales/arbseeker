#!/usr/bin/env python3
"""
Stage 2: Basis Analysis

Loads merged price data, calculates basis metrics, mean reversion statistics,
volume analysis, and capital sizing for arbitrage strategy.

Input: data/cleaned/{asset}_merged_{interval}.parquet
Output: output/backtest/{asset}_analysis.csv
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

# Assets to analyze: (name, tradfi_label, defi_label)
ASSETS = [
    ("GOLD", "CME Gold Futures (GC1!)", "Aster XAUUSDT"),
    ("GOLD_HL", "CME Gold Futures (GC1!)", "Hyperliquid PAXG"),
]

# Default interval
DEFAULT_INTERVAL = "15m"

# Capital sizing parameters and assumptions
CAPITAL_SIZING = {
    "volume_pct_conservative": 0.01,  # 1% of daily volume
    "volume_pct_moderate": 0.02,      # 2% of daily volume  
    "volume_pct_aggressive": 0.05,    # 5% of daily volume
    "basis_capture_bps": 15,          # Conservative basis capture per trade
    "cme_trading_hours": 23,          # CME open hours per day
    "trade_capture_rate": 0.50,       # % of mean reversion cycles we can capture
    
    # Cost assumptions (per leg, per trade)
    "cme_commission_per_contract": 2.50,  # CME futures commission
    "cme_contract_size_oz": 100,          # 100 oz per GC contract
    "aster_taker_fee_bps": 5,             # Aster taker fee ~5 bps
    "slippage_bps": 2,                    # Estimated slippage per leg
    "funding_rate_daily_bps": 1,          # Avg daily funding cost (perp)
    
    # Margin assumptions
    "cme_margin_pct": 0.10,               # CME initial margin ~10%
    "aster_margin_pct": 0.05,             # Aster margin at 20x = 5%
}


# ============================================
# Data Loading
# ============================================

def load_merged_data(asset: str, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Load merged price data from parquet."""
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_merged_{interval}.parquet"
    
    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return pd.DataFrame()
    
    return pd.read_parquet(filepath)


# ============================================
# Volume & Liquidity Analysis
# ============================================

def calculate_volume_stats(df: pd.DataFrame) -> dict:
    """
    Calculate volume and liquidity statistics.
    
    Args:
        df: Merged DataFrame with volume columns
    
    Returns:
        Dict with volume statistics
    """
    if df.empty:
        return {}
    
    # Filter to tradeable bars (CME open)
    tradeable = df[df["tradfi_market_open"]] if "tradfi_market_open" in df.columns else df
    
    stats = {
        "total_bars": len(df),
        "tradeable_bars": len(tradeable),
        "tradeable_pct": len(tradeable) / len(df) * 100 if len(df) > 0 else 0,
    }
    
    # TradFi volume
    if "tradfi_volume" in df.columns:
        stats["tradfi_avg_volume"] = tradeable["tradfi_volume"].mean()
        stats["tradfi_total_volume"] = tradeable["tradfi_volume"].sum()
    
    if "tradfi_dollar_volume" in df.columns:
        stats["tradfi_avg_dollar_volume"] = tradeable["tradfi_dollar_volume"].mean()
        stats["tradfi_total_dollar_volume"] = tradeable["tradfi_dollar_volume"].sum()
    
    # DeFi volume  
    if "defi_volume" in df.columns:
        stats["defi_avg_volume"] = tradeable["defi_volume"].mean()
        stats["defi_total_volume"] = tradeable["defi_volume"].sum()
    
    if "defi_dollar_volume" in df.columns:
        stats["defi_avg_dollar_volume"] = tradeable["defi_dollar_volume"].mean()
        stats["defi_total_dollar_volume"] = tradeable["defi_dollar_volume"].sum()
    
    # Volume flags
    if "both_have_volume" in df.columns:
        stats["both_volume_bars"] = tradeable["both_have_volume"].sum()
        stats["both_volume_pct"] = tradeable["both_have_volume"].mean() * 100
    
    # Estimate daily volumes
    days_covered = (df.index.max() - df.index.min()).days
    if days_covered > 0:
        stats["days_covered"] = days_covered
        stats["tradfi_daily_dollar_volume"] = stats.get("tradfi_total_dollar_volume", 0) / days_covered
        stats["defi_daily_dollar_volume"] = stats.get("defi_total_dollar_volume", 0) / days_covered
    
    return stats


# ============================================
# Capital Sizing Analysis
# ============================================

def calculate_capital_sizing(
    df: pd.DataFrame,
    volume_stats: dict,
    half_life_minutes: float,
    interval_minutes: int = 15
) -> dict:
    """
    Calculate capital sizing and return opportunities.
    
    ASSUMPTIONS:
    - Position Size: Based on % of daily DeFi volume (limiting factor)
    - Capital for Both Legs:
        * CME leg: 10% margin (initial margin requirement)
        * Aster leg: 5% margin (20x leverage available)
        * Total margin = position_size * (10% + 5%) = 15% of notional
        * BUT we show "notional" capital = position * 2 for clarity
    - PnL Calculation:
        * Gross PnL = position_size * basis_captured_bps * trades_per_day
        * Net PnL = Gross - (CME fees + Aster fees + slippage + funding)
    - Trade Frequency: 50% of theoretical mean reversion cycles
    
    Args:
        df: Merged DataFrame
        volume_stats: Volume statistics dict
        half_life_minutes: Mean reversion half-life in minutes
        interval_minutes: Bar interval in minutes
    
    Returns:
        Dict with capital sizing recommendations
    """
    if df.empty or not volume_stats:
        return {}
    
    # Get daily DeFi volume (limiting factor)
    defi_daily_vol = volume_stats.get("defi_daily_dollar_volume", 0)
    avg_price = df["tradfi_close"].mean() if "tradfi_close" in df.columns else 4500
    
    if defi_daily_vol <= 0:
        return {"error": "No DeFi volume data"}
    
    # Trading cycles per day based on half-life
    cme_hours = CAPITAL_SIZING["cme_trading_hours"]
    capture_rate = CAPITAL_SIZING["trade_capture_rate"]
    
    if half_life_minutes > 0 and half_life_minutes != float('inf'):
        cycles_per_day = (cme_hours * 60) / half_life_minutes
        realistic_trades = cycles_per_day * capture_rate
    else:
        cycles_per_day = 0
        realistic_trades = 0
    
    # Cost calculations
    aster_fee_bps = CAPITAL_SIZING["aster_taker_fee_bps"]
    slippage_bps = CAPITAL_SIZING["slippage_bps"]
    funding_bps = CAPITAL_SIZING["funding_rate_daily_bps"]
    cme_commission = CAPITAL_SIZING["cme_commission_per_contract"]
    cme_contract_oz = CAPITAL_SIZING["cme_contract_size_oz"]
    
    # CME commission in bps (per $100 oz contract at ~$4500/oz = $450K notional)
    cme_fee_bps = (cme_commission / (avg_price * cme_contract_oz)) * 10000
    
    # Total round-trip costs per trade (both legs, entry + exit)
    # CME: 2 trades (open + close) * fee
    # Aster: 2 trades (open + close) * fee
    # Slippage: 4 executions * slippage
    round_trip_cost_bps = (cme_fee_bps * 2) + (aster_fee_bps * 2) + (slippage_bps * 4)
    
    # Margin requirements
    cme_margin = CAPITAL_SIZING["cme_margin_pct"]
    aster_margin = CAPITAL_SIZING["aster_margin_pct"]
    
    results = {
        "defi_daily_volume": defi_daily_vol,
        "half_life_minutes": half_life_minutes,
        "cycles_per_day": cycles_per_day,
        "realistic_trades_per_day": realistic_trades,
        "avg_price": avg_price,
        "assumptions": {
            "basis_capture_bps": CAPITAL_SIZING["basis_capture_bps"],
            "trade_capture_rate": capture_rate,
            "round_trip_cost_bps": round_trip_cost_bps,
            "cme_margin_pct": cme_margin,
            "aster_margin_pct": aster_margin,
            "funding_daily_bps": funding_bps,
        },
        "scenarios": []
    }
    
    # Calculate scenarios
    basis_capture_gross = CAPITAL_SIZING["basis_capture_bps"]
    basis_capture_net = basis_capture_gross - round_trip_cost_bps
    
    for name, vol_pct in [("Conservative (1%)", 0.01), ("Moderate (2%)", 0.02), ("Aggressive (5%)", 0.05)]:
        position_size = defi_daily_vol * vol_pct
        
        # Capital calculations
        notional_both_legs = position_size * 2
        margin_required = position_size * (cme_margin + aster_margin)
        
        # Use moderate trade count
        trades_per_day = min(realistic_trades, 8) if realistic_trades > 0 else 5
        
        # PnL calculations
        daily_pnl_gross = position_size * (basis_capture_gross / 10000) * trades_per_day
        daily_pnl_net = position_size * (basis_capture_net / 10000) * trades_per_day
        daily_funding_cost = position_size * (funding_bps / 10000)
        daily_pnl_after_funding = daily_pnl_net - daily_funding_cost
        
        # Returns on margin (actual capital deployed)
        annual_return_on_margin = (daily_pnl_after_funding * 365) / margin_required * 100 if margin_required > 0 else 0
        # Returns on notional (for comparison)
        annual_return_on_notional = (daily_pnl_after_funding * 365) / notional_both_legs * 100 if notional_both_legs > 0 else 0
        
        results["scenarios"].append({
            "name": name,
            "volume_pct": vol_pct * 100,
            "position_size": position_size,
            "notional_both_legs": notional_both_legs,
            "margin_required": margin_required,
            "trades_per_day": trades_per_day,
            "daily_pnl_gross": daily_pnl_gross,
            "daily_pnl_net": daily_pnl_net,
            "daily_funding_cost": daily_funding_cost,
            "daily_pnl_after_funding": daily_pnl_after_funding,
            "annual_return_on_margin": annual_return_on_margin,
            "annual_return_on_notional": annual_return_on_notional,
        })
    
    # Recommendation (2% rule)
    rec_position = defi_daily_vol * 0.02
    results["recommended_position"] = rec_position
    results["recommended_margin"] = rec_position * (cme_margin + aster_margin)
    results["recommended_notional"] = rec_position * 2
    
    return results


# ============================================
# Threshold-Based Opportunity Analysis
# ============================================

def calculate_threshold_opportunities(
    df: pd.DataFrame,
    thresholds: list = [20, 30, 50, 80, 100],
    capture_rate: float = 0.50,
    round_trip_cost_bps: float = 18.1
) -> dict:
    """
    Analyze profitability at different basis entry thresholds.
    
    This addresses the key insight that while MEAN basis capture may be
    unprofitable, a SELECTIVE strategy entering only on large dislocations
    could be profitable.
    
    ASSUMPTIONS:
    - Only enter when |basis| > threshold
    - Capture X% of the basis on mean reversion
    - Round-trip costs are fixed per trade
    - Analysis uses market hours only (tradeable periods)
    
    Args:
        df: Merged DataFrame with basis_bps and tradfi_market_open columns
        thresholds: List of basis thresholds to analyze (in bps)
        capture_rate: Fraction of basis assumed to be captured (0.5 = 50%)
        round_trip_cost_bps: Total round-trip transaction costs in bps
    
    Returns:
        Dict with threshold analysis results
    """
    if df.empty:
        return {}
    
    # Filter to market hours only (tradeable periods)
    market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
    if market_col in df.columns:
        market_df = df[df[market_col]].copy()
    else:
        market_df = df.copy()
    
    if market_df.empty:
        return {}
    
    abs_basis = market_df["basis_bps"].abs()
    days = (df.index.max() - df.index.min()).days
    if days <= 0:
        days = 1
    
    results = {
        "total_tradeable_bars": len(market_df),
        "days_analyzed": days,
        "capture_rate": capture_rate,
        "round_trip_cost_bps": round_trip_cost_bps,
        "thresholds": [],
        "tail_stats": {
            "max_positive_bps": market_df["basis_bps"].max(),
            "max_negative_bps": market_df["basis_bps"].min(),
            "bars_gt_100": int((abs_basis > 100).sum()),
            "bars_gt_200": int((abs_basis > 200).sum()),
        }
    }
    
    for threshold in thresholds:
        triggered = abs_basis > threshold
        count = int(triggered.sum())
        
        if count == 0:
            results["thresholds"].append({
                "threshold_bps": threshold,
                "count": 0,
                "pct_of_bars": 0,
                "trades_per_day": 0,
                "avg_basis_when_triggered": 0,
                "captured_bps": 0,
                "net_per_trade_bps": -round_trip_cost_bps,
                "profitable": False,
            })
            continue
        
        pct_of_bars = count / len(market_df) * 100
        trades_per_day = count / days
        avg_basis_triggered = float(abs_basis[triggered].mean())
        captured_bps = avg_basis_triggered * capture_rate
        net_per_trade = captured_bps - round_trip_cost_bps
        
        results["thresholds"].append({
            "threshold_bps": threshold,
            "count": count,
            "pct_of_bars": pct_of_bars,
            "trades_per_day": trades_per_day,
            "avg_basis_when_triggered": avg_basis_triggered,
            "captured_bps": captured_bps,
            "net_per_trade_bps": net_per_trade,
            "profitable": net_per_trade > 0,
        })
    
    # Find optimal threshold (best risk/reward)
    profitable = [t for t in results["thresholds"] if t["profitable"]]
    if profitable:
        # Best = highest net per trade with reasonable frequency
        best = max(profitable, key=lambda x: x["net_per_trade_bps"] * min(x["trades_per_day"], 20))
        results["recommended_threshold"] = best["threshold_bps"]
        results["recommended_net_bps"] = best["net_per_trade_bps"]
    else:
        results["recommended_threshold"] = None
        results["recommended_net_bps"] = None
    
    return results


def print_threshold_opportunities(threshold_stats: dict, asset_name: str):
    """Print formatted threshold-based opportunity analysis."""
    if not threshold_stats:
        return
    
    print(f"\n  {asset_name} Threshold-Based Opportunity Analysis")
    print("  " + "-" * 55)
    
    print(f"  ASSUMPTIONS:")
    print(f"    Capture rate:        {threshold_stats['capture_rate']*100:.0f}% of basis when triggered")
    print(f"    Round-trip costs:    {threshold_stats['round_trip_cost_bps']:.1f} bps")
    print(f"    Tradeable bars:      {threshold_stats['total_tradeable_bars']:,}")
    print(f"    Days analyzed:       {threshold_stats['days_analyzed']}")
    
    tail = threshold_stats.get("tail_stats", {})
    print(f"\n  TAIL STATISTICS:")
    print(f"    Max positive basis:  {tail.get('max_positive_bps', 0):.0f} bps")
    print(f"    Max negative basis:  {tail.get('max_negative_bps', 0):.0f} bps")
    print(f"    Bars > 100 bps:      {tail.get('bars_gt_100', 0):,}")
    print(f"    Bars > 200 bps:      {tail.get('bars_gt_200', 0):,}")
    
    print(f"\n  THRESHOLD ANALYSIS:")
    print(f"  {'Threshold':>10} {'Count':>8} {'Freq':>8} {'Avg Basis':>10} {'Captured':>10} {'Net/Trade':>10} {'Status':>10}")
    print("  " + "-" * 68)
    
    for t in threshold_stats.get("thresholds", []):
        status = "✓ PROFIT" if t["profitable"] else "✗ LOSS"
        print(f"  {t['threshold_bps']:>8} bps {t['count']:>8,} {t['trades_per_day']:>6.1f}/d {t['avg_basis_when_triggered']:>8.0f} bps {t['captured_bps']:>8.1f} bps {t['net_per_trade_bps']:>+8.1f} bps {status:>10}")
    
    rec = threshold_stats.get("recommended_threshold")
    if rec:
        print(f"\n  RECOMMENDATION:")
        print(f"    Entry threshold:     > {rec} bps")
        print(f"    Expected net/trade:  {threshold_stats['recommended_net_bps']:+.1f} bps")
        print(f"    Note: Requires backtest validation on minute-level data")
    else:
        print(f"\n  RECOMMENDATION: No profitable threshold found with current assumptions")


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

def save_analysis_to_csv(df: pd.DataFrame, asset: str, interval: str) -> Path:
    """Save analysis data to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath = OUTPUT_DIR / f"{asset.lower()}_analysis_{interval}.csv"
    df.to_csv(filepath)
    
    return filepath


def print_volume_stats(vol_stats: dict, asset_name: str):
    """Print formatted volume statistics."""
    if not vol_stats:
        return
    
    print(f"\n  {asset_name} Volume & Liquidity")
    print("  " + "-" * 40)
    print(f"  Days covered:      {vol_stats.get('days_covered', 'N/A')}")
    print(f"  Total bars:        {vol_stats.get('total_bars', 0):,}")
    print(f"  Tradeable bars:    {vol_stats.get('tradeable_bars', 0):,} ({vol_stats.get('tradeable_pct', 0):.1f}%)")
    
    if "tradfi_daily_dollar_volume" in vol_stats:
        print(f"\n  TradFi (CME):")
        print(f"    Daily volume:    ${vol_stats['tradfi_daily_dollar_volume']/1e6:.1f}M")
    
    if "defi_daily_dollar_volume" in vol_stats:
        print(f"  DeFi (Aster):")
        print(f"    Daily volume:    ${vol_stats['defi_daily_dollar_volume']/1e6:.2f}M")
    
    if "both_volume_pct" in vol_stats:
        print(f"\n  Both have volume:  {vol_stats.get('both_volume_bars', 0):,} bars ({vol_stats['both_volume_pct']:.1f}%)")


def print_capital_sizing(cap_stats: dict, asset_name: str):
    """Print formatted capital sizing recommendations with assumptions."""
    if not cap_stats or "error" in cap_stats:
        return
    
    assumptions = cap_stats.get("assumptions", {})
    
    print(f"\n  {asset_name} Capital Sizing & Returns")
    print("  " + "-" * 55)
    
    # Assumptions section
    print(f"  ASSUMPTIONS:")
    print(f"    Basis capture (gross):   {assumptions.get('basis_capture_bps', 15)} bps/trade")
    print(f"    Round-trip costs:        {assumptions.get('round_trip_cost_bps', 0):.1f} bps")
    print(f"      (CME fees + Aster fees + slippage × 4 executions)")
    print(f"    Trade capture rate:      {assumptions.get('trade_capture_rate', 0.5)*100:.0f}% of cycles")
    print(f"    Daily funding cost:      {assumptions.get('funding_daily_bps', 1)} bps")
    print(f"    CME margin:              {assumptions.get('cme_margin_pct', 0.1)*100:.0f}%")
    print(f"    Aster margin (20x):      {assumptions.get('aster_margin_pct', 0.05)*100:.0f}%")
    
    print(f"\n  MARKET DATA:")
    print(f"    DeFi daily volume:       ${cap_stats['defi_daily_volume']/1e6:.2f}M (limiting factor)")
    print(f"    Avg gold price:          ${cap_stats.get('avg_price', 4500):,.0f}/oz")
    print(f"    Half-life:               {cap_stats['half_life_minutes']:.0f} min")
    print(f"    Cycles/day:              {cap_stats['cycles_per_day']:.0f}")
    print(f"    Realistic trades:        {cap_stats['realistic_trades_per_day']:.0f}/day")
    
    print(f"\n  SCENARIOS:")
    print(f"  {'Strategy':<18} {'Position':>10} {'Margin':>10} {'PnL Gross':>10} {'PnL Net':>10} {'ROI/Margin':>12}")
    print("  " + "-" * 72)
    
    for s in cap_stats.get("scenarios", []):
        print(f"  {s['name']:<18} ${s['position_size']:>8,.0f} ${s['margin_required']:>8,.0f} ${s['daily_pnl_gross']:>8,.0f} ${s['daily_pnl_after_funding']:>8,.0f} {s['annual_return_on_margin']:>10.0f}%")
    
    print(f"\n  RECOMMENDATION (2% volume rule):")
    print(f"    Position size:   ${cap_stats['recommended_position']:,.0f}")
    print(f"    Margin required: ${cap_stats['recommended_margin']:,.0f} (actual capital needed)")
    print(f"    Notional:        ${cap_stats['recommended_notional']:,.0f} (both legs)")


# ============================================
# Main Analysis Pipeline
# ============================================

def run_basis_analysis(interval: str = DEFAULT_INTERVAL):
    """
    Run basis analysis for all configured assets using merged data.
    
    Args:
        interval: Data interval (e.g., '15m')
    """
    print("=" * 60)
    print("Stage 2: Basis Analysis")
    print("=" * 60)
    print(f"\nInterval: {interval}")
    
    # Get interval in minutes for half-life calculation
    interval_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    interval_minutes = interval_map.get(interval, 15)
    
    all_results = {}
    
    for asset, tradfi_label, defi_label in ASSETS:
        print(f"\n[{asset}] Analyzing {tradfi_label} vs {defi_label}...")
        
        # Load merged data
        df = load_merged_data(asset, interval)
        
        if df.empty:
            print(f"  Skipping {asset} - no merged data found")
            continue
        
        print(f"  Loaded {len(df):,} merged bars")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Store results
        asset_results = {"data": df}
        
        # 1. Basis Statistics
        stats = calculate_basis_statistics(df)
        asset_results["basis_stats"] = stats
        print_statistics(stats, asset)
        
        # 2. Mean Reversion Analysis
        mr_stats = calculate_mean_reversion_stats(df)
        asset_results["mean_reversion"] = mr_stats
        print_mean_reversion_stats(mr_stats, asset)
        
        # Get half-life for capital sizing
        half_life = mr_stats.get("half_life", {}).get("half_life_minutes", 60)
        if half_life == float('inf'):
            half_life = 60  # Default fallback
        
        # Adjust half-life for interval (original calc assumes 1m data)
        half_life_adjusted = half_life * interval_minutes
        
        # 3. Volume Statistics
        vol_stats = calculate_volume_stats(df)
        asset_results["volume_stats"] = vol_stats
        print_volume_stats(vol_stats, asset)
        
        # 4. Capital Sizing (Mean-Based)
        cap_stats = calculate_capital_sizing(df, vol_stats, half_life_adjusted, interval_minutes)
        asset_results["capital_sizing"] = cap_stats
        print_capital_sizing(cap_stats, asset)
        
        # 5. Threshold-Based Opportunity Analysis
        # Get round-trip cost from capital sizing assumptions
        rt_cost = cap_stats.get("assumptions", {}).get("round_trip_cost_bps", 18.1)
        threshold_stats = calculate_threshold_opportunities(df, round_trip_cost_bps=rt_cost)
        asset_results["threshold_opportunities"] = threshold_stats
        print_threshold_opportunities(threshold_stats, asset)
        
        # 6. Save analysis CSV
        csv_path = save_analysis_to_csv(df, asset, interval)
        print(f"\n  ✓ Saved: {csv_path.name}")
        
        all_results[asset] = asset_results
    
    # Summary
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    
    for asset, _, _ in ASSETS:
        if asset in all_results:
            r = all_results[asset]
            stats = r.get("basis_stats", {})
            mr = r.get("mean_reversion", {})
            vol = r.get("volume_stats", {})
            cap = r.get("capital_sizing", {})
            thresh = r.get("threshold_opportunities", {})
            
            hl = mr.get("half_life", {}).get("half_life_minutes", None)
            hl_adj = hl * interval_minutes if hl and hl != float('inf') else None
            
            print(f"\n  {asset}:")
            print(f"    Basis:       mean={stats.get('mean_bps', 0):+.1f} bps, std={stats.get('std_bps', 0):.1f} bps")
            print(f"    Half-life:   {hl_adj:.0f} min" if hl_adj else "    Half-life:   N/A")
            print(f"    DeFi volume: ${vol.get('defi_daily_dollar_volume', 0)/1e6:.2f}M/day")
            print(f"    Max capital: ${cap.get('recommended_margin', 0):,.0f} (margin)")
            
            # Threshold recommendation
            rec_thresh = thresh.get("recommended_threshold")
            if rec_thresh:
                rec_net = thresh.get("recommended_net_bps", 0)
                print(f"    Threshold:   >{rec_thresh} bps entry → +{rec_net:.1f} bps/trade (requires backtest)")
            else:
                print(f"    Threshold:   No profitable threshold found")
    
    print("\n" + "=" * 60)
    print("Stage 2 complete. Run Stage 3 for visualization.")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basis Analysis Pipeline")
    parser.add_argument("--interval", "-i", default=DEFAULT_INTERVAL,
                        help=f"Data interval (default: {DEFAULT_INTERVAL})")
    
    args = parser.parse_args()
    
    run_basis_analysis(interval=args.interval)
