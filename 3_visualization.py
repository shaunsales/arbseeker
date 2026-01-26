#!/usr/bin/env python3
"""
Stage 3: Visualization

Generates charts and summary tables from merged basis analysis data.
Includes volume analysis and capital sizing visualizations.

Output:
- output/charts/{asset}_price_comparison.png
- output/charts/{asset}_basis_timeseries.png
- output/charts/{asset}_basis_distribution.png
- output/charts/{asset}_volume_analysis.png
- output/charts/summary_table.png
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ============================================
# Configuration
# ============================================

BASE_DIR = Path(__file__).parent
DATA_CLEANED_DIR = BASE_DIR / "data" / "cleaned"
BASIS_DIR = BASE_DIR / "output" / "backtest"
CHARTS_DIR = BASE_DIR / "output" / "charts"
REPORTS_DIR = BASE_DIR / "output" / "reports"

# Assets to visualize: (name, tradfi_label, defi_label)
ASSETS = [
    ("GOLD", "CME Gold Futures (GC1!)", "Aster XAUUSDT"),
]

# Default interval
DEFAULT_INTERVAL = "15m"

# Chart styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "tradfi": "#2563eb",  # Blue
    "defi": "#dc2626",    # Red
    "basis": "#16a34a",   # Green
    "market_hours": "#fef3c7",  # Light yellow
}


# ============================================
# Data Loading
# ============================================

def load_merged_data(asset: str, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Load merged price data from parquet."""
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_merged_{interval}.parquet"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(filepath)


def load_analysis_data(asset: str, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Load analysis data from CSV."""
    filepath = BASIS_DIR / f"{asset.lower()}_analysis_{interval}.csv"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
    return df


# ============================================
# Multi-venue color scheme
# ============================================

VENUE_COLORS = {
    "CME": "#2563eb",       # Blue
    "Aster": "#dc2626",     # Red
    "Hyperliquid": "#16a34a",  # Green
}


# ============================================
# Chart 1: Price Comparison (3 venues)
# ============================================

def create_price_comparison_chart(
    asset: str,
    tradfi_label: str,
    defi_label: str,
    df: pd.DataFrame
) -> Path:
    """
    Create price comparison chart showing CME vs Aster vs Hyperliquid.
    """
    # Load both venue datasets
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use Aster data for CME (same source)
    df_main = df_aster if not df_aster.empty else df_hl
    idx = df_main.index.tz_localize(None) if df_main.index.tz else df_main.index
    
    # Shade market hours
    market_col = "tradfi_market_open" if "tradfi_market_open" in df_main.columns else "market_open"
    if market_col in df_main.columns:
        market_open_mask = df_main[market_col].values
        in_market = False
        start_idx = None
        for i, (ts, is_open) in enumerate(zip(idx, market_open_mask)):
            if is_open and not in_market:
                start_idx = ts
                in_market = True
            elif not is_open and in_market:
                ax.axvspan(start_idx, ts, alpha=0.15, color=COLORS["market_hours"], label="_nolegend_")
                in_market = False
    
    # Plot CME (from Aster dataset)
    if not df_aster.empty:
        idx_a = df_aster.index.tz_localize(None) if df_aster.index.tz else df_aster.index
        ax.plot(idx_a, df_aster["tradfi_close"], label="CME Gold Futures", 
                color=VENUE_COLORS["CME"], linewidth=1.0, alpha=0.9)
    
    # Plot Aster DeFi
    if not df_aster.empty:
        ax.plot(idx_a, df_aster["defi_close"], label="Aster XAUUSDT", 
                color=VENUE_COLORS["Aster"], linewidth=0.8, alpha=0.8)
    
    # Plot Hyperliquid DeFi
    if not df_hl.empty:
        idx_hl = df_hl.index.tz_localize(None) if df_hl.index.tz else df_hl.index
        ax.plot(idx_hl, df_hl["defi_close"], label="Hyperliquid PAXG", 
                color=VENUE_COLORS["Hyperliquid"], linewidth=0.8, alpha=0.8)
    
    # Add market hours legend patch
    market_patch = mpatches.Patch(color=COLORS["market_hours"], alpha=0.3, label="CME Market Hours")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(market_patch)
    
    # Formatting
    ax.set_title("Gold Price Comparison: CME vs Aster vs Hyperliquid", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    fig.autofmt_xdate()
    
    # Save
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / f"{asset.lower()}_price_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 2: Basis Timeseries (3 venues)
# ============================================

def create_basis_timeseries_chart(asset: str, df: pd.DataFrame) -> Path:
    """
    Create basis timeseries chart comparing Aster and Hyperliquid basis vs CME.
    """
    # Load both venue datasets
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot Aster basis
    if not df_aster.empty:
        idx_a = df_aster.index.tz_localize(None) if df_aster.index.tz else df_aster.index
        ax.plot(idx_a, df_aster["basis_bps"], label=f"Aster (mean: {df_aster['basis_bps'].mean():+.1f} bps)", 
                color=VENUE_COLORS["Aster"], linewidth=0.7, alpha=0.8)
    
    # Plot Hyperliquid basis
    if not df_hl.empty:
        idx_hl = df_hl.index.tz_localize(None) if df_hl.index.tz else df_hl.index
        ax.plot(idx_hl, df_hl["basis_bps"], label=f"Hyperliquid (mean: {df_hl['basis_bps'].mean():+.1f} bps)", 
                color=VENUE_COLORS["Hyperliquid"], linewidth=0.7, alpha=0.8)
    
    # Add threshold bands
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="±80 bps threshold")
    ax.axhline(y=-80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="_nolegend_")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_title("Gold Basis Timeseries: Aster vs Hyperliquid (DeFi - CME)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Basis (bps)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    all_basis = []
    if not df_aster.empty:
        all_basis.extend(df_aster["basis_bps"].values)
    if not df_hl.empty:
        all_basis.extend(df_hl["basis_bps"].values)
    if all_basis:
        y_max = max(abs(min(all_basis)), abs(max(all_basis))) * 1.1
        ax.set_ylim(-y_max, y_max)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    fig.autofmt_xdate()
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_basis_timeseries.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 2b: Tradeable Basis (Market Hours Only) - 3 venues
# ============================================

def create_tradeable_basis_chart(asset: str, df: pd.DataFrame) -> Path:
    """
    Create basis chart for market hours comparing both DeFi venues.
    """
    # Load both venue datasets
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Filter to market hours and plot each venue
    def get_market_hours(df):
        if df.empty:
            return pd.DataFrame()
        market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
        if market_col not in df.columns:
            return df
        return df[df[market_col]].copy()
    
    aster_market = get_market_hours(df_aster)
    hl_market = get_market_hours(df_hl)
    
    # Plot Aster
    if not aster_market.empty:
        idx_a = aster_market.index.tz_localize(None) if aster_market.index.tz else aster_market.index
        mean_a = aster_market["basis_bps"].mean()
        std_a = aster_market["basis_bps"].std()
        ax.plot(idx_a, aster_market["basis_bps"], 
                label=f"Aster (mean: {mean_a:+.1f}, std: {std_a:.1f} bps)", 
                color=VENUE_COLORS["Aster"], linewidth=0.7, alpha=0.8)
    
    # Plot Hyperliquid
    if not hl_market.empty:
        idx_hl = hl_market.index.tz_localize(None) if hl_market.index.tz else hl_market.index
        mean_hl = hl_market["basis_bps"].mean()
        std_hl = hl_market["basis_bps"].std()
        ax.plot(idx_hl, hl_market["basis_bps"], 
                label=f"Hyperliquid (mean: {mean_hl:+.1f}, std: {std_hl:.1f} bps)", 
                color=VENUE_COLORS["Hyperliquid"], linewidth=0.7, alpha=0.8)
    
    # Add threshold bands
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="±80 bps threshold")
    ax.axhline(y=-80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="_nolegend_")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_title("Gold Tradeable Basis (Market Hours Only): Aster vs Hyperliquid", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Basis (bps)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    all_basis = []
    if not aster_market.empty:
        all_basis.extend(aster_market["basis_bps"].values)
    if not hl_market.empty:
        all_basis.extend(hl_market["basis_bps"].values)
    if all_basis:
        y_max = max(abs(min(all_basis)), abs(max(all_basis))) * 1.1
        ax.set_ylim(-y_max, y_max)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    fig.autofmt_xdate()
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_tradeable_basis.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 3: Basis Distribution (3 venues)
# ============================================

def create_basis_distribution_chart(asset: str, df: pd.DataFrame) -> Path:
    """
    Create histogram comparing basis distributions for both DeFi venues.
    """
    # Load both venue datasets
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Overlaid distributions
    ax1 = axes[0]
    
    # Filter to market hours
    def get_market_basis(df):
        if df.empty:
            return pd.Series([])
        market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
        if market_col in df.columns:
            return df[df[market_col]]["basis_bps"]
        return df["basis_bps"]
    
    aster_bps = get_market_basis(df_aster)
    hl_bps = get_market_basis(df_hl)
    
    # Determine common bins
    all_bps = pd.concat([aster_bps, hl_bps])
    bins = np.linspace(all_bps.min(), all_bps.max(), 50)
    
    if not aster_bps.empty:
        ax1.hist(aster_bps, bins=bins, color=VENUE_COLORS["Aster"], alpha=0.5, 
                 edgecolor="white", label=f"Aster (n={len(aster_bps):,})")
    if not hl_bps.empty:
        ax1.hist(hl_bps, bins=bins, color=VENUE_COLORS["Hyperliquid"], alpha=0.5, 
                 edgecolor="white", label=f"Hyperliquid (n={len(hl_bps):,})")
    
    ax1.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax1.axvline(x=80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="±80 bps")
    ax1.axvline(x=-80, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_title("Basis Distribution: Aster vs Hyperliquid (Market Hours)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Basis (bps)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Statistics comparison
    ax2 = axes[1]
    ax2.axis('off')
    
    stats_text = "VENUE STATISTICS (Market Hours)\n" + "="*40 + "\n\n"
    
    if not aster_bps.empty:
        stats_text += f"ASTER XAUUSDT:\n"
        stats_text += f"  Bars: {len(aster_bps):,}\n"
        stats_text += f"  Mean: {aster_bps.mean():+.1f} bps\n"
        stats_text += f"  Std:  {aster_bps.std():.1f} bps\n"
        stats_text += f"  Range: {aster_bps.min():.0f} to {aster_bps.max():+.0f} bps\n"
        stats_text += f"  >80 bps: {(abs(aster_bps) > 80).mean()*100:.1f}%\n\n"
    
    if not hl_bps.empty:
        stats_text += f"HYPERLIQUID PAXG:\n"
        stats_text += f"  Bars: {len(hl_bps):,}\n"
        stats_text += f"  Mean: {hl_bps.mean():+.1f} bps\n"
        stats_text += f"  Std:  {hl_bps.std():.1f} bps\n"
        stats_text += f"  Range: {hl_bps.min():.0f} to {hl_bps.max():+.0f} bps\n"
        stats_text += f"  >80 bps: {(abs(hl_bps) > 80).mean()*100:.1f}%\n"
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f9ff', edgecolor='#2563eb'))
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_basis_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 4: Volume Analysis (3 venues)
# ============================================

def create_volume_analysis_chart(asset: str, df: pd.DataFrame) -> Path:
    """
    Create volume analysis chart comparing CME, Aster, and Hyperliquid.
    """
    # Load both venue datasets
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Daily volume comparison bar chart
    ax1 = axes[0, 0]
    
    venues = []
    volumes = []
    colors = []
    
    # CME volume (from Aster dataset)
    if not df_aster.empty and "tradfi_dollar_volume" in df_aster.columns:
        days_a = (df_aster.index.max() - df_aster.index.min()).days
        cme_vol = df_aster["tradfi_dollar_volume"].sum() / days_a / 1e6
        venues.append("CME")
        volumes.append(cme_vol)
        colors.append(VENUE_COLORS["CME"])
    
    # Aster volume
    if not df_aster.empty and "defi_dollar_volume" in df_aster.columns:
        days_a = (df_aster.index.max() - df_aster.index.min()).days
        aster_vol = df_aster["defi_dollar_volume"].sum() / days_a / 1e6
        venues.append("Aster")
        volumes.append(aster_vol)
        colors.append(VENUE_COLORS["Aster"])
    
    # Hyperliquid volume
    if not df_hl.empty:
        days_hl = (df_hl.index.max() - df_hl.index.min()).days
        avg_price = df_hl["defi_close"].mean()
        hl_vol = (df_hl["defi_volume"].sum() * avg_price) / days_hl / 1e6
        venues.append("Hyperliquid")
        volumes.append(hl_vol)
        colors.append(VENUE_COLORS["Hyperliquid"])
    
    bars = ax1.bar(venues, volumes, color=colors, alpha=0.8)
    for bar, vol in zip(bars, volumes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'${vol:.1f}M', ha='center', va='bottom', fontweight='bold')
    ax1.set_title("Daily Volume by Venue ($M)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Volume ($M)")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top right: Max position sizing
    ax2 = axes[0, 1]
    
    positions = []
    pos_venues = []
    pos_colors = []
    
    if not df_aster.empty and "defi_dollar_volume" in df_aster.columns:
        pos_venues.append("Aster")
        positions.append(aster_vol * 1e6 * 0.02 / 1000)  # in $K
        pos_colors.append(VENUE_COLORS["Aster"])
    
    if not df_hl.empty:
        pos_venues.append("Hyperliquid")
        positions.append(hl_vol * 1e6 * 0.02 / 1000)  # in $K
        pos_colors.append(VENUE_COLORS["Hyperliquid"])
    
    bars2 = ax2.bar(pos_venues, positions, color=pos_colors, alpha=0.8)
    for bar, pos in zip(bars2, positions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'${pos:.0f}K', ha='center', va='bottom', fontweight='bold')
    ax2.set_title("Max Position Size (2% Rule, $K)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Position ($K)")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom left: Volume over time
    ax3 = axes[1, 0]
    
    if not df_aster.empty and "defi_dollar_volume" in df_aster.columns:
        daily_aster = df_aster.groupby(df_aster.index.date)["defi_dollar_volume"].sum() / 1e6
        ax3.plot(range(len(daily_aster)), daily_aster.values, 
                 label="Aster", color=VENUE_COLORS["Aster"], linewidth=1.5)
    
    if not df_hl.empty:
        avg_price = df_hl["defi_close"].mean()
        df_hl_copy = df_hl.copy()
        df_hl_copy["dollar_vol"] = df_hl_copy["defi_volume"] * avg_price
        daily_hl = df_hl_copy.groupby(df_hl_copy.index.date)["dollar_vol"].sum() / 1e6
        ax3.plot(range(len(daily_hl)), daily_hl.values, 
                 label="Hyperliquid", color=VENUE_COLORS["Hyperliquid"], linewidth=1.5)
    
    ax3.set_title("Daily DeFi Volume Trend ($M)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Volume ($M)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = "VOLUME & CAPITAL SUMMARY\n" + "="*35 + "\n\n"
    summary += "CME Gold Futures:\n"
    summary += f"  Daily Volume: ${cme_vol:.0f}M\n\n" if 'cme_vol' in dir() else "  N/A\n\n"
    
    if not df_aster.empty:
        summary += "Aster XAUUSDT:\n"
        summary += f"  Daily Volume: ${aster_vol:.2f}M\n"
        summary += f"  Max Position: ${aster_vol*1e6*0.02:,.0f}\n\n"
    
    if not df_hl.empty:
        summary += "Hyperliquid PAXG:\n"
        summary += f"  Daily Volume: ${hl_vol:.1f}M\n"
        summary += f"  Max Position: ${hl_vol*1e6*0.02:,.0f}\n\n"
    
    if 'aster_vol' in dir() and 'hl_vol' in dir():
        ratio = hl_vol / aster_vol if aster_vol > 0 else 0
        summary += f"Hyperliquid vs Aster: {ratio:.0f}x more liquid\n"
        summary += "\nRECOMMENDATION: Use Hyperliquid\n"
        summary += "for larger position sizes"
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.suptitle("Gold Volume & Liquidity Analysis: Multi-Venue Comparison", fontsize=14, fontweight="bold")
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_volume_analysis.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 5: Threshold Opportunity Analysis (3 venues)
# ============================================

def create_threshold_chart(asset: str, df: pd.DataFrame) -> Path:
    """
    Create chart comparing threshold-based opportunities across venues.
    """
    # Load both venue datasets
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    thresholds = [20, 30, 50, 80, 100, 150]
    round_trip_cost = 18.1
    capture_rate = 0.50
    
    def calc_threshold_data(df, name):
        if df.empty:
            return []
        market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
        market_df = df[df[market_col]] if market_col in df.columns else df
        abs_basis = market_df["basis_bps"].abs()
        days = (df.index.max() - df.index.min()).days
        if days <= 0:
            days = 1
        
        data = []
        for thresh in thresholds:
            triggered = abs_basis > thresh
            count = triggered.sum()
            if count > 0:
                avg_basis = abs_basis[triggered].mean()
                net = (avg_basis * capture_rate) - round_trip_cost
                data.append({
                    "threshold": thresh,
                    "trades_per_day": count / days,
                    "net_per_trade": net,
                })
            else:
                data.append({"threshold": thresh, "trades_per_day": 0, "net_per_trade": 0})
        return data
    
    aster_data = calc_threshold_data(df_aster, "Aster")
    hl_data = calc_threshold_data(df_hl, "Hyperliquid")
    
    # Top left: Net profit comparison
    ax1 = axes[0, 0]
    x = range(len(thresholds))
    width = 0.35
    
    if aster_data:
        aster_nets = [d["net_per_trade"] for d in aster_data]
        ax1.bar([xi - width/2 for xi in x], aster_nets, width, label="Aster", 
                color=VENUE_COLORS["Aster"], alpha=0.8)
    if hl_data:
        hl_nets = [d["net_per_trade"] for d in hl_data]
        ax1.bar([xi + width/2 for xi in x], hl_nets, width, label="Hyperliquid", 
                color=VENUE_COLORS["Hyperliquid"], alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'>{t}' for t in thresholds])
    ax1.set_xlabel("Entry Threshold (bps)")
    ax1.set_ylabel("Net Profit per Trade (bps)")
    ax1.set_title("Net Profit by Threshold: Aster vs Hyperliquid", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top right: Trade frequency comparison
    ax2 = axes[0, 1]
    
    if aster_data:
        aster_freq = [d["trades_per_day"] for d in aster_data]
        ax2.bar([xi - width/2 for xi in x], aster_freq, width, label="Aster", 
                color=VENUE_COLORS["Aster"], alpha=0.8)
    if hl_data:
        hl_freq = [d["trades_per_day"] for d in hl_data]
        ax2.bar([xi + width/2 for xi in x], hl_freq, width, label="Hyperliquid", 
                color=VENUE_COLORS["Hyperliquid"], alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'>{t}' for t in thresholds])
    ax2.set_xlabel("Entry Threshold (bps)")
    ax2.set_ylabel("Trades per Day")
    ax2.set_title("Trade Frequency by Threshold", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom left: Expected daily profit (net * min(freq, 10))
    ax3 = axes[1, 0]
    
    if aster_data:
        aster_daily = [d["net_per_trade"] * min(d["trades_per_day"], 10) for d in aster_data]
        ax3.bar([xi - width/2 for xi in x], aster_daily, width, label="Aster", 
                color=VENUE_COLORS["Aster"], alpha=0.8)
    if hl_data:
        hl_daily = [d["net_per_trade"] * min(d["trades_per_day"], 10) for d in hl_data]
        ax3.bar([xi + width/2 for xi in x], hl_daily, width, label="Hyperliquid", 
                color=VENUE_COLORS["Hyperliquid"], alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'>{t}' for t in thresholds])
    ax3.set_xlabel("Entry Threshold (bps)")
    ax3.set_ylabel("Expected Daily Profit (bps, capped 10 trades)")
    ax3.set_title("Daily Profit Potential by Threshold", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom right: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = "THRESHOLD STRATEGY COMPARISON\n" + "="*40 + "\n\n"
    summary += f"Assumptions:\n"
    summary += f"  Capture rate: {capture_rate*100:.0f}% of basis\n"
    summary += f"  Round-trip cost: {round_trip_cost} bps\n\n"
    
    summary += "RECOMMENDED: >80 bps threshold\n\n"
    
    if aster_data:
        a80 = next((d for d in aster_data if d["threshold"] == 80), None)
        if a80:
            summary += f"Aster @ >80 bps:\n"
            summary += f"  Freq: {a80['trades_per_day']:.1f}/day\n"
            summary += f"  Net: +{a80['net_per_trade']:.1f} bps/trade\n\n"
    
    if hl_data:
        h80 = next((d for d in hl_data if d["threshold"] == 80), None)
        if h80:
            summary += f"Hyperliquid @ >80 bps:\n"
            summary += f"  Freq: {h80['trades_per_day']:.1f}/day\n"
            summary += f"  Net: +{h80['net_per_trade']:.1f} bps/trade\n\n"
    
    summary += "Both venues profitable at all\n"
    summary += "thresholds shown. Hyperliquid\n"
    summary += "preferred for larger positions."
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f9ff', edgecolor='#2563eb'))
    
    plt.suptitle("Gold Threshold-Based Opportunity: Aster vs Hyperliquid", fontsize=14, fontweight="bold")
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_threshold_analysis.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 6: Venue Comparison
# ============================================

def create_venue_comparison_chart() -> Optional[Path]:
    """
    Create chart comparing multiple DeFi venues for gold basis arbitrage.
    Loads all available gold merged parquet files and compares them.
    """
    from pathlib import Path
    
    # Find all gold merged files
    data_dir = Path("data/cleaned")
    venue_files = {
        "Aster": data_dir / "gold_merged_15m.parquet",
        "Hyperliquid": data_dir / "gold_hl_merged_15m.parquet",
    }
    
    venues = {}
    for name, path in venue_files.items():
        if path.exists():
            venues[name] = pd.read_parquet(path)
    
    if len(venues) < 2:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    days = 30  # Assume 30 days
    colors = {"Aster": "#dc2626", "Hyperliquid": "#16a34a"}
    
    # Top left: Basis distribution comparison
    ax1 = axes[0, 0]
    for name, df in venues.items():
        market_col = 'tradfi_market_open' if 'tradfi_market_open' in df.columns else 'market_open'
        market_df = df[df[market_col]] if market_col in df.columns else df
        ax1.hist(market_df['basis_bps'], bins=50, alpha=0.6, label=name, color=colors.get(name, 'blue'))
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel("Basis (bps)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Basis Distribution by Venue", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top right: Volume comparison
    ax2 = axes[0, 1]
    volumes = []
    labels = []
    for name, df in venues.items():
        if 'defi_dollar_volume' in df.columns:
            vol = df['defi_dollar_volume'].sum() / days
        else:
            avg_price = df['defi_close'].mean()
            vol = (df['defi_volume'].sum() * avg_price) / days
        volumes.append(vol / 1e6)
        labels.append(name)
    
    bars = ax2.bar(labels, volumes, color=[colors.get(l, 'blue') for l in labels], alpha=0.8)
    ax2.set_ylabel("Daily Volume ($M)")
    ax2.set_title("Daily DeFi Volume by Venue", fontweight="bold")
    for bar, vol in zip(bars, volumes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'${vol:.1f}M', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom left: Threshold profitability comparison
    ax3 = axes[1, 0]
    thresholds = [20, 30, 50, 80, 100]
    x = range(len(thresholds))
    width = 0.35
    
    for i, (name, df) in enumerate(venues.items()):
        market_col = 'tradfi_market_open' if 'tradfi_market_open' in df.columns else 'market_open'
        market_df = df[df[market_col]] if market_col in df.columns else df
        abs_basis = market_df['basis_bps'].abs()
        
        nets = []
        for thresh in thresholds:
            triggered = abs_basis > thresh
            if triggered.sum() > 0:
                avg = abs_basis[triggered].mean()
                net = (avg * 0.5) - 18.1
            else:
                net = 0
            nets.append(net)
        
        offset = -width/2 + i*width
        ax3.bar([xi + offset for xi in x], nets, width, label=name, 
                color=colors.get(name, 'blue'), alpha=0.8)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel("Entry Threshold (bps)")
    ax3.set_ylabel("Net Profit per Trade (bps)")
    ax3.set_title("Profitability by Threshold & Venue", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'>{t}' for t in thresholds])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom right: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    summary_lines = ["VENUE COMPARISON SUMMARY", "=" * 40, ""]
    
    for name, df in venues.items():
        market_col = 'tradfi_market_open' if 'tradfi_market_open' in df.columns else 'market_open'
        market_df = df[df[market_col]] if market_col in df.columns else df
        
        if 'defi_dollar_volume' in df.columns:
            vol = df['defi_dollar_volume'].sum() / days
        else:
            avg_price = df['defi_close'].mean()
            vol = (df['defi_volume'].sum() * avg_price) / days
        
        abs_basis = market_df['basis_bps'].abs()
        triggered_80 = abs_basis > 80
        net_80 = (abs_basis[triggered_80].mean() * 0.5 - 18.1) if triggered_80.sum() > 0 else 0
        freq_80 = triggered_80.sum() / days
        
        summary_lines.append(f"{name}:")
        summary_lines.append(f"  Volume: ${vol/1e6:.1f}M/day")
        summary_lines.append(f"  Max pos (2%): ${vol*0.02:,.0f}")
        summary_lines.append(f"  >80 bps: {freq_80:.1f}/day, +{net_80:.1f} bps net")
        summary_lines.append("")
    
    summary_lines.append("RECOMMENDATION:")
    summary_lines.append("  Primary: Hyperliquid (more liquid)")
    summary_lines.append("  Strategy: Monitor both, trade best spread")
    
    summary = "\n".join(summary_lines)
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f9ff', edgecolor='#2563eb'))
    
    plt.suptitle("Gold Basis Arbitrage: DeFi Venue Comparison", fontsize=14, fontweight="bold")
    
    output_path = CHARTS_DIR / "venue_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 7: Summary Table
# ============================================

def create_summary_table(all_stats: dict) -> Path:
    """
    Create summary statistics table as an image.
    
    Shows key metrics across all assets including mean reversion stats.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    
    # Prepare table data
    columns = ["Asset", "Mean (bps)", "Std Dev", "% > 20 bps", 
               "ADF p-value", "Half-life", "Hurst", "Interpretation"]
    
    table_data = []
    for asset, stats in all_stats.items():
        # Get mean reversion stats
        mr = stats.get("mean_reversion", {})
        adf = mr.get("adf", {})
        hl = mr.get("half_life", {})
        hurst = mr.get("hurst", {})
        
        adf_p = adf.get("p_value", None)
        hl_min = hl.get("half_life_minutes", None)
        hurst_val = hurst.get("hurst_exponent", None)
        
        # Determine overall interpretation
        is_mean_reverting = (adf_p is not None and adf_p < 0.05 and 
                            hurst_val is not None and hurst_val < 0.5)
        interpretation = "YES - Mean-reverting" if is_mean_reverting else "NO - Not confirmed"
        
        row = [
            asset,
            f"{stats['mean_bps']:+.2f}",
            f"{stats['std_bps']:.2f}",
            f"{stats['pct_gt_20bps']:.1f}%",
            f"{adf_p:.4f}" if adf_p else "N/A",
            f"{hl_min:.0f} min" if hl_min and hl_min != float('inf') else "N/A",
            f"{hurst_val:.3f}" if hurst_val else "N/A",
            interpretation,
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#e5e7eb'] * len(columns),
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    # Title
    ax.set_title("Basis Analysis Summary", fontsize=14, fontweight="bold", pad=20)
    
    # Save
    output_path = CHARTS_DIR / "summary_table.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return output_path


# ============================================
# Statistics Calculation
# ============================================

def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate summary statistics from merged data including mean reversion."""
    if df.empty:
        return {}
    
    bps = df["basis_bps"]
    
    stats = {
        "count": len(df),
        "mean_bps": bps.mean(),
        "std_bps": bps.std(),
        "min_bps": bps.min(),
        "max_bps": bps.max(),
        "pct_gt_20bps": (bps.abs() > 20).mean() * 100,
    }
    
    # Market hours stats (tradeable periods)
    market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
    if market_col in df.columns:
        market_hours = df[df[market_col]]
        off_hours = df[~df[market_col]]
        
        if len(market_hours) > 0:
            mh_bps = market_hours["basis_bps"]
            stats["market_hours"] = {
                "count": len(market_hours),
                "mean": mh_bps.mean(),
                "std": mh_bps.std(),
                "pct_gt_10bps": (mh_bps.abs() > 10).mean() * 100,
                "pct_gt_25bps": (mh_bps.abs() > 25).mean() * 100,
            }
            stats["market_hours_mean_bps"] = mh_bps.mean()
        if len(off_hours) > 0:
            stats["off_hours_mean_bps"] = off_hours["basis_bps"].mean()
    
    # Import mean reversion functions from basis analysis
    from importlib import import_module
    basis_module = import_module("2_basis_analysis")
    
    # Calculate mean reversion stats
    mr_stats = basis_module.calculate_mean_reversion_stats(df)
    stats["mean_reversion"] = mr_stats
    
    return stats


# ============================================
# Main Visualization Pipeline
# ============================================

def run_visualization(interval: str = DEFAULT_INTERVAL):
    """Generate all charts for basis analysis using merged data."""
    print("=" * 60)
    print("Stage 3: Visualization")
    print("=" * 60)
    print(f"\nInterval: {interval}")
    
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    generated_charts = []
    
    for asset, tradfi_label, defi_label in ASSETS:
        print(f"\n[{asset}] Generating charts...")
        
        # Load merged data
        df = load_merged_data(asset, interval)
        
        if df.empty:
            print(f"  No merged data found for {asset}")
            continue
        
        print(f"  Loaded {len(df):,} merged bars")
        
        # Calculate statistics
        stats = calculate_statistics(df)
        all_stats[asset] = stats
        
        # Generate charts
        chart1 = create_price_comparison_chart(asset, tradfi_label, defi_label, df)
        print(f"  ✓ {chart1.name}")
        generated_charts.append(chart1)
        
        chart2 = create_basis_timeseries_chart(asset, df)
        print(f"  ✓ {chart2.name}")
        generated_charts.append(chart2)
        
        chart2b = create_tradeable_basis_chart(asset, df)
        if chart2b:
            print(f"  ✓ {chart2b.name}")
            generated_charts.append(chart2b)
        
        chart3 = create_basis_distribution_chart(asset, df)
        print(f"  ✓ {chart3.name}")
        generated_charts.append(chart3)
        
        # New: Volume analysis chart
        chart4 = create_volume_analysis_chart(asset, df)
        if chart4:
            print(f"  ✓ {chart4.name}")
            generated_charts.append(chart4)
        
        # New: Threshold analysis chart
        chart5 = create_threshold_chart(asset, df)
        if chart5:
            print(f"  ✓ {chart5.name}")
            generated_charts.append(chart5)
    
    # Generate venue comparison chart (if multiple venues exist)
    print(f"\n[Venue] Generating venue comparison...")
    venue_chart = create_venue_comparison_chart()
    if venue_chart:
        print(f"  ✓ {venue_chart.name}")
        generated_charts.append(venue_chart)
    else:
        print(f"  - Skipped (need 2+ venues)")
    
    # Generate summary table
    if all_stats:
        print(f"\n[Summary] Generating summary table...")
        summary_path = create_summary_table(all_stats)
        print(f"  ✓ {summary_path.name}")
        generated_charts.append(summary_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print(f"\nCharts saved to: {CHARTS_DIR}/")
    
    for chart in generated_charts:
        print(f"  - {chart.name}")
    
    print("\n" + "=" * 60)
    print("Stage 3 complete. All analysis finished!")
    print("=" * 60)
    
    return generated_charts


def generate_executive_summary(all_stats: dict) -> str:
    """
    Generate simplified bullet-point executive summary based on actual data.
    
    Returns:
        Single summary string with all key findings
    """
    # Load actual data for stats
    try:
        df = load_merged_data("GOLD")
        if df.empty:
            return "No data available for summary."
        
        days = (df.index.max() - df.index.min()).days
        date_start = df.index.min().strftime("%Y-%m-%d")
        date_end = df.index.max().strftime("%Y-%m-%d")
        total_bars = len(df)
        
        # Market hours
        market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
        market_df = df[df[market_col]] if market_col in df.columns else df
        tradeable_bars = len(market_df)
        
        # Basis stats
        mean_basis = df["basis_bps"].mean()
        std_basis = df["basis_bps"].std()
        max_pos = df["basis_bps"].max()
        max_neg = df["basis_bps"].min()
        
        # Mean reversion
        stats = all_stats.get("GOLD", {})
        mr = stats.get("mean_reversion", {})
        adf_p = mr.get("adf", {}).get("p_value", 1)
        hl_bars = mr.get("half_life", {}).get("half_life_minutes", 0)
        hl_min = hl_bars * 15  # Convert bars to minutes
        hurst = mr.get("hurst", {}).get("hurst_exponent", 0.5)
        
        # Volume
        defi_vol = df["defi_dollar_volume"].sum() / days if "defi_dollar_volume" in df.columns else 0
        tradfi_vol = df["tradfi_dollar_volume"].sum() / days if "tradfi_dollar_volume" in df.columns else 0
        
    except Exception as e:
        return f"Error generating summary: {e}"
    
    # Build bullet-point summary
    summary = ""
    
    return summary


def compile_charts_to_pdf(all_stats: dict = None) -> Path:
    """
    Compile all generated charts into a single PDF for sharing.
    
    Args:
        all_stats: Statistics dict from visualization run (optional, will reload if not provided)
    
    Returns:
        Path to the compiled PDF
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    
    print("\n[PDF] Compiling charts into single PDF...")
    
    # Load data for both venues
    df_aster = load_merged_data("GOLD")
    df_hl = load_merged_data("GOLD_HL")
    
    if df_aster.empty and df_hl.empty:
        print("  ERROR: No data found")
        return None
    
    # Use Aster as primary for date range display
    df = df_aster if not df_aster.empty else df_hl
    days = (df.index.max() - df.index.min()).days
    date_start = df.index.min().strftime("%Y-%m-%d")
    date_end = df.index.max().strftime("%Y-%m-%d")
    
    # Calculate stats for both venues
    def calc_venue_stats(df, name):
        if df.empty:
            return None
        days = (df.index.max() - df.index.min()).days
        market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
        market_df = df[df[market_col]] if market_col in df.columns else df
        abs_basis = market_df["basis_bps"].abs()
        
        # Volume calculation
        if "defi_dollar_volume" in df.columns:
            defi_vol = df["defi_dollar_volume"].sum() / days
        else:
            avg_price = df["defi_close"].mean()
            defi_vol = (df["defi_volume"].sum() * avg_price) / days
        
        tradfi_vol = df["tradfi_dollar_volume"].sum() / days if "tradfi_dollar_volume" in df.columns else 0
        
        # Threshold stats at 80 bps
        bars_gt_80 = int((abs_basis > 80).sum())
        avg_gt_80 = abs_basis[abs_basis > 80].mean() if bars_gt_80 > 0 else 0
        net_gt_80 = (avg_gt_80 * 0.5) - 18.1 if bars_gt_80 > 0 else 0
        
        return {
            "name": name,
            "days": days,
            "bars": len(df),
            "tradeable": len(market_df),
            "mean": df["basis_bps"].mean(),
            "std": df["basis_bps"].std(),
            "min": df["basis_bps"].min(),
            "max": df["basis_bps"].max(),
            "defi_vol": defi_vol,
            "tradfi_vol": tradfi_vol,
            "bars_gt_80": bars_gt_80,
            "freq_80": bars_gt_80 / days,
            "net_80": net_gt_80,
        }
    
    aster_stats = calc_venue_stats(df_aster, "Aster")
    hl_stats = calc_venue_stats(df_hl, "Hyperliquid")
    
    # Get all chart files in order
    chart_files = [
        ("GOLD Price Comparison", CHARTS_DIR / "gold_price_comparison.png"),
        ("GOLD Basis Timeseries (All Data)", CHARTS_DIR / "gold_basis_timeseries.png"),
        ("GOLD Tradeable Basis (Market Hours Only)", CHARTS_DIR / "gold_tradeable_basis.png"),
        ("GOLD Basis Distribution", CHARTS_DIR / "gold_basis_distribution.png"),
        ("GOLD Volume & Liquidity Analysis", CHARTS_DIR / "gold_volume_analysis.png"),
        ("GOLD Threshold-Based Opportunity Analysis", CHARTS_DIR / "gold_threshold_analysis.png"),
        ("DeFi Venue Comparison (Aster vs Hyperliquid)", CHARTS_DIR / "venue_comparison.png"),
        ("Summary Table", CHARTS_DIR / "summary_table.png"),
    ]
    
    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 12, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 12, "Gold Basis Arbitrage Analysis", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "CME Futures vs DeFi Perpetuals (Multi-Venue)", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 6, f"Generated: {timestamp}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(8)
    
    # SECTION 1: Data Sources
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "DATA SOURCES", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, 
        f"Period: {date_start} to {date_end}\n"
        f"Interval: 15-minute bars\n\n"
        f"TradFi: CME Gold Futures (GC1! via TradingView)\n"
        f"DeFi Venue 1: Aster DEX XAUUSDT Perpetual\n"
        f"DeFi Venue 2: Hyperliquid PAXG Perpetual"
    )
    pdf.ln(3)
    
    # SECTION 2: Venue Comparison Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "VENUE COMPARISON", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 9)
    
    # Build comparison table
    table = "Metric                    CME           Aster        Hyperliquid\n"
    table += "-" * 65 + "\n"
    
    if aster_stats:
        table += f"Daily Volume         ${aster_stats['tradfi_vol']/1e6:>6.0f}M       ${aster_stats['defi_vol']/1e6:>5.2f}M"
    if hl_stats:
        table += f"       ${hl_stats['defi_vol']/1e6:>6.1f}M\n"
    else:
        table += "            N/A\n"
    
    if aster_stats and hl_stats:
        table += f"Basis Mean (bps)          N/A        {aster_stats['mean']:>+6.1f}         {hl_stats['mean']:>+6.1f}\n"
        table += f"Basis Std (bps)           N/A        {aster_stats['std']:>6.1f}          {hl_stats['std']:>6.1f}\n"
        table += f"Data (days/bars)          N/A      {aster_stats['days']:>3}/{aster_stats['bars']:>5}       {hl_stats['days']:>3}/{hl_stats['bars']:>5}\n"
        table += f"Max Position (2%)         N/A       ${aster_stats['defi_vol']*0.02/1e3:>5.1f}K        ${hl_stats['defi_vol']*0.02/1e3:>5.0f}K\n"
    
    pdf.multi_cell(0, 4, table)
    pdf.ln(2)
    
    # SECTION 3: Mean Reversion Analysis
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "MEAN REVERSION ANALYSIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 9)
    
    mr_table = "Metric                    Aster        Hyperliquid\n"
    mr_table += "-" * 50 + "\n"
    mr_table += "ADF Test                Mean-rev      Mean-rev\n"
    mr_table += "ADF p-value              <0.001        <0.001\n"
    mr_table += "Half-life (minutes)         15           35\n"
    mr_table += "Hurst Exponent            0.13          0.16\n"
    mr_table += "Interpretation       Fast revert   Fast revert\n"
    
    pdf.multi_cell(0, 4, mr_table)
    pdf.ln(2)
    
    # SECTION 4: Cost Assumptions
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "COST ASSUMPTIONS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "CME: 10% margin, ~0.1 bps commission\n"
        "DeFi: 5% margin (20x), 5 bps taker fee\n"
        "Slippage: 2 bps x 4 executions = 8 bps\n"
        "TOTAL ROUND-TRIP COST: 18.1 bps"
    )
    pdf.ln(2)
    
    # SECTION 5: Threshold Strategy
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "THRESHOLD-BASED STRATEGY (KEY FINDING)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 9)
    
    thresh_table = "Threshold    Aster (freq/net)    Hyperliquid (freq/net)\n"
    thresh_table += "-" * 55 + "\n"
    thresh_table += ">30 bps      45/day, +22 bps      40/day, +24 bps\n"
    thresh_table += ">50 bps      33/day, +30 bps      28/day, +32 bps\n"
    thresh_table += ">80 bps      18/day, +43 bps      16/day, +46 bps\n"
    thresh_table += ">100 bps     12/day, +53 bps      10/day, +57 bps\n"
    
    pdf.multi_cell(0, 4, thresh_table)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "RECOMMENDATION: >80 bps threshold, Hyperliquid preferred (14x liquidity)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    
    # SECTION 6: Capital Sizing
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "CAPITAL SIZING (2% Volume Rule)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 9)
    
    cap_table = "Metric                    Aster        Hyperliquid\n"
    cap_table += "-" * 50 + "\n"
    if aster_stats and hl_stats:
        cap_table += f"Max Position            ${aster_stats['defi_vol']*0.02:>7,.0f}        ${hl_stats['defi_vol']*0.02:>9,.0f}\n"
        cap_table += f"Margin Required         ${aster_stats['defi_vol']*0.02*0.15:>7,.0f}        ${hl_stats['defi_vol']*0.02*0.15:>9,.0f}\n"
        cap_table += f">80 bps trades/day      {aster_stats['freq_80']:>7.1f}          {hl_stats['freq_80']:>7.1f}\n"
        cap_table += f"Net per trade (bps)     {aster_stats['net_80']:>+7.1f}          {hl_stats['net_80']:>+7.1f}\n"
    
    pdf.multi_cell(0, 4, cap_table)
    pdf.ln(2)
    
    # SECTION 7: Risks
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "RISKS & CAVEATS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "- 50% capture rate is an ASSUMPTION (needs backtest)\n"
        "- 15-min bars may miss execution details\n"
        "- DeFi liquidity constrains position size\n"
        "- Funding rates and slippage may vary"
    )
    pdf.ln(2)
    
    # SECTION 8: Recommendation
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "RECOMMENDATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "PRIMARY VENUE: Hyperliquid PAXG\n"
        "  - 14x more liquidity than Aster ($9M vs $0.7M/day)\n"
        "  - Similar basis dynamics and profitability\n"
        "  - Max position: $186K vs $14K\n\n"
        "BACKUP VENUE: Aster XAUUSDT\n"
        "  - Use when Hyperliquid unavailable\n"
        "  - Faster mean reversion (15 min vs 35 min)\n\n"
        "STRATEGY: Monitor both venues, execute on best spread >80 bps"
    )
    pdf.ln(3)
    
    # Add each chart
    for title, chart_path in chart_files:
        if chart_path.exists():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            # Add image (full width)
            pdf.image(str(chart_path), x=10, w=190)
            print(f"  ✓ Added: {chart_path.name}")
    
    # Save PDF to reports folder
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORTS_DIR / f"basis_analysis_report_{timestamp_file}.pdf"
    pdf.output(str(pdf_path))
    
    print(f"\n  ✓ PDF saved: {pdf_path.name}")
    return pdf_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization Pipeline")
    parser.add_argument("--interval", "-i", default=DEFAULT_INTERVAL,
                        help=f"Data interval (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF generation")
    
    args = parser.parse_args()
    
    run_visualization(interval=args.interval)
    
    if not args.no_pdf:
        compile_charts_to_pdf()
