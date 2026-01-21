#!/usr/bin/env python3
"""
Stage 3: Visualization

Generates charts and summary tables from basis analysis data.

Output:
- output/charts/{asset}_price_comparison.png
- output/charts/{asset}_basis_timeseries.png
- output/charts/{asset}_basis_distribution.png
- output/charts/summary_table.png
"""

from datetime import datetime, timezone
from pathlib import Path

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

# Assets to visualize
ASSETS = [
    ("TSLA", "TSLA (Stock)", "TSLAUSDT (Aster)"),
    ("GOLD", "GC=F (Gold Futures)", "XAUUSDT (Aster)"),
]

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

def load_price_data(asset: str, source: str) -> pd.DataFrame:
    """Load cleaned price data from parquet."""
    filepath = DATA_CLEANED_DIR / f"{asset.lower()}_{source}.parquet"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    return pd.read_parquet(filepath)


def load_basis_data(asset: str) -> pd.DataFrame:
    """Load basis analysis data from CSV."""
    filepath = BASIS_DIR / f"{asset.lower()}_basis.csv"
    
    if not filepath.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
    return df


# ============================================
# Chart 1: Price Comparison
# ============================================

def create_price_comparison_chart(
    asset: str,
    yahoo_label: str,
    aster_label: str,
    basis_df: pd.DataFrame
) -> Path:
    """
    Create price comparison chart with market hours shading.
    
    Shows TradFi vs DeFi prices overlaid with shaded market hours.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Normalize timezone for plotting
    idx = basis_df.index.tz_localize(None) if basis_df.index.tz else basis_df.index
    
    # Shade market hours
    market_open_mask = basis_df["market_open"].values
    
    # Find continuous market hour regions
    in_market = False
    start_idx = None
    
    for i, (ts, is_open) in enumerate(zip(idx, market_open_mask)):
        if is_open and not in_market:
            start_idx = ts
            in_market = True
        elif not is_open and in_market:
            ax.axvspan(start_idx, ts, alpha=0.3, color=COLORS["market_hours"], label="_nolegend_")
            in_market = False
    
    # Plot prices
    ax.plot(idx, basis_df["tradfi_mid"], label=yahoo_label, 
            color=COLORS["tradfi"], linewidth=0.8, alpha=0.9)
    ax.plot(idx, basis_df["defi_mid"], label=aster_label, 
            color=COLORS["defi"], linewidth=0.8, alpha=0.9)
    
    # Add market hours legend patch
    market_patch = mpatches.Patch(color=COLORS["market_hours"], alpha=0.3, label="NYSE Market Hours")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(market_patch)
    
    # Formatting
    ax.set_title(f"{asset} Price Comparison: {yahoo_label} vs {aster_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    
    # Save
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHARTS_DIR / f"{asset.lower()}_price_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 2: Basis Timeseries
# ============================================

def create_basis_timeseries_chart(asset: str, basis_df: pd.DataFrame) -> Path:
    """
    Create basis timeseries chart with threshold bands.
    
    Shows spread over time with ±20 bps threshold bands.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    idx = basis_df.index.tz_localize(None) if basis_df.index.tz else basis_df.index
    bps = basis_df["basis_bps"]
    
    # Shade market hours
    market_open_mask = basis_df["market_open"].values
    in_market = False
    start_idx = None
    
    for i, (ts, is_open) in enumerate(zip(idx, market_open_mask)):
        if is_open and not in_market:
            start_idx = ts
            in_market = True
        elif not is_open and in_market:
            ax.axvspan(start_idx, ts, alpha=0.2, color=COLORS["market_hours"], label="_nolegend_")
            in_market = False
    
    # Shade regions where spread exceeds thresholds (before plotting line)
    ax.fill_between(idx, bps, 25, where=(bps > 25), 
                    color="red", alpha=0.3, label="_nolegend_")
    ax.fill_between(idx, bps, -25, where=(bps < -25), 
                    color="red", alpha=0.3, label="_nolegend_")
    ax.fill_between(idx, bps, 10, where=((bps > 10) & (bps <= 25)), 
                    color="orange", alpha=0.2, label="_nolegend_")
    ax.fill_between(idx, bps, -10, where=((bps < -10) & (bps >= -25)), 
                    color="orange", alpha=0.2, label="_nolegend_")
    
    # Plot basis
    ax.plot(idx, bps, color=COLORS["basis"], linewidth=0.6, alpha=0.8)
    
    # Add threshold bands
    ax.axhline(y=25, color="red", linestyle="--", linewidth=1, alpha=0.7, label="±25 bps (strong)")
    ax.axhline(y=-25, color="red", linestyle="--", linewidth=1, alpha=0.7, label="_nolegend_")
    ax.axhline(y=10, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="±10 bps (moderate)")
    ax.axhline(y=-10, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="_nolegend_")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # Add mean line
    mean_bps = bps.mean()
    ax.axhline(y=mean_bps, color="purple", linestyle=":", linewidth=1.5, alpha=0.8, 
               label=f"Mean: {mean_bps:+.1f} bps")
    
    # Formatting
    ax.set_title(f"{asset} Basis Timeseries (DeFi - TradFi)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Basis (bps)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits with some padding
    y_max = max(abs(bps.min()), abs(bps.max())) * 1.1
    ax.set_ylim(-y_max, y_max)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_basis_timeseries.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 3: Basis Distribution
# ============================================

def create_basis_distribution_chart(asset: str, basis_df: pd.DataFrame) -> Path:
    """
    Create histogram of basis values.
    
    Shows distribution with market hours vs off-hours comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    bps = basis_df["basis_bps"]
    market_hours = basis_df[basis_df["market_open"]]["basis_bps"]
    off_hours = basis_df[~basis_df["market_open"]]["basis_bps"]
    
    # Left: Overall distribution
    ax1 = axes[0]
    bins = np.linspace(bps.min(), bps.max(), 50)
    
    ax1.hist(bps, bins=bins, color=COLORS["basis"], alpha=0.7, edgecolor="white")
    ax1.axvline(x=bps.mean(), color="purple", linestyle="--", linewidth=2, 
                label=f"Mean: {bps.mean():+.1f} bps")
    ax1.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax1.axvline(x=10, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="±10 bps")
    ax1.axvline(x=-10, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axvline(x=25, color="red", linestyle="--", linewidth=1, alpha=0.7, label="±25 bps")
    ax1.axvline(x=-25, color="red", linestyle="--", linewidth=1, alpha=0.7)
    
    ax1.set_title(f"{asset} Basis Distribution (All Data)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Basis (bps)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.legend(fontsize=9)
    
    # Right: Market hours vs off-hours
    ax2 = axes[1]
    
    if len(market_hours) > 0:
        ax2.hist(market_hours, bins=30, color=COLORS["tradfi"], alpha=0.6, 
                 edgecolor="white", label=f"Market Hours (n={len(market_hours):,})")
    if len(off_hours) > 0:
        ax2.hist(off_hours, bins=30, color=COLORS["defi"], alpha=0.6, 
                 edgecolor="white", label=f"Off Hours (n={len(off_hours):,})")
    
    ax2.axvline(x=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    
    ax2.set_title(f"{asset} Basis: Market Hours vs Off-Hours", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Basis (bps)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(fontsize=9)
    
    # Save
    output_path = CHARTS_DIR / f"{asset.lower()}_basis_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return output_path


# ============================================
# Chart 4: Summary Table
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

def calculate_statistics(basis_df: pd.DataFrame) -> dict:
    """Calculate summary statistics from basis data including mean reversion."""
    if basis_df.empty:
        return {}
    
    bps = basis_df["basis_bps"]
    
    stats = {
        "count": len(basis_df),
        "mean_bps": bps.mean(),
        "std_bps": bps.std(),
        "min_bps": bps.min(),
        "max_bps": bps.max(),
        "pct_gt_20bps": (bps.abs() > 20).mean() * 100,
    }
    
    # Market hours stats
    if "market_open" in basis_df.columns:
        market_hours = basis_df[basis_df["market_open"]]
        off_hours = basis_df[~basis_df["market_open"]]
        
        if len(market_hours) > 0:
            stats["market_hours_mean_bps"] = market_hours["basis_bps"].mean()
        if len(off_hours) > 0:
            stats["off_hours_mean_bps"] = off_hours["basis_bps"].mean()
    
    # Import mean reversion functions from basis analysis
    from importlib import import_module
    basis_module = import_module("2_basis_analysis")
    
    # Calculate mean reversion stats
    mr_stats = basis_module.calculate_mean_reversion_stats(basis_df)
    stats["mean_reversion"] = mr_stats
    
    return stats


# ============================================
# Main Visualization Pipeline
# ============================================

def run_visualization():
    """Generate all charts for basis analysis."""
    print("=" * 60)
    print("Stage 3: Visualization")
    print("=" * 60)
    
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    generated_charts = []
    
    for asset, yahoo_label, aster_label in ASSETS:
        print(f"\n[{asset}] Generating charts...")
        
        # Load basis data
        basis_df = load_basis_data(asset)
        
        if basis_df.empty:
            print(f"  No basis data found for {asset}")
            continue
        
        print(f"  Loaded {len(basis_df):,} basis records")
        
        # Calculate statistics
        stats = calculate_statistics(basis_df)
        all_stats[asset] = stats
        
        # Generate charts
        chart1 = create_price_comparison_chart(asset, yahoo_label, aster_label, basis_df)
        print(f"  ✓ {chart1.name}")
        generated_charts.append(chart1)
        
        chart2 = create_basis_timeseries_chart(asset, basis_df)
        print(f"  ✓ {chart2.name}")
        generated_charts.append(chart2)
        
        chart3 = create_basis_distribution_chart(asset, basis_df)
        print(f"  ✓ {chart3.name}")
        generated_charts.append(chart3)
    
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


def generate_executive_summary(all_stats: dict) -> tuple:
    """
    Generate dynamic executive summary based on actual statistics.
    
    Returns:
        Tuple of (opportunity_text, methodology_text)
    """
    # Gather statistics
    assets = list(all_stats.keys())
    
    # Check mean reversion strength across all assets
    mean_reverting_count = 0
    total_assets = len(assets)
    adf_pvalues = []
    half_lives = []
    hurst_values = []
    pct_gt_20bps_values = []
    
    for asset, stats in all_stats.items():
        mr = stats.get("mean_reversion", {})
        adf = mr.get("adf", {})
        hl = mr.get("half_life", {})
        hurst = mr.get("hurst", {})
        
        adf_p = adf.get("p_value")
        hl_min = hl.get("half_life_minutes")
        hurst_val = hurst.get("hurst_exponent")
        pct_20 = stats.get("pct_gt_20bps", 0)
        
        if adf_p is not None:
            adf_pvalues.append(adf_p)
        if hl_min is not None and hl_min != float('inf'):
            half_lives.append(hl_min)
        if hurst_val is not None:
            hurst_values.append(hurst_val)
        pct_gt_20bps_values.append(pct_20)
        
        # Check if mean-reverting
        if (adf_p is not None and adf_p < 0.05 and 
            hurst_val is not None and hurst_val < 0.5):
            mean_reverting_count += 1
    
    # Determine overall assessment
    if mean_reverting_count == total_assets:
        strength = "strong"
        conclusion = "promising"
    elif mean_reverting_count > 0:
        strength = "mixed"
        conclusion = "inconclusive"
    else:
        strength = "weak"
        conclusion = "not supported"
    
    # Build opportunity text based on findings
    asset_str = " and ".join(assets)
    
    if strength == "strong":
        opportunity_text = (
            f"This analysis identifies a potential arbitrage opportunity between traditional finance (TradFi) "
            f"and decentralized finance (DeFi) markets. Price discrepancies exist between Yahoo Finance equity/futures "
            f"prices and their synthetic perpetual counterparts on Aster DEX. Both {asset_str} show statistically "
            f"significant mean-reverting behavior, suggesting that price spreads tend to close over time rather than "
            f"persist or widen. Key risks include execution latency, transaction costs, liquidity constraints on DeFi "
            f"venues, and regulatory uncertainty around synthetic asset trading."
        )
    elif strength == "mixed":
        opportunity_text = (
            f"This analysis shows mixed results for arbitrage between TradFi and DeFi markets. "
            f"Of the assets analyzed ({asset_str}), only {mean_reverting_count} of {total_assets} show statistically "
            f"significant mean-reverting behavior. This suggests the opportunity may be asset-specific rather than "
            f"systematic. Further investigation is recommended before committing capital. Key risks include execution "
            f"latency, transaction costs, liquidity constraints, and regulatory uncertainty."
        )
    else:
        opportunity_text = (
            f"This analysis does not support a systematic arbitrage opportunity between TradFi and DeFi markets. "
            f"The assets analyzed ({asset_str}) do not show statistically significant mean-reverting behavior. "
            f"Price spreads may persist or widen rather than close, making arbitrage strategies risky. "
            f"The basis appears to follow a random walk or trending pattern rather than mean-reverting dynamics."
        )
    
    # Build methodology text with actual numbers
    avg_adf = sum(adf_pvalues) / len(adf_pvalues) if adf_pvalues else None
    avg_hl = sum(half_lives) / len(half_lives) if half_lives else None
    avg_hurst = sum(hurst_values) / len(hurst_values) if hurst_values else None
    avg_pct_20 = sum(pct_gt_20bps_values) / len(pct_gt_20bps_values) if pct_gt_20bps_values else 0
    
    # Data counts
    total_bars = sum(stats.get("count", 0) for stats in all_stats.values())
    
    methodology_text = f"We analyzed 14 days of minute-level price data ({total_bars:,} total bars) comparing "
    methodology_text += "Yahoo Finance spot prices against Aster DEX perpetual prices. "
    
    if avg_adf is not None:
        if avg_adf < 0.05:
            methodology_text += f"The Augmented Dickey-Fuller test shows p-values averaging {avg_adf:.4f}, rejecting the random walk hypothesis. "
        else:
            methodology_text += f"The Augmented Dickey-Fuller test shows p-values averaging {avg_adf:.4f}, failing to reject the random walk hypothesis. "
    
    if avg_hl is not None:
        methodology_text += f"Half-life analysis indicates spreads would revert to the mean in approximately {avg_hl:.0f} minutes ({avg_hl/60:.1f} hours). "
    
    if avg_hurst is not None:
        if avg_hurst < 0.4:
            methodology_text += f"Hurst exponents averaging {avg_hurst:.2f} confirm strongly mean-reverting dynamics. "
        elif avg_hurst < 0.5:
            methodology_text += f"Hurst exponents averaging {avg_hurst:.2f} suggest mild mean-reverting behavior. "
        else:
            methodology_text += f"Hurst exponents averaging {avg_hurst:.2f} suggest random walk or trending behavior. "
    
    methodology_text += f"Approximately {avg_pct_20:.0f}% of observations show spreads exceeding 20 basis points."
    
    return opportunity_text, methodology_text


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
    
    # Load stats if not provided
    if all_stats is None:
        all_stats = {}
        for asset, _, _ in ASSETS:
            basis_df = load_basis_data(asset)
            if not basis_df.empty:
                all_stats[asset] = calculate_statistics(basis_df)
    
    # Get all chart files in order
    chart_files = [
        ("TSLA Price Comparison", CHARTS_DIR / "tsla_price_comparison.png"),
        ("TSLA Basis Timeseries", CHARTS_DIR / "tsla_basis_timeseries.png"),
        ("TSLA Basis Distribution", CHARTS_DIR / "tsla_basis_distribution.png"),
        ("GOLD Price Comparison", CHARTS_DIR / "gold_price_comparison.png"),
        ("GOLD Basis Timeseries", CHARTS_DIR / "gold_basis_timeseries.png"),
        ("GOLD Basis Distribution", CHARTS_DIR / "gold_basis_distribution.png"),
        ("Summary Table", CHARTS_DIR / "summary_table.png"),
    ]
    
    # Create PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "", new_x=XPos.LMARGIN, new_y=YPos.NEXT)  # Spacer
    pdf.cell(0, 15, "Basis Arbitrage Analysis", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 8, "TradFi vs DeFi Price Comparison", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 6, f"Generated: {timestamp} | Assets: TSLA, GOLD", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # Generate dynamic executive summary based on actual statistics
    opportunity_text, methodology_text = generate_executive_summary(all_stats)
    
    # Executive Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    
    # Paragraph 1: Opportunity and Risk
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Opportunity & Risk", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, opportunity_text)
    pdf.ln(5)
    
    # Paragraph 2: Data and Methodology
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Data & Methodology", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, methodology_text)
    pdf.ln(5)
    
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
    run_visualization()
    compile_charts_to_pdf()
