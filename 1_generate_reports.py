#!/usr/bin/env python3
"""
Stage 1: Generate Price Comparison Reports

Creates PDF reports comparing TradFi (Yahoo) vs DeFi (Aster) price data.
Each report includes:
- Price comparison chart with both sources overlaid
- Summary statistics table
- Sample data table (last 5 rows from each source)

Output: output/reports/{symbol}_price_comparison_{timestamp}.pdf
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ============================================
# Configuration
# ============================================

BASE_DIR = Path(__file__).parent
DATA_CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REPORTS_DIR = BASE_DIR / "output" / "reports"

# Symbols to report on: (name, yahoo_symbol, yahoo_label, aster_symbol, aster_label)
SYMBOLS = [
    ("TSLA", "TSLA", "TSLA (Stock)", "TSLAUSDT", "TSLAUSDT (Aster)"),
    ("GOLD", "GC=F", "GC=F (Gold Futures)", "XAUUSDT", "XAUUSDT (Aster)"),
]


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
# Chart Generation
# ============================================

def create_price_chart(
    yahoo_df: pd.DataFrame,
    aster_df: pd.DataFrame,
    symbol: str,
    yahoo_label: str,
    aster_label: str,
    output_path: Path
) -> bool:
    """
    Create price comparison chart with both sources overlaid.
    
    Args:
        yahoo_df: Yahoo price data
        aster_df: Aster price data
        symbol: Symbol name for title
        output_path: Path to save PNG
    
    Returns:
        True if chart created successfully
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize timezone for plotting
    yahoo_idx = yahoo_df.index.tz_localize(None) if yahoo_df.index.tz else yahoo_df.index
    aster_idx = aster_df.index.tz_localize(None) if aster_df.index.tz else aster_df.index
    
    # Plot both price series
    ax.plot(yahoo_idx, yahoo_df["close"], label=yahoo_label, 
            color="#2563eb", linewidth=0.8, alpha=0.9)
    ax.plot(aster_idx, aster_df["close"], label=aster_label, 
            color="#dc2626", linewidth=0.8, alpha=0.9)
    
    # Formatting
    ax.set_title(f"{symbol} Price Comparison: {yahoo_label} vs {aster_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Save chart
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return True


# ============================================
# Statistics Calculation
# ============================================

def calculate_stats(df: pd.DataFrame, source: str) -> dict:
    """Calculate summary statistics for a price series."""
    if df.empty:
        return {}
    
    close = df["close"]
    return {
        "source": source,
        "records": len(df),
        "mean": close.mean(),
        "std": close.std(),
        "min": close.min(),
        "max": close.max(),
        "start": df.index.min(),
        "end": df.index.max(),
    }


# ============================================
# PDF Generation
# ============================================

class PriceComparisonPDF(FPDF):
    """Custom PDF class for price comparison reports."""
    
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol
        
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, f"{self.symbol} Price Comparison Report", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 10)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.cell(0, 6, f"Generated: {timestamp}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_pdf_report(
    symbol: str,
    yahoo_df: pd.DataFrame,
    aster_df: pd.DataFrame,
    yahoo_label: str,
    aster_label: str,
    chart_path: Path,
    output_path: Path
) -> bool:
    """
    Generate PDF report with chart, stats, and sample data.
    
    Args:
        symbol: Symbol name
        yahoo_df: Yahoo price data
        aster_df: Aster price data
        chart_path: Path to chart PNG
        output_path: Path to save PDF
    
    Returns:
        True if PDF created successfully
    """
    pdf = PriceComparisonPDF(symbol)
    pdf.add_page()
    
    # Price comparison chart (full width)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Price Comparison Chart", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image(str(chart_path), x=10, w=190)
    pdf.ln(5)
    
    # Summary statistics table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    yahoo_stats = calculate_stats(yahoo_df, yahoo_label)
    aster_stats = calculate_stats(aster_df, aster_label)
    
    # Table header
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [35, 25, 30, 25, 30, 30]
    headers = ["Source", "Records", "Mean", "Std Dev", "Min", "Max"]
    
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 7, header, border=1, align="C")
    pdf.ln()
    
    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for stats in [yahoo_stats, aster_stats]:
        if stats:
            pdf.cell(col_widths[0], 7, stats["source"], border=1)
            pdf.cell(col_widths[1], 7, f"{stats['records']:,}", border=1, align="R")
            pdf.cell(col_widths[2], 7, f"${stats['mean']:,.2f}", border=1, align="R")
            pdf.cell(col_widths[3], 7, f"${stats['std']:,.2f}", border=1, align="R")
            pdf.cell(col_widths[4], 7, f"${stats['min']:,.2f}", border=1, align="R")
            pdf.cell(col_widths[5], 7, f"${stats['max']:,.2f}", border=1, align="R")
            pdf.ln()
    
    pdf.ln(5)
    
    # Date range info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Data Period", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    
    if yahoo_stats:
        start_str = yahoo_stats["start"].strftime("%Y-%m-%d %H:%M UTC")
        end_str = yahoo_stats["end"].strftime("%Y-%m-%d %H:%M UTC")
        pdf.cell(0, 6, f"Start: {start_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 6, f"End:   {end_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(5)
    
    # Sample data table (15 minutes from actual trading window, not LOCF)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sample Data (Live Trading Window)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    if not yahoo_df.empty and not aster_df.empty:
        # Find rows where Yahoo has real trading data (volume > 0 means not LOCF)
        real_trading = yahoo_df[yahoo_df["volume"] > 0]
        
        if len(real_trading) >= 15:
            # Get a 15-minute window from the middle of the last trading session
            # This avoids market open/close volatility
            last_session_end = real_trading.index[-1]
            last_session_start = real_trading[real_trading.index >= last_session_end - pd.Timedelta(hours=2)].index[0]
            
            # Find middle of session and take 15 minutes
            session_data = real_trading.loc[last_session_start:last_session_end]
            mid_point = len(session_data) // 2
            sample_yahoo = session_data.iloc[mid_point:mid_point + 15]
        else:
            sample_yahoo = real_trading.tail(15)
        
        # Table header
        pdf.set_font("Helvetica", "B", 8)
        sample_cols = [50, 40, 40, 35]
        sample_headers = ["Timestamp (UTC)", yahoo_label[:15], aster_label[:15], "Diff %"]
        
        for i, header in enumerate(sample_headers):
            pdf.cell(sample_cols[i], 6, header, border=1, align="C")
        pdf.ln()
        
        # Get sample rows
        pdf.set_font("Helvetica", "", 8)
        
        for idx in sample_yahoo.index:
            ts = idx.strftime("%Y-%m-%d %H:%M UTC") if hasattr(idx, "strftime") else str(idx)[:20]
            yahoo_close = sample_yahoo.loc[idx, "close"]
            
            # Get corresponding Aster price
            if idx in aster_df.index:
                aster_close = aster_df.loc[idx, "close"]
                diff_pct = ((aster_close - yahoo_close) / yahoo_close) * 100
            else:
                aster_close = None
                diff_pct = None
            
            pdf.cell(sample_cols[0], 6, ts, border=1)
            pdf.cell(sample_cols[1], 6, f"${yahoo_close:,.2f}", border=1, align="R")
            
            if aster_close is not None:
                pdf.cell(sample_cols[2], 6, f"${aster_close:,.2f}", border=1, align="R")
                diff_str = f"{diff_pct:+.3f}%"
                pdf.cell(sample_cols[3], 6, diff_str, border=1, align="R")
            else:
                pdf.cell(sample_cols[2], 6, "N/A", border=1, align="C")
                pdf.cell(sample_cols[3], 6, "N/A", border=1, align="C")
            pdf.ln()
    
    # Save PDF
    pdf.output(str(output_path))
    return True


# ============================================
# Main Report Generation
# ============================================

def generate_symbol_report(
    symbol: str,
    yahoo_df: pd.DataFrame,
    aster_df: pd.DataFrame,
    yahoo_label: str,
    aster_label: str
) -> Path:
    """
    Generate complete PDF report for a symbol.
    
    Returns:
        Path to generated PDF
    """
    # Create temp file for chart
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        chart_path = Path(tmp.name)
    
    try:
        # Generate chart
        create_price_chart(yahoo_df, aster_df, symbol, yahoo_label, aster_label, chart_path)
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = REPORTS_DIR / f"{symbol.lower()}_price_comparison_{timestamp}.pdf"
        
        generate_pdf_report(symbol, yahoo_df, aster_df, yahoo_label, aster_label, chart_path, pdf_path)
        
        return pdf_path
        
    finally:
        # Clean up temp chart file
        if chart_path.exists():
            chart_path.unlink()


def run_report_generation():
    """Generate PDF reports for all symbols."""
    print("=" * 60)
    print("Stage 1: Price Comparison Report Generation")
    print("=" * 60)
    
    # Create reports directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    generated_reports = []
    
    for symbol, yahoo_sym, yahoo_label, aster_sym, aster_label in SYMBOLS:
        print(f"\n[{symbol}] Processing...")
        
        # Load data
        yahoo_df = load_price_data(symbol, "yahoo")
        aster_df = load_price_data(symbol, "aster")
        
        if yahoo_df.empty and aster_df.empty:
            print(f"  No data available for {symbol}")
            continue
        
        # Show data info
        print(f"  {yahoo_label}: {len(yahoo_df):,} bars")
        print(f"  {aster_label}: {len(aster_df):,} bars")
        
        # Generate report
        pdf_path = generate_symbol_report(symbol, yahoo_df, aster_df, yahoo_label, aster_label)
        generated_reports.append(pdf_path)
        print(f"  ✓ Report saved: {pdf_path.name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Report Generation Summary")
    print("=" * 60)
    print(f"\nReports saved to: {REPORTS_DIR}/")
    
    for report in generated_reports:
        print(f"  - {report.name}")
    
    print("\n" + "=" * 60)
    print("Stage 1 report generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_report_generation()
