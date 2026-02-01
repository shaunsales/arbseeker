#!/usr/bin/env python3
"""
CLI tool for downloading historical data from Binance Vision.

Usage:
    python download_data.py --ticker BTCUSDT --interval 1h --years 2023 2024
    python download_data.py --ticker ETHUSDT --interval 15m --years 2024 --market spot
    python download_data.py --list  # Show available data
"""

import argparse
import sys
from datetime import datetime

from core.data.binance import download_binance_year, list_binance_symbols, INTERVALS
from core.data.storage import list_all_data, list_available_years


def print_data_tree():
    """Print all available data in tree format."""
    data = list_all_data()
    
    if not data:
        print("No data downloaded yet.")
        print("\nExample: python download_data.py --ticker BTCUSDT --interval 1h --years 2024")
        return
    
    print("Available data:\n")
    for venue in sorted(data.keys()):
        print(f"📁 {venue}/")
        for market in sorted(data[venue].keys()):
            print(f"  └─ {market}/")
            for ticker in sorted(data[venue][market].keys()):
                print(f"      └─ {ticker}/")
                for interval in sorted(data[venue][market][ticker].keys()):
                    years = data[venue][market][ticker][interval]
                    years_str = ", ".join(str(y) for y in years)
                    print(f"          └─ {interval}/ [{years_str}]")


def download_years(ticker: str, interval: str, years: list[int], market: str):
    """Download specified years of data."""
    print(f"\n{'='*60}")
    print(f"Downloading {ticker} {interval} data from Binance Vision")
    print(f"Market: {market}")
    print(f"Years: {years}")
    print(f"{'='*60}\n")
    
    # Check existing
    existing = list_available_years("binance", market, ticker, interval)
    if existing:
        print(f"Already have: {existing}")
    
    # Download each year
    for year in years:
        if year in existing:
            print(f"\n⏭️  Skipping {year} (already exists)")
            continue
        
        print(f"\n📥 Downloading {year}...")
        try:
            path = download_binance_year(
                symbol=ticker,
                interval=interval,
                year=year,
                market=market,
            )
            if path:
                print(f"✅ Saved: {path}")
            else:
                print(f"⚠️  No data available for {year}")
        except Exception as e:
            print(f"❌ Error downloading {year}: {e}")
    
    # Show final state
    print(f"\n{'='*60}")
    print("Download complete!")
    print_data_tree()


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from Binance Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py --ticker BTCUSDT --interval 1h --years 2023 2024
  python download_data.py --ticker ETHUSDT --interval 15m --years 2024 --market spot
  python download_data.py --list
  python download_data.py --symbols  # List available symbols
        """,
    )
    
    parser.add_argument("--ticker", "-t", type=str, help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--interval", "-i", type=str, choices=INTERVALS, help="Bar interval")
    parser.add_argument("--years", "-y", type=int, nargs="+", help="Years to download")
    parser.add_argument("--market", "-m", type=str, default="futures", 
                        choices=["futures", "spot"], help="Market type")
    parser.add_argument("--list", "-l", action="store_true", help="List available data")
    parser.add_argument("--symbols", "-s", action="store_true", help="List available symbols")
    
    args = parser.parse_args()
    
    if args.list:
        print_data_tree()
        return
    
    if args.symbols:
        print(f"Fetching {args.market} symbols from Binance...")
        symbols = list_binance_symbols(args.market)
        print(f"\nAvailable symbols ({len(symbols)}):")
        # Print in columns
        cols = 6
        for i in range(0, len(symbols), cols):
            row = symbols[i:i+cols]
            print("  " + "  ".join(f"{s:12}" for s in row))
        return
    
    # Validate required args for download
    if not args.ticker:
        parser.error("--ticker is required for download")
    if not args.interval:
        parser.error("--interval is required for download")
    if not args.years:
        # Default to current year
        args.years = [datetime.now().year]
        print(f"No years specified, defaulting to {args.years}")
    
    download_years(args.ticker, args.interval, args.years, args.market)


if __name__ == "__main__":
    main()
