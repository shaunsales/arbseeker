"""
Binance Vision data downloader.

Downloads historical klines data from https://data.binance.vision/
Stores as yearly Parquet files.

URL format:
    https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip
"""

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable
import requests
import pandas as pd

from core.data.storage import save_yearly, get_data_path, list_available_years
from core.data.validator import validate_ohlcv, fill_gaps
from core.data.market_hours import add_market_open_always

# Binance Vision base URL
BASE_URL = "https://data.binance.vision"

# Supported intervals
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# Klines column names (from Binance Vision CSV headers)
# See: https://data.binance.vision/?prefix=data/futures/um/monthly/klines/
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",                # Number of trades
    "taker_buy_volume",     # Taker buy base asset volume
    "taker_buy_quote_volume",  # Taker buy quote asset volume
    "ignore",
]


def get_monthly_url(symbol: str, interval: str, year: int, month: int, market: str = "futures") -> str:
    """
    Get the URL for a monthly klines ZIP file.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Bar interval (e.g., '1h')
        year: Year
        month: Month (1-12)
        market: 'futures' (um) or 'spot'
        
    Returns:
        URL string
    """
    market_path = "futures/um" if market == "futures" else "spot"
    month_str = f"{month:02d}"
    filename = f"{symbol}-{interval}-{year}-{month_str}.zip"
    return f"{BASE_URL}/data/{market_path}/monthly/klines/{symbol}/{interval}/{filename}"


def download_month(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    market: str = "futures",
) -> Optional[pd.DataFrame]:
    """
    Download a single month of klines data.
    
    Returns:
        DataFrame with OHLCV data, or None if not available
    """
    url = get_monthly_url(symbol, interval, year, month, market)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None  # Month not available
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    
    # Extract CSV from ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # Get the first (and only) file in the ZIP
        csv_filename = zf.namelist()[0]
        with zf.open(csv_filename) as f:
            # Try reading without header first
            df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
            
            # Check if first row is a header (contains 'open' string)
            if df.iloc[0]["open"] == "open" or str(df.iloc[0]["open_time"]).lower() == "open_time":
                df = df.iloc[1:].reset_index(drop=True)
    
    # Convert types - handle both ms (pre-2025) and us (2025+) timestamps
    # Binance switched to microseconds from January 1st 2025
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    
    # Auto-detect unit based on magnitude
    sample_ts = df["open_time"].iloc[0]
    unit = "us" if sample_ts > 1e15 else "ms"
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit=unit, utc=True)
    
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    
    df["count"] = df["count"].astype(int)
    
    # Set index to open_time
    df = df.set_index("open_time")
    
    # Keep only OHLCV columns + useful extras
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]]
    
    # Add market_open (crypto is 24/7, always open)
    df = add_market_open_always(df, "market_open")
    
    return df


def download_binance_year(
    symbol: str,
    interval: str,
    year: int,
    market: str = "futures",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Download a full year of klines data from Binance Vision.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Bar interval (e.g., '1h')
        year: Year to download
        market: 'futures' or 'spot'
        progress_callback: Optional callback(month, total_months, status_msg)
        force: If True, re-download even if file exists
        
    Returns:
        Path to saved parquet file, or None if no data available
    """
    # Check if already exists
    existing_years = list_available_years("binance", market, symbol, interval)
    if year in existing_years and not force:
        print(f"Year {year} already exists for {symbol}/{interval}, skipping (use force=True to re-download)")
        return get_data_path("binance", market, symbol, interval, year)
    
    # Determine months to download
    current_date = datetime.now(timezone.utc)
    if year == current_date.year:
        # Current year: only download up to last complete month
        end_month = current_date.month - 1 if current_date.day < 15 else current_date.month
        if end_month < 1:
            print(f"No complete months available for {year} yet")
            return None
    else:
        end_month = 12
    
    # Download all months
    monthly_dfs = []
    for month in range(1, end_month + 1):
        if progress_callback:
            progress_callback(month, end_month, f"Downloading {symbol} {year}-{month:02d}...")
        
        df = download_month(symbol, interval, year, month, market)
        if df is not None and len(df) > 0:
            monthly_dfs.append(df)
            print(f"  Downloaded {year}-{month:02d}: {len(df):,} bars")
        else:
            print(f"  No data for {year}-{month:02d}")
    
    if not monthly_dfs:
        print(f"No data available for {symbol} in {year}")
        return None
    
    # Concatenate all months
    df = pd.concat(monthly_dfs, axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    print(f"Total: {len(df):,} bars for {year}")
    
    # Validate and fill gaps
    report = validate_ohlcv(df, interval)
    print(f"Validation: {report.coverage_pct:.1f}% coverage, {report.gap_count} gaps")
    
    if report.gap_count > 0:
        df = fill_gaps(df, interval)
        print(f"Filled gaps: now {len(df):,} bars")
    
    # Save to parquet
    file_path = save_yearly(df, "binance", market, symbol, interval, year)
    print(f"Saved: {file_path}")
    
    if progress_callback:
        progress_callback(end_month, end_month, "Complete")
    
    return file_path


def list_binance_symbols(market: str = "futures") -> list[str]:
    """
    Fetch list of available symbols from Binance API.
    
    Note: This uses the live API, not Binance Vision.
    """
    if market == "futures":
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    else:
        url = "https://api.binance.com/api/v3/exchangeInfo"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        symbols = [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
        return sorted(symbols)
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


def get_symbol_start_year(symbol: str, interval: str, market: str = "futures") -> Optional[int]:
    """
    Find the earliest year with data for a symbol.
    
    Checks years going backwards until no data is found.
    """
    current_year = datetime.now(timezone.utc).year
    
    for year in range(current_year, 2017, -1):  # Binance started ~2017
        # Check if January has data
        url = get_monthly_url(symbol, interval, year, 1, market)
        try:
            response = requests.head(url, timeout=5)
            if response.status_code != 200:
                # No data for this year's January, previous year was the earliest
                return year + 1 if year < current_year else None
        except:
            continue
    
    return 2017  # Binance inception


def download_binance_range(
    symbol: str,
    interval: str,
    start_year: int,
    end_year: int,
    market: str = "futures",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[Path]:
    """
    Download multiple years of data.
    
    Returns:
        List of paths to saved parquet files
    """
    paths = []
    total_years = end_year - start_year + 1
    
    for i, year in enumerate(range(start_year, end_year + 1)):
        if progress_callback:
            progress_callback(i + 1, total_years, f"Year {year}")
        
        path = download_binance_year(symbol, interval, year, market)
        if path:
            paths.append(path)
    
    return paths


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    """CLI entry point for downloading Binance Vision data."""
    import argparse
    import time

    current_year = datetime.now(timezone.utc).year

    parser = argparse.ArgumentParser(
        description="Download Binance Vision USDM futures klines and save as yearly OHLCV parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
  python -m core.data.binance --symbol BTCUSDT --year 2025
  python -m core.data.binance --symbol ETHUSDT --start-year 2023 --end-year 2025
  python -m core.data.binance --symbol SOLUSDT --year 2025 --intervals 1h,1d

Data is downloaded from https://data.binance.vision/ (no API key required).
Files are saved to: data/binance/futures/{{SYMBOL}}/{{interval}}/{{year}}.parquet""",
    )
    parser.add_argument("--symbol", "-s", required=True, help="Trading pair (e.g. BTCUSDT, ETHUSDT)")
    parser.add_argument("--year", "-y", type=int, default=None, help="Single year to download")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for range download")
    parser.add_argument("--end-year", type=int, default=None, help=f"End year for range download (default: {current_year})")
    parser.add_argument("--intervals", "-i", default="1m,1h,1d",
                        help="Comma-separated intervals to download (default: 1m,1h,1d)")
    parser.add_argument("--market", "-m", default="futures", choices=["futures", "spot"],
                        help="Market type (default: futures/USDM)")
    parser.add_argument("--force", action="store_true", help="Re-download even if data already exists")

    args = parser.parse_args()

    # Resolve year range
    if args.year:
        start_year = args.year
        end_year = args.year
    elif args.start_year:
        start_year = args.start_year
        end_year = args.end_year or current_year
    else:
        print("ERROR: Specify --year or --start-year")
        raise SystemExit(1)

    # Parse and validate intervals
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    for iv in intervals:
        if iv not in INTERVALS:
            print(f"ERROR: Invalid interval '{iv}'. Valid: {', '.join(INTERVALS)}")
            raise SystemExit(1)

    symbol = args.symbol.upper()
    years = list(range(start_year, end_year + 1))

    print(f"Downloading {symbol} from Binance Vision")
    print(f"Years: {start_year}–{end_year} | Intervals: {', '.join(intervals)} | Market: {args.market}")
    print()

    t0 = time.time()
    all_paths = []

    for interval in intervals:
        print(f"=== {interval} ===")
        for year in years:
            path = download_binance_year(
                symbol=symbol,
                interval=interval,
                year=year,
                market=args.market,
                force=args.force,
            )
            if path:
                all_paths.append(path)
        print()

    elapsed = time.time() - t0
    print(f"Done! Elapsed: {elapsed:.0f}s")

    if all_paths:
        total_size = sum(p.stat().st_size for p in all_paths if p.exists())
        print(f"Total output: {len(all_paths)} file(s), {_fmt_bytes(total_size)}")
        print("\nSaved files:")
        for p in all_paths:
            sz = p.stat().st_size if p.exists() else 0
            print(f"  {p}  ({_fmt_bytes(sz)})")


if __name__ == "__main__":
    main()
