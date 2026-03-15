"""
Binance Vision data downloader.

Downloads historical klines data from https://data.binance.vision/
Stores as monthly Parquet files.

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

from core.data.storage import save_monthly, get_data_path, list_available_periods, DATA_DIR
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


def download_binance_months(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    market: str = "futures",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    force: bool = False,
) -> list[Path]:
    """
    Download monthly klines data from Binance Vision, saving each month as a parquet.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Bar interval (e.g., '1h')
        start_month: Start month 'YYYY-MM'
        end_month: End month 'YYYY-MM' (inclusive)
        market: 'futures' or 'spot'
        progress_callback: Optional callback(current, total, status_msg)
        force: If True, re-download even if file exists
        
    Returns:
        List of paths to saved parquet files
    """
    # Build list of (year, month) tuples
    sy, sm = map(int, start_month.split("-"))
    ey, em = map(int, end_month.split("-"))
    months_to_download = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months_to_download.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1

    # Skip months in the future
    current_date = datetime.now(timezone.utc)
    months_to_download = [
        (y, m) for y, m in months_to_download
        if (y, m) < (current_date.year, current_date.month) or
           (y, m) == (current_date.year, current_date.month) and current_date.day >= 15
    ]

    if not months_to_download:
        print(f"No complete months in range {start_month} to {end_month}")
        return []

    existing = set(list_available_periods("binance", market, symbol, interval))
    total = len(months_to_download)
    saved_paths = []

    for idx, (year, month) in enumerate(months_to_download, 1):
        period = f"{year}-{month:02d}"

        if progress_callback:
            progress_callback(idx, total, f"Downloading {symbol} {period}...")

        # Skip if already exists
        if period in existing and not force:
            path = get_data_path("binance", market, symbol, interval, period)
            if path.exists():
                print(f"  {period}: already exists, skipping")
                saved_paths.append(path)
                continue

        df = download_month(symbol, interval, year, month, market)
        if df is None or len(df) == 0:
            print(f"  {period}: no data")
            continue

        # Validate and fill gaps
        report = validate_ohlcv(df, interval)
        if report.gap_count > 0:
            df = fill_gaps(df, interval)

        path = save_monthly(df, "binance", market, symbol, interval, year, month)
        saved_paths.append(path)
        print(f"  {period}: {len(df):,} bars ({report.coverage_pct:.0f}% coverage) → {path.name}")

    if progress_callback:
        progress_callback(total, total, "Complete")

    # Auto-download funding rates for futures symbols
    if market == "futures" and saved_paths:
        ensure_funding_rates(symbol, start_month, end_month)

    return saved_paths


# ---------------------------------------------------------------------------
# Funding rate downloads
# ---------------------------------------------------------------------------

def get_funding_path(ticker: str, market: str = "futures") -> Path:
    """Return the path for a symbol's funding rate Parquet."""
    return DATA_DIR / "binance" / market / ticker / "funding.parquet"


def download_funding_rates(
    symbol: str,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Download funding rate history from Binance Futures API and compute monthly medians.
    
    Args:
        symbol: Trading pair (e.g. 'BTCUSDT')
        start_month: Optional earliest month 'YYYY-MM' (default: 2019-09)
        end_month: Optional latest month 'YYYY-MM' (default: current)
        
    Returns:
        DataFrame with columns: month, median_daily_bps, mean_daily_bps, positive_pct, n_records
        Index is the month string. Returns None on failure.
    """
    import statistics
    from collections import defaultdict
    
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rates = []
    
    # Start from given month or earliest available
    if start_month:
        sy, sm = map(int, start_month.split("-"))
        start_time = int(datetime(sy, sm, 1, tzinfo=timezone.utc).timestamp() * 1000)
    else:
        start_time = int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    
    # Paginate forward
    for _ in range(30):  # Max ~30k records
        params = {"symbol": symbol, "limit": 1000, "startTime": start_time}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Funding rate fetch error for {symbol}: {e}")
            break
        
        if not data:
            break
        all_rates.extend(data)
        if len(data) < 1000:
            break
        start_time = data[-1]["fundingTime"] + 1
    
    if not all_rates:
        return None
    
    # Deduplicate and sort
    seen = set()
    unique = []
    for r in all_rates:
        if r["fundingTime"] not in seen:
            seen.add(r["fundingTime"])
            unique.append(r)
    unique.sort(key=lambda x: x["fundingTime"])
    
    # Group by month
    monthly: dict[str, list[float]] = defaultdict(list)
    for r in unique:
        dt = datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc)
        month_key = dt.strftime("%Y-%m")
        monthly[month_key].append(float(r["fundingRate"]))
    
    # Filter by end_month if specified
    if end_month:
        monthly = {k: v for k, v in monthly.items() if k <= end_month}
    
    if not monthly:
        return None
    
    # Build DataFrame
    rows = []
    for month in sorted(monthly.keys()):
        rates = monthly[month]
        median_8h = statistics.median(rates)
        mean_8h = statistics.mean(rates)
        rows.append({
            "month": month,
            "median_daily_bps": round(median_8h * 3 * 10000, 2),  # 3 funding periods/day
            "mean_daily_bps": round(mean_8h * 3 * 10000, 2),
            "positive_pct": round(sum(1 for r in rates if r > 0) / len(rates) * 100, 1),
            "n_records": len(rates),
        })
    
    df = pd.DataFrame(rows).set_index("month")
    return df


def save_funding_rates(symbol: str, df: pd.DataFrame, market: str = "futures") -> Path:
    """Save funding rate DataFrame to Parquet."""
    path = get_funding_path(symbol, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def load_funding_rates(symbol: str, market: str = "futures") -> Optional[pd.DataFrame]:
    """Load funding rate DataFrame from Parquet. Returns None if not found."""
    path = get_funding_path(symbol, market)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def ensure_funding_rates(
    symbol: str,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Ensure funding rate data exists for a symbol. Downloads if missing.
    
    Returns path to the Parquet file, or None if download failed.
    """
    path = get_funding_path(symbol)
    if path.exists() and not force:
        return path
    
    print(f"  Downloading funding rates for {symbol}...")
    df = download_funding_rates(symbol, start_month, end_month)
    if df is None or len(df) == 0:
        print(f"  No funding rate data available for {symbol}")
        return None
    
    saved = save_funding_rates(symbol, df)
    print(f"  Saved {len(df)} months of funding rates → {saved.name}")
    return saved


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

    parser = argparse.ArgumentParser(
        description="Download Binance Vision USDM futures klines and save as monthly OHLCV parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m core.data.binance --symbol BTCUSDT --start 2025-01 --end 2025-12
  python -m core.data.binance --symbol ETHUSDT --start 2024-06 --end 2025-06 --intervals 1h,1d
  python -m core.data.binance --symbol SOLUSDT --start 2025-11 --end 2025-11

Data is downloaded from https://data.binance.vision/ (no API key required).
Files are saved to: data/binance/futures/{SYMBOL}/{interval}/YYYY-MM.parquet""",
    )
    parser.add_argument("--symbol", "-s", required=True, help="Trading pair (e.g. BTCUSDT, ETHUSDT)")
    parser.add_argument("--start", required=True, help="Start month (YYYY-MM)")
    parser.add_argument("--end", required=True, help="End month (YYYY-MM, inclusive)")
    parser.add_argument("--intervals", "-i", default="1m,1h,1d",
                        help="Comma-separated intervals to download (default: 1m,1h,1d)")
    parser.add_argument("--market", "-m", default="futures", choices=["futures", "spot"],
                        help="Market type (default: futures/USDM)")
    parser.add_argument("--force", action="store_true", help="Re-download even if data already exists")

    args = parser.parse_args()

    # Parse and validate intervals
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    for iv in intervals:
        if iv not in INTERVALS:
            print(f"ERROR: Invalid interval '{iv}'. Valid: {', '.join(INTERVALS)}")
            raise SystemExit(1)

    symbol = args.symbol.upper()

    print(f"Downloading {symbol} from Binance Vision")
    print(f"Range: {args.start} to {args.end} | Intervals: {', '.join(intervals)} | Market: {args.market}")
    print()

    t0 = time.time()
    all_paths = []

    for interval in intervals:
        print(f"=== {interval} ===")
        paths = download_binance_months(
            symbol=symbol,
            interval=interval,
            start_month=args.start,
            end_month=args.end,
            market=args.market,
            force=args.force,
        )
        all_paths.extend(paths)
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
