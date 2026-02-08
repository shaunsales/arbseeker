"""
Hyperliquid LZ4-to-OHLCV builder.

Reads raw hourly LZ4 trade fill files from:
    data/sources/hyperliquid/hourly/YYYYMMDD/H.lz4

And produces per-symbol monthly OHLCV parquet files at multiple intervals:
    data/hyperliquid/perp/{SYMBOL}-USD/{interval}/YYYY-MM.parquet

This is Stage 2 of a two-stage pipeline:
    Stage 1 (hyperliquid_s3): Download raw LZ4 files from S3 -> local disk
    Stage 2 (this module): Parse LZ4 files -> per-symbol OHLCV parquets

Requires:
- lz4 package for decompression
- pandas for OHLCV resampling
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import lz4.frame
import pandas as pd

from core.data.storage import DATA_DIR, save_monthly
from core.data.market_hours import add_market_open_always


SOURCES_DIR = DATA_DIR / "sources" / "hyperliquid" / "hourly"

# Map user-friendly intervals to pandas resample format
RESAMPLE_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}

DEFAULT_INTERVALS = ["1m", "1h", "1d"]


def list_source_dates() -> list[str]:
    """List YYYYMMDD date folders available in sources."""
    if not SOURCES_DIR.exists():
        return []
    return sorted(
        d.name for d in SOURCES_DIR.iterdir()
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
    )


def _parse_lz4_file(
    path: Path, symbols: Optional[set[str]] = None
) -> dict[str, list[dict]]:
    """Parse an LZ4 trade fill file and extract trades grouped by symbol.

    Args:
        path: Path to the .lz4 file
        symbols: Set of symbols to extract, or None for all

    Returns:
        dict of symbol -> list of trade dicts
    """
    compressed = path.read_bytes()

    try:
        raw = lz4.frame.decompress(compressed)
    except Exception:
        return {}

    trades_by_symbol: dict[str, list[dict]] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            block = json.loads(line)
            for event in block.get("events", []):
                if len(event) >= 2:
                    fill = event[1]
                    coin = fill.get("coin")
                    if coin and (symbols is None or coin in symbols):
                        if coin not in trades_by_symbol:
                            trades_by_symbol[coin] = []
                        trades_by_symbol[coin].append({
                            "time": fill["time"],
                            "price": float(fill["px"]),
                            "size": float(fill["sz"]),
                            "side": fill["side"],
                        })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    return trades_by_symbol


def _trades_to_ohlcv(trades: list[dict]) -> pd.DataFrame:
    """Convert raw trades to 1-minute OHLCV DataFrame."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    ohlcv = df["price"].resample("1min").ohlc()
    ohlcv["volume"] = df["size"].resample("1min").sum()
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    ohlcv = ohlcv.dropna()

    return ohlcv


def _resample_ohlcv(df_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV to a larger interval."""
    if df_1m.empty:
        return df_1m

    resample_interval = RESAMPLE_MAP.get(interval, interval)

    resampled = df_1m[["open", "high", "low", "close", "volume"]].resample(resample_interval).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def _symbol_to_ticker(coin: str) -> str:
    """Convert Hyperliquid coin name to ticker with quote suffix."""
    return f"{coin}-USD"


def build_parquets(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symbols: Optional[list[str]] = None,
    intervals: Optional[list[str]] = None,
    force: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """
    Build per-symbol OHLCV parquet files from downloaded LZ4 sources.

    Args:
        start_date: Start date filter (None = all available)
        end_date: End date filter (None = all available)
        symbols: List of symbols to extract (None = all found)
        intervals: List of OHLCV intervals to build (default: 1m, 1h, 1d)
        force: Overwrite existing parquet files
        log_callback: Called with each log line

    Returns:
        List of paths to saved parquet files
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    if intervals is None:
        intervals = DEFAULT_INTERVALS
    symbols_set = set(symbols) if symbols else None
    symbols_label = ", ".join(sorted(symbols)) if symbols else "ALL"
    intervals_label = ", ".join(intervals)

    # Find available source dates
    all_dates = list_source_dates()
    if not all_dates:
        log("No source files found in data/sources/hyperliquid/hourly/")
        log("Run the downloader first: python -m core.data.hyperliquid_s3 --help")
        return []

    # Filter by date range
    if start_date:
        start_str = start_date.strftime("%Y%m%d")
        all_dates = [d for d in all_dates if d >= start_str]
    if end_date:
        end_str = end_date.strftime("%Y%m%d")
        all_dates = [d for d in all_dates if d <= end_str]

    if not all_dates:
        log("No source files in the specified date range.")
        return []

    log(f"Building OHLCV from {len(all_dates)} day(s) of source data")
    log(f"Symbols: {symbols_label} | Intervals: {intervals_label}")
    log(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    log("")

    # symbol -> month_key -> list of 1m OHLCV DataFrames
    monthly_1m: dict[str, dict[str, list[pd.DataFrame]]] = {}
    all_symbols_seen: set[str] = set()

    for day_idx, date_str in enumerate(all_dates, 1):
        day_dir = SOURCES_DIR / date_str
        month_key = f"{date_str[:4]}-{date_str[4:6]}"

        log(f"[Day {day_idx}/{len(all_dates)}] {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

        day_trades_by_symbol: dict[str, list[dict]] = {}
        hours_ok = 0

        for hour in range(24):
            lz4_path = day_dir / f"{hour}.lz4"
            if not lz4_path.exists():
                continue

            hour_data = _parse_lz4_file(lz4_path, symbols_set)
            if hour_data:
                hours_ok += 1
                for sym, trades in hour_data.items():
                    if sym not in day_trades_by_symbol:
                        day_trades_by_symbol[sym] = []
                    day_trades_by_symbol[sym].extend(trades)

        if day_trades_by_symbol:
            total_trades = sum(len(t) for t in day_trades_by_symbol.values())
            total_bars = 0

            for sym, trades in day_trades_by_symbol.items():
                all_symbols_seen.add(sym)
                day_df = _trades_to_ohlcv(trades)
                if day_df.empty:
                    continue
                total_bars += len(day_df)
                if sym not in monthly_1m:
                    monthly_1m[sym] = {}
                if month_key not in monthly_1m[sym]:
                    monthly_1m[sym][month_key] = []
                monthly_1m[sym][month_key].append(day_df)

            log(f"  {hours_ok}/24 hours | {len(day_trades_by_symbol)} symbols | {total_trades:,} trades | {total_bars:,} 1m bars")
        else:
            log(f"  {hours_ok}/24 hours | no trades found")

    # Save monthly files per symbol per interval
    log("")
    if all_symbols_seen:
        log(f"Found {len(all_symbols_seen)} symbols: {', '.join(sorted(all_symbols_seen))}")
    log(f"Saving OHLCV parquets for {len(intervals)} interval(s): {intervals_label}")

    saved_paths = []
    for sym in sorted(monthly_1m.keys()):
        ticker = _symbol_to_ticker(sym)
        months = monthly_1m[sym]
        for month_key in sorted(months.keys()):
            dfs = months[month_key]
            if not dfs:
                continue

            # Combine 1m data for this symbol/month
            base_1m = pd.concat(dfs).sort_index()
            base_1m = base_1m[~base_1m.index.duplicated(keep="first")]

            year, month = map(int, month_key.split("-"))

            for interval in intervals:
                if interval == "1m":
                    resampled = base_1m
                else:
                    resampled = _resample_ohlcv(base_1m, interval)

                if resampled.empty:
                    continue

                # Add market_open column
                out = add_market_open_always(resampled, "market_open")

                path = save_monthly(out, "hyperliquid", "perp", ticker, interval, year, month)
                saved_paths.append(path)
                log(f"  {ticker}/{interval}/{month_key}: {len(out):,} bars → {path.name}")

    log("")
    log(f"Done! Saved {len(saved_paths)} file(s) across {len(monthly_1m)} symbol(s) × {len(intervals)} interval(s).")

    return saved_paths


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    """CLI entry point for building OHLCV parquets from LZ4 sources."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Build per-symbol OHLCV parquet files from downloaded Hyperliquid LZ4 sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m core.data.hyperliquid_build --symbol ALL
  python -m core.data.hyperliquid_build --symbol BTC,ETH,SOL --start 2026-01-01 --end 2026-01-31
  python -m core.data.hyperliquid_build --symbol BTC --intervals 1h,1d

Source files must exist at: data/sources/hyperliquid/hourly/YYYYMMDD/H.lz4
Download them first with: python -m core.data.hyperliquid_s3 --help""",
    )
    parser.add_argument("--symbol", "-s", default="ALL",
                        help="Symbol(s) to extract: ALL for every symbol, or comma-separated (e.g. BTC,ETH,SOL)")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--intervals", "-i", default="1m,1h,1d",
                        help="Comma-separated OHLCV intervals to build (default: 1m,1h,1d)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files")

    args = parser.parse_args()

    # Parse symbols
    if args.symbol.upper() == "ALL":
        symbols = None
    else:
        symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]

    # Parse and validate intervals
    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]
    valid_intervals = set(RESAMPLE_MAP.keys())
    for iv in intervals:
        if iv not in valid_intervals:
            print(f"ERROR: Invalid interval '{iv}'. Valid: {', '.join(sorted(valid_intervals))}")
            raise SystemExit(1)

    start_date = None
    end_date = None
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Show available sources
    available = list_source_dates()
    if not available:
        print("No source files found. Download them first:")
        print("  python -m core.data.hyperliquid_s3 --start YYYY-MM-DD --end YYYY-MM-DD")
        raise SystemExit(1)

    print(f"Source data available: {available[0]} to {available[-1]} ({len(available)} days)")
    print()

    t0 = time.time()

    paths = build_parquets(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        intervals=intervals,
        force=args.force,
        log_callback=print,
    )

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s")

    if paths:
        total_size = sum(p.stat().st_size for p in paths if p.exists())
        print(f"Total output size: {_fmt_bytes(total_size)}")
        print("\nSaved files:")
        for p in paths:
            sz = p.stat().st_size if p.exists() else 0
            print(f"  {p}  ({_fmt_bytes(sz)})")


if __name__ == "__main__":
    main()
