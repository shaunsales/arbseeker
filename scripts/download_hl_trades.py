"""
Download Hyperliquid trade data from S3 and convert to 1-minute OHLCV.

Requires:
- AWS credentials configured (or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN)
- lz4 package: pip install lz4

Usage:
    python scripts/download_hl_trades.py --symbol PAXG --start 2025-08-01 --end 2025-12-31
"""

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("Warning: lz4 package not installed. Using command-line unlz4.")


S3_BUCKET = "s3://hl-mainnet-node-data/node_fills_by_block/hourly"


def download_hour(date_str: str, hour: int, tmp_dir: Path) -> Path | None:
    """Download one hour of trade data from S3."""
    s3_path = f"{S3_BUCKET}/{date_str}/{hour}.lz4"
    local_path = tmp_dir / f"{date_str}_{hour}.lz4"
    
    cmd = [
        "aws", "s3", "cp", s3_path, str(local_path),
        "--request-payer", "requester"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    
    return local_path


def decompress_lz4(lz4_path: Path) -> Path:
    """Decompress an LZ4 file."""
    output_path = lz4_path.with_suffix("")
    
    if HAS_LZ4:
        with lz4.frame.open(lz4_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                f_out.write(f_in.read())
    else:
        subprocess.run(["unlz4", "--rm", str(lz4_path)], check=True)
    
    return output_path


def extract_trades(json_path: Path, symbol: str) -> list[dict]:
    """Extract trades for a specific symbol from the JSON file."""
    trades = []
    
    with open(json_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                block = json.loads(line)
                for event in block.get("events", []):
                    if len(event) >= 2:
                        fill = event[1]
                        if fill.get("coin") == symbol:
                            trades.append({
                                "time": fill["time"],
                                "price": float(fill["px"]),
                                "size": float(fill["sz"]),
                                "side": fill["side"],
                            })
            except json.JSONDecodeError:
                continue
    
    return trades


def trades_to_ohlcv(trades: list[dict], interval: str = "1min") -> pd.DataFrame:
    """Convert trades to OHLCV DataFrame."""
    if not trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    
    # Map user-friendly intervals to pandas resample format
    interval_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "4h": "4h", "1d": "1D",
    }
    resample_interval = interval_map.get(interval, interval)
    
    # Resample to interval
    ohlcv = df["price"].resample(resample_interval).ohlc()
    ohlcv["volume"] = df["size"].resample(resample_interval).sum()
    
    # Rename columns
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    
    # Drop rows with no trades
    ohlcv = ohlcv.dropna()
    
    return ohlcv


def download_day(date: datetime, symbol: str, tmp_dir: Path) -> pd.DataFrame:
    """Download and process one day of data."""
    date_str = date.strftime("%Y%m%d")
    all_trades = []
    
    for hour in range(24):
        print(f"  Hour {hour:02d}...", end=" ", flush=True)
        
        lz4_path = download_hour(date_str, hour, tmp_dir)
        if lz4_path is None:
            print("not found")
            continue
        
        json_path = decompress_lz4(lz4_path)
        trades = extract_trades(json_path, symbol)
        all_trades.extend(trades)
        
        # Cleanup
        json_path.unlink(missing_ok=True)
        lz4_path.unlink(missing_ok=True)
        
        print(f"{len(trades)} trades")
    
    return trades_to_ohlcv(all_trades)


def download_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    interval: str = "1m",
) -> list[Path]:
    """Download a range of dates and save as monthly parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_date = start_date
    monthly_data = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        while current_date <= end_date:
            print(f"Downloading {symbol} {current_date.date()}...")
            
            df = download_day(current_date, symbol, tmp_path)
            
            if not df.empty:
                month_key = current_date.strftime("%Y-%m")
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                monthly_data[month_key].append(df)
                print(f"  Total: {len(df)} candles")
            else:
                print(f"  No data")
            
            current_date += timedelta(days=1)
    
    # Save monthly files
    saved_paths = []
    for month_key, dfs in monthly_data.items():
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        
        output_path = output_dir / f"{month_key}.parquet"
        combined.to_parquet(output_path)
        saved_paths.append(output_path)
        print(f"Saved: {output_path} ({len(combined)} candles)")
    
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Download Hyperliquid trade data")
    parser.add_argument("--symbol", default="PAXG", help="Symbol to download")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1m", help="OHLCV interval (default: 1m)")
    parser.add_argument("--output", default="data/hyperliquid_trades", help="Output directory")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    output_dir = Path(args.output) / args.symbol / args.interval
    
    print(f"Downloading {args.symbol} from {start_date.date()} to {end_date.date()}")
    print(f"Output: {output_dir}")
    print()
    
    paths = download_range(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        interval=args.interval,
    )
    
    print()
    print(f"Done! Saved {len(paths)} files.")


if __name__ == "__main__":
    main()
