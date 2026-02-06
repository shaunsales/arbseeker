"""
Hyperliquid S3 historical trade data downloader.

Downloads raw trade fills from the public Hyperliquid S3 bucket,
decompresses LZ4 files, extracts trades for a given symbol,
and converts them to OHLCV bars stored in the standard parquet format.

Uses boto3 for S3 access — no AWS CLI dependency required.

Requires:
- AWS credentials (passed as parameters, not from env)
- boto3 package for S3 access
- lz4 package for decompression
"""

import io
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable

import boto3
import botocore.exceptions
import lz4.frame
import pandas as pd

from core.data.storage import save_monthly, get_data_path, list_available_periods
from core.data.market_hours import add_market_open_always


S3_BUCKET_NAME = "hl-mainnet-node-data"
S3_PREFIX = "node_fills_by_block/hourly"
S3_REGION = "eu-west-2"

# Map user-friendly intervals to pandas resample format
RESAMPLE_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}


def _make_session(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
) -> boto3.Session:
    """Create a boto3 session with the given credentials."""
    return boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token or None,
        region_name=S3_REGION,
    )


def validate_aws_credentials(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate AWS credentials using STS get_caller_identity.

    Returns:
        (success, message) tuple
    """
    try:
        session = _make_session(aws_access_key_id, aws_secret_access_key, aws_session_token)
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        account = identity.get("Account", "unknown")
        return True, f"AWS credentials valid (account {account})"

    except botocore.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "InvalidClientTokenId":
            return False, "Invalid AWS Access Key ID"
        elif code == "SignatureDoesNotMatch":
            return False, "Invalid AWS Secret Access Key"
        elif code == "ExpiredTokenException":
            return False, "AWS session token has expired"
        else:
            return False, f"AWS auth error: {e.response['Error']['Message'][:200]}"
    except botocore.exceptions.NoCredentialsError:
        return False, "No credentials provided"
    except botocore.exceptions.EndpointConnectionError:
        return False, "Cannot connect to AWS — check network connection"
    except Exception as e:
        return False, f"Error: {str(e)[:200]}"


def probe_s3_date(
    date: datetime,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
    s3_client=None,
) -> bool:
    """
    Check if a specific date has data in the S3 bucket.
    Lists the date prefix — fast because it only checks for 1 key.

    Returns:
        True if data exists for this date
    """
    try:
        if s3_client is None:
            session = _make_session(aws_access_key_id, aws_secret_access_key, aws_session_token)
            s3_client = session.client("s3")

        date_str = date.strftime("%Y%m%d")
        prefix = f"{S3_PREFIX}/{date_str}/"

        resp = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=prefix,
            MaxKeys=1,
            RequestPayer="requester",
        )
        return resp.get("KeyCount", 0) > 0

    except Exception:
        return False


def find_earliest_s3_date(
    start_date: datetime,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
    max_probes: int = 8,
) -> Optional[datetime]:
    """
    Find the earliest date with data, searching forward from start_date.
    Uses increasing steps to avoid too many probes.

    Returns:
        First date with data found, or None if nothing found within probes
    """
    session = _make_session(aws_access_key_id, aws_secret_access_key, aws_session_token)
    s3_client = session.client("s3")

    if probe_s3_date(start_date, "", "", None, s3_client=s3_client):
        return start_date

    steps = [7, 30, 90, 180, 365]
    probes_used = 1

    for step in steps:
        if probes_used >= max_probes:
            break
        candidate = start_date + timedelta(days=step)
        probes_used += 1
        if probe_s3_date(candidate, "", "", None, s3_client=s3_client):
            return candidate

    return None


def download_s3_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
    force: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[Path]:
    """
    Download trade data from Hyperliquid S3 and save as monthly OHLCV parquets.

    Args:
        symbol: Hyperliquid symbol (e.g., 'PAXG')
        start_date: Start date (UTC)
        end_date: End date (UTC, inclusive)
        interval: OHLCV bar interval (e.g., '15m', '1h')
        aws_access_key_id: AWS access key
        aws_secret_access_key: AWS secret key
        aws_session_token: Optional AWS session token
        force: Re-download even if data exists
        log_callback: Called with each log line for UI streaming
        progress_callback: Called with (current_day, total_days, message)
        cancel_event: If set, download will stop when this event is triggered

    Returns:
        List of paths to saved parquet files
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    def is_cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    session = _make_session(aws_access_key_id, aws_secret_access_key, aws_session_token)
    s3_client = session.client("s3")

    # Calculate total days
    total_days = (end_date - start_date).days + 1
    log(f"Downloading {symbol} from S3: {start_date.date()} to {end_date.date()} ({total_days} days)")
    log(f"Interval: {interval} | Bucket: {S3_BUCKET_NAME}/{S3_PREFIX}")
    log("")

    # Probe start date to validate data availability
    log("Checking data availability...")
    if not probe_s3_date(start_date, "", "", None, s3_client=s3_client):
        log(f"  ⚠ No data found for start date {start_date.date()}")
        log(f"  Searching forward for earliest available data...")
        earliest = find_earliest_s3_date(
            start_date, aws_access_key_id, aws_secret_access_key, aws_session_token,
        )
        if earliest:
            log(f"  Found data starting at {earliest.date()}")
            log(f"  Adjusting start date from {start_date.date()} → {earliest.date()}")
            start_date = earliest
            total_days = (end_date - start_date).days + 1
            log(f"  New range: {total_days} days")
        else:
            log(f"  No data found within search range. The bucket may not have data for these dates.")
            log(f"  Proceeding anyway — individual days will be skipped if empty.")
    else:
        log(f"  ✓ Start date {start_date.date()} has data")
    log("")

    if is_cancelled():
        log("Download cancelled.")
        return []

    monthly_data: dict[str, list[pd.DataFrame]] = {}
    current_date = start_date
    day_num = 0

    while current_date <= end_date:
        if is_cancelled():
            log("")
            log("⚠ Download cancelled by user.")
            break

        day_num += 1
        date_str = current_date.strftime("%Y%m%d")
        month_key = current_date.strftime("%Y-%m")

        if progress_callback:
            progress_callback(day_num, total_days, f"Day {day_num}/{total_days}: {current_date.date()}")

        log(f"[Day {day_num}/{total_days}] {current_date.date()}")

        # Check if this month already exists and we're not forcing
        if not force:
            existing = list_available_periods("hyperliquid", "perp", symbol, interval)
            if month_key in existing and _month_complete(current_date, end_date):
                log(f"  Month {month_key} already exists, skipping...")
                if month_key not in monthly_data:
                    monthly_data[month_key] = []  # mark as handled
                current_date += timedelta(days=1)
                continue

        day_trades = []
        hours_ok = 0
        hours_missing = 0

        for hour in range(24):
            if is_cancelled():
                break

            trades = _download_and_parse_hour(s3_client, date_str, hour, symbol)
            if trades is None:
                hours_missing += 1
                continue

            day_trades.extend(trades)
            hours_ok += 1

        if is_cancelled():
            current_date += timedelta(days=1)
            continue

        if day_trades:
            day_df = _trades_to_ohlcv(day_trades, interval)
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(day_df)
            log(f"  {hours_ok}/24 hours | {len(day_trades):,} trades | {len(day_df):,} bars")
        else:
            log(f"  {hours_ok}/24 hours | no trades for {symbol}")
            if hours_missing == 24:
                log(f"  ⚠ No data available for this date (future or too old?)")

        current_date += timedelta(days=1)

    # Save monthly files
    log("")
    log("Saving monthly files...")

    saved_paths = []
    for month_key in sorted(monthly_data.keys()):
        dfs = monthly_data[month_key]
        if not dfs:
            # Month was skipped (already existed)
            path = get_data_path("hyperliquid", "perp", symbol, interval, month_key)
            if path.exists():
                saved_paths.append(path)
                log(f"  {month_key}: already exists (skipped)")
            continue

        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # Add market_open column
        combined = add_market_open_always(combined, "market_open")

        year, month = map(int, month_key.split("-"))
        path = save_monthly(combined, "hyperliquid", "perp", symbol, interval, year, month)
        saved_paths.append(path)
        log(f"  {month_key}: {len(combined):,} bars → {path.name}")

    log("")
    log(f"Done! Saved {len(saved_paths)} monthly file(s).")

    if progress_callback:
        progress_callback(total_days, total_days, "Complete")

    return saved_paths


def _download_and_parse_hour(
    s3_client, date_str: str, hour: int, symbol: str
) -> Optional[list[dict]]:
    """Download one hour of trade data from S3, decompress, and extract trades in memory."""
    key = f"{S3_PREFIX}/{date_str}/{hour}.lz4"

    try:
        resp = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            RequestPayer="requester",
        )
        compressed = resp["Body"].read()
    except botocore.exceptions.ClientError:
        return None

    # Decompress LZ4 in memory
    try:
        raw = lz4.frame.decompress(compressed)
    except Exception:
        return None

    # Parse trades from JSON lines
    trades = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
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
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    return trades


def _trades_to_ohlcv(trades: list[dict], interval: str = "1m") -> pd.DataFrame:
    """Convert raw trades to OHLCV DataFrame."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    resample_interval = RESAMPLE_MAP.get(interval, interval)

    ohlcv = df["price"].resample(resample_interval).ohlc()
    ohlcv["volume"] = df["size"].resample(resample_interval).sum()
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    ohlcv = ohlcv.dropna()

    return ohlcv


def _month_complete(current_date: datetime, end_date: datetime) -> bool:
    """Check if we'd download the entire month (skip optimization)."""
    return current_date.day == 1
