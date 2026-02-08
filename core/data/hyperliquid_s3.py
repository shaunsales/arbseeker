"""
Hyperliquid S3 raw data downloader.

Downloads raw hourly LZ4 trade fill files from the public Hyperliquid S3
bucket and saves them to disk at:
    data/sources/hyperliquid/hourly/YYYYMMDD/H.lz4

This is Stage 1 of a two-stage pipeline:
    Stage 1 (this module): Download raw LZ4 files from S3 -> local disk
    Stage 2 (hyperliquid_build): Parse LZ4 files -> per-symbol OHLCV parquets

Uses boto3 for S3 access — no AWS CLI dependency required.

Requires:
- AWS credentials (via env vars or CLI flags)
- boto3 package for S3 access
"""

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable

import boto3
import botocore.exceptions

from core.data.storage import DATA_DIR


S3_BUCKET_NAME = "hl-mainnet-node-data"
S3_PREFIX = "node_fills_by_block/hourly"
S3_REGION = "eu-west-2"

SOURCES_DIR = DATA_DIR / "sources" / "hyperliquid" / "hourly"


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


def list_local_dates() -> list[str]:
    """List YYYYMMDD date folders already downloaded to sources."""
    if not SOURCES_DIR.exists():
        return []
    return sorted(
        d.name for d in SOURCES_DIR.iterdir()
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
    )


def download_s3_range(
    start_date: datetime,
    end_date: datetime,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
    force: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[Path]:
    """
    Download raw LZ4 trade files from Hyperliquid S3 to local disk.

    Saves files to: data/sources/hyperliquid/hourly/YYYYMMDD/H.lz4
    Skips hours that already exist on disk unless force=True.

    Args:
        start_date: Start date (UTC)
        end_date: End date (UTC, inclusive)
        aws_access_key_id: AWS access key
        aws_secret_access_key: AWS secret key
        aws_session_token: Optional AWS session token
        force: Re-download even if file exists on disk
        log_callback: Called with each log line
        progress_callback: Called with (current_day, total_days, message)
        cancel_event: If set, download will stop when this event is triggered

    Returns:
        List of paths to saved LZ4 files
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
    log(f"Downloading raw LZ4 from S3: {start_date.date()} to {end_date.date()} ({total_days} days)")
    log(f"Bucket: {S3_BUCKET_NAME}/{S3_PREFIX}")
    log(f"Output: {SOURCES_DIR}/")
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
            if earliest > end_date:
                log(f"  Earliest data ({earliest.date()}) is after end date ({end_date.date()}) — no data in range.")
                return []
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

    saved_paths: list[Path] = []
    total_bytes = 0
    current_date = start_date
    day_num = 0

    while current_date <= end_date:
        if is_cancelled():
            log("")
            log("⚠ Download cancelled by user.")
            break

        day_num += 1
        date_str = current_date.strftime("%Y%m%d")
        day_dir = SOURCES_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(day_num, total_days, f"Day {day_num}/{total_days}: {current_date.date()}")

        log(f"[Day {day_num}/{total_days}] {current_date.date()}")

        hours_ok = 0
        hours_skipped = 0
        hours_missing = 0
        day_bytes = 0

        for hour in range(24):
            if is_cancelled():
                break

            dest = day_dir / f"{hour}.lz4"

            # Skip if already downloaded
            if dest.exists() and not force:
                hours_skipped += 1
                saved_paths.append(dest)
                continue

            nbytes = _download_hour(s3_client, date_str, hour, dest)
            if nbytes > 0:
                hours_ok += 1
                day_bytes += nbytes
                total_bytes += nbytes
                saved_paths.append(dest)
            else:
                hours_missing += 1

        parts = []
        if hours_ok:
            parts.append(f"{hours_ok} downloaded")
        if hours_skipped:
            parts.append(f"{hours_skipped} cached")
        if hours_missing:
            parts.append(f"{hours_missing} missing")
        status = " | ".join(parts)

        if day_bytes:
            log(f"  {status} | {_fmt_bytes(day_bytes)}")
        else:
            log(f"  {status}")
            if hours_missing == 24:
                log(f"  ⚠ No data available for this date (future or too old?)")

        current_date += timedelta(days=1)

    log("")
    log(f"Done! {len(saved_paths)} files | {_fmt_bytes(total_bytes)} downloaded")
    log(f"Source files saved to: {SOURCES_DIR}/")
    log("")
    log("Next step: run the builder to create OHLCV parquets:")
    log("  python -m core.data.hyperliquid_build --help")

    if progress_callback:
        progress_callback(total_days, total_days, "Complete")

    return saved_paths


def _download_hour(s3_client, date_str: str, hour: int, dest: Path) -> int:
    """Download one hourly LZ4 file from S3 and save to disk.

    Returns:
        Number of bytes downloaded, or 0 if file doesn't exist.
    """
    key = f"{S3_PREFIX}/{date_str}/{hour}.lz4"

    try:
        resp = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            RequestPayer="requester",
        )
        data = resp["Body"].read()
    except botocore.exceptions.ClientError:
        return 0

    dest.write_bytes(data)
    return len(data)


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _load_env_file(path: Optional[Path] = None) -> dict[str, str]:
    """Load key=value pairs from a .env file. Ignores comments and blank lines."""
    if path is None:
        path = Path(__file__).parent.parent.parent / ".env"
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            # Strip optional quotes
            value = value.strip().strip("'\"")
            env[key.strip()] = value
    return env


def main():
    """CLI entry point for downloading Hyperliquid S3 raw data."""
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser(
        description="Download raw Hyperliquid trade data (LZ4) from S3 to local disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m core.data.hyperliquid_s3 --start 2026-01-01 --end 2026-01-02
  python -m core.data.hyperliquid_s3 --start 2026-01-01 --end 2026-01-31 --force

Files are saved to: data/sources/hyperliquid/hourly/YYYYMMDD/H.lz4
Then use the builder to create OHLCV parquets:
  python -m core.data.hyperliquid_build --symbol BTC,ETH --start 2026-01-01 --end 2026-01-02

AWS credentials are resolved in order:
  1. --aws-key / --aws-secret CLI flags
  2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables
  3. .env file in the project root""",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist on disk")
    parser.add_argument("--aws-key", default=None, help="AWS Access Key ID (or set AWS_ACCESS_KEY_ID env var)")
    parser.add_argument("--aws-secret", default=None, help="AWS Secret Access Key (or set AWS_SECRET_ACCESS_KEY env var)")
    parser.add_argument("--aws-token", default=None, help="AWS Session Token (optional)")

    args = parser.parse_args()

    # Resolve AWS credentials: CLI flags > env vars > .env file
    env_file = _load_env_file()
    aws_key = args.aws_key or os.environ.get("AWS_ACCESS_KEY_ID") or env_file.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = args.aws_secret or os.environ.get("AWS_SECRET_ACCESS_KEY") or env_file.get("AWS_SECRET_ACCESS_KEY", "")
    aws_token = args.aws_token or os.environ.get("AWS_SESSION_TOKEN") or env_file.get("AWS_SESSION_TOKEN")

    if not aws_key or not aws_secret:
        print("ERROR: AWS credentials required.")
        print("  Option 1: Add to .env file in project root:")
        print("    AWS_ACCESS_KEY_ID=your_key")
        print("    AWS_SECRET_ACCESS_KEY=your_secret")
        print("  Option 2: Export as environment variables")
        print("  Option 3: Pass --aws-key and --aws-secret flags")
        raise SystemExit(1)

    source = "CLI flags" if args.aws_key else (".env file" if env_file.get("AWS_ACCESS_KEY_ID") else "environment variables")
    print(f"Using AWS credentials from: {source}")

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Validate credentials first
    print("Validating AWS credentials...")
    valid, msg = validate_aws_credentials(aws_key, aws_secret, aws_token)
    if not valid:
        print(f"ERROR: {msg}")
        raise SystemExit(1)
    print(f"  {msg}")
    print()

    t0 = time.time()

    paths = download_s3_range(
        start_date=start_date,
        end_date=end_date,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        force=args.force,
        log_callback=print,
    )

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s")

    if paths:
        total_size = sum(p.stat().st_size for p in paths if p.exists())
        print(f"Total source data on disk: {_fmt_bytes(total_size)}")


if __name__ == "__main__":
    main()
