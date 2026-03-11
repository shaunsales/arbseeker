"""
Strategy data infrastructure.

Provides:
- StrategyDataSpec: Declares what data a strategy needs (venue, ticker, intervals, indicators)
- StrategyData: Look-ahead-safe accessor for multi-interval data during backtesting
- StrategyDataBuilder: Builds parquet files + manifest from a strategy's data_spec()
- StrategyDataValidator: Validates that built data matches a strategy's spec
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import compute_indicators, get_warmup_bars


# ---------------------------------------------------------------------------
# StrategyDataSpec — what data a strategy needs
# ---------------------------------------------------------------------------

@dataclass
class StrategyDataSpec:
    """
    Declares the data dependencies for a strategy.

    Returned by SingleAssetStrategy.data_spec(). The web UI reads this to know
    what data to build, and the engine validates it before running.

    Attributes:
        venue: Exchange name (e.g. "binance")
        market: Market type (e.g. "futures")
        ticker: Trading pair (e.g. "BTCUSDT")
        intervals: Dict mapping interval string to list of (indicator_name, params) tuples.
                   "1m" is always required (execution interval).
                   Example: {"1m": [], "1h": [("adx", {"length": 14}), ("sma", {"length": 50})]}
    """
    venue: str
    market: str
    ticker: str
    intervals: dict[str, list[tuple[str, dict]]] = field(default_factory=dict)

    def __post_init__(self):
        if "1m" not in self.intervals:
            raise ValueError("StrategyDataSpec must include '1m' interval (execution interval)")

    def indicator_columns(self, interval: str) -> list[str]:
        """Return expected indicator column names for an interval.

        This is a best-effort mapping based on how core.indicators names columns.
        Used for validation — checks that built parquets contain the right columns.
        """
        from core.indicators.indicators import _add_indicator
        cols = []
        for ind_name, params in self.intervals.get(interval, []):
            cols.extend(_indicator_column_names(ind_name, params))
        return cols

    def warmup_bars(self, interval: str) -> int:
        """
        Calculate the number of warmup bars needed for an interval.

        Returns the maximum lookback across all indicators for this interval,
        with a 10% safety margin. Returns 0 for intervals with no indicators.
        """
        indicators = self.intervals.get(interval, [])
        return get_warmup_bars(indicators)

    def max_warmup_seconds(self) -> int:
        """
        Calculate the maximum warmup time in seconds across all intervals.

        This is used by the data builder to know how much extra data to fetch
        before the user's requested start_date.
        """
        max_seconds = 0
        for interval, indicators in self.intervals.items():
            wb = get_warmup_bars(indicators)
            if wb > 0:
                secs = wb * _interval_seconds(interval)
                max_seconds = max(max_seconds, secs)
        return max_seconds

    def warmup_periods(self) -> int:
        """
        Number of extra monthly periods to prepend for indicator warmup.

        Converts max_warmup_seconds to whole months (rounded up + 1 for safety).
        """
        secs = self.max_warmup_seconds()
        if secs == 0:
            return 0
        days = secs / 86400
        months = int(days / 30) + 2  # +2: round up + safety margin
        return months

    def to_dict(self) -> dict:
        """Serialise to dict for JSON manifest."""
        return {
            "venue": self.venue,
            "market": self.market,
            "ticker": self.ticker,
            "intervals": {
                k: [(name, params) for name, params in v]
                for k, v in self.intervals.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyDataSpec":
        """Deserialise from JSON manifest dict."""
        intervals = {
            k: [(name, params) for name, params in v]
            for k, v in d["intervals"].items()
        }
        return cls(
            venue=d["venue"],
            market=d["market"],
            ticker=d["ticker"],
            intervals=intervals,
        )


# ---------------------------------------------------------------------------
# Interval duration helpers
# ---------------------------------------------------------------------------

_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def _interval_seconds(interval: str) -> int:
    """Return duration of an interval in seconds."""
    if interval in _INTERVAL_SECONDS:
        return _INTERVAL_SECONDS[interval]
    raise ValueError(f"Unknown interval: {interval}")


# ---------------------------------------------------------------------------
# StrategyData — look-ahead-safe multi-interval accessor
# ---------------------------------------------------------------------------

class StrategyData:
    """
    Look-ahead-safe accessor for multi-interval data during backtesting.

    Wraps loaded parquet DataFrames and provides timestamp-based lookups
    that only return fully closed bars (preventing look-ahead bias).

    Usage in a strategy's on_bar():
        m1 = data.bar("1m", timestamp)     # current 1m bar
        h1 = data.bar("1h", timestamp)     # last fully closed 1h bar
        last_5h = data.bars("1h", timestamp, n=5)  # last 5 closed 1h bars
    """

    def __init__(self, frames: dict[str, pd.DataFrame], intervals: dict[str, int]):
        """
        Args:
            frames: Dict mapping interval string to DataFrame (DatetimeIndex).
            intervals: Dict mapping interval string to duration in seconds.
        """
        self._frames = frames
        self._intervals = intervals
        # Pre-compute numpy arrays of timestamps for fast searchsorted
        self._ts_arrays: dict[str, np.ndarray] = {}
        for interval, df in frames.items():
            self._ts_arrays[interval] = df.index.values.astype("int64")

    @classmethod
    def from_strategy_folder(cls, folder: Path, spec: StrategyDataSpec) -> "StrategyData":
        """Load StrategyData from a strategy's data folder."""
        data_dir = folder / "data"
        frames = {}
        intervals = {}
        for interval in spec.intervals:
            parquet_path = data_dir / f"{interval}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Missing {interval}.parquet — build data first "
                    f"(expected at {parquet_path})"
                )
            df = pd.read_parquet(parquet_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{interval}.parquet index must be DatetimeIndex")
            frames[interval] = df
            intervals[interval] = _interval_seconds(interval)
        return cls(frames, intervals)

    def bar(self, interval: str, timestamp: pd.Timestamp) -> pd.Series:
        """
        Return the last FULLY CLOSED bar at or before timestamp.

        For the execution interval (1m), this returns the bar whose open_time
        equals timestamp (the current bar — its close is the current price).

        For larger intervals, a bar is only available after it closes:
        bar("1h", 14:32) returns the 13:00 bar (closed at 14:00),
        NOT the in-progress 14:00 bar.
        """
        if interval not in self._frames:
            raise KeyError(f"Interval '{interval}' not loaded. Available: {list(self._frames.keys())}")

        df = self._frames[interval]
        duration_ns = self._intervals[interval] * 1_000_000_000  # seconds → nanoseconds
        ts_ns = timestamp.value  # nanosecond timestamp

        # The latest bar whose close time <= timestamp
        # close_time = open_time + duration
        # So we need: open_time + duration <= timestamp
        # i.e. open_time <= timestamp - duration
        cutoff_ns = ts_ns - duration_ns

        ts_array = self._ts_arrays[interval]
        # searchsorted: find rightmost index where open_time <= cutoff
        idx = np.searchsorted(ts_array, cutoff_ns, side="right") - 1

        if idx < 0:
            raise ValueError(
                f"No closed {interval} bar available at {timestamp}. "
                f"First bar opens at {df.index[0]}, closes at offset +{self._intervals[interval]}s"
            )

        return df.iloc[idx]

    def bars(self, interval: str, timestamp: pd.Timestamp, n: int) -> pd.DataFrame:
        """
        Return the last N fully closed bars ending at or before timestamp.

        Returns a DataFrame with N rows (or fewer if not enough history).
        """
        if interval not in self._frames:
            raise KeyError(f"Interval '{interval}' not loaded. Available: {list(self._frames.keys())}")

        df = self._frames[interval]
        duration_ns = self._intervals[interval] * 1_000_000_000
        ts_ns = timestamp.value
        cutoff_ns = ts_ns - duration_ns

        ts_array = self._ts_arrays[interval]
        end_idx = np.searchsorted(ts_array, cutoff_ns, side="right")
        start_idx = max(0, end_idx - n)

        return df.iloc[start_idx:end_idx]

    def current_bar(self, timestamp: pd.Timestamp) -> pd.Series:
        """Shortcut: return the current 1m bar (execution bar)."""
        df = self._frames["1m"]
        ts_array = self._ts_arrays["1m"]
        ts_ns = timestamp.value
        idx = np.searchsorted(ts_array, ts_ns, side="right") - 1
        if idx < 0:
            raise ValueError(f"No 1m bar at {timestamp}")
        return df.iloc[idx]

    def snapshot(self, timestamp: pd.Timestamp) -> dict:
        """
        Capture a snapshot of all intervals at the current timestamp.

        Used for decision context capture — records what data was available
        when a trade signal was generated.
        """
        result = {}
        for interval in self._frames:
            try:
                bar = self.bar(interval, timestamp)
                result[interval] = {
                    "open_time": str(bar.name),
                    **{col: _serialise_value(bar[col]) for col in bar.index},
                }
            except ValueError:
                result[interval] = None
        return result


# ---------------------------------------------------------------------------
# Strategy folder & manifest helpers
# ---------------------------------------------------------------------------

STRATEGIES_OUTPUT_DIR = Path("output/strategies")


def strategy_folder(strategy_name: str) -> Path:
    """Return the output folder for a strategy."""
    return STRATEGIES_OUTPUT_DIR / strategy_name


def load_manifest(strategy_name: str) -> Optional[dict]:
    """Load manifest.json for a strategy, or None if it doesn't exist."""
    manifest_path = strategy_folder(strategy_name) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(strategy_name: str, manifest: dict) -> Path:
    """Save manifest.json for a strategy."""
    folder = strategy_folder(strategy_name)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


# ---------------------------------------------------------------------------
# StrategyDataBuilder — builds parquet files from a data_spec
# ---------------------------------------------------------------------------

class StrategyDataBuilder:
    """
    Builds parquet data files for a strategy.

    Reads the strategy's data_spec(), downloads/loads OHLCV for each interval,
    computes indicators, validates, and saves to the strategy folder.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def build(
        self,
        strategy_name: str,
        spec: StrategyDataSpec,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Build all data files for a strategy.

        Args:
            strategy_name: Name of the strategy (used as folder name)
            spec: StrategyDataSpec from the strategy class
            start_date: Start date string (e.g. "2024-01")
            end_date: End date string (e.g. "2025-01")

        Returns:
            Manifest dict with build metadata
        """
        from core.data.storage import load_ohlcv

        folder = strategy_folder(strategy_name)
        data_dir = folder / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (folder / "results").mkdir(parents=True, exist_ok=True)

        # Parse date range into list of periods
        periods = _date_range_to_periods(start_date, end_date)

        # Prepend extra periods for indicator warmup
        warmup_months = spec.warmup_periods()
        if warmup_months > 0:
            extended_start = _subtract_months(start_date, warmup_months)
            warmup_periods = _date_range_to_periods(extended_start, start_date)
            # Remove last element to avoid duplicating start_date
            warmup_periods = [p for p in warmup_periods if p not in periods]
            all_periods = warmup_periods + periods
            if self.verbose:
                print(f"Indicator warmup: prepending {len(warmup_periods)} extra months "
                      f"({extended_start} → {start_date})")
        else:
            all_periods = periods

        # The user's requested start timestamp — indicators must be warm by here
        requested_start = pd.Timestamp(f"{start_date}-01", tz="UTC")

        quality_summary = {}

        for interval, indicators in spec.intervals.items():
            if self.verbose:
                wb = spec.warmup_bars(interval)
                print(f"Building {interval} data ({len(indicators)} indicators, "
                      f"{wb} warmup bars needed)...")

            # Load OHLCV (including warmup periods)
            df = load_ohlcv(
                spec.venue, spec.market, spec.ticker, interval,
                periods=all_periods,
            )

            if df is None or len(df) == 0:
                raise ValueError(
                    f"No OHLCV data for {spec.venue}/{spec.market}/{spec.ticker}/{interval} "
                    f"periods={periods}"
                )

            if self.verbose:
                print(f"  Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

            # Compute indicators (on full data including warmup)
            if indicators:
                df = compute_indicators(df, indicators)
                if self.verbose:
                    ind_names = [name for name, _ in indicators]
                    print(f"  Computed indicators: {', '.join(ind_names)}")

            # Check that indicators are warm at requested_start
            if indicators:
                all_valid_mask = df.notna().all(axis=1)
                if all_valid_mask.any():
                    first_valid_ts = df.index[all_valid_mask.values.argmax()]
                    if first_valid_ts > requested_start:
                        if self.verbose:
                            print(f"  WARNING: {interval} indicators not fully warm at "
                                  f"{requested_start}. First valid bar: {first_valid_ts}. "
                                  f"Need more historical data for warmup.")

            bars_before_trim = len(df)

            # Trim to requested date range — discard warmup-only bars.
            # Output parquets contain only the user's requested range with
            # all indicators fully warmed from bar 0.
            df = df[df.index >= requested_start]

            if len(df) == 0:
                raise ValueError(
                    f"No data for {interval} after trimming to requested start "
                    f"{requested_start}. Loaded {bars_before_trim} bars total."
                )

            trimmed = bars_before_trim - len(df)
            if self.verbose and trimmed > 0:
                print(f"  Trimmed {trimmed} warmup bars, "
                      f"{len(df):,} bars in output range")

            # Save parquet — clean data, indicators warm from bar 0
            parquet_path = data_dir / f"{interval}.parquet"
            df.to_parquet(parquet_path)

            # Quality stats on the trimmed (output) data
            total_bars = len(df)
            ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            null_ohlcv_bars = int(df[ohlcv_cols].isnull().any(axis=1).sum()) if ohlcv_cols else 0
            null_any_bars = int(df.isnull().any(axis=1).sum())

            quality_summary[interval] = {
                "bars": total_bars,
                "start": str(df.index[0]),
                "end": str(df.index[-1]),
                "null_bars": null_ohlcv_bars,
                "null_indicator_bars": max(0, null_any_bars - null_ohlcv_bars),
                "warmup_bars_fetched": trimmed,
                "coverage_pct": round((total_bars - null_ohlcv_bars) / total_bars * 100, 1) if total_bars > 0 else 0,
                "columns": list(df.columns),
            }

            if self.verbose:
                print(f"  Saved {parquet_path} ({total_bars:,} bars)")
                if null_any_bars > 0:
                    print(f"  WARNING: {null_any_bars} bars still have NaN values "
                          f"after trimming — may need more warmup data")

        # Build manifest
        manifest = {
            "strategy_name": strategy_name,
            "spec": spec.to_dict(),
            "date_range": {"start": start_date, "end": end_date},
            "periods": periods,
            "built_at": datetime.utcnow().isoformat(),
            "quality": quality_summary,
        }

        manifest_path = save_manifest(strategy_name, manifest)

        if self.verbose:
            print(f"\nManifest saved to {manifest_path}")
            print(f"Strategy data ready at {folder}")

        return manifest


# ---------------------------------------------------------------------------
# StrategyDataValidator — validates data matches spec
# ---------------------------------------------------------------------------

class StrategyDataValidator:
    """Validates that built data files match a strategy's data_spec."""

    @staticmethod
    def validate(strategy_name: str, spec: StrategyDataSpec) -> list[str]:
        """
        Validate that all data files exist and match the spec.

        Returns:
            List of error messages (empty = valid)
        """
        errors = []
        folder = strategy_folder(strategy_name)
        data_dir = folder / "data"

        # Check folder exists
        if not folder.exists():
            errors.append(f"Strategy folder not found: {folder}")
            return errors

        # Check manifest exists
        manifest = load_manifest(strategy_name)
        if manifest is None:
            errors.append(f"manifest.json not found — build data first")
            return errors

        # Check each interval parquet exists
        start_timestamps = []
        for interval in spec.intervals:
            parquet_path = data_dir / f"{interval}.parquet"
            if not parquet_path.exists():
                errors.append(f"Missing {interval}.parquet — build data first")
                continue

            # Load and check columns
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                errors.append(f"Failed to read {interval}.parquet: {e}")
                continue

            # Check OHLCV columns
            required_ohlcv = {"open", "high", "low", "close", "volume"}
            missing_ohlcv = required_ohlcv - set(df.columns)
            if missing_ohlcv:
                errors.append(f"{interval}.parquet missing OHLCV columns: {missing_ohlcv}")

            # Check indicator columns
            expected_indicators = _expected_indicator_columns(spec.intervals.get(interval, []))
            missing_indicators = set(expected_indicators) - set(df.columns)
            if missing_indicators:
                errors.append(
                    f"{interval}.parquet missing indicator columns: {missing_indicators}. "
                    f"Rebuild data to include them."
                )

            # Track start timestamps for alignment check
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                start_timestamps.append((interval, df.index[0]))

        # Check all intervals share the same start date (within 1 day tolerance)
        if len(start_timestamps) > 1:
            first_start = start_timestamps[0][1]
            for interval, start in start_timestamps[1:]:
                diff = abs((start - first_start).total_seconds())
                if diff > 86400:  # > 1 day difference
                    errors.append(
                        f"Start timestamp mismatch: {start_timestamps[0][0]} starts at "
                        f"{first_start}, but {interval} starts at {start}. Rebuild all data."
                    )

        return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subtract_months(date_str: str, months: int) -> str:
    """
    Subtract N months from a 'YYYY-MM' date string.

    Returns a new 'YYYY-MM' string.
    """
    if re.match(r"^\d{4}-\d{2}$", date_str):
        year, month = int(date_str[:4]), int(date_str[5:7])
    elif re.match(r"^\d{4}$", date_str):
        year, month = int(date_str), 1
    else:
        raise ValueError(f"Invalid date format: {date_str}")

    # Subtract months
    total_months = year * 12 + (month - 1) - months
    new_year = total_months // 12
    new_month = total_months % 12 + 1
    return f"{new_year:04d}-{new_month:02d}"


def _date_range_to_periods(start: str, end: str) -> list[str]:
    """
    Convert start/end date strings to a list of period strings.

    Supports formats: "2024-01" (monthly) or "2024" (yearly).
    Returns monthly periods between start and end (inclusive).
    """
    # Parse start
    if re.match(r"^\d{4}-\d{2}$", start):
        start_year, start_month = int(start[:4]), int(start[5:7])
    elif re.match(r"^\d{4}$", start):
        start_year, start_month = int(start), 1
    else:
        raise ValueError(f"Invalid start date format: {start}. Use 'YYYY-MM' or 'YYYY'")

    # Parse end
    if re.match(r"^\d{4}-\d{2}$", end):
        end_year, end_month = int(end[:4]), int(end[5:7])
    elif re.match(r"^\d{4}$", end):
        end_year, end_month = int(end), 12
    else:
        raise ValueError(f"Invalid end date format: {end}. Use 'YYYY-MM' or 'YYYY'")

    periods = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        periods.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    return periods


def _expected_indicator_columns(indicators: list[tuple[str, dict]]) -> list[str]:
    """Return expected column names for a list of indicator specs."""
    cols = []
    for name, params in indicators:
        cols.extend(_indicator_column_names(name, params))
    return cols


def _indicator_column_names(name: str, params: dict) -> list[str]:
    """Return the column names that an indicator will produce."""
    n = name.lower()
    if n == "sma":
        return [f"SMA_{params.get('length', 20)}"]
    elif n == "ema":
        return [f"EMA_{params.get('length', 20)}"]
    elif n == "wma":
        return [f"WMA_{params.get('length', 20)}"]
    elif n == "rsi":
        return [f"RSI_{params.get('length', 14)}"]
    elif n == "adx":
        length = params.get("length", 14)
        return [f"ADX_{length}"]
    elif n == "dmp":
        length = params.get("length", 14)
        return [f"DMP_{length}"]
    elif n == "dmn":
        length = params.get("length", 14)
        return [f"DMN_{length}"]
    elif n == "macd":
        f, s, sig = params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)
        return [f"MACD_{f}_{s}_{sig}", f"MACDh_{f}_{s}_{sig}", f"MACDs_{f}_{s}_{sig}"]
    elif n == "bbands":
        length, std = params.get("length", 20), params.get("std", 2)
        return [f"BBL_{length}_{std}", f"BBM_{length}_{std}", f"BBU_{length}_{std}",
                f"BBB_{length}_{std}", f"BBP_{length}_{std}"]
    elif n == "atr":
        return [f"ATR_{params.get('length', 14)}"]
    elif n == "natr":
        return [f"NATR_{params.get('length', 14)}"]
    elif n == "stoch":
        k, d = params.get("k", 14), params.get("d", 3)
        return [f"STOCHk_{k}_{d}", f"STOCHd_{k}_{d}"]
    elif n == "cci":
        return [f"CCI_{params.get('length', 20)}"]
    elif n == "willr":
        return [f"WILLR_{params.get('length', 14)}"]
    elif n == "roc":
        return [f"ROC_{params.get('length', 10)}"]
    elif n == "mom":
        return [f"MOM_{params.get('length', 10)}"]
    elif n == "kc":
        length = params.get("length", 20)
        return [f"KCL_{length}", f"KCM_{length}", f"KCU_{length}"]
    elif n == "obv":
        return ["OBV"]
    elif n == "vwap":
        return ["VWAP"]
    elif n == "mfi":
        return [f"MFI_{params.get('length', 14)}"]
    elif n == "ad":
        return ["AD"]
    else:
        return [name.upper()]


def _serialise_value(v):
    """Convert a value to JSON-safe format."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if not np.isnan(v) else None
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v
