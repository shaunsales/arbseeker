"""
Strategy Data Builder Routes

Web UI for discovering strategies, building data files, and managing strategy folders.
"""

import importlib
import inspect
import json
import pkgutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.routes.data import _NumpyEncoder

from core.strategy.base import SingleAssetStrategy
from core.data.storage import list_available_periods
from core.strategy.data import (
    StrategyDataSpec,
    StrategyDataBuilder,
    StrategyDataValidator,
    strategy_folder,
    load_manifest,
    _subtract_months,
)
from core.indicators import (
    get_indicator_metadata,
    compute_indicators,
    INDICATOR_REGISTRY,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Strategy auto-discovery
# ---------------------------------------------------------------------------

def discover_strategies() -> list[dict]:
    """
    Scan the strategies/ directory and find all SingleAssetStrategy subclasses
    that implement data_spec(). Returns a list of dicts with strategy metadata.
    """
    import strategies  # top-level strategies package

    results = []

    for importer, modname, ispkg in pkgutil.iter_modules(strategies.__path__):
        # Skip test/internal modules
        if modname.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"strategies.{modname}")
        except Exception:
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, SingleAssetStrategy)
                and obj is not SingleAssetStrategy
                and not attr_name.startswith("_")
            ):
                # Try to instantiate and get data_spec
                try:
                    instance = obj()
                    spec = instance.data_spec()
                except Exception:
                    spec = None

                # Load manifest to get available data date range
                data_date_range = None
                if spec is not None:
                    manifest = load_manifest(attr_name)
                    if manifest and "date_range" in manifest:
                        data_date_range = manifest["date_range"]

                results.append({
                    "class_name": attr_name,
                    "module": f"strategies.{modname}",
                    "name": instance.name if 'instance' in dir() else attr_name,
                    "has_data_spec": spec is not None,
                    "spec": spec,
                    "data_date_range": data_date_range,
                })

    return results


def _get_strategy_instance(class_name: str) -> Optional[SingleAssetStrategy]:
    """Get a strategy instance by class name."""
    for info in discover_strategies():
        if info["class_name"] == class_name:
            module = importlib.import_module(info["module"])
            cls = getattr(module, class_name)
            return cls()
    return None


def _get_strategy_status(strategy_name: str, spec: Optional[StrategyDataSpec]) -> dict:
    """Get the current data build status for a strategy."""
    folder = strategy_folder(strategy_name)
    manifest = load_manifest(strategy_name)

    status = {
        "has_folder": folder.exists(),
        "has_manifest": manifest is not None,
        "manifest": manifest,
        "errors": [],
        "ready": False,
    }

    if manifest and spec:
        errors = StrategyDataValidator.validate(strategy_name, spec)
        status["errors"] = errors
        status["ready"] = len(errors) == 0

    return status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
async def strategy_list():
    """List all discovered strategies."""
    strategies = discover_strategies()
    return {
        "strategies": [
            {
                "class_name": s["class_name"],
                "module": s["module"],
                "has_data_spec": s["has_data_spec"],
            }
            for s in strategies
        ]
    }


@router.get("/spec/{class_name}")
async def strategy_spec(class_name: str):
    """Return strategy spec + build status as JSON."""
    instance = _get_strategy_instance(class_name)
    if instance is None:
        return {"error": "Strategy not found"}

    spec = instance.data_spec()
    if spec is None:
        return {"error": "Strategy does not define data_spec()"}

    status = _get_strategy_status(instance.name, spec)

    doc = instance.__class__.__doc__ or ""
    description = doc.strip().split("\n")[0] if doc.strip() else ""

    # Build intervals with indicator details
    intervals = []
    for interval, indicators in spec.intervals.items():
        ind_list = []
        for ind_name, ind_params in indicators:
            ind_list.append({
                "name": ind_name,
                "params": ind_params,
                "warmup_bars": spec.warmup_bars(interval),
            })
        intervals.append({
            "interval": interval,
            "indicators": ind_list,
            "is_price_only": len(indicators) == 0,
        })

    return {
        "class_name": class_name,
        "description": description,
        "spec": {
            "venue": spec.venue,
            "market": spec.market,
            "ticker": spec.ticker,
            "intervals": intervals,
        },
        "has_manifest": status["has_manifest"],
        "manifest": status["manifest"],
        "errors": status["errors"],
    }


@router.get("/available-dates/{class_name}")
async def available_dates(class_name: str):
    """
    Return the available date range for a strategy's data requirements.

    Computes the intersection of available periods across all intervals,
    then offsets the earliest start by the warmup requirement so the user
    can only select start dates that guarantee warm indicators.
    """
    instance = _get_strategy_instance(class_name)
    if instance is None:
        return {"error": "Strategy not found"}

    spec = instance.data_spec()
    if spec is None:
        return {"error": "No data_spec()"}

    # Find available monthly periods per interval
    per_interval = {}
    for interval in spec.intervals:
        periods = list_available_periods(spec.venue, spec.market, spec.ticker, interval)
        # Normalise yearly periods to monthly (e.g. "2024" → "2024-01" .. "2024-12")
        monthly = []
        for p in periods:
            if len(p) == 4:  # yearly
                for m in range(1, 13):
                    monthly.append(f"{p}-{m:02d}")
            else:
                monthly.append(p)
        per_interval[interval] = set(monthly)

    if not per_interval:
        return {"months": [], "earliest_start": None, "latest_end": None}

    # Intersection: months available for ALL intervals
    common = set.intersection(*per_interval.values()) if per_interval else set()
    if not common:
        return {"months": [], "earliest_start": None, "latest_end": None}

    all_months = sorted(common)
    raw_earliest = all_months[0]
    latest_end = all_months[-1]

    # Offset earliest by warmup requirement
    warmup_months = spec.warmup_periods()
    if warmup_months > 0 and len(all_months) > warmup_months:
        earliest_start = all_months[warmup_months]
    else:
        earliest_start = raw_earliest

    # Per-interval availability (for the legend / tooltip)
    interval_ranges = {}
    for iv, months_set in per_interval.items():
        s = sorted(months_set)
        interval_ranges[iv] = {
            "months": s,
            "start": s[0] if s else None,
            "end": s[-1] if s else None,
            "count": len(s),
        }

    return {
        "months": all_months,
        "raw_earliest": raw_earliest,
        "earliest_start": earliest_start,
        "latest_end": latest_end,
        "warmup_months": warmup_months,
        "per_interval": interval_ranges,
    }


class DeleteRequest(BaseModel):
    class_name: str


@router.post("/delete")
async def delete_strategy_data(req: DeleteRequest):
    """Delete built data files for a strategy, allowing a fresh build."""
    import shutil

    instance = _get_strategy_instance(req.class_name)
    if instance is None:
        return {"success": False, "error": "Strategy not found"}

    folder = strategy_folder(instance.name)
    data_dir = folder / "data"
    results_dir = folder / "results"
    manifest_path = folder / "manifest.json"

    deleted = []
    for target in [data_dir, results_dir, manifest_path]:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(str(target.name))

    return {"success": True, "deleted": deleted, "strategy_name": instance.name}


class BuildRequest(BaseModel):
    class_name: str
    start_date: str
    end_date: str


@router.post("/build")
async def build_strategy_data(req: BuildRequest):
    """Build data files for a strategy."""
    instance = _get_strategy_instance(req.class_name)
    if instance is None:
        return {"success": False, "error": "Strategy not found"}

    spec = instance.data_spec()
    if spec is None:
        return {"success": False, "error": "Strategy does not define data_spec()"}

    try:
        builder = StrategyDataBuilder(verbose=True)
        manifest = builder.build(
            strategy_name=instance.name,
            spec=spec,
            start_date=req.start_date,
            end_date=req.end_date,
        )

        # Validate after build
        errors = StrategyDataValidator.validate(instance.name, spec)

        return {
            "success": len(errors) == 0,
            "strategy_name": instance.name,
            "quality": manifest.get("quality", {}),
            "errors": errors,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Strategy data preview
# ---------------------------------------------------------------------------

# Indicator columns that should overlay on the price chart (same scale as price)
_PRICE_OVERLAY_PREFIXES = ("SMA_", "EMA_", "BB_", "VWAP")


# ---------------------------------------------------------------------------
# Indicator endpoints (for ad-hoc indicator overlay UI)
# ---------------------------------------------------------------------------

@router.get("/indicators")
async def indicator_list():
    """Return all available indicators with their metadata for the UI picker."""
    registry = get_indicator_metadata()
    return {"indicators": registry}


class ComputeIndicatorsRequest(BaseModel):
    class_name: str
    interval: str
    indicators: list[dict]  # [{"name": "sma", "params": {"length": 20}}, ...]


@router.post("/indicators/compute")
async def compute_adhoc_indicators(req: ComputeIndicatorsRequest):
    """
    Compute ad-hoc indicators on an existing strategy's OHLCV data.
    Returns per-indicator grouped results with render specs for the chart.
    """
    instance = _get_strategy_instance(req.class_name)
    if instance is None:
        return JSONResponse({"error": "Strategy not found"}, status_code=404)

    parquet_path = strategy_folder(instance.name) / "data" / f"{req.interval}.parquet"
    if not parquet_path.exists():
        return JSONResponse({"error": f"No data file for {req.interval}"}, status_code=404)

    df = pd.read_parquet(parquet_path)

    # Compute each indicator individually so we can track which columns it produces
    _RESAMPLE_LADDER = [
        ("1min", "1m"), ("5min", "5m"), ("15min", "15m"),
        ("1h", "1h"), ("4h", "4h"), ("1D", "1d"),
    ]
    max_bars = 8000

    # Resample OHLCV first (shared across all indicators)
    chart_df = df.copy()
    chart_interval = None
    if len(chart_df) > max_bars:
        ohlcv_agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        for rule, label in _RESAMPLE_LADDER:
            candidate = chart_df.resample(rule).agg(ohlcv_agg).dropna(subset=["open"])
            if len(candidate) <= max_bars:
                chart_df = candidate
                chart_interval = label
                break
        else:
            chart_df = chart_df.resample("1D").agg(ohlcv_agg).dropna(subset=["open"])
            chart_interval = "1d"

    results = []

    for ind_req in req.indicators:
        ind_name = ind_req["name"].lower()
        ind_params = ind_req.get("params", {})
        meta = INDICATOR_REGISTRY.get(ind_name, {})
        render = meta.get("render", {"type": "line"})
        display = meta.get("display", "panel")

        # Compute on the full-resolution df, then resample
        work_df = df.copy()
        before_cols = set(work_df.columns)
        try:
            compute_indicators(work_df, [(ind_name, ind_params)], inplace=True)
        except Exception as e:
            results.append({"name": ind_name, "error": str(e)})
            continue

        new_cols = [c for c in work_df.columns if c not in before_cols]
        if not new_cols:
            continue

        # Resample indicator columns onto the chart timeframe
        if chart_interval and chart_interval != "1m":
            ind_agg = {c: "mean" for c in new_cols}
            # For markers (PSAR), use "last" instead of "mean"
            if render.get("type") == "markers":
                ind_agg = {c: "last" for c in new_cols}
            resampled = work_df[new_cols].resample(
                {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}.get(chart_interval, "1D")
            ).agg(ind_agg)
            # Align to chart_df index
            ind_chart = resampled.reindex(chart_df.index)
        else:
            ind_chart = work_df[new_cols].reindex(chart_df.index)

        # Build series data per column
        series: dict[str, list] = {}
        for col in new_cols:
            col_data = []
            for ts, v in ind_chart[col].items():
                if pd.isna(v):
                    continue
                col_data.append({"time": int(ts.timestamp()), "value": round(float(v), 6)})
            series[col] = col_data

        # For markers render type, include close prices so frontend can position dots
        if render.get("type") == "markers":
            close_data = []
            for ts, row in chart_df.iterrows():
                close_data.append({"time": int(ts.timestamp()), "value": round(float(row["close"]), 6)})
            series["_close"] = close_data

        # For colored_line (SuperTrend), include close for color comparison
        if render.get("type") == "colored_line":
            close_data = []
            for ts, row in chart_df.iterrows():
                close_data.append({"time": int(ts.timestamp()), "value": round(float(row["close"]), 6)})
            series["_close"] = close_data

        results.append({
            "name": ind_name,
            "label": meta.get("label", ind_name),
            "display": display,
            "render": render,
            "columns": new_cols,
            "series": series,
        })

    output = {"results": results, "chart_interval": chart_interval}
    content = json.loads(json.dumps(output, cls=_NumpyEncoder))
    return JSONResponse(content=content)


@router.get("/preview/{class_name}/{interval}")
async def strategy_data_preview(
    class_name: str,
    interval: str,
    page: int = 1,
    page_size: int = 100,
):
    """Preview a strategy's built parquet file — returns JSON for React."""
    instance = _get_strategy_instance(class_name)
    if instance is None:
        return {"error": "Strategy not found"}

    parquet_path = strategy_folder(instance.name) / "data" / f"{interval}.parquet"
    if not parquet_path.exists():
        return {"error": f"No data file for {interval}"}

    df = pd.read_parquet(parquet_path)

    # Identify column groups
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    _raw_extras = {"quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "market_open"}
    indicator_cols = [c for c in df.columns if c not in ohlcv_cols and c not in _raw_extras]
    overlay_cols = [c for c in indicator_cols if any(c.startswith(p) for p in _PRICE_OVERLAY_PREFIXES)]
    separate_cols = [c for c in indicator_cols if c not in overlay_cols]

    # Prepare chart data — resample if needed for performance
    _RESAMPLE_LADDER = [
        ("1min", "1m"), ("5min", "5m"), ("15min", "15m"),
        ("1h", "1h"), ("4h", "4h"), ("1D", "1d"),
    ]
    max_bars = 8000
    chart_df = df
    original_bars = len(df)
    chart_interval = None

    if len(chart_df) > max_bars:
        ohlcv_agg = {
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }
        # Also aggregate indicator columns with mean
        extra_agg = {c: "mean" for c in indicator_cols if c in chart_df.columns}
        agg_dict = {**ohlcv_agg, **extra_agg}

        for rule, label in _RESAMPLE_LADDER:
            candidate = chart_df.resample(rule).agg(agg_dict).dropna(subset=["open"])
            if len(candidate) <= max_bars:
                chart_df = candidate
                chart_interval = label
                break
        else:
            chart_df = chart_df.resample("1D").agg(agg_dict).dropna(subset=["open"])
            chart_interval = "1d"

    chart_df = chart_df.fillna(0)

    # Build lightweight-charts-compatible data with unix epoch timestamps
    ohlcv_data = []
    volume_data = []
    for ts, row in chart_df.iterrows():
        t = int(ts.timestamp())
        ohlcv_data.append({
            "time": t,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
        if "volume" in chart_df.columns:
            color = "#22c55e80" if row["close"] >= row["open"] else "#ef444480"
            volume_data.append({"time": t, "value": float(row["volume"]), "color": color})

    overlays_data = {}
    for col in overlay_cols:
        overlays_data[col] = [
            {"time": int(ts.timestamp()), "value": float(v)}
            for ts, v in chart_df[col].items()
        ]

    indicators_data = {}
    for col in separate_cols:
        indicators_data[col] = [
            {"time": int(ts.timestamp()), "value": float(v)}
            for ts, v in chart_df[col].items()
        ]

    chart_data = {
        "ohlcv": ohlcv_data,
        "volume": volume_data,
        "overlays": overlays_data,
        "indicators": indicators_data,
        "resampled": chart_interval is not None,
        "chart_interval": chart_interval,
        "original_bars": original_bars,
    }

    # Prepare table data (paginated)
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    table_df = df.iloc[start_idx:end_idx]

    all_cols = ohlcv_cols + indicator_cols
    table_data = []
    for ts, row in table_df.iterrows():
        rec = {"timestamp": ts.strftime("%Y-%m-%d %H:%M")}
        for col in all_cols:
            val = row[col]
            if pd.isna(val):
                rec[col] = "—"
            elif col == "volume":
                rec[col] = f"{val:,.0f}"
            elif isinstance(val, float):
                rec[col] = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
            else:
                rec[col] = str(val)
        table_data.append(rec)

    resp = {
        "class_name": class_name,
        "strategy_name": instance.name,
        "interval": interval,
        "chart_data": chart_data,
        "all_cols": all_cols,
        "overlay_cols": overlay_cols,
        "separate_cols": separate_cols,
        "table_data": table_data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        },
    }
    content = json.loads(json.dumps(resp, cls=_NumpyEncoder))
    return JSONResponse(content=content)
