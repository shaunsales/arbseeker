"""
Backtest routes.

Handles listing, viewing, and running backtests.
Reads results from output/strategies/{name}/results/.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.routes.strategy import discover_strategies, _get_strategy_instance
from core.strategy.data import strategy_folder, STRATEGIES_OUTPUT_DIR, _indicator_column_names
from core.strategy.engine import BacktestEngine
from core.strategy.position import CostModel, DEFAULT_COSTS

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_runs() -> list[dict]:
    """Scan all strategy folders for backtest runs."""
    runs = []
    if not STRATEGIES_OUTPUT_DIR.exists():
        return runs

    for strat_dir in sorted(STRATEGIES_OUTPUT_DIR.iterdir()):
        results_dir = strat_dir / "results"
        if not results_dir.is_dir():
            continue

        for meta_file in sorted(results_dir.glob("*_meta.json"), reverse=True):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                run_id = meta.get("run_id", meta_file.stem.replace("_meta", ""))
                runs.append({
                    "strategy_name": strat_dir.name,
                    "run_id": run_id,
                    "meta": meta,
                    "folder": str(results_dir),
                })
            except Exception:
                continue

    return runs


RESAMPLE_CANDIDATES = ["1min", "5min", "15min", "1h"]
MAX_CHART_BARS = 20_000

OHLCV_COLS = {"open", "high", "low", "close", "volume", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "market_open"}


def _pick_resample_interval(n_bars_1m: int) -> str:
    """Pick the smallest resample interval that keeps bars under MAX_CHART_BARS.

    n_bars_1m is the number of 1-minute bars in the dataset.
    """
    for interval in RESAMPLE_CANDIDATES:
        divisor = pd.Timedelta(interval).total_seconds() / 60
        estimated = n_bars_1m / max(divisor, 1)
        if estimated <= MAX_CHART_BARS:
            return interval
    return "1h"


def _resample_bars(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1m OHLCV bars to a coarser interval."""
    if interval == "1min":
        return df
    agg = {}
    if "close" in df.columns:
        agg["close"] = "last"
    if "open" in df.columns:
        agg["open"] = "first"
    if "high" in df.columns:
        agg["high"] = "max"
    if "low" in df.columns:
        agg["low"] = "min"
    if "volume" in df.columns:
        agg["volume"] = "sum"
    # nav / drawdown from bar-level results
    if "nav" in df.columns:
        agg["nav"] = "last"
    if "drawdown_pct" in df.columns:
        agg["drawdown_pct"] = "min"
    return df.resample(interval).agg(agg).dropna(subset=["close"])


def _series_to_json(timestamps, values) -> list[dict]:
    """Convert aligned timestamps + values to [{time, value}] for lightweight-charts."""
    out = []
    for t, v in zip(timestamps, values):
        if isinstance(v, float) and np.isnan(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isnan(fv):
            continue
        epoch = int(t.timestamp()) if hasattr(t, "timestamp") else int(pd.Timestamp(t).timestamp())
        out.append({"time": epoch, "value": round(fv, 4)})
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/")
async def backtest_page():
    """List all backtest runs and strategies."""
    runs = _list_runs()
    strategies = discover_strategies()
    return {"runs": runs, "strategies": strategies}


@router.get("/view/{strategy_name}/{run_id}")
async def view_run(strategy_name: str, run_id: str):
    """Load and display a backtest run — returns JSON."""
    results_dir = strategy_folder(strategy_name) / "results"

    meta_path = results_dir / f"{run_id}_meta.json"
    if not meta_path.exists():
        return {"error": "Run not found"}

    with open(meta_path) as f:
        meta = json.load(f)

    # ------------------------------------------------------------------
    # Load bar-level data and resample to a sensible chart interval
    # ------------------------------------------------------------------
    bars_path = results_dir / f"{run_id}_bars.parquet"
    chart_data: dict = {}
    resampled_index: pd.DatetimeIndex | None = None

    if bars_path.exists():
        bars_df = pd.read_parquet(bars_path)
        interval = _pick_resample_interval(len(bars_df))
        bars_r = _resample_bars(bars_df, interval)
        resampled_index = bars_r.index

        chart_data["interval"] = interval
        chart_data["price"] = _series_to_json(bars_r.index, bars_r["close"])
        if "nav" in bars_r.columns:
            chart_data["equity"] = _series_to_json(bars_r.index, bars_r["nav"])
        if "drawdown_pct" in bars_r.columns:
            chart_data["drawdown"] = _series_to_json(bars_r.index, bars_r["drawdown_pct"])

    # ------------------------------------------------------------------
    # Load trades
    # ------------------------------------------------------------------
    trades = []
    trades_path = results_dir / f"{run_id}_trades.parquet"
    if trades_path.exists():
        trades_df = pd.read_parquet(trades_path)
        for _, row in trades_df.iterrows():
            raw_meta = row.get("metadata", None)
            trade_metadata = None
            if raw_meta is not None:
                if isinstance(raw_meta, str):
                    try:
                        trade_metadata = json.loads(raw_meta)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(raw_meta, dict):
                    trade_metadata = raw_meta

            trades.append({
                "side": row.get("side", ""),
                "entry_time": str(row["entry_time"]),
                "entry_price": round(float(row["entry_price"]), 2),
                "exit_time": str(row["exit_time"]),
                "exit_price": round(float(row["exit_price"]), 2),
                "size": round(float(row["size"]), 0),
                "gross_pnl": round(float(row["gross_pnl"]), 2),
                "costs": round(float(row["costs"]), 2),
                "net_pnl": round(float(row["net_pnl"]), 2),
                "bars_held": int(row["bars_held"]),
                "entry_reason": row.get("entry_reason", ""),
                "exit_reason": row.get("exit_reason", ""),
                "metadata": trade_metadata,
            })

        # Build trade markers for chart
        chart_data["markers"] = []
        for _, row in trades_df.iterrows():
            entry_ts = pd.Timestamp(row["entry_time"])
            exit_ts = pd.Timestamp(row["exit_time"])
            is_long = row.get("side", "") == "long"
            chart_data["markers"].append({
                "time": int(entry_ts.timestamp()),
                "position": "belowBar" if is_long else "aboveBar",
                "color": "#22c55e" if is_long else "#ef4444",
                "shape": "arrowUp" if is_long else "arrowDown",
                "text": f"{'L' if is_long else 'S'} {row.get('entry_reason', '')}",
            })
            chart_data["markers"].append({
                "time": int(exit_ts.timestamp()),
                "position": "aboveBar" if is_long else "belowBar",
                "color": "#a855f7",
                "shape": "circle",
                "text": f"X {row.get('exit_reason', '')}",
            })

    # ------------------------------------------------------------------
    # Load indicator data, LOCF-resample onto the same chart index
    # ------------------------------------------------------------------
    indicators: list[dict] = []
    data_dir = strategy_folder(strategy_name) / "data"
    manifest_path = strategy_folder(strategy_name) / "manifest.json"

    if manifest_path.exists() and data_dir.exists() and resampled_index is not None:
        with open(manifest_path) as f:
            manifest = json.load(f)
        spec_intervals = manifest.get("spec", {}).get("intervals", {})

        for src_interval, ind_list in spec_intervals.items():
            if not ind_list:
                continue
            parquet_path = data_dir / f"{src_interval}.parquet"
            if not parquet_path.exists():
                continue
            ind_df = pd.read_parquet(parquet_path)

            # Build whitelist: only the PRIMARY column per indicator.
            # e.g. ("adx", {"length":14}) → "ADX_14" only (skip DMP/DMN)
            # ("sma", {"length":50}) → "SMA_50"
            wanted_cols: set[str] = set()
            for ind_name, params in ind_list:
                all_cols = _indicator_column_names(ind_name, params)
                if all_cols:
                    wanted_cols.add(all_cols[0])

            ind_cols = [c for c in ind_df.columns if c in wanted_cols]
            if not ind_cols:
                continue

            # Reindex indicator data onto the chart's resampled timestamps (LOCF)
            ind_aligned = ind_df[ind_cols].reindex(resampled_index, method="ffill")

            for col in ind_cols:
                indicators.append({
                    "name": col,
                    "interval": src_interval,
                    "series": _series_to_json(ind_aligned.index, ind_aligned[col]),
                })

    # ------------------------------------------------------------------
    tearsheet_exists = (results_dir / f"{run_id}_tearsheet.html").exists()

    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "meta": meta,
        "chart_data": chart_data,
        "trades": trades,
        "indicators": indicators,
        "tearsheet_exists": tearsheet_exists,
    }


@router.get("/tearsheet/{strategy_name}/{run_id}")
async def serve_tearsheet(strategy_name: str, run_id: str):
    """Serve the QuantStats HTML tearsheet."""
    path = strategy_folder(strategy_name) / "results" / f"{run_id}_tearsheet.html"
    if not path.exists():
        return JSONResponse({"error": "Tearsheet not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")


class RunBacktestRequest(BaseModel):
    class_name: str
    capital: float = 100_000
    commission_bps: float = 3.5
    slippage_bps: float = 2.0
    funding_daily_bps: float = 5.0


@router.post("/run")
async def run_backtest(req: RunBacktestRequest):
    """Run a backtest for a strategy."""
    instance = _get_strategy_instance(req.class_name)
    if instance is None:
        return {"success": False, "error": "Strategy not found"}

    spec = instance.data_spec()
    if spec is None:
        return {"success": False, "error": "Strategy does not define data_spec()"}

    try:
        costs = CostModel(
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
            funding_daily_bps=req.funding_daily_bps,
        )
        engine = BacktestEngine(verbose=False)
        result = engine.run(instance, capital=req.capital, costs=costs)

        return {
            "success": True,
            "strategy_name": instance.name,
            "run_id": result.config.get("run_id", ""),
            "metrics": result.summary(),
            "total_trades": len(result.trades),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
