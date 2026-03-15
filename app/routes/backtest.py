"""
Backtest routes.

Handles listing, viewing, and running backtests.
Reads results from output/strategies/{name}/results/.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.routes.strategy import discover_strategies, _get_strategy_instance
from core.strategy.data import strategy_folder, STRATEGIES_OUTPUT_DIR, _indicator_column_names
from core.data.binance import load_funding_rates
from core.indicators.indicators import INDICATOR_REGISTRY
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

        # OHLCV for candlestick rendering — bars parquet may only have close,
        # so load raw 1m data from strategy data folder and resample it.
        ohlcv_data = []
        volume_data = []
        ohlcv_source = bars_r
        has_ohlc = all(c in bars_r.columns for c in ["open", "high", "low", "close"])
        if not has_ohlc:
            raw_1m_path = strategy_folder(strategy_name) / "data" / "1m.parquet"
            if raw_1m_path.exists():
                raw_1m = pd.read_parquet(raw_1m_path, columns=["open", "high", "low", "close", "volume"])
                ohlcv_source = _resample_bars(raw_1m, interval)
                has_ohlc = all(c in ohlcv_source.columns for c in ["open", "high", "low", "close"])

        if has_ohlc:
            for ts, row in ohlcv_source.iterrows():
                t = int(ts.timestamp())
                o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                if any(np.isnan(v) for v in [o, h, l, c]):
                    continue
                ohlcv_data.append({"time": t, "open": round(o, 2), "high": round(h, 2), "low": round(l, 2), "close": round(c, 2)})
                if "volume" in ohlcv_source.columns:
                    vol = float(row["volume"])
                    if not np.isnan(vol):
                        volume_data.append({"time": t, "value": round(vol, 2), "color": "#22c55e80" if c >= o else "#ef444480"})
        chart_data["ohlcv"] = ohlcv_data
        chart_data["volume"] = volume_data

        if "nav" in bars_r.columns:
            chart_data["equity"] = _series_to_json(bars_r.index, bars_r["nav"])
        if "drawdown_pct" in bars_r.columns:
            chart_data["drawdown"] = _series_to_json(bars_r.index, bars_r["drawdown_pct"])

        # Daily-resampled NAV + rolling max drawdown for the Performance chart
        if "nav" in bars_df.columns:
            daily = bars_df[["nav"]].resample("1D").last().dropna()
            initial_nav = daily["nav"].iloc[0]
            if initial_nav and initial_nav != 0:
                daily_nav = daily["nav"] / initial_nav
            else:
                daily_nav = daily["nav"]
            peak = daily_nav.cummax()
            daily_dd = ((daily_nav - peak) / peak) * 100
            rolling_mdd = daily_dd.cummin()
            chart_data["daily_nav"] = _series_to_json(daily.index, daily_nav)
            chart_data["daily_mdd"] = _series_to_json(daily.index, rolling_mdd)

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

            funding_cost = round(float(row.get("funding_cost", 0)), 2)
            total_costs = round(float(row["costs"]), 2)
            fees = round(total_costs - funding_cost, 2)

            trades.append({
                "side": row.get("side", ""),
                "entry_time": str(row["entry_time"]),
                "entry_price": round(float(row["entry_price"]), 2),
                "exit_time": str(row["exit_time"]),
                "exit_price": round(float(row["exit_price"]), 2),
                "size": round(float(row["size"]), 0),
                "gross_pnl": round(float(row["gross_pnl"]), 2),
                "costs": total_costs,
                "fees": fees,
                "funding_cost": funding_cost,
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
                "position": "aboveBar",
                "color": "#06b6d4" if is_long else "#f97316",
                "shape": "arrowDown" if is_long else "arrowDown",
                "text": f"{'▲L' if is_long else '▼S'}",
            })
            chart_data["markers"].append({
                "time": int(exit_ts.timestamp()),
                "position": "belowBar",
                "color": "#a855f7",
                "shape": "arrowUp",
                "text": f"✕",
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
                # Look up display/render metadata from the indicator registry
                ind_display = "panel"
                ind_render = {"type": "line"}
                for ind_name, _params in ind_list:
                    reg = INDICATOR_REGISTRY.get(ind_name.lower(), {})
                    expected_cols = _indicator_column_names(ind_name, _params)
                    if col in expected_cols:
                        ind_display = reg.get("display", "panel")
                        ind_render = reg.get("render", {"type": "line"})
                        break

                indicators.append({
                    "name": col,
                    "interval": src_interval,
                    "display": ind_display,
                    "render": ind_render,
                    "series": _series_to_json(ind_aligned.index, ind_aligned[col]),
                })

    # (max_drawdown series now computed as daily_mdd above)

    # ------------------------------------------------------------------
    # Load funding rate data for the strategy's symbol
    # ------------------------------------------------------------------
    # Only include funding rates if trades actually used per-symbol funding
    has_funding_costs = any(t.get("funding_cost", 0) != 0 for t in trades)
    funding_rates_data: list[dict] | None = None
    if has_funding_costs and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        ticker = manifest.get("spec", {}).get("ticker")
        market = manifest.get("spec", {}).get("market", "futures")
        if ticker and market == "futures":
            fr_df = load_funding_rates(ticker, market)
            if fr_df is not None:
                funding_rates_data = [
                    {"month": str(idx), "rate_bps": round(row["median_daily_bps"], 2)}
                    for idx, row in fr_df.iterrows()
                ]

    # ------------------------------------------------------------------
    tearsheet_exists = (results_dir / f"{run_id}_tearsheet.html").exists()

    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "meta": meta,
        "chart_data": chart_data,
        "trades": trades,
        "indicators": indicators,
        "funding_rates": funding_rates_data,
        "tearsheet_exists": tearsheet_exists,
    }


@router.delete("/delete/{strategy_name}/{run_id}")
async def delete_run(strategy_name: str, run_id: str):
    """Delete a backtest run's files."""
    results_dir = strategy_folder(strategy_name) / "results"
    prefix = f"{run_id}"
    deleted = []
    for f in results_dir.glob(f"{prefix}*"):
        f.unlink()
        deleted.append(f.name)
    if not deleted:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return {"success": True, "deleted": deleted}


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
    funding_daily_bps: float = 3.0
    start_date: str | None = None  # "YYYY-MM" — inclusive start
    end_date: str | None = None    # "YYYY-MM" — inclusive end (last day of month)
    stop_loss_pct: float | None = None        # Fixed SL % from entry (None = use strategy default)
    trailing_stop_pct: float | None = None    # TSL % from best price (None = use strategy default)


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
        result = engine.run(
            instance,
            capital=req.capital,
            costs=costs,
            start_date=req.start_date,
            end_date=req.end_date,
            stop_loss_pct=req.stop_loss_pct,
            trailing_stop_pct=req.trailing_stop_pct,
        )

        return {
            "success": True,
            "strategy_name": instance.name,
            "run_id": result.config.get("run_id", ""),
            "metrics": result.summary(),
            "total_trades": len(result.trades),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
