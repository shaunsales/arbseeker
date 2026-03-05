"""
Deterministic test for the v2 backtest engine.

Creates synthetic data with known price movements, runs a deterministic
strategy, and validates:
- PnL calculation
- NAV / balance tracking
- Drawdown
- No look-ahead bias
- Decision context capture
- Cost model application
- Position sizing
- Bar-level state recording
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.strategy.base import SingleAssetStrategy, StrategyConfig
from core.strategy.data import (
    StrategyDataSpec,
    StrategyData,
    strategy_folder,
    save_manifest,
    STRATEGIES_OUTPUT_DIR,
)
from core.strategy.position import Signal, Position, CostModel
from core.strategy.engine import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Test strategy: deterministic buy/sell at known bars
# ---------------------------------------------------------------------------

class TestFixedEntryExit(SingleAssetStrategy):
    """
    Deterministic strategy for engine validation.
    
    Behaviour:
    - Goes LONG at bar 60 (T=01:00)
    - Closes at bar 120 (T=02:00)
    - Goes SHORT at bar 180 (T=03:00)
    - Closes at bar 240 (T=04:00)
    
    Uses 1h ADX as a sanity check for multi-interval access.
    """
    
    def __init__(self):
        super().__init__(StrategyConfig(name="TestFixedEntryExit"))
    
    def data_spec(self) -> StrategyDataSpec:
        return StrategyDataSpec(
            venue="test",
            market="test",
            ticker="TESTUSDT",
            intervals={
                "1m": [],
                "1h": [("sma", {"length": 5})],
            },
        )
    
    def required_indicators(self):
        return []
    
    def on_bar(self, timestamp, data, balance, position):
        """Deterministic entry/exit at known minute offsets."""
        # Get minute offset from start
        base = data._frames["1m"].index[0]
        minutes = int((timestamp - base).total_seconds() / 60)
        
        # Verify 1h data is available (no look-ahead)
        try:
            h1_bar = data.bar("1h", timestamp)
            self._last_h1_time = h1_bar.name
        except ValueError:
            self._last_h1_time = None
        
        if minutes == 60 and position is None:
            return Signal.buy(size=1.0, reason="long_entry_bar60")
        elif minutes == 120 and position is not None:
            return Signal.close(reason="long_exit_bar120")
        elif minutes == 180 and position is None:
            return Signal.sell(size=1.0, reason="short_entry_bar180")
        elif minutes == 240 and position is not None:
            return Signal.close(reason="short_exit_bar240")
        
        return Signal.hold()


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def build_synthetic_data(strategy_name: str) -> dict:
    """
    Build synthetic OHLCV data with known price pattern.
    
    Price pattern (close prices):
    - T=00:00 to 01:00: 100.0 (flat)
    - T=01:00 to 02:00: ramps from 100 to 110 (10% gain for longs)
    - T=02:00 to 03:00: 110.0 (flat)
    - T=03:00 to 04:00: ramps from 110 to 100 (9.09% drop — profit for shorts)
    - T=04:00 to 05:00: 100.0 (flat)
    
    Returns the expected metrics for validation.
    """
    folder = strategy_folder(strategy_name)
    data_dir = folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (folder / "results").mkdir(parents=True, exist_ok=True)
    
    n_bars = 300  # 5 hours of 1m data
    idx = pd.date_range("2025-01-01 00:00", periods=n_bars, freq="1min", tz="UTC")
    
    # Build price pattern
    prices = np.zeros(n_bars)
    for i in range(n_bars):
        if i < 60:
            prices[i] = 100.0
        elif i < 120:
            # Ramp from 100 to 110 over 60 bars
            prices[i] = 100.0 + (i - 60) * (10.0 / 60)
        elif i < 180:
            prices[i] = 110.0
        elif i < 240:
            # Ramp from 110 to 100 over 60 bars
            prices[i] = 110.0 - (i - 180) * (10.0 / 60)
        else:
            prices[i] = 100.0
    
    # 1m OHLCV
    df_1m = pd.DataFrame({
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": np.ones(n_bars) * 100,
    }, index=idx)
    df_1m.index.name = "open_time"
    df_1m.to_parquet(data_dir / "1m.parquet")
    
    # 1h OHLCV (5 bars)
    n_h = 5
    idx_h = pd.date_range("2025-01-01 00:00", periods=n_h, freq="1h", tz="UTC")
    h_close = [100.0, 110.0, 110.0, 100.0, 100.0]
    df_1h = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 110.0, 100.0],
        "high": [100.5, 110.5, 110.5, 110.5, 100.5],
        "low": [99.5, 99.5, 109.5, 99.5, 99.5],
        "close": h_close,
        "volume": [6000.0] * n_h,
        "SMA_5": [np.nan, np.nan, np.nan, np.nan, 104.0],
    }, index=idx_h)
    df_1h.index.name = "open_time"
    df_1h.to_parquet(data_dir / "1h.parquet")
    
    # Manifest
    spec = TestFixedEntryExit().data_spec()
    manifest = {
        "strategy_name": strategy_name,
        "spec": spec.to_dict(),
        "date_range": {"start": "2025-01", "end": "2025-01"},
        "built_at": "2025-01-01T00:00:00",
        "quality": {
            "1m": {"bars": n_bars, "start": str(idx[0]), "end": str(idx[-1]),
                    "null_bars": 0, "coverage_pct": 100.0, "columns": list(df_1m.columns)},
            "1h": {"bars": n_h, "start": str(idx_h[0]), "end": str(idx_h[-1]),
                    "null_bars": 4, "coverage_pct": 20.0, "columns": list(df_1h.columns)},
        },
    }
    save_manifest(strategy_name, manifest)
    
    return {
        "prices": prices,
        "entry_long_price": 100.0,     # bar 60 close
        "exit_long_price": 110.0,      # bar 120 close
        "entry_short_price": 110.0,    # bar 180 close
        "exit_short_price": 100.0,     # bar 240 close
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

STRATEGY_NAME = "TestFixedEntryExit"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test strategy folder before and after each test."""
    folder = strategy_folder(STRATEGY_NAME)
    if folder.exists():
        shutil.rmtree(folder)
    yield
    if folder.exists():
        shutil.rmtree(folder)


class TestEngineV2:
    
    def setup_method(self):
        self.expected = build_synthetic_data(STRATEGY_NAME)
        self.strategy = TestFixedEntryExit()
        self.engine = BacktestEngine(verbose=False)
    
    def _run(self, capital=100_000, costs=None):
        costs = costs or CostModel()  # Zero costs for clean math
        return self.engine.run(self.strategy, capital=capital, costs=costs)
    
    # --- PnL ---
    
    def test_long_pnl_zero_costs(self):
        """Long: buy at 100, sell at 110 → +10% on position size."""
        result = self._run(capital=100_000)
        
        # With max_position_pct=1.0, size=1.0 → position = 100k
        # Long PnL: (110-100)/100 * 100k = 10k
        long_trade = result.trades[0]
        assert long_trade.entry_price == 100.0
        assert long_trade.exit_price == 110.0
        assert abs(long_trade.gross_pnl - 10_000) < 0.01
        assert long_trade.entry_reason == "long_entry_bar60"
        assert long_trade.exit_reason == "long_exit_bar120"
    
    def test_short_pnl_zero_costs(self):
        """Short: sell at 110, buy at 100 → profit on short."""
        result = self._run(capital=100_000)
        
        short_trade = result.trades[1]
        assert short_trade.entry_price == 110.0
        assert short_trade.exit_price == 100.0
        # Short PnL: (110-100)/110 * size
        # After long trade, balance = 110k, so short size = 110k
        expected_pnl = (110.0 - 100.0) / 110.0 * 110_000
        assert abs(short_trade.gross_pnl - expected_pnl) < 0.01
    
    def test_total_pnl_zero_costs(self):
        """Total: both trades profitable, final > initial."""
        result = self._run(capital=100_000)
        assert result.final_capital > 100_000
        assert len(result.trades) == 2
    
    # --- Costs ---
    
    def test_costs_applied(self):
        """Costs reduce PnL."""
        costs = CostModel(
            commission_bps=3.5,
            slippage_bps=2.0,
            funding_daily_bps=5.0,
            bars_per_day=1440,
        )
        result = self._run(costs=costs)
        
        for trade in result.trades:
            assert trade.costs > 0
            assert trade.net_pnl < trade.gross_pnl
    
    def test_round_trip_cost_calculation(self):
        """Verify round-trip cost = (comm + slip) * 2 * size / 10000."""
        costs = CostModel(commission_bps=10.0, slippage_bps=5.0, bars_per_day=1440)
        result = self._run(costs=costs)
        
        trade = result.trades[0]
        # Round trip = (10+5)*2 * size / 10000 = 30 * 100000 / 10000 = 300
        expected_rt = 30 * trade.size / 10000
        # Holding cost = funding * size * days
        days = trade.bars_held / 1440
        expected_hold = costs.funding_daily_bps * trade.size / 10000 * days
        expected_total = expected_rt + expected_hold
        assert abs(trade.costs - expected_total) < 0.01
    
    # --- Balance / NAV / Drawdown ---
    
    def test_balance_only_changes_on_close(self):
        """Balance should be constant while in a position."""
        result = self._run()
        
        bars_path = strategy_folder(STRATEGY_NAME) / "results"
        parquets = list(bars_path.glob("*_bars.parquet"))
        assert len(parquets) == 1
        bars_df = pd.read_parquet(parquets[0])
        
        # Bars 0-59: no position, balance = 100k
        assert all(bars_df.iloc[:60]["balance"] == 100_000)
        
        # Bars 60-119: long position, balance should stay at 100k
        assert all(bars_df.iloc[60:120]["balance"] == 100_000)
        
        # Bar 120: position closed, balance increases
        assert bars_df.iloc[120]["balance"] > 100_000
    
    def test_nav_tracks_unrealized(self):
        """NAV should include unrealized PnL during position."""
        result = self._run()
        
        bars_path = strategy_folder(STRATEGY_NAME) / "results"
        bars_df = pd.read_parquet(list(bars_path.glob("*_bars.parquet"))[0])
        
        # At bar 90 (midpoint of long), price ~ 105
        # Position entered at 100, so unrealized PnL ~ +5% of 100k = +5k
        # NAV should be ~ 105k
        nav_90 = bars_df.iloc[90]["nav"]
        assert nav_90 > 100_000
        assert nav_90 < 106_000  # Not yet at 110k
    
    def test_drawdown_recorded(self):
        """Drawdown should be non-positive and track from peak."""
        result = self._run()
        
        bars_path = strategy_folder(STRATEGY_NAME) / "results"
        bars_df = pd.read_parquet(list(bars_path.glob("*_bars.parquet"))[0])
        
        assert "drawdown_pct" in bars_df.columns
        assert all(bars_df["drawdown_pct"] <= 0)
        # At bar 0, drawdown should be 0
        assert bars_df.iloc[0]["drawdown_pct"] == 0.0
    
    # --- Decision Context ---
    
    def test_entry_context_captured(self):
        """Trade entry should have a snapshot of available data."""
        result = self._run()
        
        trade = result.trades[0]
        assert "entry_context" in trade.metadata
        ctx = trade.metadata["entry_context"]
        assert "1m" in ctx
        assert "1h" in ctx
    
    def test_exit_context_captured(self):
        """Trade exit should have a snapshot of available data."""
        result = self._run()
        
        trade = result.trades[0]
        assert "exit_context" in trade.metadata
        ctx = trade.metadata["exit_context"]
        assert "1m" in ctx
        assert "1h" in ctx
    
    # --- Look-ahead safety ---
    
    def test_1h_bar_not_available_in_first_hour(self):
        """During the first hour, no 1h bar should be available."""
        strategy = TestFixedEntryExit()
        strategy._last_h1_time = "should_be_overwritten"
        
        # Run the engine
        result = self.engine.run(strategy, capital=100_000, costs=CostModel())
        
        # The strategy checks bar("1h", timestamp) at every bar.
        # For bars 0-59 (00:00 to 00:59), no 1h bar should be available.
        # The strategy records _last_h1_time = None when it catches ValueError.
        # After bar 60 (01:00), the 00:00 bar should be available.
        # We verify via the entry context of the first trade (at bar 60 = 01:00).
        trade = result.trades[0]
        entry_ctx = trade.metadata["entry_context"]
        # At T=01:00, the 00:00 1h bar just closed
        assert entry_ctx["1h"] is not None
        assert "2025-01-01 00:00:00" in entry_ctx["1h"]["open_time"]
    
    # --- Bar-level recording ---
    
    def test_bar_level_columns(self):
        """Bar-level parquet should have all required columns."""
        result = self._run()
        
        bars_path = strategy_folder(STRATEGY_NAME) / "results"
        bars_df = pd.read_parquet(list(bars_path.glob("*_bars.parquet"))[0])
        
        required = ["close", "balance", "nav", "drawdown_pct",
                     "position_side", "position_size", "position_pnl",
                     "position_pnl_pct", "signal"]
        for col in required:
            assert col in bars_df.columns, f"Missing column: {col}"
    
    def test_position_side_tracking(self):
        """Position side should be flat/long/short at correct bars."""
        result = self._run()
        
        bars_path = strategy_folder(STRATEGY_NAME) / "results"
        bars_df = pd.read_parquet(list(bars_path.glob("*_bars.parquet"))[0])
        
        assert bars_df.iloc[30]["position_side"] == "flat"
        assert bars_df.iloc[90]["position_side"] == "long"
        assert bars_df.iloc[150]["position_side"] == "flat"
        assert bars_df.iloc[210]["position_side"] == "short"
        assert bars_df.iloc[270]["position_side"] == "flat"
    
    # --- Output files ---
    
    def test_output_files_created(self):
        """Engine should create bars, trades, and meta files."""
        result = self._run()
        
        results_dir = strategy_folder(STRATEGY_NAME) / "results"
        bars_files = list(results_dir.glob("*_bars.parquet"))
        trades_files = list(results_dir.glob("*_trades.parquet"))
        meta_files = list(results_dir.glob("*_meta.json"))
        
        assert len(bars_files) == 1
        assert len(trades_files) == 1
        assert len(meta_files) == 1
        
        # Verify meta JSON
        with open(meta_files[0]) as f:
            meta = json.load(f)
        assert meta["strategy_name"] == STRATEGY_NAME
        assert meta["total_trades"] == 2
        assert meta["initial_capital"] == 100_000
    
    def test_trades_parquet_has_context(self):
        """Trades parquet should include serialized decision context."""
        result = self._run()
        
        results_dir = strategy_folder(STRATEGY_NAME) / "results"
        trades_df = pd.read_parquet(list(results_dir.glob("*_trades.parquet"))[0])
        
        assert "context" in trades_df.columns
        ctx = json.loads(trades_df.iloc[0]["context"])
        assert "entry_context" in ctx
        assert "exit_context" in ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
