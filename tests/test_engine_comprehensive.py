"""
Comprehensive backtest engine tests with synthetic data.

Tests cover:
1. Technical indicators triggering entry/exit
2. Price-level triggers for entry/exit
3. NAV and drawdown correctness
4. Start and end equity
5. Equity zero — strategy loses all capital
6. Fixed sizing vs compound (reinvest) sizing
7. Stop loss fires at correct 1m bar
8. Trailing stop loss tracks best price and fires correctly
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from core.strategy.base import SingleAssetStrategy, StrategyConfig
from core.strategy.data import (
    StrategyDataSpec,
    StrategyData,
    strategy_folder,
    save_manifest,
)
from core.strategy.position import Signal, Position, CostModel, Side
from core.strategy.engine import BacktestEngine, BacktestResult


# ===================================================================
# Helpers: synthetic data builder
# ===================================================================

def _make_spec(intervals=None):
    """Default test spec: 1m + 1h with SMA_5."""
    intervals = intervals or {"1m": [], "1h": [("sma", {"length": 5})]}
    return StrategyDataSpec(
        venue="test", market="test", ticker="TESTUSDT", intervals=intervals,
    )


def _build_data(name: str, prices_1m: np.ndarray, spec: StrategyDataSpec,
                h1_close: list = None, h1_extras: dict = None):
    """
    Build synthetic data files for a test strategy.
    
    Args:
        name: strategy name
        prices_1m: 1m close prices array
        spec: data spec
        h1_close: optional hourly close overrides
        h1_extras: optional dict of extra columns for 1h (e.g. {"SMA_5": [...]})
    """
    folder = strategy_folder(name)
    data_dir = folder / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (folder / "results").mkdir(parents=True, exist_ok=True)

    n = len(prices_1m)
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="1min", tz="UTC")

    df_1m = pd.DataFrame({
        "open": prices_1m,
        "high": prices_1m + 0.5,
        "low": prices_1m - 0.5,
        "close": prices_1m,
        "volume": 100.0,
    }, index=idx)
    df_1m.index.name = "open_time"
    df_1m.to_parquet(data_dir / "1m.parquet")

    # Build 1h if spec requires it
    if "1h" in spec.intervals:
        n_h = max(1, n // 60)
        idx_h = pd.date_range("2025-01-01 00:00", periods=n_h, freq="1h", tz="UTC")
        if h1_close is None:
            # Resample from 1m
            h1_close = [float(prices_1m[min(i * 60 + 59, n - 1)]) for i in range(n_h)]
        
        h1_data = {
            "open": [float(prices_1m[i * 60]) if i * 60 < n else h1_close[i] for i in range(n_h)],
            "high": [c + 1.0 for c in h1_close],
            "low": [c - 1.0 for c in h1_close],
            "close": h1_close,
            "volume": [6000.0] * n_h,
        }
        if h1_extras:
            h1_data.update(h1_extras)
        
        df_1h = pd.DataFrame(h1_data, index=idx_h)
        df_1h.index.name = "open_time"
        df_1h.to_parquet(data_dir / "1h.parquet")

    # Manifest
    manifest = {
        "strategy_name": name,
        "spec": spec.to_dict(),
        "date_range": {"start": "2025-01", "end": "2025-01"},
        "built_at": "2025-01-01T00:00:00",
        "quality": {
            iv: {"bars": n if iv == "1m" else n // 60,
                 "start": "2025-01-01T00:00:00+00:00",
                 "end": "2025-01-01T05:00:00+00:00",
                 "null_bars": 0, "coverage_pct": 100.0,
                 "columns": []}
            for iv in spec.intervals
        },
    }
    save_manifest(name, manifest)


def _cleanup(name: str):
    folder = strategy_folder(name)
    if folder.exists():
        shutil.rmtree(folder)


# ===================================================================
# Test strategies with deterministic behaviour
# ===================================================================

class IndicatorStrategy(SingleAssetStrategy):
    """
    Buys when 1h SMA_5 > threshold, sells when SMA_5 < threshold.
    Tests that indicator values correctly trigger entry/exit.
    """
    def __init__(self, threshold=105.0, **kw):
        super().__init__(StrategyConfig(name="IndicatorStrategy"))
        self.threshold = threshold

    def data_spec(self):
        return _make_spec()

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        try:
            h1 = data.bar("1h", ts)
        except (ValueError, KeyError):
            return Signal.hold()
        sma = h1.get("SMA_5")
        if sma is None or pd.isna(sma):
            return Signal.hold()
        if position is None and sma > self.threshold:
            return Signal.buy(reason=f"sma={sma:.1f}>{self.threshold}")
        if position is not None and sma < self.threshold:
            return Signal.close(reason=f"sma={sma:.1f}<{self.threshold}")
        return Signal.hold()


class PriceTriggerStrategy(SingleAssetStrategy):
    """
    Buys when 1m close > buy_price, closes when 1m close < sell_price.
    Tests price-level triggers on 1m bars.
    """
    def __init__(self, buy_price=105.0, sell_price=103.0):
        super().__init__(StrategyConfig(name="PriceTriggerStrategy"))
        self.buy_price = buy_price
        self.sell_price = sell_price

    def data_spec(self):
        return _make_spec({"1m": []})

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        try:
            m1 = data.bar("1m", ts)
        except (ValueError, KeyError):
            return Signal.hold()
        price = m1["close"]
        if position is None and price > self.buy_price:
            return Signal.buy(reason=f"price={price:.1f}>{self.buy_price}")
        if position is not None and price < self.sell_price:
            return Signal.close(reason=f"price={price:.1f}<{self.sell_price}")
        return Signal.hold()


class AlwaysLongStrategy(SingleAssetStrategy):
    """
    Enters long on bar 10, never exits. Useful for NAV/drawdown tests.
    """
    def __init__(self):
        super().__init__(StrategyConfig(name="AlwaysLongStrategy"))

    def data_spec(self):
        return _make_spec({"1m": []})

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        base = data._frames["1m"].index[0]
        m = int((ts - base).total_seconds() / 60)
        if m == 10 and position is None:
            return Signal.buy(reason="enter")
        return Signal.hold()


class LosingStrategy(SingleAssetStrategy):
    """
    Repeatedly buys at high, sells at low to drain all capital.
    """
    def __init__(self):
        super().__init__(StrategyConfig(name="LosingStrategy"))

    def data_spec(self):
        return _make_spec({"1m": []})

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        base = data._frames["1m"].index[0]
        m = int((ts - base).total_seconds() / 60)
        # Cycle: buy at peak (m%20==5), sell at trough (m%20==15)
        if position is None and m % 20 == 5:
            return Signal.buy(reason="buy_high")
        if position is not None and m % 20 == 15:
            return Signal.close(reason="sell_low")
        return Signal.hold()


class FixedEntryExitStrategy(SingleAssetStrategy):
    """
    Simple: long at bar 10, close at bar 70.
    Used for sizing tests (both fixed and compound).
    """
    def __init__(self, config=None, sl_pct=None, tsl_pct=None):
        config = config or StrategyConfig(name="FixedEntryExitStrategy")
        super().__init__(config)
        self.sl_pct = sl_pct
        self.tsl_pct = tsl_pct

    def data_spec(self):
        return _make_spec({"1m": []})

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        base = data._frames["1m"].index[0]
        m = int((ts - base).total_seconds() / 60)
        if m == 10 and position is None:
            return Signal.buy(
                reason="enter",
                stop_loss_pct=self.sl_pct,
                trailing_stop_pct=self.tsl_pct,
            )
        if m == 70 and position is not None:
            return Signal.close(reason="exit")
        return Signal.hold()


class ShortWithStopLoss(SingleAssetStrategy):
    """Short at bar 10 with stop loss. Used to test SL on short positions."""
    def __init__(self, sl_pct=5.0, tsl_pct=None):
        super().__init__(StrategyConfig(name="ShortWithStopLoss"))
        self.sl_pct = sl_pct
        self.tsl_pct = tsl_pct

    def data_spec(self):
        return _make_spec({"1m": []})

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        base = data._frames["1m"].index[0]
        m = int((ts - base).total_seconds() / 60)
        if m == 10 and position is None:
            return Signal.sell(
                reason="short_enter",
                stop_loss_pct=self.sl_pct,
                trailing_stop_pct=self.tsl_pct,
            )
        if m == 200 and position is not None:
            return Signal.close(reason="manual_exit")
        return Signal.hold()


class RepeatedTradeStrategy(SingleAssetStrategy):
    """
    Makes multiple trades: buy at bar 10, close at 30, buy at 50, close at 70, etc.
    Used for compound sizing tests.
    """
    def __init__(self, config=None):
        config = config or StrategyConfig(name="RepeatedTradeStrategy")
        super().__init__(config)

    def data_spec(self):
        return _make_spec({"1m": []})

    def required_indicators(self):
        return []

    def on_bar(self, ts, data, balance, position):
        base = data._frames["1m"].index[0]
        m = int((ts - base).total_seconds() / 60)
        # Trade pattern: buy at 10,50,90,... close at 30,70,110,...
        cycle = m % 40
        if cycle == 10 and position is None:
            return Signal.buy(reason=f"enter_m{m}")
        if cycle == 30 and position is not None:
            return Signal.close(reason=f"exit_m{m}")
        return Signal.hold()


# ===================================================================
# 1. Technical indicators triggering entry/exit
# ===================================================================

class TestIndicatorTriggers:
    NAME = "test_indicators"

    def setup_method(self):
        _cleanup(self.NAME)
        # Price flat at 100, but SMA_5 crosses threshold at specific hours
        n = 300  # 5 hours
        prices = np.full(n, 100.0)
        # SMA values: [90, 110, 110, 90, 90] — crosses 105 at hour 1, back below at hour 3
        _build_data(self.NAME, prices, _make_spec(),
                    h1_close=[100.0, 100.0, 100.0, 100.0, 100.0],
                    h1_extras={"SMA_5": [90.0, 110.0, 110.0, 90.0, 90.0]})

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_entry_on_indicator_cross_above(self):
        """Should enter long when SMA > threshold after hour 1 closes."""
        strat = IndicatorStrategy(threshold=105.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        assert len(result.trades) >= 1
        # First entry should happen at bar 60 (when 1h bar with SMA=110 becomes visible)
        t = result.trades[0]
        assert "sma=110" in t.entry_reason

    def test_exit_on_indicator_cross_below(self):
        """Should exit when SMA drops below threshold."""
        strat = IndicatorStrategy(threshold=105.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        # Should have at least one closed trade with SMA-based exit
        closed = [t for t in result.trades if "sma=90" in t.exit_reason]
        assert len(closed) >= 1

    def test_no_trade_when_indicator_below_threshold(self):
        """No entry when SMA stays below threshold."""
        strat = IndicatorStrategy(threshold=200.0)  # SMA never reaches 200
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        assert len(result.trades) == 0
        assert result.final_capital == 100_000


# ===================================================================
# 2. Price-level triggers for entry/exit
# ===================================================================

class TestPriceTriggers:
    NAME = "test_price"

    def setup_method(self):
        _cleanup(self.NAME)
        # Price: flat 100 for 50 bars, ramp to 110 over 50, flat 110 for 50,
        # ramp down to 100 over 50, flat 100 for 100 bars
        prices = np.concatenate([
            np.full(50, 100.0),
            np.linspace(100, 110, 50),
            np.full(50, 110.0),
            np.linspace(110, 100, 50),
            np.full(100, 100.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_entry_at_price_level(self):
        """Should enter when price crosses above buy_price."""
        strat = PriceTriggerStrategy(buy_price=105.0, sell_price=103.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        assert len(result.trades) >= 1
        # Entry price should be just above 105
        t = result.trades[0]
        assert t.entry_price >= 105.0

    def test_exit_at_price_level(self):
        """Should exit when price drops below sell_price."""
        strat = PriceTriggerStrategy(buy_price=105.0, sell_price=103.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        closed = [t for t in result.trades if t.exit_reason != "end_of_data"]
        assert len(closed) >= 1
        # Exit should happen during the ramp down, price ~ 103
        assert closed[0].exit_price <= 103.2

    def test_exact_bar_entry(self):
        """Entry happens on the exact minute bar where price crosses threshold."""
        strat = PriceTriggerStrategy(buy_price=105.0, sell_price=103.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        # Price ramps from 100 to 110 in bars 50-99, 105 is at bar 75
        entry_bar = int((t.entry_time - pd.Timestamp("2025-01-01", tz="UTC")).total_seconds() / 60)
        assert 74 <= entry_bar <= 76  # Allow ±1 for rounding


# ===================================================================
# 3. NAV and drawdown
# ===================================================================

class TestNAVAndDrawdown:
    NAME = "test_nav"

    def setup_method(self):
        _cleanup(self.NAME)
        # Price: 100 for 10 bars, ramp to 120, ramp down to 90, back to 100
        prices = np.concatenate([
            np.full(10, 100.0),       # 0-9: flat
            np.linspace(100, 120, 40), # 10-49: up to 120
            np.linspace(120, 90, 60),  # 50-109: down to 90
            np.linspace(90, 100, 40),  # 110-149: recovery
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_nav_equals_balance_when_flat(self):
        """NAV should equal balance when no position is open."""
        strat = AlwaysLongStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        # Before entry (bars 0-9), NAV == balance == 100k
        for i in range(10):
            assert bars.iloc[i]["nav"] == bars.iloc[i]["balance"] == 100_000

    def test_nav_includes_unrealized(self):
        """NAV should include unrealized PnL during position."""
        strat = AlwaysLongStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        # At bar 49 (price=120), position entered at 100
        # Unrealized = 100k * (120-100)/100 = +20k, NAV = 120k
        nav_49 = bars.iloc[49]["nav"]
        assert abs(nav_49 - 120_000) < 100  # Allow small rounding

    def test_drawdown_at_peak(self):
        """Drawdown should be 0 at the peak NAV."""
        strat = AlwaysLongStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        # Peak NAV is around bar 49 (price=120)
        peak_idx = bars["nav"].idxmax()
        assert bars.loc[peak_idx, "drawdown_pct"] == 0.0

    def test_drawdown_after_decline(self):
        """Drawdown should be negative after price drops from peak."""
        strat = AlwaysLongStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        # At bar 109 (price=90), entered at 100, peak was 120
        # NAV = 100k * 90/100 = 90k, peak NAV = 120k
        # Drawdown = (90k - 120k) / 120k = -25%
        dd_109 = bars.iloc[109]["drawdown_pct"]
        assert dd_109 < -20  # Should be around -25%
        assert dd_109 > -30

    def test_drawdown_always_non_positive(self):
        """Drawdown should never be positive."""
        strat = AlwaysLongStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        assert all(bars["drawdown_pct"] <= 0.001)  # tiny float tolerance


# ===================================================================
# 4. Start and end equity
# ===================================================================

class TestStartEndEquity:
    NAME = "test_equity"

    def setup_method(self):
        _cleanup(self.NAME)
        # Price: 100 → 110 → 100
        prices = np.concatenate([
            np.full(10, 100.0),
            np.linspace(100, 110, 60),
            np.full(80, 110.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_initial_equity_matches_capital(self):
        """First bar NAV should equal initial capital."""
        strat = FixedEntryExitStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=50_000, costs=CostModel())
        assert result.initial_capital == 50_000
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        assert bars.iloc[0]["nav"] == 50_000

    def test_final_equity_matches_result(self):
        """Final bar NAV should equal result.final_capital after all positions closed."""
        strat = FixedEntryExitStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=50_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        # After trade closed, final equity = final_capital
        assert abs(result.final_capital - bars.iloc[-1]["nav"]) < 1.0

    def test_total_return_calculation(self):
        """Total return should match (final - initial) / initial."""
        strat = FixedEntryExitStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        expected_pct = (result.final_capital / 100_000 - 1) * 100
        assert abs(result.total_return_pct - expected_pct) < 0.01

    def test_different_initial_capitals(self):
        """Engine should work correctly with different starting capitals."""
        for cap in [10_000, 100_000, 1_000_000]:
            _cleanup(self.NAME)
            _build_data(self.NAME, np.concatenate([
                np.full(10, 100.0), np.linspace(100, 110, 60), np.full(80, 110.0),
            ]), _make_spec({"1m": []}))
            strat = FixedEntryExitStrategy()
            strat.config.name = self.NAME
            result = BacktestEngine(verbose=False).run(strat, capital=cap, costs=CostModel())
            # Trade PnL should scale proportionally
            if result.trades:
                pnl_pct = result.trades[0].gross_pnl / cap * 100
                assert 5 < pnl_pct < 15  # ~10% gain


# ===================================================================
# 5. Equity zero — strategy loses all capital
# ===================================================================

class TestEquityZero:
    NAME = "test_zero"

    def setup_method(self):
        _cleanup(self.NAME)
        # Sawtooth: rises from 100 to 105 (bars 0-4), drops to 80 (bars 5-14),
        # repeats every 20 bars. Strategy buys at peaks, sells at troughs.
        cycle = np.concatenate([
            np.linspace(100, 105, 5),   # 0-4: small rise
            np.linspace(105, 80, 10),   # 5-14: big drop
            np.linspace(80, 100, 5),    # 15-19: recovery
        ])
        prices = np.tile(cycle, 15)  # 300 bars, 15 cycles
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_capital_drains_with_losses(self):
        """Repeated losing trades should drain capital."""
        strat = LosingStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=10_000, costs=CostModel())
        # Should have multiple losing trades
        losing = [t for t in result.trades if t.net_pnl < 0]
        assert len(losing) > 3
        # Capital should be significantly reduced
        assert result.final_capital < 5_000

    def test_no_trades_after_zero_capital(self):
        """Engine should not open positions when balance <= 0."""
        strat = LosingStrategy()
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=1_000, costs=CostModel())
        bars = pd.read_parquet(list((strategy_folder(self.NAME) / "results").glob("*_bars.parquet"))[0])
        # Once balance goes very low, no more positions should open
        # Check that late bars are all flat
        last_50 = bars.iloc[-50:]
        # There should be some flat bars at the end (depleted capital)
        flat_count = (last_50["position_side"] == "flat").sum()
        assert flat_count > 10  # Most should be flat


# ===================================================================
# 6. Fixed sizing vs compound (reinvest)
# ===================================================================

class TestSizing:
    NAME = "test_sizing"

    def setup_method(self):
        _cleanup(self.NAME)
        # Steady uptrend: 100 → 120 over 300 bars (each trade profitable)
        prices = np.linspace(100, 120, 300)
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_fixed_sizing_same_size_each_trade(self):
        """With fixed sizing, every trade should have the same notional size."""
        config = StrategyConfig(
            name=self.NAME,
            fixed_size=True,
            fixed_size_amount=50_000,
        )
        strat = RepeatedTradeStrategy(config=config)
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        assert len(result.trades) >= 2
        # All trades should be exactly 50k
        for t in result.trades:
            assert abs(t.size - 50_000) < 0.01

    def test_compound_sizing_grows_with_balance(self):
        """With compound sizing, position size should grow as balance increases."""
        config = StrategyConfig(name=self.NAME)
        strat = RepeatedTradeStrategy(config=config)
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        assert len(result.trades) >= 2
        # Since every trade is profitable in uptrend, sizes should increase
        sizes = [t.size for t in result.trades]
        # Second trade should be larger than first
        assert sizes[1] > sizes[0]

    def test_fixed_vs_compound_final_capital(self):
        """Compound sizing should produce higher final capital in uptrend."""
        # Fixed
        config_fixed = StrategyConfig(
            name=self.NAME,
            fixed_size=True,
            fixed_size_amount=100_000,
        )
        strat_fixed = RepeatedTradeStrategy(config=config_fixed)
        r_fixed = BacktestEngine(verbose=False).run(strat_fixed, capital=100_000, costs=CostModel())

        _cleanup(self.NAME)
        _build_data(self.NAME, np.linspace(100, 120, 300), _make_spec({"1m": []}))

        # Compound
        config_compound = StrategyConfig(name=self.NAME)
        strat_compound = RepeatedTradeStrategy(config=config_compound)
        r_compound = BacktestEngine(verbose=False).run(strat_compound, capital=100_000, costs=CostModel())

        # Compound should outperform in trending market
        assert r_compound.final_capital > r_fixed.final_capital


# ===================================================================
# 7. Stop loss
# ===================================================================

class TestStopLoss:
    NAME = "test_sl"

    def _build_long_sl_data(self):
        """Price: flat 100, enters at bar 10, drops to 90 at bar 60."""
        prices = np.concatenate([
            np.full(10, 100.0),          # 0-9: flat
            np.full(20, 100.0),          # 10-29: flat (position open)
            np.linspace(100, 90, 30),    # 30-59: drops to 90
            np.full(90, 90.0),           # 60-149: flat
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def _build_short_sl_data(self):
        """Price: flat 100, enters short at bar 10, rises to 110 at bar 60."""
        prices = np.concatenate([
            np.full(10, 100.0),
            np.full(20, 100.0),
            np.linspace(100, 110, 30),   # rises to 110
            np.full(90, 110.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))

    def setup_method(self):
        _cleanup(self.NAME)

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_long_stop_loss_fires(self):
        """5% SL on long should fire when price drops 5% from 100."""
        self._build_long_sl_data()
        strat = FixedEntryExitStrategy(sl_pct=5.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        assert len(result.trades) == 1
        t = result.trades[0]
        assert "stop_loss_hit" in t.exit_reason
        # SL price = 100 * (1 - 0.05) = 95
        assert t.exit_price <= 95.0

    def test_long_stop_loss_exact_bar(self):
        """SL should fire on the exact bar where price breaches SL level."""
        self._build_long_sl_data()
        strat = FixedEntryExitStrategy(sl_pct=5.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        # Price ramps from 100 to 90 in bars 30-59 (30 bars for 10 point drop)
        # 5% = 5 points, so SL at bar 30 + 15 = 45
        exit_bar = int((t.exit_time - pd.Timestamp("2025-01-01", tz="UTC")).total_seconds() / 60)
        assert 44 <= exit_bar <= 46

    def test_long_stop_loss_does_not_fire_early(self):
        """SL should not fire while price is above SL level."""
        self._build_long_sl_data()
        strat = FixedEntryExitStrategy(sl_pct=20.0)  # Very wide SL
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        # With 20% SL (SL at 80), price only drops to 90 — SL should NOT fire
        # Strategy closes at bar 70 instead
        t = result.trades[0]
        assert "stop_loss" not in t.exit_reason
        assert t.exit_reason == "exit"

    def test_short_stop_loss_fires(self):
        """SL on short should fire when price rises above SL level."""
        self._build_short_sl_data()
        strat = ShortWithStopLoss(sl_pct=5.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        assert "stop_loss_hit" in t.exit_reason
        # SL price = 100 * (1 + 0.05) = 105
        assert t.exit_price >= 105.0

    def test_stop_loss_pnl_correct(self):
        """PnL on SL trade should match the loss at SL level."""
        self._build_long_sl_data()
        strat = FixedEntryExitStrategy(sl_pct=5.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        # Loss should be ~5% of position size
        expected_loss_pct = (t.exit_price - t.entry_price) / t.entry_price
        actual_loss_pct = t.gross_pnl / t.size
        assert abs(expected_loss_pct - actual_loss_pct) < 0.001


# ===================================================================
# 8. Trailing stop loss
# ===================================================================

class TestTrailingStopLoss:
    NAME = "test_tsl"

    def setup_method(self):
        _cleanup(self.NAME)

    def teardown_method(self):
        _cleanup(self.NAME)

    def test_tsl_long_fires_after_peak(self):
        """TSL should fire when price retraces from peak by TSL %."""
        # Price: flat 100, ramp to 120, then drop to 100
        prices = np.concatenate([
            np.full(10, 100.0),          # 0-9
            np.linspace(100, 120, 40),   # 10-49: ramp up
            np.linspace(120, 100, 40),   # 50-89: drop
            np.full(60, 100.0),          # 90-149: flat
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))
        strat = FixedEntryExitStrategy(tsl_pct=5.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        assert "trailing_stop_hit" in t.exit_reason
        # Best price should be ~120, TSL at 120*0.95 = 114
        assert t.exit_price <= 114.5
        assert t.exit_price >= 113.0

    def test_tsl_tracks_best_price(self):
        """TSL price should move up with price, never down (for longs)."""
        # Price: rise to 130 quickly, then sharp drop — all before bar 70 exit
        prices = np.concatenate([
            np.full(10, 100.0),           # 0-9: flat
            np.linspace(100, 130, 30),    # 10-39: ramp to 130
            np.linspace(130, 110, 20),    # 40-59: drop to 110 (>10% from 130)
            np.full(90, 110.0),           # 60-149: flat
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))
        strat = FixedEntryExitStrategy(tsl_pct=10.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        assert "trailing_stop_hit" in t.exit_reason
        # Best price = ~130, TSL = 130*0.90 = 117
        assert t.exit_price <= 117.5
        assert t.exit_price >= 116.0

    def test_tsl_does_not_fire_in_uptrend(self):
        """TSL should not fire while price keeps making new highs."""
        # Steady uptrend only, no pullback > TSL%
        prices = np.concatenate([
            np.full(10, 100.0),
            np.linspace(100, 110, 60),  # Steady 10% rise
            np.full(80, 110.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))
        strat = FixedEntryExitStrategy(tsl_pct=5.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        # No pullback > 5%, so TSL should NOT fire; strategy exits at bar 70
        assert t.exit_reason == "exit"

    def test_tsl_short_fires_on_bounce(self):
        """TSL on short should fire when price bounces up from best (lowest) price."""
        prices = np.concatenate([
            np.full(10, 100.0),
            np.linspace(100, 80, 50),    # Drop to 80
            np.linspace(80, 95, 30),     # Bounce to 95
            np.full(60, 95.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))
        strat = ShortWithStopLoss(sl_pct=None, tsl_pct=10.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        assert "trailing_stop_hit" in t.exit_reason
        # Best price (lowest for short) = ~80, TSL = 80*1.10 = 88
        assert t.exit_price >= 87.5
        assert t.exit_price <= 89.0

    def test_both_sl_and_tsl_sl_fires_first(self):
        """When both SL and TSL are set, SL should fire if hit first."""
        # Immediate drop — SL should fire before TSL gets a chance
        prices = np.concatenate([
            np.full(10, 100.0),
            np.linspace(100, 80, 30),   # Sharp drop to 80 (20%)
            np.full(110, 80.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))
        strat = FixedEntryExitStrategy(sl_pct=5.0, tsl_pct=10.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        # SL at 95 should fire before TSL (which starts at 95 too, but SL checks first)
        assert "stop_loss_hit" in t.exit_reason

    def test_both_sl_and_tsl_tsl_fires_after_rally(self):
        """TSL should fire on pullback even when SL was never hit."""
        # Rally then pullback — SL never hit, TSL fires on pullback from peak
        prices = np.concatenate([
            np.full(10, 100.0),
            np.linspace(100, 130, 40),   # Rally to 130
            np.linspace(130, 115, 20),   # Pullback to 115 (~11.5% from peak)
            np.full(80, 115.0),
        ])
        _build_data(self.NAME, prices, _make_spec({"1m": []}))
        strat = FixedEntryExitStrategy(sl_pct=15.0, tsl_pct=10.0)
        strat.config.name = self.NAME
        result = BacktestEngine(verbose=False).run(strat, capital=100_000, costs=CostModel())
        t = result.trades[0]
        # SL at 85 (never hit), TSL from peak 130 at 117 → fires during pullback
        assert "trailing_stop_hit" in t.exit_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
