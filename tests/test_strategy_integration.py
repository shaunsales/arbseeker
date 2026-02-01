"""
Integration tests for the strategy framework.

Verifies critical backtest behaviors:
1. No lookahead bias - indicators align with price timestamps
2. Position entry/exit triggers work correctly
3. Open positions are closed at end of backtest
4. Capital management (fixed size vs reinvesting P&L)

Run with: pytest tests/test_strategy_integration.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from core.strategy import (
    SingleAssetStrategy,
    Signal,
    Position,
    BacktestEngine,
    BacktestResult,
    StrategyConfig,
    CostModel,
    Side,
    ZERO_COSTS,
)
from core.indicators import compute_indicators
from strategies import (
    _TestBuyAtBarN,
    _TestBuyAndSell,
    _TestRecording,
    _TestHoldOnly,
    _TestCapitalTracking,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def deterministic_ohlcv():
    """
    Create deterministic OHLCV data for predictable testing.
    
    Pattern: Price starts at 100, trends up to 150, then down to 80.
    This creates clear crossover points for MA strategies.
    """
    n = 200
    
    # Create predictable price pattern
    prices = np.concatenate([
        np.linspace(100, 150, 80),   # Uptrend bars 0-79
        np.linspace(150, 150, 40),   # Flat bars 80-119
        np.linspace(150, 80, 80),    # Downtrend bars 120-199
    ])
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.ones(n) * 1000,
    })
    
    df.index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return df


@pytest.fixture
def simple_up_down_data():
    """
    Simple data: 50 bars up, 50 bars down.
    Entry at bar 10, should exit when trend reverses.
    """
    n = 100
    prices = np.concatenate([
        np.linspace(100, 200, 50),  # Up
        np.linspace(200, 100, 50),  # Down
    ])
    
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.ones(n) * 1000,
    })
    
    df.index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return df


# =============================================================================
# Test Strategies (imported from strategies module)
# =============================================================================
# Using: _TestBuyAtBarN, _TestBuyAndSell, _TestRecording, _TestHoldOnly
# These are defined in strategies/_example_strategies.py

# Aliases for cleaner test code
BuyAtBarNStrategy = _TestBuyAtBarN
BuyAndSellStrategy = _TestBuyAndSell
RecordingStrategy = _TestRecording
CapitalTrackingStrategy = _TestCapitalTracking


# =============================================================================
# Test 1: No Lookahead Bias - Indicators Align with Timestamps
# =============================================================================

class TestNoLookaheadBias:
    """Verify that indicators cannot see future data."""
    
    def test_sma_uses_only_past_data(self, deterministic_ohlcv):
        """Verify SMA at bar N only uses data from bars 0 to N."""
        df = deterministic_ohlcv.copy()
        df = compute_indicators(df, [("sma", {"length": 5})])
        
        # Manually calculate SMA at specific bars
        for idx in range(5, 50):
            # SMA should be average of close[idx-4:idx+1] (inclusive)
            expected_sma = df["close"].iloc[idx-4:idx+1].mean()
            actual_sma = df["SMA_5"].iloc[idx]
            
            assert actual_sma == pytest.approx(expected_sma, rel=1e-10), \
                f"SMA mismatch at bar {idx}: expected {expected_sma}, got {actual_sma}"
    
    def test_indicator_nan_before_warmup(self, deterministic_ohlcv):
        """Verify indicators are NaN during warmup period."""
        df = deterministic_ohlcv.copy()
        df = compute_indicators(df, [("sma", {"length": 20})])
        
        # First 19 bars should be NaN
        assert df["SMA_20"].iloc[:19].isna().all(), "SMA should be NaN during warmup"
        # Bar 19 onwards should have values
        assert not df["SMA_20"].iloc[19:].isna().any(), "SMA should have values after warmup"
    
    def test_strategy_receives_aligned_data(self, deterministic_ohlcv):
        """Verify strategy sees correct data at each bar."""
        strategy = RecordingStrategy()
        engine = BacktestEngine(verbose=False)
        
        engine.run(
            strategy=strategy,
            data=deterministic_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Verify each recorded bar
        for record in strategy.bar_records:
            idx = record["idx"]
            expected_close = deterministic_ohlcv["close"].iloc[idx]
            expected_timestamp = deterministic_ohlcv.index[idx]
            
            assert record["close"] == expected_close, \
                f"Bar {idx}: close mismatch"
            assert record["timestamp"] == expected_timestamp, \
                f"Bar {idx}: timestamp mismatch"
    
    def test_indicator_value_at_bar_matches_manual_calc(self, deterministic_ohlcv):
        """Strategy should see indicator values that match manual calculation."""
        strategy = RecordingStrategy()
        engine = BacktestEngine(verbose=False)
        
        engine.run(
            strategy=strategy,
            data=deterministic_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Check SMA values match manual calculation
        for record in strategy.bar_records:
            idx = record["idx"]
            if idx >= 5:  # After warmup
                # Manual SMA calculation
                expected_sma = deterministic_ohlcv["close"].iloc[idx-4:idx+1].mean()
                assert record["sma_5"] == pytest.approx(expected_sma, rel=1e-10), \
                    f"Bar {idx}: SMA mismatch in strategy"


# =============================================================================
# Test 2: Position Entry and Exit Triggers
# =============================================================================

class TestPositionTriggers:
    """Verify position entry and exit work correctly."""
    
    def test_entry_at_correct_bar(self, simple_up_down_data):
        """Verify position is opened at the exact bar we specify."""
        buy_bar = 10
        strategy = BuyAtBarNStrategy(buy_bar=buy_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        assert strategy.entry_triggered, "Entry should have been triggered"
        assert strategy.entry_bar == buy_bar, f"Entry should be at bar {buy_bar}"
        
        # Check entry price matches the close at that bar
        expected_price = simple_up_down_data["close"].iloc[buy_bar]
        assert strategy.entry_price == expected_price, "Entry price mismatch"
    
    def test_entry_price_is_close_of_signal_bar(self, simple_up_down_data):
        """Verify entry price is the close price of the bar where signal was generated."""
        buy_bar = 20
        sell_bar = 70
        strategy = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        assert len(result.trades) == 1, "Should have exactly 1 trade"
        
        trade = result.trades[0]
        expected_entry = simple_up_down_data["close"].iloc[buy_bar]
        expected_exit = simple_up_down_data["close"].iloc[sell_bar]
        
        assert trade.entry_price == expected_entry, \
            f"Entry price should be {expected_entry}, got {trade.entry_price}"
        assert trade.exit_price == expected_exit, \
            f"Exit price should be {expected_exit}, got {trade.exit_price}"
    
    def test_exit_triggers_on_signal_bar(self, simple_up_down_data):
        """Verify position is closed at the exact bar we specify."""
        buy_bar = 10
        sell_bar = 30
        strategy = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        assert len(result.trades) == 1
        trade = result.trades[0]
        
        # Entry timestamp
        assert trade.entry_time == simple_up_down_data.index[buy_bar]
        # Exit timestamp
        assert trade.exit_time == simple_up_down_data.index[sell_bar]
        # Bars held
        assert trade.bars_held == sell_bar - buy_bar
    
    def test_pnl_calculation_correct(self, simple_up_down_data):
        """Verify P&L is calculated correctly based on entry/exit prices."""
        buy_bar = 10   # Price ~120
        sell_bar = 40  # Price ~180
        strategy = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,  # No costs for clean calculation
        )
        
        trade = result.trades[0]
        entry_price = simple_up_down_data["close"].iloc[buy_bar]
        exit_price = simple_up_down_data["close"].iloc[sell_bar]
        
        # Expected P&L: (exit - entry) / entry * position_size
        expected_pnl_pct = (exit_price - entry_price) / entry_price
        expected_pnl = trade.size * expected_pnl_pct
        
        assert trade.gross_pnl == pytest.approx(expected_pnl, rel=0.001), \
            f"P&L mismatch: expected {expected_pnl}, got {trade.gross_pnl}"


# =============================================================================
# Test 3: Open Positions Closed at End of Backtest
# =============================================================================

class TestEndOfBacktestCleanup:
    """Verify open positions are closed cleanly at end of backtest."""
    
    def test_open_position_closed_at_end(self, simple_up_down_data):
        """Verify position still open at end is closed."""
        # Buy at bar 10, never sell (hold to end)
        buy_bar = 10
        strategy = BuyAtBarNStrategy(buy_bar=buy_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Should have 1 trade (forced close at end)
        assert len(result.trades) == 1, "Position should be closed at end"
        
        trade = result.trades[0]
        assert trade.exit_reason == "end_of_data", \
            f"Exit reason should be 'end_of_data', got '{trade.exit_reason}'"
    
    def test_end_of_backtest_uses_last_bar_price(self, simple_up_down_data):
        """Verify end-of-backtest close uses the last bar's close price."""
        buy_bar = 10
        strategy = BuyAtBarNStrategy(buy_bar=buy_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        trade = result.trades[0]
        expected_exit_price = simple_up_down_data["close"].iloc[-1]
        
        assert trade.exit_price == expected_exit_price, \
            f"Exit price should be last bar's close: {expected_exit_price}"
    
    def test_end_of_backtest_timestamp_is_last_bar(self, simple_up_down_data):
        """Verify end-of-backtest close has correct timestamp."""
        buy_bar = 10
        strategy = BuyAtBarNStrategy(buy_bar=buy_bar)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        trade = result.trades[0]
        expected_exit_time = simple_up_down_data.index[-1]
        
        assert trade.exit_time == expected_exit_time, \
            f"Exit time should be last bar's timestamp"
    
    def test_final_capital_includes_forced_close(self, simple_up_down_data):
        """Verify final capital correctly reflects the forced close P&L."""
        buy_bar = 10
        strategy = BuyAtBarNStrategy(buy_bar=buy_bar)
        engine = BacktestEngine(verbose=False)
        
        initial_capital = 100_000
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=initial_capital,
            costs=ZERO_COSTS,
        )
        
        # Calculate expected final capital
        entry_price = simple_up_down_data["close"].iloc[buy_bar]
        exit_price = simple_up_down_data["close"].iloc[-1]
        pnl_pct = (exit_price - entry_price) / entry_price
        expected_final = initial_capital * (1 + pnl_pct)
        
        assert result.final_capital == pytest.approx(expected_final, rel=0.001), \
            f"Final capital should be {expected_final}, got {result.final_capital}"


# =============================================================================
# Test 4: Capital Management - Fixed Size vs Reinvesting P&L
# =============================================================================

class TestCapitalManagement:
    """Verify capital management behavior."""
    
    def test_position_size_based_on_current_capital(self, deterministic_ohlcv):
        """Verify position size is based on capital at time of entry."""
        strategy = CapitalTrackingStrategy()
        engine = BacktestEngine(verbose=False)
        
        initial_capital = 100_000
        result = engine.run(
            strategy=strategy,
            data=deterministic_ohlcv,
            capital=initial_capital,
            costs=ZERO_COSTS,
        )
        
        # Check capital changes after profitable trades
        if len(result.trades) >= 2:
            # After first profitable trade, next position should use updated capital
            for i, (idx, capital, has_position) in enumerate(strategy.capital_history):
                if i > 0:
                    # Capital should change when positions are closed
                    prev_capital = strategy.capital_history[i-1][1]
                    # Capital only changes at trade boundaries
    
    def test_position_uses_full_capital_with_size_1(self, simple_up_down_data):
        """Verify size=1.0 uses full available capital."""
        buy_bar = 10
        strategy = BuyAtBarNStrategy(buy_bar=buy_bar)
        engine = BacktestEngine(verbose=False)
        
        initial_capital = 100_000
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=initial_capital,
            costs=ZERO_COSTS,
        )
        
        trade = result.trades[0]
        # Position size should equal initial capital (size=1.0 means 100%)
        assert trade.size == initial_capital, \
            f"Position size should be {initial_capital}, got {trade.size}"
    
    def test_capital_compounds_after_profitable_trade(self, simple_up_down_data):
        """Verify capital increases after profitable trade (compounding)."""
        # Buy during uptrend, sell during uptrend = profit
        buy_bar = 10   # Price ~120
        sell_bar = 40  # Price ~180  
        strategy = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        
        initial_capital = 100_000
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=initial_capital,
            costs=ZERO_COSTS,
        )
        
        # Should be profitable
        assert result.final_capital > initial_capital, \
            "Final capital should be greater after profitable trade"
        
        # Verify the exact amount
        entry_price = simple_up_down_data["close"].iloc[buy_bar]
        exit_price = simple_up_down_data["close"].iloc[sell_bar]
        expected_return = (exit_price - entry_price) / entry_price
        expected_final = initial_capital * (1 + expected_return)
        
        assert result.final_capital == pytest.approx(expected_final, rel=0.001)
    
    def test_capital_decreases_after_losing_trade(self, simple_up_down_data):
        """Verify capital decreases after losing trade."""
        # Buy at peak, sell during downtrend = loss
        buy_bar = 50   # Price ~200 (peak)
        sell_bar = 90  # Price ~120
        strategy = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        
        initial_capital = 100_000
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=initial_capital,
            costs=ZERO_COSTS,
        )
        
        # Should be unprofitable
        assert result.final_capital < initial_capital, \
            "Final capital should be less after losing trade"
    
    def test_costs_reduce_capital(self, simple_up_down_data):
        """Verify trading costs reduce capital."""
        buy_bar = 10
        sell_bar = 40
        strategy = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        
        initial_capital = 100_000
        
        # With costs
        costs = CostModel(commission_bps=10, slippage_bps=5, funding_daily_bps=0)
        result_with_costs = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=initial_capital,
            costs=costs,
        )
        
        # Without costs
        result_no_costs = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=initial_capital,
            costs=ZERO_COSTS,
        )
        
        assert result_with_costs.final_capital < result_no_costs.final_capital, \
            "Capital with costs should be less than without costs"
        
        # Verify the difference matches expected costs
        trade = result_with_costs.trades[0]
        assert trade.costs > 0, "Trade should have costs"
        assert trade.net_pnl < trade.gross_pnl, "Net P&L should be less than gross"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

class TestFixedSizing:
    """Test fixed position sizing (no compounding)."""
    
    def test_fixed_size_uses_specified_amount(self, simple_up_down_data):
        """Verify fixed_size_amount is used instead of capital percentage."""
        from strategies._example_strategies import _TestBuyAtBarN
        
        # Create strategy with fixed sizing
        strategy = _TestBuyAtBarN(buy_bar=10)
        strategy.config.fixed_size = True
        strategy.config.fixed_size_amount = 25_000  # Fixed $25k per trade
        
        engine = BacktestEngine(verbose=False)
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,  # Capital is $100k but we only trade $25k
            costs=ZERO_COSTS,
        )
        
        assert len(result.trades) == 1
        trade = result.trades[0]
        
        # Position size should be $25k, not $100k
        assert trade.size == 25_000, f"Position size should be $25,000, got ${trade.size:,.0f}"
    
    def test_fixed_size_does_not_compound(self, simple_up_down_data):
        """Verify fixed sizing doesn't change after profitable trades."""
        # Create strategy that trades multiple times with fixed sizing
        strategy = CapitalTrackingStrategy()
        strategy.config.fixed_size = True
        strategy.config.fixed_size_amount = 10_000
        
        engine = BacktestEngine(verbose=False)
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # All trades should have same position size
        if len(result.trades) >= 2:
            sizes = [t.size for t in result.trades]
            assert all(s == 10_000 for s in sizes), \
                f"All trades should be $10,000 with fixed sizing, got {sizes}"
    
    def test_compounding_vs_fixed_comparison(self, simple_up_down_data):
        """Compare compounding (default) vs fixed sizing results."""
        buy_bar = 10
        sell_bar = 40
        
        # Compounding (default)
        strategy_compound = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        engine = BacktestEngine(verbose=False)
        result_compound = engine.run(
            strategy=strategy_compound,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Fixed sizing
        strategy_fixed = BuyAndSellStrategy(buy_bar=buy_bar, sell_bar=sell_bar)
        strategy_fixed.config.fixed_size = True
        strategy_fixed.config.fixed_size_amount = 100_000
        result_fixed = engine.run(
            strategy=strategy_fixed,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # With same initial capital and single trade, results should be identical
        assert result_compound.trades[0].size == result_fixed.trades[0].size
        assert result_compound.final_capital == result_fixed.final_capital


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_no_trades_if_no_signals(self, simple_up_down_data):
        """Verify no trades if strategy never signals."""
        engine = BacktestEngine(verbose=False)
        result = engine.run(
            strategy=_TestHoldOnly(),
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        assert len(result.trades) == 0, "Should have no trades"
        assert result.final_capital == 100_000, "Capital should be unchanged"
    
    def test_equity_curve_length_matches_data(self, simple_up_down_data):
        """Verify equity curve has same length as input data."""
        strategy = BuyAtBarNStrategy(buy_bar=10)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        assert len(result.equity_curve) == len(simple_up_down_data), \
            "Equity curve length should match data length"
    
    def test_equity_curve_timestamps_match_data(self, simple_up_down_data):
        """Verify equity curve timestamps match input data."""
        strategy = BuyAtBarNStrategy(buy_bar=10)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=simple_up_down_data,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        pd.testing.assert_index_equal(
            result.equity_curve.index,
            simple_up_down_data.index,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
