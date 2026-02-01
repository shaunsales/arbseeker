"""
Tests for the strategy framework.

Run with: pytest tests/test_strategy.py -v
"""

import pytest
import pandas as pd
import numpy as np

from core.strategy import (
    SingleAssetStrategy,
    Signal,
    Position,
    Trade,
    CostModel,
    Side,
    BacktestEngine,
    BacktestResult,
    StrategyConfig,
    ZERO_COSTS,
    DEFAULT_COSTS,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data with a trend for testing."""
    np.random.seed(42)
    n = 500
    
    # Create trending data (up then down)
    trend = np.concatenate([
        np.linspace(100, 150, n // 2),  # Uptrend
        np.linspace(150, 120, n // 2),  # Downtrend
    ])
    noise = np.random.randn(n) * 2
    close = trend + noise
    
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.002),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    
    df.index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return df


class SimpleMAStrategy(SingleAssetStrategy):
    """Simple MA crossover strategy for testing."""
    
    def __init__(self, fast=10, slow=30):
        super().__init__(StrategyConfig(name="SimpleMA"))
        self.fast = fast
        self.slow = slow
    
    def required_indicators(self):
        return [
            ("sma", {"length": self.fast}),
            ("sma", {"length": self.slow}),
        ]
    
    def on_bar(self, idx, data, capital, position):
        if idx < self.slow:
            return Signal.hold()
        
        fast_ma = data[f"SMA_{self.fast}"].iloc[idx]
        slow_ma = data[f"SMA_{self.slow}"].iloc[idx]
        
        if fast_ma > slow_ma and position is None:
            return Signal.buy(size=1.0, reason="MA cross up")
        elif fast_ma < slow_ma and position is not None:
            return Signal.close(reason="MA cross down")
        
        return Signal.hold()


class AlwaysLongStrategy(SingleAssetStrategy):
    """Strategy that buys on first bar and holds."""
    
    def required_indicators(self):
        return []
    
    def on_bar(self, idx, data, capital, position):
        if idx == 0 and position is None:
            return Signal.buy(size=1.0, reason="Initial buy")
        return Signal.hold()


class TestSignal:
    """Test Signal class."""
    
    def test_buy_signal(self):
        signal = Signal.buy(size=0.5, reason="test")
        assert signal.action == "buy"
        assert signal.size == 0.5
        assert signal.reason == "test"
    
    def test_sell_signal(self):
        signal = Signal.sell(size=1.0, reason="short")
        assert signal.action == "sell"
        assert signal.size == 1.0
    
    def test_close_signal(self):
        signal = Signal.close(reason="exit")
        assert signal.action == "close"
    
    def test_hold_signal(self):
        signal = Signal.hold()
        assert signal.action == "hold"


class TestPosition:
    """Test Position class."""
    
    def test_position_long_pnl(self):
        pos = Position(
            symbol="TEST",
            side=Side.LONG,
            entry_price=100.0,
            size=1000.0,
        )
        
        # Price goes up 10%
        pos.update_price(110.0)
        assert pos.unrealized_pnl == pytest.approx(100.0, rel=0.01)  # 10% of 1000
    
    def test_position_short_pnl(self):
        pos = Position(
            symbol="TEST",
            side=Side.SHORT,
            entry_price=100.0,
            size=1000.0,
        )
        
        # Price goes down 10%
        pos.update_price(90.0)
        assert pos.unrealized_pnl == pytest.approx(100.0, rel=0.01)  # 10% of 1000
    
    def test_position_close(self):
        from datetime import datetime, timezone
        
        pos = Position(
            symbol="TEST",
            side=Side.LONG,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            entry_price=100.0,
            size=1000.0,
        )
        
        trade = pos.close(110.0, datetime(2024, 1, 2, tzinfo=timezone.utc))
        
        assert trade.gross_pnl == pytest.approx(100.0, rel=0.01)
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0


class TestCostModel:
    """Test CostModel class."""
    
    def test_round_trip_cost(self):
        costs = CostModel(commission_bps=5.0, slippage_bps=2.0)
        
        # (5 + 2) * 2 = 14 bps round trip
        # 14 bps of 10000 = 14
        cost = costs.round_trip_cost(10000)
        assert cost == pytest.approx(14.0, rel=0.01)
    
    def test_holding_cost(self):
        costs = CostModel(funding_daily_bps=10.0, bars_per_day=24)
        
        # 10 bps per day, hold for 24 bars (1 day)
        cost = costs.holding_cost(10000, bars_held=24)
        assert cost == pytest.approx(10.0, rel=0.01)
    
    def test_zero_costs(self):
        assert ZERO_COSTS.round_trip_cost(10000) == 0.0
        assert ZERO_COSTS.holding_cost(10000, 100) == 0.0


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    def test_backtest_runs(self, sample_ohlcv):
        strategy = SimpleMAStrategy(fast=10, slow=30)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=sample_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        assert isinstance(result, BacktestResult)
        assert result.total_bars == len(sample_ohlcv)
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(sample_ohlcv)
    
    def test_backtest_makes_trades(self, sample_ohlcv):
        strategy = SimpleMAStrategy(fast=10, slow=30)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=sample_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Should make some trades on trending data
        assert len(result.trades) > 0
    
    def test_backtest_respects_costs(self, sample_ohlcv):
        strategy = AlwaysLongStrategy()
        engine = BacktestEngine(verbose=False)
        
        # Run with zero costs
        result_no_cost = engine.run(
            strategy=strategy,
            data=sample_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Run with costs
        result_with_cost = engine.run(
            strategy=strategy,
            data=sample_ohlcv,
            capital=100_000,
            costs=DEFAULT_COSTS,
        )
        
        # With costs should have lower final capital
        assert result_with_cost.final_capital < result_no_cost.final_capital
    
    def test_metrics_calculated(self, sample_ohlcv):
        strategy = SimpleMAStrategy(fast=10, slow=30)
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=sample_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        # Metrics should be populated
        assert result.total_return_pct != 0.0 or result.total_trades == 0
        assert result.max_drawdown_pct >= 0.0
        assert 0.0 <= result.win_rate <= 100.0


class TestBacktestResult:
    """Test BacktestResult class."""
    
    def test_summary(self, sample_ohlcv):
        strategy = SimpleMAStrategy()
        engine = BacktestEngine(verbose=False)
        
        result = engine.run(
            strategy=strategy,
            data=sample_ohlcv,
            capital=100_000,
            costs=ZERO_COSTS,
        )
        
        summary = result.summary()
        
        assert "strategy" in summary
        assert "total_return_pct" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown_pct" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
