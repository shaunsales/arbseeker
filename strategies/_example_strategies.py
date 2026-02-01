"""
Example strategies for testing and demonstration.

WARNING: These strategies are used by the test suite.
         Do not modify without updating corresponding tests.

Prefix: _example_ indicates these are for testing/reference only.
"""

from typing import Optional
import pandas as pd

from core.strategy import (
    SingleAssetStrategy,
    Signal,
    Position,
    StrategyConfig,
)


# =============================================================================
# Simple MA Crossover Strategy
# =============================================================================

class ExampleMACrossover(SingleAssetStrategy):
    """
    Simple Moving Average Crossover strategy.
    
    - Buys when fast MA crosses above slow MA
    - Sells when fast MA crosses below slow MA
    
    Usage:
        strategy = ExampleMACrossover(fast=10, slow=30)
        engine.run(strategy=strategy, data=df, capital=100_000)
    """
    
    def __init__(self, fast: int = 10, slow: int = 30, config: Optional[StrategyConfig] = None):
        config = config or StrategyConfig(name=f"MA_Crossover_{fast}_{slow}")
        super().__init__(config)
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
            return Signal.buy(size=1.0, reason="MA crossover up")
        elif fast_ma < slow_ma and position is not None:
            return Signal.close(reason="MA crossover down")
        
        return Signal.hold()


# =============================================================================
# RSI Mean Reversion Strategy
# =============================================================================

class ExampleRSIMeanReversion(SingleAssetStrategy):
    """
    RSI-based mean reversion strategy.
    
    - Buys when RSI drops below oversold level
    - Sells when RSI rises above overbought level
    
    Usage:
        strategy = ExampleRSIMeanReversion(period=14, oversold=30, overbought=70)
    """
    
    def __init__(
        self, 
        period: int = 14, 
        oversold: float = 30, 
        overbought: float = 70,
        config: Optional[StrategyConfig] = None,
    ):
        config = config or StrategyConfig(name=f"RSI_MeanRev_{period}")
        super().__init__(config)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def required_indicators(self):
        return [("rsi", {"length": self.period})]
    
    def on_bar(self, idx, data, capital, position):
        if idx < self.period:
            return Signal.hold()
        
        rsi = data[f"RSI_{self.period}"].iloc[idx]
        
        if pd.isna(rsi):
            return Signal.hold()
        
        if rsi < self.oversold and position is None:
            return Signal.buy(size=1.0, reason=f"RSI oversold ({rsi:.1f})")
        elif rsi > self.overbought and position is not None:
            return Signal.close(reason=f"RSI overbought ({rsi:.1f})")
        
        return Signal.hold()


# =============================================================================
# Bollinger Band Breakout Strategy  
# =============================================================================

class ExampleBollingerBreakout(SingleAssetStrategy):
    """
    Bollinger Band breakout strategy.
    
    - Buys when price breaks above upper band
    - Sells when price drops below lower band
    
    Usage:
        strategy = ExampleBollingerBreakout(length=20, std=2)
    """
    
    def __init__(
        self,
        length: int = 20,
        std: float = 2.0,
        config: Optional[StrategyConfig] = None,
    ):
        config = config or StrategyConfig(name=f"BB_Breakout_{length}")
        super().__init__(config)
        self.length = length
        self.std = int(std)  # For column naming
    
    def required_indicators(self):
        return [("bbands", {"length": self.length, "std": self.std})]
    
    def on_bar(self, idx, data, capital, position):
        if idx < self.length:
            return Signal.hold()
        
        close = data["close"].iloc[idx]
        upper = data[f"BBU_{self.length}_{self.std}"].iloc[idx]
        lower = data[f"BBL_{self.length}_{self.std}"].iloc[idx]
        
        if pd.isna(upper) or pd.isna(lower):
            return Signal.hold()
        
        if close > upper and position is None:
            return Signal.buy(size=1.0, reason="Price above upper BB")
        elif close < lower and position is not None:
            return Signal.close(reason="Price below lower BB")
        
        return Signal.hold()


# =============================================================================
# Test Utility Strategies (used by test suite)
# =============================================================================

class _TestBuyAtBarN(SingleAssetStrategy):
    """Test utility: Buy at specific bar N and hold. DO NOT MODIFY."""
    
    def __init__(self, buy_bar: int):
        super().__init__(StrategyConfig(name=f"_TestBuyAtBar{buy_bar}"))
        self.buy_bar = buy_bar
        self.entry_triggered = False
        self.entry_bar = None
        self.entry_price = None
        
    def required_indicators(self):
        return []
    
    def on_bar(self, idx, data, capital, position):
        if idx == self.buy_bar and position is None:
            self.entry_triggered = True
            self.entry_bar = idx
            self.entry_price = data["close"].iloc[idx]
            return Signal.buy(size=1.0, reason=f"Buy at bar {idx}")
        return Signal.hold()


class _TestBuyAndSell(SingleAssetStrategy):
    """Test utility: Buy at bar N, sell at bar M. DO NOT MODIFY."""
    
    def __init__(self, buy_bar: int, sell_bar: int):
        super().__init__(StrategyConfig(name="_TestBuySell"))
        self.buy_bar = buy_bar
        self.sell_bar = sell_bar
        self.events = []
        
    def required_indicators(self):
        return []
    
    def on_bar(self, idx, data, capital, position):
        price = data["close"].iloc[idx]
        
        if idx == self.buy_bar and position is None:
            self.events.append(("buy", idx, price, capital))
            return Signal.buy(size=1.0, reason="Scheduled buy")
        
        if idx == self.sell_bar and position is not None:
            self.events.append(("sell", idx, price, capital))
            return Signal.close(reason="Scheduled sell")
        
        return Signal.hold()


class _TestRecording(SingleAssetStrategy):
    """Test utility: Records data at each bar. DO NOT MODIFY."""
    
    def __init__(self):
        super().__init__(StrategyConfig(name="_TestRecording"))
        self.bar_records = []
        
    def required_indicators(self):
        return [("sma", {"length": 5})]
    
    def on_bar(self, idx, data, capital, position):
        record = {
            "idx": idx,
            "timestamp": data.index[idx],
            "close": data["close"].iloc[idx],
            "sma_5": data["SMA_5"].iloc[idx] if "SMA_5" in data.columns else None,
        }
        self.bar_records.append(record)
        return Signal.hold()


class _TestHoldOnly(SingleAssetStrategy):
    """Test utility: Never trades, just holds. DO NOT MODIFY."""
    
    def __init__(self):
        super().__init__(StrategyConfig(name="_TestHoldOnly"))
    
    def required_indicators(self):
        return []
    
    def on_bar(self, idx, data, capital, position):
        return Signal.hold()


class _TestCapitalTracking(SingleAssetStrategy):
    """Test utility: Tracks capital changes after each trade. DO NOT MODIFY."""
    
    def __init__(self):
        super().__init__(StrategyConfig(name="_TestCapitalTracking"))
        self.capital_history = []
        self.trade_count = 0
        
    def required_indicators(self):
        return [("sma", {"length": 5}), ("sma", {"length": 10})]
    
    def on_bar(self, idx, data, capital, position):
        self.capital_history.append((idx, capital, position is not None))
        
        if idx < 10:
            return Signal.hold()
        
        fast = data["SMA_5"].iloc[idx]
        slow = data["SMA_10"].iloc[idx]
        
        if fast > slow and position is None:
            self.trade_count += 1
            return Signal.buy(size=1.0, reason="Cross up")
        elif fast < slow and position is not None:
            return Signal.close(reason="Cross down")
        
        return Signal.hold()
