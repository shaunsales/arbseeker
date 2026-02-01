"""
Strategy base classes for backtesting.

Provides abstract base classes for:
- SingleAssetStrategy: Long/short on one asset
- MultiLeggedStrategy: Multiple simultaneous positions (e.g., basis arb)

Strategies define:
1. required_indicators() - What indicators to pre-compute
2. on_bar() - Logic called for each bar, returns Signal(s)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from core.strategy.position import Signal, Position, CostModel


@dataclass
class DataSpec:
    """Specification for data required by a strategy."""
    venue: str
    market: str
    ticker: str
    interval: str
    
    def __str__(self):
        return f"{self.venue}/{self.market}/{self.ticker}/{self.interval}"


@dataclass 
class StrategyConfig:
    """Base configuration for all strategies."""
    name: str = "unnamed"
    capital: float = 100_000
    max_position_pct: float = 1.0  # Max position as % of capital
    costs: CostModel = field(default_factory=CostModel)
    
    # Position sizing mode
    fixed_size: bool = False  # If True, use fixed position size (no compounding)
    fixed_size_amount: float = 0.0  # Fixed $ amount per trade (0 = use capital * max_position_pct)
    
    # Spread trading mode (for basis arb)
    spread_pnl_mode: bool = False  # If True, calculate P&L based on spread convergence


class SingleAssetStrategy(ABC):
    """
    Base class for single-asset strategies.
    
    Supports long and short positions on one asset.
    Indicators are pre-computed before backtest for efficient array access.
    
    Example:
        class MACrossover(SingleAssetStrategy):
            def __init__(self, fast=10, slow=20):
                self.fast = fast
                self.slow = slow
            
            def required_indicators(self):
                return [
                    ("sma", {"length": self.fast}),
                    ("sma", {"length": self.slow}),
                ]
            
            def on_bar(self, idx, data, capital, position):
                fast_ma = data[f"SMA_{self.fast}"].iloc[idx]
                slow_ma = data[f"SMA_{self.slow}"].iloc[idx]
                
                if fast_ma > slow_ma and position is None:
                    return Signal.buy(size=1.0, reason="MA crossover up")
                elif fast_ma < slow_ma and position is not None:
                    return Signal.close(reason="MA crossover down")
                
                return Signal.hold()
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
    
    @property
    def name(self) -> str:
        """Strategy name for display."""
        return self.config.name or self.__class__.__name__
    
    @abstractmethod
    def required_indicators(self) -> list[tuple[str, dict]]:
        """
        Return list of indicators to pre-compute.
        
        Returns:
            List of (indicator_name, params) tuples
            
        Example:
            return [
                ("sma", {"length": 20}),
                ("rsi", {"length": 14}),
            ]
        """
        pass
    
    @abstractmethod
    def on_bar(
        self,
        idx: int,
        data: pd.DataFrame,
        capital: float,
        position: Optional[Position],
    ) -> Signal:
        """
        Called for each bar during backtest.
        
        Args:
            idx: Current bar index (0-based). Use data.iloc[idx] for current bar.
                 Can look back with data.iloc[idx-N:idx] for N bars of history.
            data: Full DataFrame with OHLCV + pre-computed indicators.
            capital: Current available capital.
            position: Current open position (None if flat).
            
        Returns:
            Signal indicating action to take (buy, sell, close, hold)
        """
        pass
    
    def on_start(self, data: pd.DataFrame) -> None:
        """Called once before backtest starts. Override for setup logic."""
        pass
    
    def on_end(self, data: pd.DataFrame) -> None:
        """Called once after backtest ends. Override for cleanup logic."""
        pass


class MultiLeggedStrategy(ABC):
    """
    Base class for multi-legged strategies.
    
    Supports multiple simultaneous positions across different assets/venues.
    Useful for arbitrage, pairs trading, spread strategies.
    
    Example:
        class BasisArb(MultiLeggedStrategy):
            def __init__(self, threshold_bps=50):
                self.threshold_bps = threshold_bps
            
            def required_data(self):
                return {
                    "spot": DataSpec("binance", "spot", "BTCUSDT", "1h"),
                    "perp": DataSpec("binance", "futures", "BTCUSDT", "1h"),
                }
            
            def required_indicators(self):
                return {"spot": [], "perp": []}  # No indicators needed
            
            def on_bar(self, idx, data, capital, positions):
                spot_price = data["spot"].iloc[idx]["close"]
                perp_price = data["perp"].iloc[idx]["close"]
                basis_bps = (perp_price - spot_price) / spot_price * 10000
                
                if abs(basis_bps) > self.threshold_bps and not positions:
                    if basis_bps > 0:  # Perp premium
                        return {
                            "spot": Signal.buy(size=0.5, reason="Long spot"),
                            "perp": Signal.sell(size=0.5, reason="Short perp"),
                        }
                    else:  # Perp discount
                        return {
                            "spot": Signal.sell(size=0.5, reason="Short spot"),
                            "perp": Signal.buy(size=0.5, reason="Long perp"),
                        }
                
                # Check for mean reversion (close positions)
                if positions and abs(basis_bps) < 10:
                    return {
                        "spot": Signal.close(reason="Basis reverted"),
                        "perp": Signal.close(reason="Basis reverted"),
                    }
                
                return {}  # Hold
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
    
    @property
    def name(self) -> str:
        """Strategy name for display."""
        return self.config.name or self.__class__.__name__
    
    @abstractmethod
    def required_data(self) -> dict[str, DataSpec]:
        """
        Return data specifications for each leg.
        
        Returns:
            Dict mapping leg name to DataSpec
            
        Example:
            return {
                "cme": DataSpec("tradingview", "futures", "GC1!", "15m"),
                "defi": DataSpec("hyperliquid", "perp", "PAXG", "15m"),
            }
        """
        pass
    
    @abstractmethod
    def required_indicators(self) -> dict[str, list[tuple[str, dict]]]:
        """
        Return indicators to pre-compute for each leg.
        
        Returns:
            Dict mapping leg name to list of (indicator_name, params)
        """
        pass
    
    @abstractmethod
    def on_bar(
        self,
        idx: int,
        data: dict[str, pd.DataFrame],
        capital: float,
        positions: dict[str, Optional[Position]],
    ) -> dict[str, Signal]:
        """
        Called for each bar during backtest.
        
        Args:
            idx: Current bar index.
            data: Dict mapping leg name to DataFrame with OHLCV + indicators.
            capital: Current available capital.
            positions: Dict mapping leg name to current Position (or None).
            
        Returns:
            Dict mapping leg name to Signal. Empty dict = hold all.
        """
        pass
    
    def on_start(self, data: dict[str, pd.DataFrame]) -> None:
        """Called once before backtest starts."""
        pass
    
    def on_end(self, data: dict[str, pd.DataFrame]) -> None:
        """Called once after backtest ends."""
        pass
    
    def calculate_basis(
        self,
        data: dict[str, pd.DataFrame],
        idx: int,
        leg1: str,
        leg2: str,
    ) -> float:
        """
        Helper to calculate basis between two legs in bps.
        
        Returns:
            (leg2_price - leg1_price) / leg1_price * 10000
        """
        price1 = data[leg1].iloc[idx]["close"]
        price2 = data[leg2].iloc[idx]["close"]
        return (price2 - price1) / price1 * 10000
    
    def get_entry_basis(self) -> float:
        """
        Return the entry basis in bps for spread P&L calculation.
        Override in subclass to track entry basis.
        """
        return 0.0
    
    def calculate_spread_pnl(
        self,
        entry_basis_bps: float,
        exit_basis_bps: float,
        notional: float,
    ) -> float:
        """
        Calculate P&L for a spread trade based on basis convergence.
        
        Args:
            entry_basis_bps: Basis at entry (in bps)
            exit_basis_bps: Basis at exit (in bps)
            notional: Position notional value
            
        Returns:
            Gross P&L from basis convergence
        """
        # P&L = basis captured × notional / 10000
        # We profit when |basis| decreases (spread converges)
        captured_bps = abs(entry_basis_bps) - abs(exit_basis_bps)
        return captured_bps * notional / 10000
