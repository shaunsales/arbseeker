"""
BasisStrategy - Strategy class for pre-computed basis files.

Unlike MultiLeggedStrategy which loads raw OHLCV from multiple venues,
BasisStrategy works with pre-computed basis files that already have:
- base_price
- {quote}_price
- {quote}_basis_bps
- data_quality

This simplifies strategy logic since alignment and basis calculation are done.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from core.strategy.position import Signal, Position, CostModel
from core.data.basis import load_basis


@dataclass
class BasisStrategyConfig:
    """Configuration for basis strategies."""
    name: str = "unnamed"
    capital: float = 100_000
    position_size: float = 1.0  # Position size as fraction of capital per leg
    
    # Basis file parameters
    base_ticker: str = ""
    interval: str = "1h"
    quote_venue: str = ""  # Which quote venue to trade
    
    # Cost model
    costs: CostModel = field(default_factory=CostModel)
    
    # Position limits
    max_trades_per_day: int = 0  # 0 = unlimited
    max_position_hold_bars: int = 0  # 0 = unlimited


@dataclass
class BasisPosition:
    """Tracks a basis position (long or short the spread)."""
    direction: int  # 1 = long spread (long quote, short base), -1 = short spread
    entry_bar: int
    entry_basis_bps: float
    entry_base_price: float
    entry_quote_price: float
    size: float  # Notional per leg
    reason: str = ""
    
    def unrealized_pnl(self, current_basis_bps: float) -> float:
        """Calculate unrealized P&L based on basis change."""
        if self.direction == 1:
            # Long spread: profit when basis increases (quote outperforms)
            captured_bps = current_basis_bps - self.entry_basis_bps
        else:
            # Short spread: profit when basis decreases (base outperforms)
            captured_bps = self.entry_basis_bps - current_basis_bps
        return captured_bps * self.size / 10000
    
    def bars_held(self, current_bar: int) -> int:
        return current_bar - self.entry_bar


@dataclass
class BasisSignal:
    """Signal for basis strategy."""
    action: str  # "open_long", "open_short", "close", "hold"
    size: float = 1.0  # Position size as fraction of capital
    reason: str = ""
    
    @classmethod
    def open_long(cls, size: float = 1.0, reason: str = "") -> "BasisSignal":
        """Open long spread position (long quote, short base)."""
        return cls(action="open_long", size=size, reason=reason)
    
    @classmethod
    def open_short(cls, size: float = 1.0, reason: str = "") -> "BasisSignal":
        """Open short spread position (short quote, long base)."""
        return cls(action="open_short", size=size, reason=reason)
    
    @classmethod
    def close(cls, reason: str = "") -> "BasisSignal":
        """Close current position."""
        return cls(action="close", reason=reason)
    
    @classmethod
    def hold(cls) -> "BasisSignal":
        """Hold current position (or stay flat)."""
        return cls(action="hold")


class BasisStrategy(ABC):
    """
    Base class for strategies trading pre-computed basis files.
    
    Simpler than MultiLeggedStrategy because:
    - Data is already aligned and basis is pre-computed
    - Single DataFrame with base_price, quote_price, basis_bps
    - Position tracking is spread-based (long/short the basis)
    
    Example:
        class SimpleBasisArb(BasisStrategy):
            def __init__(self, threshold_bps=50, take_profit_bps=20):
                config = BasisStrategyConfig(
                    name="SimpleBasisArb",
                    base_ticker="BTCUSDT",
                    quote_venue="hyperliquid",
                )
                super().__init__(config)
                self.threshold_bps = threshold_bps
                self.take_profit_bps = take_profit_bps
            
            def required_indicators(self):
                return []  # No additional indicators needed
            
            def on_bar(self, idx, data, capital, position):
                basis_bps = data["hyperliquid_basis_bps"].iloc[idx]
                
                # Entry: basis exceeds threshold
                if position is None and abs(basis_bps) > self.threshold_bps:
                    if basis_bps > 0:
                        return BasisSignal.open_short(reason=f"Basis {basis_bps:.0f} bps")
                    else:
                        return BasisSignal.open_long(reason=f"Basis {basis_bps:.0f} bps")
                
                # Exit: basis reverted
                if position is not None:
                    captured = position.unrealized_pnl(basis_bps) / position.size * 10000
                    if captured >= self.take_profit_bps:
                        return BasisSignal.close(reason=f"Take profit {captured:.0f} bps")
                
                return BasisSignal.hold()
    """
    
    def __init__(self, config: Optional[BasisStrategyConfig] = None):
        self.config = config or BasisStrategyConfig()
        self._data: Optional[pd.DataFrame] = None
        self._quote_venue: str = ""
    
    @property
    def name(self) -> str:
        return self.config.name or self.__class__.__name__
    
    def load_data(self, periods: Optional[list[str]] = None) -> pd.DataFrame:
        """Load basis data for this strategy."""
        df = load_basis(
            self.config.base_ticker,
            self.config.interval,
            periods=periods,
        )
        if df is None:
            raise ValueError(f"No basis data found for {self.config.base_ticker}/{self.config.interval}")
        
        self._data = df
        
        # Auto-detect quote venue from columns if not specified
        if not self.config.quote_venue:
            for col in df.columns:
                if col.endswith("_basis_bps"):
                    self._quote_venue = col.replace("_basis_bps", "")
                    break
        else:
            self._quote_venue = self.config.quote_venue
        
        return df
    
    @property
    def quote_venue(self) -> str:
        return self._quote_venue or self.config.quote_venue
    
    def get_basis_bps(self, idx: int) -> float:
        """Get basis in bps at given bar index."""
        if self._data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self._data[f"{self.quote_venue}_basis_bps"].iloc[idx]
    
    def get_base_price(self, idx: int) -> float:
        """Get base price at given bar index."""
        if self._data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self._data["base_price"].iloc[idx]
    
    def get_quote_price(self, idx: int) -> float:
        """Get quote price at given bar index."""
        if self._data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        return self._data[f"{self.quote_venue}_price"].iloc[idx]
    
    @abstractmethod
    def required_indicators(self) -> list[tuple[str, dict]]:
        """
        Return list of indicators to add to the basis data.
        
        Indicators are computed on the basis_bps column by default.
        
        Returns:
            List of (indicator_name, params) tuples
        """
        pass
    
    @abstractmethod
    def on_bar(
        self,
        idx: int,
        data: pd.DataFrame,
        capital: float,
        position: Optional[BasisPosition],
    ) -> BasisSignal:
        """
        Called for each bar during backtest.
        
        Args:
            idx: Current bar index
            data: Basis DataFrame with prices, basis_bps, and indicators
            capital: Current available capital
            position: Current BasisPosition or None if flat
            
        Returns:
            BasisSignal indicating action to take
        """
        pass
    
    def on_start(self, data: pd.DataFrame) -> None:
        """Called once before backtest starts."""
        pass
    
    def on_end(self, data: pd.DataFrame) -> None:
        """Called once after backtest ends."""
        pass


class SimpleBasisMeanReversion(BasisStrategy):
    """
    Simple mean-reversion strategy on basis.
    
    Entry: When |basis| exceeds threshold
    Exit: When basis reverts by take_profit amount or max hold time reached
    """
    
    def __init__(
        self,
        base_ticker: str,
        interval: str = "1h",
        quote_venue: str = "",
        threshold_bps: float = 50.0,
        take_profit_bps: float = 25.0,
        stop_loss_bps: float = 100.0,
        max_hold_bars: int = 96,  # 4 days at 1h
        capital: float = 100_000,
        position_size: float = 0.5,
    ):
        config = BasisStrategyConfig(
            name="SimpleBasisMeanReversion",
            base_ticker=base_ticker,
            interval=interval,
            quote_venue=quote_venue,
            capital=capital,
            position_size=position_size,
            max_position_hold_bars=max_hold_bars,
        )
        super().__init__(config)
        self.threshold_bps = threshold_bps
        self.take_profit_bps = take_profit_bps
        self.stop_loss_bps = stop_loss_bps
    
    def required_indicators(self) -> list[tuple[str, dict]]:
        return []  # No additional indicators
    
    def on_bar(
        self,
        idx: int,
        data: pd.DataFrame,
        capital: float,
        position: Optional[BasisPosition],
    ) -> BasisSignal:
        basis_bps = self.get_basis_bps(idx)
        
        # Exit logic
        if position is not None:
            bars_held = position.bars_held(idx)
            pnl_bps = position.unrealized_pnl(basis_bps) / position.size * 10000
            
            # Take profit
            if pnl_bps >= self.take_profit_bps:
                return BasisSignal.close(reason=f"TP {pnl_bps:.0f} bps after {bars_held} bars")
            
            # Stop loss
            if pnl_bps <= -self.stop_loss_bps:
                return BasisSignal.close(reason=f"SL {pnl_bps:.0f} bps after {bars_held} bars")
            
            # Max hold time
            if self.config.max_position_hold_bars > 0 and bars_held >= self.config.max_position_hold_bars:
                return BasisSignal.close(reason=f"Max hold {bars_held} bars, PnL {pnl_bps:.0f} bps")
            
            return BasisSignal.hold()
        
        # Entry logic
        if abs(basis_bps) > self.threshold_bps:
            if basis_bps > 0:
                # Basis positive = quote expensive, short the spread
                return BasisSignal.open_short(
                    size=self.config.position_size,
                    reason=f"Short spread at {basis_bps:.0f} bps"
                )
            else:
                # Basis negative = quote cheap, long the spread
                return BasisSignal.open_long(
                    size=self.config.position_size,
                    reason=f"Long spread at {basis_bps:.0f} bps"
                )
        
        return BasisSignal.hold()
