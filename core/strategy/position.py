"""
Position and Trade tracking for backtesting.

Provides dataclasses for tracking:
- Open positions (with mark-to-market P&L)
- Closed trades (with realized P&L)
- Multi-leg positions (for strategies like basis arb)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import uuid


class Side(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Position:
    """
    An open position in a single asset.
    
    Tracks entry details and current mark-to-market P&L.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Entry details
    symbol: str = ""
    side: Side = Side.LONG
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0
    size: float = 0.0  # Notional value in quote currency
    
    # Optional leg identifier (for multi-leg strategies)
    leg: str = ""
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    
    # Metadata
    entry_reason: str = ""
    metadata: dict = field(default_factory=dict)
    
    def update_price(self, price: float) -> float:
        """Update current price and recalculate unrealized P&L."""
        self.current_price = price
        
        if self.side == Side.LONG:
            self.unrealized_pnl = self.size * (price - self.entry_price) / self.entry_price
        else:  # SHORT
            self.unrealized_pnl = self.size * (self.entry_price - price) / self.entry_price
        
        return self.unrealized_pnl
    
    def close(self, exit_price: float, exit_time: datetime) -> "Trade":
        """
        Close the position and return a Trade record.
        
        Returns:
            Trade object with realized P&L
        """
        self.status = PositionStatus.CLOSED
        self.current_price = exit_price
        
        # Calculate realized P&L
        if self.side == Side.LONG:
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        realized_pnl = self.size * pnl_pct
        
        return Trade(
            position_id=self.id,
            symbol=self.symbol,
            side=self.side,
            leg=self.leg,
            entry_time=self.entry_time,
            entry_price=self.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            size=self.size,
            gross_pnl=realized_pnl,
            entry_reason=self.entry_reason,
            metadata=self.metadata.copy(),
        )


@dataclass
class Trade:
    """
    A completed (closed) trade.
    
    Records entry/exit details and realized P&L.
    """
    position_id: str = ""
    
    # Trade details
    symbol: str = ""
    side: Side = Side.LONG
    leg: str = ""
    
    # Entry
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0
    
    # Exit
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    
    # Size and P&L
    size: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    
    # Metadata
    exit_reason: str = ""
    entry_reason: str = ""
    bars_held: int = 0
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate net P&L after initialization."""
        if self.net_pnl == 0.0 and self.gross_pnl != 0.0:
            self.net_pnl = self.gross_pnl - self.costs
    
    @property
    def pnl_pct(self) -> float:
        """Return P&L as percentage of position size."""
        if self.size == 0:
            return 0.0
        return (self.net_pnl / self.size) * 100
    
    @property
    def duration(self) -> Optional[float]:
        """Return trade duration in hours."""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        return None


@dataclass
class Signal:
    """
    A trading signal from a strategy.
    
    Tells the engine what action to take.
    """
    action: str  # "buy", "sell", "close", "hold"
    size: float = 1.0  # Position size as fraction of capital (0.0 - 1.0)
    
    # Optional details
    symbol: str = ""
    leg: str = ""  # For multi-leg strategies
    price: Optional[float] = None  # Limit price (None = market)
    reason: str = ""
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def buy(cls, size: float = 1.0, reason: str = "", **kwargs) -> "Signal":
        """Create a buy signal."""
        return cls(action="buy", size=size, reason=reason, **kwargs)
    
    @classmethod
    def sell(cls, size: float = 1.0, reason: str = "", **kwargs) -> "Signal":
        """Create a sell signal."""
        return cls(action="sell", size=size, reason=reason, **kwargs)
    
    @classmethod
    def close(cls, reason: str = "", **kwargs) -> "Signal":
        """Create a close signal (exit all positions)."""
        return cls(action="close", size=0.0, reason=reason, **kwargs)
    
    @classmethod
    def hold(cls) -> "Signal":
        """Create a hold signal (do nothing)."""
        return cls(action="hold", size=0.0)


@dataclass
class CostModel:
    """
    Trading cost model for backtesting.
    
    All costs in basis points (bps) unless otherwise noted.
    """
    # Per-trade costs (applied on entry and exit)
    commission_bps: float = 0.0  # Commission/fee
    slippage_bps: float = 0.0    # Slippage estimate
    
    # Holding costs (applied per bar while position is open)
    funding_daily_bps: float = 0.0  # Daily funding rate
    
    # Computed
    bars_per_day: int = 24  # For funding calculation (default: hourly bars)
    
    def round_trip_cost(self, size: float) -> float:
        """Calculate total round-trip cost for a position."""
        # Entry + exit costs
        cost_bps = (self.commission_bps + self.slippage_bps) * 2
        return size * (cost_bps / 10000)
    
    def holding_cost(self, size: float, bars_held: int) -> float:
        """Calculate holding cost for a position."""
        days_held = bars_held / self.bars_per_day
        return size * (self.funding_daily_bps / 10000) * days_held
    
    def total_cost(self, size: float, bars_held: int) -> float:
        """Calculate total cost (round-trip + holding)."""
        return self.round_trip_cost(size) + self.holding_cost(size, bars_held)


# Default cost models for common scenarios
DEFAULT_COSTS = CostModel(
    commission_bps=3.5,   # Typical crypto taker fee
    slippage_bps=2.0,     # Conservative slippage
    funding_daily_bps=5.0,  # Typical funding rate
    bars_per_day=24,
)

ZERO_COSTS = CostModel()  # For testing without costs
