"""
Position and Trade tracking for backtesting.

Provides dataclasses for tracking:
- Open positions (with mark-to-market P&L)
- Closed trades (with realized P&L)
- Multi-leg positions (for strategies like basis arb)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
from pathlib import Path
import json
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
    
    # Stop loss / trailing stop loss
    stop_loss_pct: Optional[float] = None      # Fixed SL as % from entry (e.g. 2.0 = 2%)
    trailing_stop_pct: Optional[float] = None   # TSL as % from best price
    best_price: float = 0.0                     # Tracks best price since entry (for TSL)
    
    # Metadata
    entry_reason: str = ""
    metadata: dict = field(default_factory=dict)
    
    def update_price(self, price: float) -> float:
        """Update current price, best price, and recalculate unrealized P&L."""
        self.current_price = price
        
        # Track best price for trailing stop
        if self.side == Side.LONG:
            self.best_price = max(self.best_price, price)
            self.unrealized_pnl = self.size * (price - self.entry_price) / self.entry_price
        else:  # SHORT
            if self.best_price == 0:
                self.best_price = price
            self.best_price = min(self.best_price, price)
            self.unrealized_pnl = self.size * (self.entry_price - price) / self.entry_price
        
        return self.unrealized_pnl
    
    @property
    def stop_loss_price(self) -> Optional[float]:
        """Compute the fixed stop loss price level."""
        if self.stop_loss_pct is None:
            return None
        if self.side == Side.LONG:
            return self.entry_price * (1 - self.stop_loss_pct / 100)
        else:
            return self.entry_price * (1 + self.stop_loss_pct / 100)
    
    @property
    def trailing_stop_price(self) -> Optional[float]:
        """Compute the trailing stop price from best price."""
        if self.trailing_stop_pct is None or self.best_price == 0:
            return None
        if self.side == Side.LONG:
            return self.best_price * (1 - self.trailing_stop_pct / 100)
        else:
            return self.best_price * (1 + self.trailing_stop_pct / 100)
    
    def check_stop_loss(self, price: float) -> Optional[str]:
        """
        Check if price triggers stop loss or trailing stop.
        
        Returns exit reason string if triggered, None otherwise.
        Checks fixed SL first, then TSL.
        """
        # Fixed stop loss
        sl = self.stop_loss_price
        if sl is not None:
            if self.side == Side.LONG and price <= sl:
                return f"stop_loss_hit@{price:.2f}<=sl@{sl:.2f}"
            elif self.side == Side.SHORT and price >= sl:
                return f"stop_loss_hit@{price:.2f}>=sl@{sl:.2f}"
        
        # Trailing stop loss
        tsl = self.trailing_stop_price
        if tsl is not None:
            if self.side == Side.LONG and price <= tsl:
                return f"trailing_stop_hit@{price:.2f}<=tsl@{tsl:.2f}(best={self.best_price:.2f})"
            elif self.side == Side.SHORT and price >= tsl:
                return f"trailing_stop_hit@{price:.2f}>=tsl@{tsl:.2f}(best={self.best_price:.2f})"
        
        return None
    
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
    
    # Risk management (applied by engine when position is opened)
    stop_loss_pct: Optional[float] = None       # Fixed SL % from entry
    trailing_stop_pct: Optional[float] = None    # TSL % from best price
    
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


class FundingSchedule:
    """
    Direction-aware funding rate schedule loaded from historical data.
    
    Positive rate = longs pay, shorts receive.
    Negative rate = shorts pay, longs receive.
    
    Uses monthly median rates from static JSON data.
    """
    
    _DATA_FILE = Path(__file__).parent.parent / "data" / "funding_rates_btcusdt.json"
    _instance: Optional["FundingSchedule"] = None
    
    def __init__(self, data_path: Optional[Path] = None):
        path = data_path or self._DATA_FILE
        if not path.exists():
            self._months: dict[str, dict] = {}
            return
        with open(path) as f:
            raw = json.load(f)
        self._months = raw.get("months", {})
    
    @classmethod
    def default(cls) -> "FundingSchedule":
        """Singleton access to the default BTCUSDT funding schedule."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def daily_rate_bps(self, month: str) -> Optional[float]:
        """
        Get the median daily funding rate (bps) for a given month.
        
        Args:
            month: "YYYY-MM" string
            
        Returns:
            Median daily rate in bps (positive = longs pay), or None if no data.
        """
        entry = self._months.get(month)
        if entry is None:
            return None
        return entry["median_daily_bps"]
    
    def funding_cost(
        self,
        side: "Side",
        size: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> float:
        """
        Compute direction-aware funding cost for a position.
        
        Prorates across months when a trade spans multiple months.
        
        Returns:
            Cost in USD. Positive = you paid, negative = you received.
        """
        if entry_time >= exit_time:
            return 0.0
        
        total_cost = 0.0
        current = entry_time
        
        while current < exit_time:
            month_key = current.strftime("%Y-%m")
            
            # End of this month or exit_time, whichever is first
            if current.month == 12:
                month_end = current.replace(year=current.year + 1, month=1, day=1,
                                            hour=0, minute=0, second=0, microsecond=0)
            else:
                month_end = current.replace(month=current.month + 1, day=1,
                                            hour=0, minute=0, second=0, microsecond=0)
            
            period_end = min(month_end, exit_time)
            days_in_period = (period_end - current).total_seconds() / 86400
            
            rate_bps = self.daily_rate_bps(month_key)
            if rate_bps is not None and days_in_period > 0:
                # Positive rate = longs pay, shorts receive
                if side == Side.LONG:
                    total_cost += size * (rate_bps / 10000) * days_in_period
                else:  # SHORT
                    total_cost -= size * (rate_bps / 10000) * days_in_period
            
            current = period_end
        
        return total_cost
    
    @property
    def available(self) -> bool:
        """Whether historical funding data is loaded."""
        return len(self._months) > 0


@dataclass
class CostModel:
    """
    Trading cost model for backtesting.
    
    All costs in basis points (bps) unless otherwise noted.
    Uses historical funding rates when available (direction-aware),
    falls back to flat funding_daily_bps otherwise.
    """
    # Per-trade costs (applied on entry and exit)
    commission_bps: float = 0.0  # Commission/fee
    slippage_bps: float = 0.0    # Slippage estimate
    
    # Flat funding fallback (used when no historical data available)
    funding_daily_bps: float = 0.0  # Daily funding rate
    
    # For flat funding calculation
    bars_per_day: int = 24  # Default: hourly bars
    
    # Historical funding schedule (loaded on first use)
    _funding_schedule: Optional[FundingSchedule] = field(default=None, repr=False)
    
    @property
    def funding_schedule(self) -> FundingSchedule:
        if self._funding_schedule is None:
            self._funding_schedule = FundingSchedule.default()
        return self._funding_schedule
    
    def round_trip_cost(self, size: float) -> float:
        """Calculate total round-trip cost for a position."""
        cost_bps = (self.commission_bps + self.slippage_bps) * 2
        return size * (cost_bps / 10000)
    
    def holding_cost_flat(self, size: float, bars_held: int) -> float:
        """Calculate holding cost using flat daily rate (fallback)."""
        days_held = bars_held / self.bars_per_day
        return size * (self.funding_daily_bps / 10000) * days_held
    
    def holding_cost(
        self,
        size: float,
        bars_held: int,
        side: Optional["Side"] = None,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate holding cost.
        
        Uses historical funding rates (direction-aware) when side and
        timestamps are provided and data is available.
        Falls back to flat funding_daily_bps otherwise.
        """
        schedule = self.funding_schedule
        if side is not None and entry_time is not None and exit_time is not None and schedule.available:
            return schedule.funding_cost(side, size, entry_time, exit_time)
        return self.holding_cost_flat(size, bars_held)
    
    def total_cost(
        self,
        size: float,
        bars_held: int,
        side: Optional["Side"] = None,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
    ) -> float:
        """Calculate total cost (round-trip + holding)."""
        return self.round_trip_cost(size) + self.holding_cost(
            size, bars_held, side, entry_time, exit_time
        )


# Default cost models for common scenarios
DEFAULT_COSTS = CostModel(
    commission_bps=3.5,   # Typical crypto taker fee
    slippage_bps=2.0,     # Conservative slippage
    funding_daily_bps=3.0,  # Fallback: Binance BTCUSDT median (0.01% per 8h)
    bars_per_day=24,
)

ZERO_COSTS = CostModel()  # For testing without costs
