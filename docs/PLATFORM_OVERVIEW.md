# Backtesting Platform Overview

A modular Python framework for strategy backtesting with support for single-asset and multi-legged strategies.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Downloaders              Storage                   Loader              │
│  ┌─────────────┐         ┌─────────────┐          ┌─────────────┐      │
│  │ yahoo.py    │ ──────► │ Parquet     │ ◄─────── │ load_ohlcv()│      │
│  │ hyperliquid │         │ Files       │          │             │      │
│  │ .py         │         │ (yearly/    │          │ Handles:    │      │
│  └─────────────┘         │  monthly)   │          │ - Multi-file│      │
│                          └─────────────┘          │ - Concat    │      │
│                                                   │ - Dedup     │      │
│                                                   └─────────────┘      │
├─────────────────────────────────────────────────────────────────────────┤
│                          STRATEGY LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────┐    ┌───────────────────────┐                │
│  │ SingleAssetStrategy   │    │ MultiLeggedStrategy   │                │
│  │ ───────────────────── │    │ ───────────────────── │                │
│  │ • One asset           │    │ • Multiple assets     │                │
│  │ • Long/Short          │    │ • Simultaneous pos    │                │
│  │ • required_indicators │    │ • required_data()     │                │
│  │ • on_bar() → Signal   │    │ • on_bar() → Signals  │                │
│  └───────────────────────┘    └───────────────────────┘                │
├─────────────────────────────────────────────────────────────────────────┤
│                          ENGINE LAYER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      BacktestEngine                              │   │
│  │  ─────────────────────────────────────────────────────────────── │   │
│  │  1. Load data (per strategy.required_data())                     │   │
│  │  2. Compute indicators (per strategy.required_indicators())      │   │
│  │  3. Align timestamps (for multi-leg)                             │   │
│  │  4. Bar-by-bar iteration:                                        │   │
│  │     • Call strategy.on_bar() → Signal(s)                         │   │
│  │     • Execute trades (Position → Trade)                          │   │
│  │     • Apply costs (CostModel)                                    │   │
│  │     • Track equity curve                                         │   │
│  │  5. Compute metrics → BacktestResult                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                          OUTPUT                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  BacktestResult                                                         │
│  ├── trades: list[Trade]      # All executed trades                    │
│  ├── equity_curve: Series     # Portfolio value over time              │
│  ├── metrics: Sharpe, DD, etc # Computed performance metrics           │
│  └── config: dict             # Strategy + cost configuration          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Layer

### 1.1 Storage Structure

All data is stored as Parquet files in a hierarchical directory structure:

```
data/{venue}/{market}/{ticker}/{interval}/{period}.parquet
```

| Component | Description | Examples |
|-----------|-------------|----------|
| `venue` | Data source | `yahoo`, `hyperliquid`, `binance` |
| `market` | Market type | `futures`, `spot`, `perp` |
| `ticker` | Symbol | `GC=F`, `BTCUSDT`, `PAXG` |
| `interval` | Bar size | `1m`, `15m`, `1h`, `1d` |
| `period` | Time range | `2024` (year), `2024-10` (month) |

**Example paths:**
```
data/yahoo/futures/GC=F/15m/2025-12.parquet
data/hyperliquid/perp/PAXG/15m/2025-12.parquet
```

### 1.2 DataFrame Schema

All OHLCV data uses a consistent schema:

| Column | Type | Description |
|--------|------|-------------|
| `index` | `DatetimeIndex` (UTC) | Bar timestamp |
| `open` | `float64` | Open price |
| `high` | `float64` | High price |
| `low` | `float64` | Low price |
| `close` | `float64` | Close price |
| `volume` | `float64` | Volume |

Optional columns (added by downloaders):
- `market_open`: `bool` - Whether market is open
- `near_close`: `bool` - Whether market is closing soon (for auto-close logic)

### 1.3 Core Functions

**`core/data/storage.py`**

```python
# Save data
save_ohlcv(df, venue, market, ticker, interval, period)
save_yearly(df, venue, market, ticker, interval, year)
save_monthly(df, venue, market, ticker, interval, year, month)

# Load data (concatenates multiple files automatically)
df = load_ohlcv(venue, market, ticker, interval, periods=None)
# periods=None loads all available; or specify: ["2024", "2025-01", "2025-02"]

# Discovery
periods = list_available_periods(venue, market, ticker, interval)
# Returns: ["2024", "2025-01", "2025-02", ...]

all_data = list_all_data()
# Returns: {venue: {market: {ticker: {interval: [periods]}}}}
```

### 1.4 Data Downloaders

| Module | Source | Assets |
|--------|--------|--------|
| `core/data/yahoo.py` | Yahoo Finance | Futures (GC=F, CL=F), Stocks |
| `core/data/hyperliquid.py` | Hyperliquid API | Perpetuals (PAXG, BTC, ETH) |

Each downloader follows the pattern:
```python
df = download_{source}_month(ticker, year, month, interval)
save_monthly(df, venue, market, ticker, interval, year, month)
```

---

## 2. Strategy Layer

### 2.1 Strategy Types

#### SingleAssetStrategy

For strategies that trade one asset (long/short):

```python
class SingleAssetStrategy(ABC):
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
    
    @abstractmethod
    def required_indicators(self) -> list[tuple[str, dict]]:
        """Return indicators to pre-compute."""
        # Example: [("sma", {"length": 20}), ("rsi", {"length": 14})]
        pass
    
    @abstractmethod
    def on_bar(
        self,
        idx: int,                    # Current bar index
        data: pd.DataFrame,          # Full OHLCV + indicators
        capital: float,              # Current capital
        position: Optional[Position], # Current position (or None)
    ) -> Signal:
        """Called each bar. Return Signal (buy/sell/close/hold)."""
        pass
```

#### MultiLeggedStrategy

For strategies with multiple simultaneous positions (arbitrage, pairs, spreads):

```python
class MultiLeggedStrategy(ABC):
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
    
    @abstractmethod
    def required_data(self) -> dict[str, DataSpec]:
        """Return data specs for each leg."""
        # Example: {"tradfi": DataSpec("yahoo", "futures", "GC=F", "15m"),
        #           "defi": DataSpec("hyperliquid", "perp", "PAXG", "15m")}
        pass
    
    @abstractmethod
    def required_indicators(self) -> dict[str, list[tuple[str, dict]]]:
        """Return indicators for each leg."""
        pass
    
    @abstractmethod
    def on_bar(
        self,
        idx: int,
        data: dict[str, pd.DataFrame],      # {leg_name: DataFrame}
        capital: float,
        positions: dict[str, Optional[Position]],  # {leg_name: Position}
    ) -> dict[str, Signal]:
        """Called each bar. Return {leg_name: Signal}."""
        pass
```

### 2.2 DataSpec

Specifies data requirements for a strategy leg:

```python
@dataclass
class DataSpec:
    venue: str      # "yahoo", "hyperliquid", etc.
    market: str     # "futures", "perp", "spot"
    ticker: str     # "GC=F", "PAXG"
    interval: str   # "15m", "1h"
```

### 2.3 StrategyConfig

Configuration shared by all strategies:

```python
@dataclass
class StrategyConfig:
    name: str = "unnamed"
    capital: float = 100_000
    max_position_pct: float = 1.0     # Max position as % of capital
    costs: CostModel = CostModel()
    
    # Position sizing mode
    fixed_size: bool = False           # If True, use fixed $ amount
    fixed_size_amount: float = 0.0     # Fixed $ per trade
    
    # Spread trading mode
    spread_pnl_mode: bool = False      # Calculate P&L from spread convergence
```

### 2.4 Signal

Strategies return `Signal` objects to indicate actions:

```python
@dataclass
class Signal:
    action: str   # "buy", "sell", "close", "hold"
    size: float   # Position size as fraction (0.0-1.0)
    reason: str   # For logging/debugging

# Factory methods
Signal.buy(size=1.0, reason="entry_signal")
Signal.sell(size=1.0, reason="short_entry")
Signal.close(reason="take_profit")
Signal.hold()
```

---

## 3. Engine Layer

### 3.1 BacktestEngine

The engine orchestrates the entire backtest:

```python
engine = BacktestEngine(verbose=True)

result = engine.run(
    strategy=MyStrategy(),
    capital=100_000,
    costs=CostModel(...),
    # Optional: load data automatically
    venue="yahoo", market="futures", ticker="GC=F", interval="15m",
)
```

### 3.2 Execution Flow

#### Single-Asset Strategy

```
1. Load Data
   └── load_ohlcv(venue, market, ticker, interval)

2. Compute Indicators
   └── compute_indicators(df, strategy.required_indicators())

3. Initialize State
   ├── capital = initial_capital
   ├── position = None
   └── equity_curve = []

4. Bar-by-Bar Loop (for idx in range(len(data))):
   │
   ├── Update position mark-to-market
   │   └── position.update_price(close)
   │
   ├── Get signal from strategy
   │   └── signal = strategy.on_bar(idx, data, capital, position)
   │
   ├── Execute signal
   │   ├── BUY  → Create Position(LONG), deduct entry cost
   │   ├── SELL → Create Position(SHORT), deduct entry cost
   │   └── CLOSE → Close position, create Trade, apply full costs
   │
   └── Record equity (capital + unrealized P&L)

5. Force-close any open positions at end

6. Compute metrics → BacktestResult
```

#### Multi-Legged Strategy

```
1. Load Data (for each leg)
   └── for leg, spec in strategy.required_data():
         data[leg] = load_ohlcv(spec.venue, spec.market, ...)

2. Align Timestamps
   └── common_index = intersection of all leg indexes

3. Compute Indicators (for each leg)
   └── compute_indicators(data[leg], indicators[leg])

4. Initialize State
   ├── capital = initial_capital
   ├── positions = {leg: None for leg in legs}
   └── equity_curve = []

5. Bar-by-Bar Loop (for idx in range(len(common_index))):
   │
   ├── Check market hours
   │   ├── all_markets_open = all(data[leg]["market_open"])
   │   └── any_near_close = any(data[leg]["near_close"])
   │
   ├── Force close if market closing
   │   └── signals = {leg: Signal.close() for leg with position}
   │
   ├── Skip if any market closed
   │
   ├── Get signals from strategy
   │   └── signals = strategy.on_bar(idx, data, capital, positions)
   │
   ├── Execute signals (for each leg)
   │   ├── BUY/SELL → Create Position for leg
   │   └── CLOSE → Close position, create Trade
   │
   └── Record equity (capital + sum of unrealized P&L)

6. Force-close any open positions at end

7. Compute metrics → BacktestResult
```

---

## 4. Position & Trade Tracking

### 4.1 Position

An open position in a single asset:

```python
@dataclass
class Position:
    id: str                    # Unique ID
    symbol: str                # Asset symbol
    side: Side                 # LONG or SHORT
    leg: str                   # Leg name (for multi-leg)
    
    entry_time: datetime
    entry_price: float
    size: float                # Notional value ($)
    
    current_price: float       # Mark-to-market price
    unrealized_pnl: float      # Current unrealized P&L
    status: PositionStatus     # OPEN or CLOSED
```

**P&L Calculation:**
```python
# LONG position
unrealized_pnl = size * (current_price - entry_price) / entry_price

# SHORT position
unrealized_pnl = size * (entry_price - current_price) / entry_price
```

### 4.2 Trade

A closed position with realized P&L:

```python
@dataclass
class Trade:
    position_id: str
    symbol: str
    side: Side
    leg: str
    
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    
    size: float           # Notional value
    gross_pnl: float      # P&L before costs
    costs: float          # Total costs
    net_pnl: float        # gross_pnl - costs
    
    exit_reason: str      # "take_profit", "stop_loss", etc.
    bars_held: int        # Duration in bars
```

---

## 5. Cost Model

### 5.1 CostModel

Defines trading costs applied to each trade:

```python
@dataclass
class CostModel:
    commission_bps: float = 0.0    # Commission per trade (bps)
    slippage_bps: float = 0.0      # Slippage estimate (bps)
    funding_daily_bps: float = 0.0 # Daily funding rate (bps)
    bars_per_day: int = 24         # For funding calculation
```

### 5.2 Cost Calculations

```python
# Round-trip cost (entry + exit)
round_trip_cost = size * (commission_bps + slippage_bps) * 2 / 10000

# Holding cost (funding)
days_held = bars_held / bars_per_day
holding_cost = size * (funding_daily_bps / 10000) * days_held

# Total cost
total_cost = round_trip_cost + holding_cost
```

**Example:**
```
Position: $100,000 notional, held 96 bars (1 day at 15-min interval)
CostModel: commission=5 bps, slippage=2 bps, funding=5 bps/day

Round-trip: $100,000 × (5 + 2) × 2 / 10000 = $140
Holding:    $100,000 × 5 / 10000 × 1 day = $50
Total:      $190
```

### 5.3 Default Cost Models

```python
DEFAULT_COSTS = CostModel(
    commission_bps=3.5,
    slippage_bps=2.0,
    funding_daily_bps=5.0,
    bars_per_day=24,
)

ZERO_COSTS = CostModel()  # For testing
```

---

## 6. BacktestResult

### 6.1 Structure

```python
@dataclass
class BacktestResult:
    # Configuration
    strategy_name: str
    config: dict
    
    # Time range
    start_time: datetime
    end_time: datetime
    total_bars: int
    
    # Capital
    initial_capital: float
    final_capital: float
    
    # Trades
    trades: list[Trade]
    
    # Time series
    equity_curve: pd.Series    # Portfolio value indexed by timestamp
    drawdown_curve: pd.Series  # Drawdown % indexed by timestamp
    
    # Metrics
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
```

### 6.2 Metrics Calculation

```python
# Total Return
total_return_pct = (final_capital / initial_capital - 1) * 100

# Annualized Return (CAGR)
days = total_bars / bars_per_day
annual_return_pct = ((1 + total_return_pct/100) ** (365/days) - 1) * 100

# Sharpe Ratio (annualized)
returns = equity_curve.pct_change()
sharpe_ratio = (returns.mean() / returns.std()) * sqrt(bars_per_day * 365)

# Max Drawdown
rolling_max = equity_curve.cummax()
drawdown = (equity_curve - rolling_max) / rolling_max
max_drawdown_pct = abs(drawdown.min()) * 100

# Win Rate
win_rate = winning_trades / total_trades * 100

# Profit Factor
profit_factor = sum(winning_pnl) / abs(sum(losing_pnl))
```

---

## 7. Quick Reference

### Minimal Single-Asset Strategy

```python
from core.strategy import SingleAssetStrategy, StrategyConfig, BacktestEngine
from core.strategy.position import Signal

class MACrossover(SingleAssetStrategy):
    def __init__(self, fast=10, slow=20):
        super().__init__(StrategyConfig(name="MA Crossover"))
        self.fast, self.slow = fast, slow
    
    def required_indicators(self):
        return [("sma", {"length": self.fast}), ("sma", {"length": self.slow})]
    
    def on_bar(self, idx, data, capital, position):
        if idx < self.slow:
            return Signal.hold()
        
        fast = data[f"SMA_{self.fast}"].iloc[idx]
        slow = data[f"SMA_{self.slow}"].iloc[idx]
        
        if fast > slow and position is None:
            return Signal.buy(reason="MA cross up")
        elif fast < slow and position is not None:
            return Signal.close(reason="MA cross down")
        return Signal.hold()

# Run
result = BacktestEngine().run(
    strategy=MACrossover(),
    venue="yahoo", market="futures", ticker="GC=F", interval="1h",
    capital=100_000,
)
result.print_report()
```

### Minimal Multi-Legged Strategy

```python
from core.strategy import MultiLeggedStrategy, StrategyConfig, DataSpec, BacktestEngine
from core.strategy.position import Signal

class SpreadStrategy(MultiLeggedStrategy):
    def __init__(self, threshold_bps=50):
        super().__init__(StrategyConfig(name="Spread"))
        self.threshold = threshold_bps
    
    def required_data(self):
        return {
            "leg1": DataSpec("venue1", "market", "TICKER1", "15m"),
            "leg2": DataSpec("venue2", "market", "TICKER2", "15m"),
        }
    
    def required_indicators(self):
        return {"leg1": [], "leg2": []}
    
    def on_bar(self, idx, data, capital, positions):
        p1 = data["leg1"].iloc[idx]["close"]
        p2 = data["leg2"].iloc[idx]["close"]
        spread_bps = (p2 - p1) / p1 * 10000
        
        if abs(spread_bps) > self.threshold and not any(positions.values()):
            return {
                "leg1": Signal.buy(size=0.5, reason="Long leg1"),
                "leg2": Signal.sell(size=0.5, reason="Short leg2"),
            }
        elif any(positions.values()) and abs(spread_bps) < 10:
            return {
                "leg1": Signal.close(reason="Spread converged"),
                "leg2": Signal.close(reason="Spread converged"),
            }
        return {}

# Run
result = BacktestEngine().run(strategy=SpreadStrategy(), capital=100_000)
result.print_report()
```

---

## 8. File Reference

| Module | Purpose |
|--------|---------|
| `core/data/storage.py` | Parquet I/O, `load_ohlcv()`, `save_ohlcv()` |
| `core/data/yahoo.py` | Yahoo Finance downloader |
| `core/data/hyperliquid.py` | Hyperliquid API downloader |
| `core/data/market_hours.py` | CME market hours, near-close detection |
| `core/indicators/indicators.py` | pandas-ta wrapper, `compute_indicators()` |
| `core/strategy/base.py` | `SingleAssetStrategy`, `MultiLeggedStrategy`, `DataSpec`, `StrategyConfig` |
| `core/strategy/position.py` | `Position`, `Trade`, `Signal`, `CostModel` |
| `core/strategy/engine.py` | `BacktestEngine`, `BacktestResult` |
