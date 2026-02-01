# Strategy Framework Refactor Plan

Refactor the codebase into a modular trading strategy framework with Binance Vision data, pre-computed indicators, multi-legged strategy support, **unified web app**, and interactive Plotly charts.

---

## Key Requirements (Confirmed)

| Requirement | Decision |
|-------------|----------|
| **Pandas** | 3.0 GA |
| **Data Source** | Binance Vision (no API keys needed) |
| **Data Format** | Parquet only (no SQLite) |
| **Data Chunking** | Yearly files (2023.parquet, 2024.parquet, ...) |
| **Strategy Scope** | Long/short, multi-asset, multi-legged |
| **Indicators** | Pre-computed before backtest (array access) |
| **Charts** | Plotly with show/hide controls |
| **Interface** | Unified Python web app (data + backtest + export) |

---

## Proposed Architecture

```
basisarb-research/
├── core/                       # Core framework
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── binance.py          # Binance Vision downloader
│   │   ├── validator.py        # Gap detection, LOCF
│   │   └── storage.py          # Parquet I/O (yearly files)
│   ├── indicators/
│   │   ├── __init__.py
│   │   └── indicators.py       # pandas-ta wrapper
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py             # Strategy base classes
│   │   ├── position.py         # Position/Trade/Leg tracking
│   │   └── engine.py           # Backtest engine
│   └── viz/
│       ├── __init__.py
│       └── charts.py           # Plotly charts
│
├── app/                        # NEW: Web application
│   ├── __init__.py
│   ├── main.py                 # FastAPI/Flask app
│   ├── routes/
│   │   ├── data.py             # Data browser endpoints
│   │   ├── backtest.py         # Backtest runner endpoints
│   │   └── export.py           # QuantStats export
│   ├── templates/              # HTML templates (if Flask)
│   └── static/                 # CSS/JS assets
│
├── strategies/                 # Strategy library
│   ├── __init__.py
│   ├── basis_arb.py            # Multi-legged basis arb
│   └── trend_following.py      # Example single-asset
│
├── data/                       # Structured storage (yearly Parquet)
│   └── {venue}/{market}/{ticker}/{interval}/
│       ├── 2023.parquet
│       ├── 2024.parquet
│       └── 2025.parquet
│
├── output/
│   ├── backtests/              # Results JSON + trades CSV
│   └── charts/                 # Interactive HTML
│
├── run_app.py                  # Launch web app
└── requirements.txt
```

---

## Phase 1: Binance Vision Data Infrastructure

### 1.1 Downloader (`core/data/binance.py`)

**Binance Vision URL structure:**
```
https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip
https://data.binance.vision/data/futures/um/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}-{DD}.zip
```

**Features:**
- Download monthly ZIP files (more efficient for 2-3 years)
- Auto-extract and concatenate
- Resume/skip already downloaded months
- Checksum verification

**Klines columns:**
```
open_time, open, high, low, close, volume, close_time, 
quote_volume, trades, taker_buy_base, taker_buy_quote, ignore
```

### 1.2 Validator (`core/data/validator.py`)

- **Gap detection**: Identify missing timestamps
- **LOCF**: Forward-fill gaps
- **Report**: Coverage %, gaps list, date range

### 1.3 Storage (`core/data/storage.py`)

**Yearly file structure:**
```
data/binance/futures/BTCUSDT/1h/
├── 2023.parquet    # Jan 1 - Dec 31, 2023
├── 2024.parquet    # Jan 1 - Dec 31, 2024
└── 2025.parquet    # Jan 1 - Dec 31, 2025
```

```python
# Save (always full year)
save_yearly(df, venue="binance", market="futures", ticker="BTCUSDT", interval="1h", year=2024)
# -> data/binance/futures/BTCUSDT/1h/2024.parquet

# Load (select years)
df = load_ohlcv(venue="binance", market="futures", ticker="BTCUSDT", interval="1h",
                years=[2023, 2024, 2025])  # Concatenates all selected years

# List available years
years = list_available_years(venue="binance", market="futures", ticker="BTCUSDT", interval="1h")
# -> [2023, 2024, 2025]
```

### 1.4 Download Logic

- Download entire year at once (Jan 1 - Dec 31)
- If year file exists → skip (idempotent)
- Validate data after download (gaps, LOCF)
- Current year: download up to yesterday

---

## Phase 2: Technical Indicators

### 2.1 Wrapper (`core/indicators/indicators.py`)

```python
def compute_indicators(df: pd.DataFrame, indicators: list[tuple]) -> pd.DataFrame:
    """
    Pre-compute all indicators before backtest.
    
    Args:
        df: OHLCV DataFrame
        indicators: List of (name, params) tuples
        
    Returns:
        DataFrame with indicator columns added
    """
    import pandas_ta as ta
    
    for name, params in indicators:
        df.ta(kind=name, **params, append=True)
    
    return df
```

**Example usage:**
```python
indicators = [
    ("sma", {"length": 20}),
    ("sma", {"length": 50}),
    ("rsi", {"length": 14}),
    ("macd", {"fast": 12, "slow": 26, "signal": 9}),
    ("adx", {"length": 14}),
    ("bbands", {"length": 20, "std": 2}),
]
df = compute_indicators(df, indicators)
# Columns added: SMA_20, SMA_50, RSI_14, MACD_12_26_9, ADX_14, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
```

---

## Phase 3: Strategy Framework

### 3.1 Strategy Types

**Type A: Single-Asset Strategy**
```python
class SingleAssetStrategy(ABC):
    """Long/short on one asset."""
    
    @abstractmethod
    def required_indicators(self) -> list[tuple]:
        """Return [(indicator_name, params), ...]"""
        pass
    
    @abstractmethod
    def on_bar(self, idx: int, data: pd.DataFrame, capital: float, 
               positions: list[Position]) -> Signal:
        """
        Args:
            idx: Current bar index (can access data.iloc[idx-N:idx] for lookback)
            data: Full DataFrame with OHLCV + indicators (pre-computed)
            capital: Available capital
            positions: Current open positions
        """
        pass
```

**Type B: Multi-Legged Strategy (for Basis Arb)**
```python
class MultiLeggedStrategy(ABC):
    """Strategies with multiple simultaneous positions (e.g., long + short)."""
    
    @abstractmethod
    def required_data(self) -> dict[str, DataSpec]:
        """Return {leg_name: DataSpec(venue, ticker, interval), ...}"""
        pass
    
    @abstractmethod
    def on_bar(self, idx: int, data: dict[str, pd.DataFrame], capital: float,
               positions: dict[str, list[Position]]) -> dict[str, Signal]:
        """Return {leg_name: Signal, ...}"""
        pass
```

### 3.2 Position & Trade Tracking

```python
@dataclass
class Position:
    id: str
    leg: str                    # Leg name (e.g., "cme", "defi")
    entry_time: datetime
    entry_price: float
    size: float                 # Notional value
    side: str                   # "long" or "short"
    
@dataclass
class Trade:
    position: Position
    exit_time: datetime
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str
    bars_held: int
```

### 3.3 Backtest Engine

```python
class BacktestEngine:
    def run(self, strategy: Strategy, capital: float, 
            costs: CostModel = None) -> BacktestResult:
        """
        1. Load data per strategy.required_data()
        2. Compute indicators per strategy.required_indicators()
        3. Iterate bars, call strategy.on_bar()
        4. Execute signals, track positions
        5. Generate equity curve, trades list
        """
        
class BacktestResult:
    config: dict
    trades: list[Trade]
    equity_curve: pd.Series      # Indexed by timestamp
    positions_history: pd.DataFrame  # All position snapshots
    metrics: dict                # Sharpe, drawdown, etc.
```

---

## Phase 4: Web Application

### 4.1 Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Backend | **FastAPI** | Async, modern, auto-docs |
| Frontend | **Jinja2 + HTMX + Plotly** | Simple, no build step |
| Styling | **TailwindCSS (CDN)** | Quick, modern UI |

### 4.2 Features

**Data Tab:**
- Browse data directory tree (venue → market → ticker → interval)
- Show available years with checkboxes
- Download new years (shows progress)
- Click year → instant OHLCV chart + stats panel
- Flag data issues (gaps, outliers) visually

**Backtest Tab:**
- Select strategy from dropdown
- Configure parameters (sliders/inputs)
- Select data (ticker, years)
- Run backtest → show progress
- Results: equity curve, trades table, metrics

**Export Tab:**
- Generate QuantStats HTML/PDF
- Download trades CSV
- Download equity curve

### 4.3 API Endpoints

```python
# Data
GET  /api/data/tree                    # List all data
GET  /api/data/{venue}/{market}/{ticker}/{interval}/years  # Available years
POST /api/data/download                # Download year (async job)
GET  /api/data/preview/{path}          # OHLCV preview + stats

# Backtest
GET  /api/strategies                   # List available strategies
GET  /api/strategies/{name}/params     # Get strategy parameters
POST /api/backtest/run                 # Run backtest (async job)
GET  /api/backtest/{id}/status         # Job status
GET  /api/backtest/{id}/result         # Get results

# Export
GET  /api/export/quantstats/{backtest_id}  # Generate report
```

### 4.4 UI Mockup

```
┌─────────────────────────────────────────────────────────────┐
│  📊 Strategy Lab                    [Data] [Backtest] [Export] │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────────────────────────────────┐│
│ │ 📁 binance  │  │                                         ││
│ │  └─ futures │  │     [Candlestick Chart - Plotly]        ││
│ │     └─ BTCUSDT│ │                                         ││
│ │        └─ 1h │  │                                         ││
│ │           ☑ 2023│                                         ││
│ │           ☑ 2024│ ├─────────────────────────────────────────┤│
│ │           ☐ 2025│ │  Bars: 17,520 | Gaps: 2 | Range: 99.9% ││
│ │                │  │  Open: $42,100 | Close: $67,200        ││
│ └─────────────┘  └─────────────────────────────────────────┘│
│ [Download 2025]  [Clear All]                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Migrate Basis Arb

### 5.1 Refactor to MultiLeggedStrategy

```python
# strategies/basis_arb.py
class BasisArbStrategy(MultiLeggedStrategy):
    def __init__(self, threshold_bps=50, take_profit_bps=15, ...):
        self.threshold_bps = threshold_bps
        ...
    
    def required_data(self) -> dict[str, DataSpec]:
        return {
            "cme": DataSpec(venue="tradingview", ticker="GC1!", interval="15m"),
            "defi": DataSpec(venue="hyperliquid", ticker="PAXG", interval="15m"),
        }
    
    def required_indicators(self) -> list[tuple]:
        return []  # Basis arb uses basis_bps, computed in preprocessing
    
    def on_bar(self, idx, data, capital, positions) -> dict[str, Signal]:
        cme_price = data["cme"].iloc[idx]["close"]
        defi_price = data["defi"].iloc[idx]["close"]
        basis_bps = (defi_price - cme_price) / cme_price * 10000
        
        if abs(basis_bps) > self.threshold_bps and not positions:
            # Enter: long cheaper, short expensive
            ...
```

---

## Dependencies

```txt
# requirements.txt
pandas>=3.0.0
numpy>=2.0.0
pandas-ta>=0.3.14b
plotly>=5.18.0
pyarrow>=14.0.0
requests>=2.31.0

# Web app
fastapi>=0.109.0
uvicorn>=0.27.0
jinja2>=3.1.0
python-multipart>=0.0.6

# Existing (keep)
quantstats>=0.0.62
```

---

## Implementation Order

| # | Phase | Deliverable | Est. |
|---|-------|-------------|------|
| 1 | Data Infrastructure | `core/data/` (Binance, validator, yearly storage) | 2h |
| 2 | Indicators | `core/indicators/` (pandas-ta wrapper) | 1h |
| 3 | Strategy Framework | `core/strategy/` (base classes, engine) | 2h |
| 4 | Web Application | `app/` (FastAPI + HTMX + Plotly) | 3h |
| 5 | Migrate Basis Arb | `strategies/basis_arb.py` | 1h |
| **Total** | | | **9h** |

---

## Suggested Build Sequence

1. **Phase 1**: Data infra first (foundation for everything)
2. **Phase 4**: Web app early (enables visual testing of data)
3. **Phase 2**: Indicators (quick, needed for strategies)
4. **Phase 3**: Strategy framework (core logic)
5. **Phase 5**: Migrate basis arb (validate framework works)

---

## Ready to Implement

All requirements confirmed. Awaiting approval to start Phase 1.
