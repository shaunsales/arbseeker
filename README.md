# Basis Arbitrage Research Platform

A modular Python platform for cross-venue basis arbitrage research, backtesting, and data management. Includes a web-based UI for data browsing, basis file creation, and strategy backtesting.

## Features

- **Web application** - FastAPI + HTMX 2.0 + Alpine.js UI for data management and analysis
- **Multi-venue data pipeline** - Binance futures, Hyperliquid perps (S3 bulk + LZ4-to-Parquet builder)
- **Basis file builder** - Cross-venue spread computation with data quality tracking
- **Strategy framework** - SingleAsset, MultiLegged, and BasisStrategy types
- **Cost modeling** - Commission, slippage, funding rates
- **Pre-computed indicators** - pandas-ta integration

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch web app
python run_app.py
# Open http://localhost:8000

# Download Binance data (CLI)
python -m core.data.binance --symbol BTCUSDT --start 2025-07 --end 2025-12

# Download Hyperliquid raw trades from S3
python -m core.data.hyperliquid_s3 --start 2025-10-01 --end 2025-12-31

# Build Hyperliquid OHLCV parquets (uses top-liquidity symbol config)
python -m core.data.hyperliquid_build --cleanup
```

## Architecture

```
app/                    # Web application (FastAPI + HTMX 2.0 + Alpine.js)
├── main.py                 # FastAPI app entry point
├── routes/
│   ├── data.py             # Data browser, download, preview
│   ├── basis.py            # Basis file builder UI
│   └── backtest.py         # Backtest runner UI
├── templates/              # Jinja2 + HTMX templates
└── static/                 # CSS, JS assets

core/                   # Core framework
├── data/               # Data infrastructure
│   ├── storage.py          # Parquet I/O (load_ohlcv, save_ohlcv)
│   ├── binance.py          # Binance Vision monthly klines downloader
│   ├── hyperliquid_s3.py   # Hyperliquid S3 raw LZ4 trade downloader
│   ├── hyperliquid_build.py # LZ4-to-OHLCV Parquet builder
│   ├── hyperliquid_symbols.json # Top-liquidity symbol config
│   ├── basis.py            # Basis file builder (cross-venue spreads)
│   ├── market_hours.py     # Market hours & near-close detection
│   └── validator.py        # OHLCV data validation & gap filling
├── indicators/         # Technical indicators (pandas-ta)
│   └── indicators.py       # compute_indicators() wrapper
├── strategy/           # Backtesting engine
│   ├── base.py             # SingleAssetStrategy, MultiLeggedStrategy
│   ├── basis_strategy.py   # BasisStrategy, BasisPosition, BasisSignal
│   ├── position.py         # Position, Trade, Signal, CostModel
│   └── engine.py           # BacktestEngine, BacktestResult
└── analysis/           # Analysis utilities

strategies/             # Strategy implementations
├── basis_arb.py            # BasisArbitrage strategy
└── _example_strategies.py  # Example strategies

tests/                  # Test suite
```

## Data Pipeline

### Binance Futures
```bash
python -m core.data.binance --symbol BTCUSDT --start 2025-07 --end 2026-01
```
Downloads monthly OHLCV klines from Binance Vision at 1m/1h/1d intervals.

### Hyperliquid Perps (two-stage pipeline)
```bash
# Stage 1: Download raw hourly LZ4 trade fills from S3
python -m core.data.hyperliquid_s3 --start 2025-10-01 --end 2025-12-31

# Stage 2: Build per-symbol OHLCV parquets from LZ4 sources
python -m core.data.hyperliquid_build              # default 20 symbols from config
python -m core.data.hyperliquid_build --symbol ALL  # extract all symbols
python -m core.data.hyperliquid_build --cleanup     # delete LZ4 sources after build
```

### Data Storage
```
data/{venue}/{market}/{ticker}/{interval}/
├── 2025-07.parquet    # Monthly parquet files
├── 2025-08.parquet
└── ...

data/basis/{base_ticker}/{interval}/
├── 2025-10.parquet    # Pre-computed basis spreads
└── ...
```

## Strategy Types

| Type | Use Case |
|------|----------|
| `SingleAssetStrategy` | Trend following, mean reversion |
| `MultiLeggedStrategy` | Pairs trading, spreads |
| `BasisStrategy` | Pre-computed basis file arbitrage |

## Cost Model

```python
CostModel(
    commission_bps=5.0,     # Per-trade commission
    slippage_bps=2.0,       # Slippage estimate
    funding_daily_bps=5.0,  # Daily funding rate
    bars_per_day=24,        # For funding calculation
)
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

See `docs/PLATFORM_OVERVIEW.md` for complete framework documentation:
- Data layer architecture & pipeline
- Strategy base classes
- Backtest engine flow
- Cost calculations
- BacktestResult metrics

## Requirements

- Python 3.10+
- See `requirements.txt`

## License

MIT
