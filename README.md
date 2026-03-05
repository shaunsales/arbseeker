# Basis Arbitrage Research Platform

A modular Python platform for cross-venue basis arbitrage research, backtesting, and data management. Includes a React SPA web UI for data browsing, strategy data building, and backtesting.

## Features

- **Web application** — React SPA frontend + FastAPI JSON API backend
- **Multi-venue data pipeline** — Binance futures, Hyperliquid perps (S3 bulk + LZ4-to-Parquet builder)
- **Basis file builder** — Cross-venue spread computation with data quality tracking
- **Strategy framework** — SingleAsset, MultiLegged, and BasisStrategy types
- **Strategy data builder** — Multi-interval data with pre-computed indicators
- **Cost modeling** — Commission, slippage, funding rates
- **Pre-computed indicators** — pandas-ta integration

## Quick Start

```bash
# Backend setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd frontend && npm install && cd ..

# Launch (two terminals)
python run_app.py                    # API on http://localhost:8000
cd frontend && npm run dev           # React on http://localhost:5173

# Download Binance data (CLI)
python -m core.data.binance --symbol BTCUSDT --start 2025-07 --end 2025-12

# Download Hyperliquid raw trades from S3
python -m core.data.hyperliquid_s3 --start 2025-10-01 --end 2025-12-31

# Build Hyperliquid OHLCV parquets (uses top-liquidity symbol config)
python -m core.data.hyperliquid_build --cleanup
```

## Architecture

```
app/                    # Backend API (FastAPI, JSON only)
├── main.py                 # FastAPI app entry point
├── routes/
│   ├── data.py             # Data browser, download, preview endpoints
│   ├── strategy.py         # Strategy spec, build, preview endpoints
│   ├── basis.py            # Basis file builder endpoints
│   └── backtest.py         # Backtest runner endpoints

frontend/               # React SPA (Vite + TypeScript)
├── src/
│   ├── api/                # Typed API client (fetch wrappers)
│   ├── components/
│   │   ├── layout/         # AppLayout with shadcn sidebar nav
│   │   ├── data/           # OhlcvChart, DataTable, DataPreview
│   │   ├── strategy/       # StrategyChart, MonthRangePicker, BuildControls
│   │   └── ui/             # shadcn/ui primitives (button, badge, sidebar, etc.)
│   ├── pages/              # DataPage, DownloadPage, StrategyPage, BasisPage, BacktestPage
│   ├── types/              # TypeScript interfaces matching API responses
│   └── hooks/              # Custom React hooks
├── index.html
└── vite.config.ts          # Dev proxy: /api → localhost:8000

core/                   # Core framework (pure Python, no web dependencies)
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
│   ├── data.py             # StrategyDataSpec, StrategyDataBuilder, manifest
│   ├── position.py         # Position, Trade, Signal, CostModel
│   └── engine.py           # BacktestEngine, BacktestResult
└── analysis/           # Analysis utilities

strategies/             # Strategy implementations
├── basis_arb.py            # BasisArbitrage strategy
├── adx_trend.py            # ADXTrend strategy (SingleAsset example)
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

## Frontend Stack

| Layer | Choice |
|-------|--------|
| Framework | React 19 + Vite |
| Language | TypeScript |
| Styling | TailwindCSS v4 |
| Components | shadcn/ui (sidebar, collapsible, tooltip, etc.) |
| API State | TanStack Query |
| Tables | TanStack Table |
| Charts | lightweight-charts (TradingView) |
| Routing | React Router v7 |
| Icons | Lucide React |

### Navigation

Hierarchical sidebar with collapsible groups:
- **Data** → Browser (data tree + OHLCV chart/table), Download (Binance downloader)
- **Strategies** → Single Asset, Basis, Multi-Leg
- **Backtest** (top-level)

### Key Frontend Components
- **OhlcvChart** — candlestick + volume chart with server-side resampling for large datasets
- **DataTable** — TanStack Table with column sorting and pagination
- **StrategyChart** — price + overlay indicators + separate indicator panel + volume, synced time scales
- **MonthRangePicker** — calendar grid of months, click-to-select range, warmup indication, per-interval availability

## Documentation

See `docs/PLATFORM_OVERVIEW.md` for complete framework documentation:
- Data layer architecture & pipeline
- Strategy base classes
- Backtest engine flow
- Cost calculations
- BacktestResult metrics

See `docs/SINGLE_ASSET_STRATEGY_PLAN.md` for:
- Single asset strategy workflow (Data → Backtest → Visualisation)
- Frontend architecture & component details
- Implementation status tracker

## Requirements

- Python 3.10+
- Node.js 18+
- See `requirements.txt` (Python) and `frontend/package.json` (JS)

## License

MIT
