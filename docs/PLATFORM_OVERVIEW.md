# Platform Overview

High-level architecture of Antlyr.

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│  Frontend — React SPA                                 │
│  Vite + TypeScript + TailwindCSS + shadcn/ui          │
│  lightweight-charts · TanStack Query/Table             │
├───────────────────────────────────────────────────────┤
│  Backend — FastAPI (JSON API only)                    │
│  Routes: /data  /strategy  /basis  /backtest          │
├───────────────────────────────────────────────────────┤
│  Core — Python framework                              │
│  Data pipeline · Indicators · Strategy · Engine        │
├───────────────────────────────────────────────────────┤
│  Storage — Parquet files                              │
│  data/{venue}/{market}/{ticker}/{interval}/*.parquet  │
│  output/strategies/{name}/data/ + results/            │
└───────────────────────────────────────────────────────┘
```

---

## Data Layer

### Storage
All OHLCV data stored as monthly Parquet files:
```
data/{venue}/{market}/{ticker}/{interval}/{period}.parquet
```
- **Venues**: Binance, Hyperliquid
- **Schema**: DatetimeIndex (UTC) + open/high/low/close/volume
- **Loader**: `load_ohlcv()` auto-concatenates multi-file ranges

### Downloaders
| Module | Source |
|--------|--------|
| `core/data/binance.py` | Binance Vision monthly klines |
| `core/data/hyperliquid_s3.py` | Hyperliquid S3 raw LZ4 trades |
| `core/data/hyperliquid_build.py` | LZ4 → per-symbol OHLCV parquets |
| `core/data/basis.py` | Cross-venue basis spread computation |

---

## Strategy Layer

### Strategy Types
| Type | Use Case |
|------|----------|
| `SingleAssetStrategy` | One asset, long/short (trend, mean reversion) |
| `MultiLeggedStrategy` | Multiple simultaneous positions (pairs, spreads) |
| `BasisStrategy` | Pre-computed basis file arbitrage |

### SingleAsset Workflow
1. Strategy declares data needs via `data_spec()` (intervals + indicators)
2. Web UI builds multi-interval parquets with pre-computed indicators
3. Engine iterates 1m bars, calling `strategy.on_bar()` each bar
4. Strategy accesses larger intervals via `StrategyData` (look-ahead-safe)
5. Engine records bar-level state, captures decision context on trades

### Key Classes
- **`StrategyDataSpec`** — declares venue, ticker, intervals, indicators
- **`StrategyData`** — look-ahead-safe multi-interval data accessor
- **`Signal`** — buy / sell / close / hold with reason string
- **`Position`** — open position with mark-to-market tracking
- **`Trade`** — closed position with realized PnL + decision context

---

## Engine Layer

### BacktestEngine
Orchestrates the backtest loop:
1. Load strategy data from parquets
2. Iterate 1m bars, call `strategy.on_bar()` → `Signal`
3. Execute trades, apply costs (`CostModel`)
4. Record bar-level state (nav, drawdown, position)
5. Compute metrics via QuantStats

### Cost Model
```python
CostModel(commission_bps=3.5, slippage_bps=2.0, funding_daily_bps=5.0)
```
Applied as round-trip commission + slippage on entry/exit, plus daily funding on open positions.

### Output
Each backtest run produces:
- `{run_id}_bars.parquet` — bar-level state (nav, drawdown, position)
- `{run_id}_trades.parquet` — trade log with decision context JSON
- `{run_id}_meta.json` — config + summary metrics
- `{run_id}_tearsheet.html` — QuantStats HTML tearsheet

---

## Web Application

### Backend (FastAPI)
Pure JSON API, no HTML rendering.

| Route | Purpose |
|-------|---------|
| `/data/` | Browse data tree, preview OHLCV, download from Binance |
| `/strategy/` | List strategies, view spec, build data, preview parquets |
| `/basis/` | Build cross-venue basis files |
| `/backtest/` | List runs, view results, run backtests, serve tearsheets |

### Frontend (React SPA)
| Tech | Purpose |
|------|---------|
| React 19 + Vite | Framework + build |
| TypeScript | Type safety |
| TailwindCSS v4 + shadcn/ui | Styling + components |
| TanStack Query | API state management |
| TanStack Table | Data tables with sorting/pagination |
| lightweight-charts | TradingView-style OHLCV + line charts |
| React Router v7 | Client-side routing |

### Pages
- **Data Browser** — tree of available data, OHLCV chart + table preview
- **Download** — Binance data downloader with progress tracking
- **Strategy** — select strategy, view spec, build data, preview charts
- **Backtest** — run list sidebar, metrics cards, price/equity/drawdown charts, trades table
