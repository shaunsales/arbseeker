# Project Cleanup & Standardization Plan

A phased refactor to standardize data formats, create blended basis files, improve backtest outputs, and build an interactive performance dashboard.

See `README.md` for quick start and `docs/PLATFORM_OVERVIEW.md` for architecture details.

---

## Completed

- [x] Core framework (`core/data`, `core/strategy`, `core/indicators`)
- [x] BasisArbitrage strategy with parameter optimization
- [x] HTML report generator (`scripts/generate_basis_report.py`)
- [x] Hyperliquid S3 trade downloader (`core/data/hyperliquid_s3.py`)
- [x] Hyperliquid LZ4-to-OHLCV builder (`core/data/hyperliquid_build.py`)
- [x] Binance Vision klines downloader (`core/data/binance.py`)
- [x] Web app (FastAPI JSON API + React SPA frontend)
- [x] Remove legacy files

---

## Phase 1: Data Layer Standardization ✅

### 1.1 Standardize Raw OHLCV Schema ✅
```
Index: open_time (DatetimeIndex, UTC)

Columns (required):
- open, high, low, close (float64)
- volume (float64)

Columns (optional):
- market_open (bool) - True if market is open at this bar
- quote_volume (float64) - volume in quote currency
- count (int) - number of trades in bar
- taker_buy_volume (float64) - taker buy volume

Computed at runtime (NOT stored):
- near_close - compute from market_open transitions
- mid - compute as (high + low) / 2
```

### 1.2 Enhance Web App Data Browser ✅
- [x] Add pagination controls (page size: 100/500/1000 rows)
- [x] Add date range filter (start/end date pickers)
- [x] Show data quality metrics (gaps, coverage %, bar count)
- [x] Downloadable CSV/Parquet export buttons
- [x] Chart/Table tabs with paginated data table
- [x] React SPA with TanStack Table + lightweight-charts (migrated from HTMX/Alpine)

### 1.3 Update Downloaders ✅
- [x] Binance Vision monthly klines downloader (`core/data/binance.py`)
- [x] Hyperliquid S3 raw LZ4 trade downloader (`core/data/hyperliquid_s3.py`)
- [x] Hyperliquid LZ4-to-OHLCV Parquet builder (`core/data/hyperliquid_build.py`)
- [x] Top-liquidity symbol config (`core/data/hyperliquid_symbols.json`)
- [x] Memory-efficient month-by-month processing
- [x] Skip-if-exists logic for output parquets
- [x] `--cleanup` flag to delete LZ4 sources after build
- [x] Add `market_open` to all venues (24/7 = always True for crypto)
- [x] Consistent `open_time` index name across all venues
- [x] All timestamps aligned to UTC bar open time

### 1.4 Data Validation ✅
- [x] `core/data/validator.py` - OHLCV validation & gap filling
- [x] Coverage %, gap detection, bar count metrics

---

## Phase 2: Basis Files ✅

### 2.1 Basis Data Schema ✅
Pre-computed basis data with one base venue and multiple quote venues:

```
Path: data/basis/{base_ticker}/{interval}/{period}.parquet

Columns:
- open_time (DatetimeIndex, UTC) - aligned to bar open within 1 second
- base_price (float64) - open price from base venue
- {quote_venue}_price (float64) - open price from each quote venue
- {quote_venue}_basis_abs (float64) - quote_price - base_price
- {quote_venue}_basis_bps (float64) - (quote_price - base_price) / base_price * 10000
- data_quality (str) - "ok", "base_stale", "quote_stale", "gap"
```

### 2.2 Basis Builder Tool ✅
Module: `core/data/basis.py` with `create_basis_file()` function.
- Flexible period loading (handles yearly vs monthly format mismatches)
- `load_basis()` and `list_basis_files()` utilities

### 2.3 Web App Basis Builder UI ✅
- [x] Select base venue/ticker
- [x] Add one or more quote venues/tickers
- [x] Auto-detect overlapping periods
- [x] Preview aligned data with basis chart
- [x] Create & save basis file

### 2.4 BasisStrategy Class ✅
- **MultiLeggedStrategy** - unchanged, supports any multi-leg trades
- **BasisStrategy** - new class for pre-computed basis files
  - `BasisPosition` tracks spread positions (direction, entry basis, size)
  - `BasisSignal` for open_long/open_short/close/hold actions
  - `SimpleBasisMeanReversion` example strategy included
  - `BacktestEngine._run_basis()` for running basis backtests

---

## Research: Cross-Venue Gold Basis Arbitrage

See [`docs/GOLD_BASIS_ARB.md`](docs/GOLD_BASIS_ARB.md) — separate project for data collection, output feeds into this platform's basis pipeline.

---

## Phase 3: Single Asset Strategy Backtesting

See [`docs/SINGLE_ASSET_STRATEGY_PLAN.md`](docs/SINGLE_ASSET_STRATEGY_PLAN.md) for full detail.

### 3A — Strategy Data Builder ✅
- [x] `StrategyDataSpec` + manifest schema
- [x] Data builder (download OHLCV, compute indicators, save parquets per interval)
- [x] Backend API — strategy endpoints (JSON)
- [x] Web UI — React strategy data builder page

### 3B — Backtest Engine v2 (partial)
- [x] `StrategyData` multi-interval accessor (look-ahead-safe)
- [x] Refactored strategy interface (always 1m execution, multi-interval reads)
- [x] Deterministic test strategy + engine validation
- [ ] Decision context capture on entry/exit
- [ ] Bar-level state recording
- [ ] QuantStats metrics (replace hand-rolled `compute_metrics`)
- [ ] Parameter optimisation (grid search)

### FE — React SPA Frontend ✅
- [x] Scaffold Vite + React 19 + TypeScript + TailwindCSS v4 + shadcn/ui
- [x] Convert backend routes to pure JSON API (remove Jinja2)
- [x] Hierarchical sidebar navigation (shadcn sidebar with collapsible groups)
- [x] Data Browser page (tree + OHLCV chart + TanStack Table)
- [x] Download page (Binance data downloader)
- [x] Strategy page (spec, build, data preview with charts)
- [x] Custom MonthRangePicker (calendar grid, warmup, per-interval availability)
- [x] Server-side OHLCV resampling for large datasets (300k+ bars)
- [x] Strategy preview charts (price + overlays + indicators + volume, synced)
- [x] Dark mode with shadcn theming

### 3C — Visualisation
- [x] Source data viewer (chart + table preview for strategy parquets)
- [ ] Backtest viewer (price + indicators + equity + trade markers, synced charts)
- [ ] Trade inspector (click trade → see decision context)
- [ ] QuantStats tearsheet embed
- [ ] Optimisation results table

---

## Phase 4: Performance Dashboard

**Build iteratively** - add views as each pipeline stage completes.

*Needs further specification before implementation. Will define after Phase 3 is complete.*

---

## Implementation Order

| Phase | Deliverable |
|-------|-------------|
| 0 | ✅ Delete legacy files, update PLAN.md |
| 1.1 | ✅ Standardize OHLCV schema |
| 1.2 | ✅ Web app data browser + TradingView charts |
| 1.3 | ✅ Binance + Hyperliquid data pipelines |
| 1.4 | ✅ Data validation |
| 2.1-2.2 | ✅ Basis file schema + builder |
| 2.3 | ✅ Web app basis builder UI |
| 2.4 | ✅ BasisStrategy class |
| 3A | ✅ Strategy data builder (multi-interval parquets + JSON API + React UI) |
| FE | ✅ React SPA frontend (sidebar nav, charts, tables, MonthRangePicker, dark mode) |
| 3B | Backtest engine v2 (context capture, bar recording, QuantStats) — partial |
| 3C | Visualisation (backtest viewer, trade inspector, tearsheets) — partial |
| 4.x | Performance dashboard (iterative) |
