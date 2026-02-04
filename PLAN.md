# Project Cleanup & Standardization Plan

A phased refactor to standardize data formats, create blended basis files, improve backtest outputs, and build an interactive performance dashboard.

See `README.md` for quick start and `docs/PLATFORM_OVERVIEW.md` for architecture details.

---

## Completed

- [x] Core framework (`core/data`, `core/strategy`, `core/indicators`)
- [x] BasisArbitrage strategy with parameter optimization
- [x] HTML report generator (`scripts/generate_basis_report.py`)
- [x] Hyperliquid S3 trade downloader (`scripts/download_hl_trades.py`)
- [x] Remove legacy files (1_data_acquisition.py, 1_generate_reports.py, etc.)

---

## Phase 1: Data Layer Standardization

### 1.1 Standardize Raw OHLCV Schema
```
Columns (required):
- timestamp (DatetimeIndex, UTC) - bar open time, aligned to within 1 second
- open, high, low, close (float64)
- volume (float64)

Columns (optional):
- market_open (bool) - True if market is open at this bar
- trades (int) - number of trades in bar
- quote_volume (float64) - volume in quote currency

Remove:
- near_close flag (compute at runtime from market hours)
- mid (compute at runtime)
```

### 1.2 Enhance Web App Data Browser
- [ ] Add pagination controls (page size: 100/500/1000 rows)
- [ ] Add date range filter
- [ ] Show data quality metrics (gaps, coverage %)
- [ ] Downloadable CSV/Parquet export button

### 1.3 Update Downloaders
- [ ] Standardize output format across all downloaders (yahoo, hyperliquid, binance)
- [ ] Remove `near_close` computation
- [ ] Align all timestamps to UTC bar open time (within 1 second tolerance)

---

## Phase 2: Basis Files

### 2.1 Basis Data Schema
Pre-computed basis data with one base venue and multiple quote venues:

```
Path: data/basis/{base_ticker}/{interval}/{period}.parquet

Columns:
- timestamp (DatetimeIndex, UTC) - aligned to bar open within 1 second
- base_price (float64) - open price from base venue
- {quote_venue}_price (float64) - open price from each quote venue
- {quote_venue}_basis_abs (float64) - quote_price - base_price
- {quote_venue}_basis_bps (float64) - (quote_price - base_price) / base_price * 10000
- data_quality (str) - "ok", "base_stale", "quote_stale", "gap"
```

### 2.2 Basis Builder Tool
New module: `core/data/basis.py` with `create_basis_file()` function.

### 2.3 Web App Basis Builder UI
- [ ] Select base venue/ticker
- [ ] Add one or more quote venues/tickers
- [ ] Choose time periods
- [ ] Preview aligned data with basis chart
- [ ] Create & save basis file

### 2.4 Strategy Split
- **MultiLeggedStrategy** - unchanged, supports any multi-leg trades
- **BasisStrategy** - new class for pre-computed basis files, same indicator pattern

---

## Phase 3: Backtest Output Standardization

**Key principle:** Input data stays in `data/`, backtest results go to `output/backtests/`.

### 3.1 Backtest Result Schema
Bar-by-bar output: `output/backtests/{strategy_name}_{timestamp}.parquet`
- Simplified price data (reference only)
- Indicators (ONLY those used in strategy decision logic)
- Position state, signals, P&L, drawdown, costs

### 3.2 Trade-Level Summary
`output/backtests/{strategy_name}_{timestamp}_trades.parquet`

### 3.3 Metadata File
`output/backtests/{strategy_name}_{timestamp}.json`

### 3.4 Update BacktestEngine
- [ ] Collect bar-level data during run
- [ ] Track only indicators from strategy.config.indicators
- [ ] Add `save_results(path)` method
- [ ] Unified format for SingleAsset, MultiLegged, and BasisStrategy

---

## Phase 4: Performance Dashboard

**Build iteratively** - add views as each pipeline stage completes.

*Needs further specification before implementation. Will define after Phase 3 is complete.*

---

## Implementation Order

| Phase | Deliverable |
|-------|-------------|
| 0 | ✅ Delete legacy files, update PLAN.md |
| 1.1 | Standardize OHLCV schema |
| 1.2 | Web app pagination + filters |
| 2.1-2.2 | Basis file schema + builder |
| 2.3 | BasisStrategy class |
| 2.4 | Web app basis builder UI |
| 3.1-3.4 | Backtest output schema + engine updates |
| 4.x | Performance dashboard (iterative) |
