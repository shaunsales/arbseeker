# Single Asset Strategy Backtesting Plan

End-to-end workflow for backtesting single-asset strategies on Binance Futures perp data.

Three stages: **Data → Backtest → Visualisation**

---

## Architecture

### Backend — Python (FastAPI)
Pure JSON API. All data processing, indicator computation, backtest execution, and file management stays in Python. No HTML rendering — the backend is a headless API server.

- **FastAPI** — async, fast, auto-generated OpenAPI docs
- **Endpoints return JSON only** — no Jinja2 templates
- CORS enabled for local dev (React dev server on different port)
- In production, FastAPI serves the React build as static files (single deployment)

### Frontend — React SPA
Separate app in `frontend/` directory. Handles all UI rendering, state management, and user interaction.

**Stack:**
| Layer | Choice | Why |
|-------|--------|-----|
| Framework | React 19 + Vite | Fastest DX, dominant ecosystem |
| Language | TypeScript | Type safety, API contract enforcement |
| Styling | TailwindCSS v4 | Already used, utility-first, fast iteration |
| Components | shadcn/ui | Beautiful defaults, copy-paste ownership, Radix primitives |
| API State | TanStack Query | Caching, loading/error states, background refetch |
| Charts | lightweight-charts-react | TradingView charts as React components (already using LW Charts) |
| Routing | React Router v7 | Client-side page navigation |
| Icons | Lucide React | Clean, consistent icon set |

**Project structure:**
```
frontend/
  src/
    api/              # API client functions (typed fetch wrappers)
    components/       # Reusable UI components (charts, tables, etc.)
    pages/            # Page-level components (Data, Strategy, Backtest)
    hooks/            # Custom React hooks
    types/            # TypeScript interfaces matching API responses
    lib/              # Utilities
  public/
  index.html
  vite.config.ts
  tailwind.config.ts
```

**Dev workflow:**
- `uvicorn app.main:app --reload` → API on :8000
- `npm run dev` (in `frontend/`) → React on :5173 (proxies API calls to :8000)
- Single terminal script or `concurrently` to run both

**Migration approach:**
The existing Jinja2/HTMX/Alpine frontend is replaced entirely. Each page becomes a React route with components that fetch from the same API endpoints. The backend routes only need their `TemplateResponse()` calls changed to return JSON (the logic stays the same). The `core/` Python package is completely untouched.

---

## Design Principles

### No Look-Ahead Bias
- Backtest iterates on **1m bars** for precise stop-loss/TSL execution.
- Larger-interval data (15m, 1h) is accessed via the **last fully closed bar** only. A 1h bar's close price is not available until the bar closes — so at T=14:32, the latest usable 1h bar is the one that closed at 14:00 (covering 13:00–14:00), not the in-progress 14:00–15:00 bar.
- This is enforced by the data accessor, not by the strategy author.

### Strategy Code Location
Strategy implementations live in `strategies/` as Python classes:
```
strategies/
  _example_strategies.py     # ExampleMACrossover etc.
  adx_trend.py               # ADX-based trend following
  my_strategy.py             # user strategies
```

The strategy class defines its data needs (intervals, indicators) and trading logic. It is separate from the data and results, which go into `output/`.

### Strategy Data & Results Folder
Each strategy gets its own output folder. Data files are generated into it. If anything changes (new indicator, different date range), regenerate the relevant parquet files.

```
output/strategies/{strategy_name}/
  manifest.json              # strategy config, intervals, indicators, date range
  data/
    1m.parquet               # OHLCV only — for TSL/SL execution
    15m.parquet              # OHLCV + indicators (e.g. RSI_14, ATR_14)
    1h.parquet               # OHLCV + indicators (e.g. ADX_14, SMA_50)
  results/
    {run_id}_bars.parquet    # bar-level backtest state (1m)
    {run_id}_trades.parquet  # trade log with decision context
    {run_id}_meta.json       # run config, summary metrics, timestamps
    {run_id}_quantstats.html # QuantStats tearsheet
```

### Time Range Validation
- All parquet files in a strategy folder must cover the **same date range**.
- The manifest stores the canonical date range.
- If the user changes the backtest range, the UI flags that data files need regeneration.
- On build, each interval's data is validated for coverage and gaps before saving.

### Metrics via QuantStats
Replace hand-rolled `compute_metrics()` with `quantstats.stats` for industry-standard calculations:
- CAGR, annualised return, annualised volatility
- Sharpe, Sortino, Calmar ratios
- Max drawdown, avg drawdown, drawdown duration
- Win rate, profit factor, payoff ratio, expectancy
- Full HTML tearsheet via `quantstats.reports.html()`

### Data Source
Binance Futures only (initially). Big-cap USDT-M perpetuals (BTCUSDT, ETHUSDT, etc.).

---

## Phase 3A: Strategy Data Builder

Build the multi-interval data files that a strategy needs.

### 3A.1 Strategy Data Interface
Each strategy declares its data dependencies via a `data_spec()` method on the base class. This is the **single source of truth** — the web UI reads it to know what to build, and the engine validates it before running.

```python
class SingleAssetStrategy(ABC):

    @abstractmethod
    def data_spec(self) -> StrategyDataSpec:
        """Declare the data this strategy requires."""
        ...

@dataclass
class StrategyDataSpec:
    venue: str                         # "binance"
    market: str                        # "futures"
    ticker: str                        # "BTCUSDT"
    intervals: dict[str, list]         # {"1m": [], "1h": [("adx", {"length": 14})]}
```

- `intervals` maps each interval to a list of `(indicator_name, params)` tuples.
- `1m` is always required (execution interval) — typically no indicators, just OHLCV.
- Larger intervals carry indicators relevant to that timeframe.
- **Date range is NOT part of the spec** — it's chosen in the web UI at build time and stored in the manifest. The strategy only validates that all data files share the same start timestamp.

Example:
```python
class ADXTrend(SingleAssetStrategy):
    def data_spec(self) -> StrategyDataSpec:
        return StrategyDataSpec(
            venue="binance",
            market="futures",
            ticker="BTCUSDT",
            intervals={
                "1m": [],
                "1h": [("adx", {"length": 14}), ("sma", {"length": 50})],
            },
        )
```

### Validation
Before a backtest runs, the engine checks:
1. Strategy folder exists at `output/strategies/{strategy.name}/`
2. A parquet file exists for every interval in `data_spec().intervals`
3. Each parquet contains the expected OHLCV columns + indicator columns
4. All parquet files share the same start timestamp

If any check fails, the engine raises a clear error (e.g. "Missing 1h.parquet — build data first").

### 3A.2 Data Builder
- Download/load OHLCV for each interval from Binance (reuse existing `core/data/binance.py`).
- Compute indicators per interval (reuse existing `core/indicators/` module).
- Validate: check date range coverage, gap %, bar counts match expectations.
- Save each interval as a separate parquet file.
- Write `manifest.json` with full spec + build timestamp + data quality summary.

### 3A.3 Web UI — Strategy Data Builder Page
React page at `/strategy`. Components:
- **`<StrategySidebar />`** — auto-discovers strategy classes, click to select
- **`<StrategySpec />`** — displays venue, market, ticker, intervals, indicators (read-only from `data_spec()`)
- **`<BuildControls />`** — calendar month grid picker for date range, "Build Data" button
- **`<CurrentData />`** — shows manifest info, clickable file rows to preview data, delete button
- **`<DataPreview />`** — inline chart + table preview of a parquet file (see 3C.2)

API endpoints (JSON only):
- `GET /strategy/` — list all discovered strategies
- `GET /strategy/spec/{class_name}` — spec + status + manifest
- `GET /strategy/available-dates/{class_name}` — available date ranges per interval
- `POST /strategy/build` — build data files
- `POST /strategy/delete` — delete data files
- `GET /strategy/preview/{class_name}/{interval}` — chart + table data for a parquet file

---

## Phase 3B: Backtest Engine v2

Run the strategy on 1m bars with multi-interval data access.

### 3B.1 Multi-Interval Data Accessor
```python
class StrategyData:
    """Look-ahead-safe accessor for multi-interval data."""

    def bar(self, interval: str, timestamp: pd.Timestamp) -> pd.Series:
        """Return the last FULLY CLOSED bar at or before timestamp."""
        ...

    def bars(self, interval: str, timestamp: pd.Timestamp, n: int) -> pd.DataFrame:
        """Return last N fully closed bars ending at or before timestamp."""
        ...
```

- Loads all parquets from the strategy folder.
- `bar("1h", T)` returns the 1h bar whose `open_time + 1h <= T`.
- This is the **single point of look-ahead prevention**.
- Caches interval DataFrames in memory; uses searchsorted for O(log n) lookups.

### 3B.2 Refactored Strategy Interface
```python
class SingleAssetStrategy(ABC):
    @abstractmethod
    def on_bar(self, timestamp: pd.Timestamp, data: StrategyData,
               capital: float, position: Optional[Position]) -> Signal:
        """Called every 1m bar. Use data.bar('1h', timestamp) for larger intervals."""
```

- Strategy receives `StrategyData` instead of raw DataFrame.
- Strategy decides which intervals to query.
- Engine always iterates 1m bars.

### 3B.3 Decision Context Capture
When a signal fires (buy/sell/close), capture a snapshot:
```python
{
    "signal": "buy",
    "reason": "ADX > 25 and price crossed above SMA_50",
    "context": {
        "1m":  {"close": 42150.0, "open_time": "2025-01-15T14:32:00Z"},
        "1h":  {"ADX_14": 28.3, "SMA_50": 41800.0, "close": 42100.0,
                "open_time": "2025-01-15T13:00:00Z"},
    }
}
```

- Stored on each `Trade` object.
- Exported as JSON column in trades parquet.
- Critical for understanding *why* a decision was made and *what data* was available at that moment.

### 3B.4 Bar-Level State Recording
Every 1m bar, record:
- `timestamp`, `close` (1m price)
- `balance` — cash balance (realised capital, changes only on trade close)
- `nav` — net asset value (balance + unrealized PnL), this is the equity curve
- `drawdown_pct` — drawdown from peak NAV
- `position_side` (long/short/flat)
- `position_size` — current position notional
- `position_pnl` — unrealized PnL ($)
- `position_pnl_pct` — unrealized PnL (%)
- `signal` (if any fired this bar)

The user sets **starting equity** before each run (e.g. 10,000 USDT). The engine initialises `balance = starting_equity` and tracks NAV from there.

Saved as `{run_id}_bars.parquet`.

### 3B.5 Metrics & Output
- Compute all metrics via `quantstats.stats` from the equity series.
- Generate QuantStats HTML tearsheet.
- Save: bars parquet, trades parquet, metadata JSON, tearsheet HTML.

### 3B.6 Parameter Optimisation (Deferred)
Grid search over named parameters. **Deferred until single-run workflow is proven.**

When ready:
- Define parameter grid (e.g. `{"adx_threshold": [20, 25, 30], "tsl_pct": [0.5, 1.0, 2.0]}`).
- Engine runs all combinations, saves each as a separate run.
- Summary table ranks runs by Sharpe / return / drawdown.
- Can be parallelised with multiprocessing.

---

## Phase 3C: Visualisation (React Frontend)

All visualisation is built as React components using `lightweight-charts-react` and shadcn/ui.

### 3C.1 Backtest Viewer
React page at `/backtest/:runId`. Components:
- **`<PriceChart />`** — 1m OHLCV (or downsampled) with trade entry/exit markers overlaid
- **`<IndicatorPanel />`** — one panel per indicator, stacked below price, synced time scales
- **`<EquityCurve />`** — NAV over time from bars parquet
- **`<DrawdownChart />`** — drawdown % from peak
- All chart components share a synced time scale via a common hook (`useChartSync`)

### 3C.2 Source Data Viewer
React component in the Strategy page. Click a data file → inline preview with:
- **`<StrategyChart />`** — price + overlay indicators + separate indicator panel + volume
- **`<DataTable />`** — paginated table with color-coded indicator columns
- API endpoint: `GET /strategy/preview/{class_name}/{interval}?page=N`

### 3C.3 Trade Inspector
- Click a trade marker on the price chart → side panel or modal shows decision context
- Uses the context snapshot from 3B.3 (indicator values + timestamps at entry/exit)
- React component: `<TradeInspector trade={selectedTrade} />`

### 3C.4 QuantStats Tearsheet
- Embed generated HTML tearsheet in an iframe, or link to open in new tab
- API endpoint: `GET /backtest/tearsheet/{run_id}` returns the HTML file

### 3C.5 Optimisation Results (Deferred)
- shadcn/ui `<DataTable />` with sortable columns for parameter sweep results
- Click a row to navigate to its full backtest view

*Further detail to be defined as we approach this stage.*

---

## Testing

### Deterministic Test Strategy
A minimal strategy with **known, predictable behaviour** to validate the engine:

```python
class TestFixedEntryExit(SingleAssetStrategy):
    """
    Buys when 1h SMA_10 > SMA_20 (first crossover).
    Sells when 1h SMA_10 < SMA_20 (first crossunder).
    Uses synthetic OHLCV data where crossover points are known.
    """
```

### What to verify
- **PnL calculation** — given known entry/exit prices, verify exact gross PnL, costs, net PnL.
- **NAV / balance tracking** — balance changes only on trade close; NAV includes unrealized PnL.
- **Drawdown** — verify peak-to-trough calculation against hand-computed values.
- **No look-ahead** — confirm 1h bar data at T=14:32 uses the 13:00–14:00 bar, not 14:00–15:00.
- **Decision context** — verify the snapshot captures correct indicator values and timestamps.
- **Cost model** — verify commission, slippage, and funding costs are applied correctly.
- **Position sizing** — verify size = starting_equity × position_pct.

### Synthetic data approach
Generate a small DataFrame (~500 1m bars, ~8 1h bars) with known prices:
- Bars 1–200: price rises steadily (triggers long entry at known bar).
- Bars 200–350: price falls (triggers exit at known bar, then short entry).
- Bars 350–500: price rises (triggers short exit).

With known entry/exit prices we can hand-calculate exact expected PnL, costs, NAV at every bar, and assert against engine output.

---

## Implementation Order

| Step | Deliverable | Status |
|------|-------------|--------|
| 3A.1 | `StrategyDataSpec` + manifest schema | ✅ Done |
| 3A.2 | Data builder (download, compute indicators, save parquets) | ✅ Done |
| 3A.3 | Backend API — strategy endpoints (JSON) | ✅ Done |
| 3B.1 | `StrategyData` multi-interval accessor | ✅ Done |
| 3B.2 | Refactored strategy interface + engine loop | ✅ Done |
| Test | Deterministic test strategy + engine validation | ✅ Done |
| **FE** | **Migrate frontend to React SPA** | **✅ Done** |
|  | → Scaffold Vite + React + TypeScript + TailwindCSS + shadcn/ui | ✅ Done |
|  | → Convert backend routes to pure JSON API (remove Jinja2) | ✅ Done |
|  | → Build React pages: Data, Basis, Strategy, Backtest | ✅ Done |
|  | → Reimplement strategy builder with React components | ✅ Done |
|  | → Chart components using lightweight-charts | ✅ Done |
|  | → Hierarchical sidebar navigation (shadcn sidebar) | ✅ Done |
|  | → Custom MonthRangePicker (calendar grid, warmup, availability) | ✅ Done |
|  | → Server-side OHLCV resampling for large datasets | ✅ Done |
|  | → Standalone Download page (Binance) | ✅ Done |
| 3B.3 | Decision context capture | **← Next** |
| 3B.4 | Bar-level state recording | High |
| 3B.5 | QuantStats metrics + output files | Medium |
| 3B.6 | Parameter optimisation (grid search) | Deferred |
| 3C.1 | Backtest viewer (React — charts + trades) | Medium |
| 3C.3 | Trade inspector (click to see context) | Low |
| 3C.4 | QuantStats tearsheet embed | Low |
| 3C.5 | Optimisation results table | Deferred |
