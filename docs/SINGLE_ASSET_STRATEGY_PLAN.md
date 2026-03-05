# Single Asset Strategy Backtesting Plan

End-to-end workflow for backtesting single-asset strategies on Binance Futures perp data.

Three stages: **Data → Backtest → Visualisation**

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

### 3A.1 StrategyDataSpec
Define what data a strategy requires:
```python
@dataclass
class StrategyDataSpec:
    name: str                          # strategy name (folder name)
    venue: str                         # "binance"
    market: str                        # "futures"
    ticker: str                        # "BTCUSDT"
    date_range: tuple[str, str]        # ("2024-01", "2025-01")
    intervals: dict[str, list]         # {"1m": [], "1h": [("adx", {"length": 14})]}
```

- `intervals` maps each interval to a list of indicator specs.
- `1m` is always required (execution interval) — typically no indicators, just OHLCV.
- Larger intervals carry indicators relevant to that timeframe.

### 3A.2 Data Builder
- Download/load OHLCV for each interval from Binance (reuse existing `core/data/binance.py`).
- Compute indicators per interval (reuse existing `core/indicators/` module).
- Validate: check date range coverage, gap %, bar counts match expectations.
- Save each interval as a separate parquet file.
- Write `manifest.json` with full spec + build timestamp + data quality summary.

### 3A.3 Web UI — Strategy Data Builder Page
- Create/select a strategy by name.
- Pick venue (Binance) + ticker (from available data or download).
- Pick date range.
- Per interval: select which indicators to compute (with parameters).
- "Build" button → generates parquet files, shows progress + quality summary.
- If files already exist and date range matches, show current status. Flag if rebuild needed.

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

## Phase 3C: Visualisation

### 3C.1 Backtest Viewer (Web UI)
Load a saved backtest run and render:
- **Price chart** (1m OHLCV or downsampled) with trade entry/exit markers
- **Indicator charts** (one per indicator, stacked below price, synced horizontally)
- **Equity curve** chart
- **Drawdown** chart
- All charts use Lightweight Charts, synced time scales (same pattern as basis preview).

### 3C.2 Source Data Viewer
- View any of the strategy's data parquet files directly (OHLCV + indicators per interval).
- Useful for validating data quality before running backtests.

### 3C.3 Trade Inspector
- Click a trade on the chart to see its decision context (the snapshot from 3B.3).
- Shows all indicator values at entry and exit with their timestamps.

### 3C.4 QuantStats Tearsheet
- Embed or link to the generated HTML tearsheet.
- Includes: monthly returns heatmap, drawdown periods, rolling Sharpe, etc.

### 3C.5 Optimisation Results (Deferred)
- Table view of parameter sweep results, sortable by any metric.
- Click a run to view its full backtest visualisation.

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

| Step | Deliverable | Priority |
|------|-------------|----------|
| 3A.1 | `StrategyDataSpec` + manifest schema | High |
| 3A.2 | Data builder (download, compute indicators, save parquets) | High |
| 3A.3 | Web UI — strategy data builder page | High |
| 3B.1 | `StrategyData` multi-interval accessor | High |
| 3B.2 | Refactored strategy interface + engine loop | High |
| 3B.3 | Decision context capture | High |
| 3B.4 | Bar-level state recording | High |
| 3B.5 | QuantStats metrics + output files | Medium |
| 3B.6 | Parameter optimisation (grid search) | Deferred |
| 3C.1 | Backtest viewer (charts + trades) | Medium |
| 3C.2 | Source data viewer | Low |
| 3C.3 | Trade inspector (click to see context) | Low |
| 3C.4 | QuantStats tearsheet embed | Low |
| 3C.5 | Optimisation results table | Deferred |
| Test | Deterministic test strategy + engine validation | High |
