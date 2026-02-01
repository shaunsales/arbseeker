# Basis Arbitrage Research

## Overview

A modular backtesting framework for TradFi-DeFi basis arbitrage, with a fully optimized Gold Basis Arbitrage strategy trading CME Gold Futures vs Hyperliquid perpetuals.

---

## Project Status: ✅ FRAMEWORK & STRATEGY COMPLETE

### Optimized BasisArb Strategy Results

| Metric | Value |
|--------|-------|
| **Annual Return** | 72% |
| **Sharpe Ratio** | 2.14 |
| **Win Rate** | 69% |
| **Max Drawdown** | 1.4% |
| Period | 51 days |
| Trade Pairs | 94 |

*Results with $500k/leg (2x leverage) on $1M capital*

### Optimal Parameters (from 1000-iteration sweep)
```python
threshold_bps = 80.0          # Entry when |basis| > 80 bps
take_profit_captured_bps = 55.0  # Exit when captured 55 bps
half_life_bars = 2.5          # Mean reversion half-life (~37 min)
max_half_lives = 8.0          # Max hold ~20 bars (~5 hours)
```

---

## Project Structure

```
basisarb-research/
├── core/                       # Core framework
│   ├── data/
│   │   ├── storage.py          # Parquet I/O (yearly/monthly files)
│   │   ├── yahoo.py            # Yahoo Finance downloader
│   │   ├── hyperliquid.py      # Hyperliquid API downloader
│   │   ├── market_hours.py     # CME market hours + near_close detection
│   │   └── validator.py        # Gap detection, LOCF
│   ├── indicators/
│   │   └── indicators.py       # pandas-ta wrapper
│   └── strategy/
│       ├── base.py             # SingleAssetStrategy, MultiLeggedStrategy
│       ├── position.py         # Position, Trade, CostModel
│       └── engine.py           # BacktestEngine
│
├── strategies/
│   └── basis_arb.py            # BasisArbitrage(MultiLeggedStrategy)
│
├── scripts/
│   ├── param_sweep.py          # Parameter optimization
│   └── debug_basis_pnl.py      # P&L analysis tools
│
├── data/
│   ├── test/                   # Test data (Yahoo GC + Hyperliquid PAXG)
│   │   ├── futures/GC/15m/     # CME Gold via Yahoo
│   │   └── perp/PAXG/15m/      # Hyperliquid PAXG
│   └── cleaned/                # Legacy merged data
│
├── output/
│   ├── param_sweep_results.csv # Parameter sweep results
│   ├── basis_arb_returns.csv   # Daily returns for QuantStats
│   └── basis_arb_quantstats.html  # QuantStats tearsheet
│
├── docs/
│   └── STRATEGY_FRAMEWORK.md   # Framework design document
│
└── tests/                      # Test suite
```

---

## Quick Start

```bash
source venv/bin/activate

# Run optimized BasisArb backtest
PYTHONPATH=. python -c "
from strategies.basis_arb import BasisArbitrage, BasisArbConfig
from core.strategy import BacktestEngine, StrategyConfig, DataSpec
from core.strategy.position import CostModel

strategy = BasisArbitrage(
    config=StrategyConfig(
        name='BasisArb',
        fixed_size=True,
        fixed_size_amount=500000,  # $500k per leg
        costs=CostModel(commission_bps=5.1, slippage_bps=4.0, funding_daily_bps=5.0),
    ),
    arb_config=BasisArbConfig(),  # Uses optimized defaults
    tradfi_spec=DataSpec('test', 'futures', 'GC', '15m'),
    defi_spec=DataSpec('test', 'perp', 'PAXG', '15m'),
)

result = BacktestEngine().run(strategy=strategy, capital=1_000_000)
print(f'Net P&L: \${sum(t.net_pnl for t in result.trades):,.0f}')
print(f'Sharpe: {result.sharpe_ratio:.2f}')
"

# Run parameter sweep
PYTHONPATH=. python scripts/param_sweep.py

# Generate QuantStats report
# (automatically saved to output/basis_arb_quantstats.html)
```

---

## Strategy Framework

### Strategy Types

| Type | Use Case | Example |
|------|----------|---------|
| `SingleAssetStrategy` | Long/short one asset | Trend following |
| `MultiLeggedStrategy` | Pairs, spreads, arb | **BasisArbitrage** |

### BasisArbitrage Strategy

**Entry:** `|basis| > threshold_bps` (default 80 bps)

**Exit:**
1. **Take-profit:** Captured `take_profit_captured_bps` (default 55 bps)
2. **Time-expiry:** Held `max_half_lives × half_life_bars` bars
3. **Market-closing:** Auto-close 15 min before CME close

**No stop-loss needed** - basis arb locks the spread at entry; spread widening just means wait longer for convergence.

---

## Cost Model

| Component | Value | Notes |
|-----------|-------|-------|
| CME Commission | 0.1 bps | ~$2.50/contract |
| DeFi Taker Fee | 5.0 bps | Hyperliquid |
| Slippage | 4.0 bps | 2 bps × 2 (entry + exit) |
| Funding | 5.0 bps/day | Prorated by bars held |
| **Total Round-trip** | ~18 bps | |

---

## Data Infrastructure

### Supported Data Sources
- **Yahoo Finance** (`core/data/yahoo.py`) - CME Gold Futures (GC=F)
- **Hyperliquid** (`core/data/hyperliquid.py`) - PAXG perpetual

### File Format
```
data/{venue}/{market}/{ticker}/{interval}/
├── 2025.parquet           # Full year
├── 2025-12.parquet        # Monthly
└── 2026-01.parquet
```

### Market Hours
- CME Gold: Sun 6PM - Fri 5PM ET (23-hour trading)
- Daily break: 5-6 PM ET
- Auto-close positions 15 min before market close

---

## Leverage & Position Sizing

| Capital | Position/Leg | Leverage | Margin Used | Annual Return |
|---------|--------------|----------|-------------|---------------|
| $1M | $250k | 0.5x | ~4% | 36% |
| $1M | $500k | 1.0x | ~8% | 72% |
| $1M | $1.75M | 3.5x | ~26% | 203% |

**Constraints:**
- Max leverage: 3-4x
- Max margin utilization: 75%
- Hyperliquid Gold volume: ~$50M/day (use 2% rule)

---

## Next Steps

### Immediate
- [ ] Get real Hyperliquid synthetic gold data (replace PAXG proxy)
- [ ] Paper trade to measure execution quality
- [ ] Monitor live funding rates

### Future
- [ ] Live trading system (CME + Hyperliquid APIs)
- [ ] Web application (Phase 4 of framework plan)
- [ ] Additional arb strategies (funding arb, cross-exchange)

---

## Documentation

- `docs/STRATEGY_FRAMEWORK.md` - Full framework design document
- `output/basis_arb_quantstats.html` - QuantStats performance report
- `output/param_sweep_results.csv` - Parameter optimization results

---

## Market Hours Reference

| Market | Hours (UTC) | Notes |
|--------|-------------|-------|
| CME Globex | Sun 23:00 - Fri 22:00 | 23-hour trading |
| CME Break | 22:00 - 23:00 daily | 1-hour maintenance |
| DeFi | 24/7 | No breaks |

**Tradeable Window:** Only during CME hours (basis is stale otherwise)
