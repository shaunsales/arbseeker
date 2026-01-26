# Basis Arbitrage Research - Project Plan

## Goal
Validate TradFi-DeFi basis arbitrage opportunities by comparing prices between 
CME Gold Futures (TradFi) and DeFi perpetuals on Aster and Hyperliquid.

**Current Focus:** Gold (XAUUSDT / PAXG) basis arbitrage - the most promising opportunity identified.

---

## Project Status: ✅ BACKTEST COMPLETE

### Key Findings

| Metric | Aster XAUUSDT | Hyperliquid PAXG |
|--------|---------------|------------------|
| Daily Volume | $0.68M | $9.3M |
| Mean Basis | -24 bps | +6 bps |
| Half-life | 15 min | 35 min |
| Hurst Exponent | 0.13 | 0.16 |
| Max Position (2%) | $13.5K | $186K |
| Profitable >50 bps | ✅ Yes | ✅ Yes |

**Recommendation:** Use Hyperliquid PAXG as primary venue (14x more liquid).

---

## Project Structure

```
basisarb-research/
├── PLAN.md                 # This file - project documentation
├── README.md               # Quick start guide
├── requirements.txt        # Python dependencies
│
├── 1_data_acquisition.py   # Stage 1: Fetch CME + DeFi data
├── 2_basis_analysis.py     # Stage 2: Calculate basis + mean reversion
├── 3_visualization.py      # Stage 3: Generate 3-venue comparison charts
├── 4_backtest.py           # Stage 4: Backtest simulation + QuantStats
├── 1_generate_reports.py   # Generate price comparison PDFs
│
├── data/
│   ├── raw/                # Raw API responses (parquet)
│   └── cleaned/            # Merged TradFi-DeFi data
│       ├── gold_merged_15m.parquet      # CME vs Aster
│       └── gold_hl_merged_15m.parquet   # CME vs Hyperliquid
│
├── output/
│   ├── backtest/           # Backtest results + QuantStats reports
│   ├── charts/             # PNG visualizations (3-venue)
│   └── reports/            # PDF reports with executive summary
│
└── tests/                  # Test suite (65 tests)
```

---

## Quick Start

```bash
source venv/bin/activate

# Full pipeline
python 1_data_acquisition.py    # Fetch 30 days of 15-min data
python 2_basis_analysis.py      # Calculate basis + mean reversion stats
python 3_visualization.py       # Generate charts + PDF report
python 4_backtest.py            # Run backtest simulation

# Backtest options
python 4_backtest.py --capital 1000000 --threshold 50 --venue hyperliquid
python 4_backtest.py --quantstats pdf    # Generate QuantStats tearsheet
python 4_backtest.py --scenario          # Run scenario analysis
```

---

## Data Sources

### TradFi (CME Gold Futures)
| Source | Symbol | Method | Limit |
|--------|--------|--------|-------|
| TradingView | GC1! | tvDatafeed | 5000 bars |

### DeFi Venues
| Venue | Symbol | API | Daily Volume |
|-------|--------|-----|--------------|
| Aster DEX | XAUUSDT | REST /klines | ~$0.68M |
| Hyperliquid | PAXG | REST /info | ~$9.3M |

### Data Format
15-minute OHLCV bars with:
- `tradfi_close`, `defi_close` - Closing prices
- `basis_bps` - Spread in basis points
- `tradfi_market_open` - CME market hours flag
- `defi_dollar_volume` - Volume in USD

---

## Analysis Pipeline

### Stage 1: Data Acquisition
- Fetches CME gold futures via TradingView
- Fetches DeFi perps from Aster and Hyperliquid
- Applies LOCF (Last Observation Carried Forward) for alignment
- Merges into single parquet per venue pair

### Stage 2: Basis Analysis
- Calculates basis: `(defi_close - tradfi_close) / tradfi_close * 10000`
- Runs mean reversion tests (ADF, half-life, Hurst)
- Calculates volume statistics
- Generates capital sizing recommendations

### Stage 3: Visualization
- **Price Comparison:** CME vs Aster vs Hyperliquid (3 lines)
- **Basis Timeseries:** Both DeFi venues overlaid
- **Basis Distribution:** Histograms with stats panel
- **Volume Analysis:** Bar charts comparing all venues
- **Threshold Analysis:** Profitability at different entry thresholds
- **Venue Comparison:** Side-by-side metrics
- **PDF Report:** Executive summary + all charts

### Stage 4: Backtest Simulation
- Threshold-based entry/exit simulation
- Realistic position sizing (2% volume rule)
- Cost modeling (fees, slippage, funding)
- Equity curve generation
- QuantStats integration for tearsheet reports
- Scenario analysis across thresholds

---

## Mean Reversion Statistics

| Test | Aster | Hyperliquid | Interpretation |
|------|-------|-------------|----------------|
| ADF p-value | <0.001 | <0.001 | Both stationary ✅ |
| Half-life | 15 min | 35 min | Aster faster |
| Hurst | 0.13 | 0.16 | Both mean-reverting ✅ |

---

## Trading Strategy

### Threshold-Based Entry
Enter when |basis| > threshold, exit at mean reversion.

| Threshold | Net/Trade | Freq/Day | Daily Profit |
|-----------|-----------|----------|--------------|
| >50 bps | +24 bps | 8-10 | ~$350/day* |
| >80 bps | +35 bps | 3-5 | ~$260/day* |
| >100 bps | +45 bps | 1-2 | ~$135/day* |

*Per $186K position on Hyperliquid

### Cost Assumptions
- CME commission: $2.50/contract (~0.5 bps)
- DeFi taker fee: 3.5 bps
- Slippage: 2 bps/execution
- Funding: 5 bps/day
- **Total round-trip: ~18 bps**

---

## Capital Sizing

### Position Limits (2% of daily volume rule)
| Venue | Daily Volume | Max Position |
|-------|--------------|--------------|
| Aster | $0.68M | $13,500 |
| Hyperliquid | $9.3M | $186,000 |

### Margin Requirements
| Leg | Margin | Example ($186K position) |
|-----|--------|--------------------------|
| CME | 10% | $18,600 |
| DeFi | 5% | $9,300 |
| **Total** | **15%** | **$27,900** |

---

## Backtest Results

### Configuration
- **Capital:** $1,000,000
- **Venue:** Hyperliquid PAXG
- **Threshold:** >50 bps
- **Capture Rate:** 50%
- **Position Size:** $186,290 (volume-limited)

### Performance (51 days)
| Metric | Value |
|--------|-------|
| Net P&L | +$175,129 (+17.5%) |
| Annualized | 125% |
| Sharpe Ratio | 21.64 |
| Max Drawdown | 0.0% |
| Total Trades | 332 |
| Win Rate | 100%* |
| Avg Trade | $527 |

*100% win rate due to simplified exit logic (instant mean reversion assumed)

### QuantStats Report
```bash
python 4_backtest.py --quantstats pdf
```
Generates HTML tearsheet with:
- CAGR, Sharpe, Sortino, Calmar
- Drawdown analysis
- Monthly returns heatmap
- Benchmark comparison (GLD)

---

## Risk & Caveats

1. **Simplified exit logic** - assumes instant mean reversion
2. **15-min bars miss execution details** - consider 1-min data
3. **DeFi liquidity constrains position size** - monitor order book depth
4. **Funding rates vary** - can spike in volatile markets
5. **Slippage may be higher** - especially for larger positions

---

## Next Steps

### Immediate
- [ ] Add realistic exit logic with partial fills
- [ ] Monitor live funding rates on both venues
- [ ] Paper trade to measure actual execution quality

### Future
- [ ] Add more DeFi venues (dYdX, GMX)
- [ ] Real-time websocket data collection
- [ ] Automated signal generation
- [ ] IBKR integration for CME execution

---

## Market Hours Reference

| Market | Hours (UTC) | Notes |
|--------|-------------|-------|
| CME Globex | Sun 23:00 - Fri 22:00 | 23-hour trading |
| CME Break | 22:00 - 23:00 daily | 1-hour maintenance |
| DeFi | 24/7 | No breaks |

**Tradeable Window:** Only during CME hours (basis is stale otherwise)
