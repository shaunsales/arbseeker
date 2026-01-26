# ArbSeeker

**CME-DeFi Gold Basis Arbitrage Analysis**

Analyzes basis arbitrage opportunities between CME Gold Futures and DeFi perpetuals on **Aster** and **Hyperliquid**, with full 3-venue comparison.

## Key Results

| Venue | Daily Volume | Mean Basis | Half-life | Profitable |
|-------|--------------|------------|-----------|------------|
| **Hyperliquid PAXG** | $9.3M | +6 bps | 35 min | ✅ >50 bps |
| **Aster XAUUSDT** | $0.68M | -24 bps | 15 min | ✅ >50 bps |

**Recommendation:** Hyperliquid PAXG as primary venue (14x more liquid).

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python 1_data_acquisition.py    # Fetch CME + Aster + Hyperliquid data
python 2_basis_analysis.py      # Calculate basis + mean reversion stats
python 3_visualization.py       # Generate 3-venue comparison + PDF report
```

## Pipeline

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  1. Data Acquisition│────▶│  2. Basis Analysis  │────▶│  3. Visualization   │
│                     │     │                     │     │                     │
│  • CME (TradingView)│     │  • Spread calc      │     │  • 3-venue charts   │
│  • Aster DEX        │     │  • ADF test         │     │  • Volume analysis  │
│  • Hyperliquid      │     │  • Half-life        │     │  • Threshold study  │
│  • LOCF alignment   │     │  • Hurst exponent   │     │  • PDF report       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Output

```
output/
├── reports/
│   └── basis_analysis_report_*.pdf   # Executive summary + all charts
├── charts/
│   ├── gold_price_comparison.png     # CME vs Aster vs Hyperliquid
│   ├── gold_basis_timeseries.png     # Both DeFi venues overlaid
│   ├── gold_tradeable_basis.png      # Market hours only
│   ├── gold_basis_distribution.png   # Histograms + stats
│   ├── gold_volume_analysis.png      # Volume by venue
│   ├── gold_threshold_analysis.png   # Profitability thresholds
│   ├── venue_comparison.png          # Side-by-side metrics
│   └── summary_table.png
└── backtest/
    ├── gold_analysis_15m.csv
    └── gold_hl_analysis_15m.csv
```

## Data Sources

| Source | Symbol | Type | API |
|--------|--------|------|-----|
| TradingView | GC1! | CME Futures | tvDatafeed |
| Aster DEX | XAUUSDT | DeFi Perp | REST /klines |
| Hyperliquid | PAXG | DeFi Perp | REST /info |

## Mean Reversion Tests

| Test | Aster | Hyperliquid | Meaning |
|------|-------|-------------|---------|
| **ADF p-value** | <0.001 | <0.001 | Stationary ✅ |
| **Half-life** | 15 min | 35 min | Reversion speed |
| **Hurst** | 0.13 | 0.16 | Mean-reverting ✅ |

## Trading Economics

### Cost Structure (~18 bps round-trip)
- CME commission: ~0.5 bps
- DeFi taker fee: 3.5 bps
- Slippage: 8 bps (4 executions)
- Funding: 5 bps/day

### Position Sizing (2% volume rule)
| Venue | Max Position | Margin Required |
|-------|--------------|-----------------|
| Aster | $13.5K | $2K |
| Hyperliquid | $186K | $28K |

## Risks & Caveats

1. **50% capture rate assumed** - needs backtest validation
2. **15-min bars** - may miss execution details
3. **Liquidity constrains size** - monitor order books
4. **Funding rates vary** - can spike in volatile markets

## Testing

```bash
python -m pytest tests/ -v    # 65 tests
```

## Requirements

- Python 3.10+
- See `requirements.txt`

## Documentation

See `PLAN.md` for detailed project documentation including:
- Full analysis results
- Capital sizing calculations
- Strategy recommendations
- Next steps

## License

MIT
