# ArbSeeker

**TradFi-DeFi Basis Arbitrage Analysis**

Validates arbitrage opportunities between traditional finance (Yahoo Finance) and decentralized finance (Aster DEX) by analyzing price spreads and mean reversion characteristics.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python 1_data_acquisition.py    # Fetch 14 days of minute data
python 1_generate_reports.py    # Generate price comparison PDFs
python 2_basis_analysis.py      # Calculate basis + mean reversion stats
python 3_visualization.py       # Generate charts + final PDF report
```

## Pipeline Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  1. Data Acquisition│────▶│  2. Basis Analysis  │────▶│  3. Visualization   │
│                     │     │                     │     │                     │
│  • Yahoo Finance    │     │  • Spread calc      │     │  • Price charts     │
│  • Aster DEX        │     │  • ADF test         │     │  • Basis timeseries │
│  • LOCF alignment   │     │  • Half-life        │     │  • Distribution     │
│                     │     │  • Hurst exponent   │     │  • PDF report       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Output

```
output/
├── reports/
│   ├── basis_analysis_report_*.pdf   # Main report with executive summary
│   ├── tsla_price_comparison_*.pdf
│   └── gold_price_comparison_*.pdf
├── charts/
│   ├── *_price_comparison.png
│   ├── *_basis_timeseries.png
│   ├── *_basis_distribution.png
│   └── summary_table.png
└── backtest/
    ├── tsla_basis.csv
    └── gold_basis.csv
```

## Statistical Tests

| Test | Purpose | Threshold |
|------|---------|-----------|
| **ADF** | Stationarity (mean-reverting?) | p < 0.05 |
| **Half-life** | Reversion speed | Lower = faster |
| **Hurst** | Mean-reverting vs trending | H < 0.5 = mean-reverting |

## Data Sources

| Source | Type | Assets | Frequency |
|--------|------|--------|-----------|
| Yahoo Finance | TradFi | TSLA, GC=F | 1-minute |
| Aster DEX | DeFi | TSLAUSDT, XAUUSDT | 1-minute |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific stage
python -m pytest tests/test_basis_analysis.py -v
```

65 tests covering data acquisition, basis calculations, and visualization.

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## License

MIT
