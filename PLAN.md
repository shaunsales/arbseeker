# Basis Arbitrage Research - Project Plan

## Goal
Validate TradFi-DeFi basis arbitrage opportunities by comparing prices between 
Yahoo Finance (TradFi) and Aster/Hyperliquid (DeFi).

---

## Project Structure

```
basisarb-research/
├── PLAN.md                 # This file - project documentation
├── requirements.txt        # Python dependencies
├── test_connections.py     # SDK connection validation
│
├── 1_data_acquisition.py   # Stage 1: Fetch and clean data
├── 2_basis_analysis.py     # Stage 2: Calculate basis and backtest metrics
├── 3_visualization.py      # Stage 3: Generate charts and tables
│
├── data/                   # Raw and cleaned data files
│   ├── raw/                # Raw API responses
│   └── cleaned/            # Normalized parquet files
│
├── output/                 # Analysis results
│   ├── backtest/           # Basis analysis CSV files
│   └── charts/             # PNG visualizations
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_data_acquisition.py
│   ├── test_basis_analysis.py
│   ├── test_visualization.py
│   └── fixtures/           # Test data fixtures
│       ├── yahoo_sample.parquet
│       ├── aster_sample.parquet
│       └── basis_sample.csv
│
└── venv/                   # Python virtual environment
```

---

## Development Workflow

Each stage follows this process:
1. **Build** — Implement code based on specs
2. **Unit Tests** — Test individual functions
3. **Integration Tests** — Test component interactions (where appropriate)
4. **Fixtures** — Create sample data in `tests/fixtures/`
5. **Verify** — Run all tests, ensure passing before next stage

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific stage tests
python -m pytest tests/test_data_acquisition.py -v
```

---

## Environment Setup ✅

### Requirements
```bash
pip install -r requirements.txt
```

### SDK References
| Source | Package | Docs |
|--------|---------|------|
| Yahoo Finance | `yfinance` | https://pypi.org/project/yfinance/ |
| Hyperliquid | `hyperliquid-python-sdk` | https://github.com/hyperliquid-dex/hyperliquid-python-sdk |
| Aster | `aster-connector-python` | https://github.com/asterdex/aster-connector-python |

### Quick Start
```bash
source venv/bin/activate
python test_connections.py      # Verify SDK connections
python 1_data_acquisition.py    # Stage 1: Fetch data
python 2_basis_analysis.py      # Stage 2: Run analysis
python 3_visualization.py       # Stage 3: Generate charts
```

---

## Stage 1: Data Acquisition

**Script:** `1_data_acquisition.py`

### Purpose
Fetch raw price data from all sources, clean/normalize it, and save in a common format.

### Data Sources
| Source | Type | Assets |
|--------|------|--------|
| Yahoo Finance | TradFi | TSLA, AAPL, NVDA, GC=F, SI=F, QQQ |
| Aster (primary) | DeFi | TSLAUSDT, AAPLUSDT, NVDAUSDT, XAUUSDT, XAGUSDT, QQQUSDT |
| Hyperliquid (fallback) | DeFi | Crypto only (no RWA yet) |

### Data Limits
- **Yahoo Finance**: 7 days @ 1m intervals
- **Aster**: 5000 candles @ 1m (~3.5 days)
- **Hyperliquid**: 5000 candles @ 1m (~3.5 days)

### Output Format (Parquet)
```
data/cleaned/{asset}_{source}.parquet
```

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime64[ns, UTC] | Index, UTC normalized |
| `open` | float64 | Open price |
| `high` | float64 | High price |
| `low` | float64 | Low price |
| `close` | float64 | Close price |
| `mid` | float64 | (high + low) / 2 |
| `volume` | float64 | Trading volume |
| `source` | string | yahoo / aster / hyperliquid |
| `symbol` | string | Original symbol |

---

## Stage 2: Basis Analysis

**Script:** `2_basis_analysis.py`

### Purpose
Load cleaned data, align timestamps, calculate basis metrics, and generate backtest output.

### Input
```
data/cleaned/{asset}_yahoo.parquet
data/cleaned/{asset}_aster.parquet
```

### Processing
1. Load TradFi and DeFi data for each asset
2. Resample to 1-minute alignment
3. Calculate basis: `basis = defi_mid - tradfi_mid`
4. Flag market hours (NYSE: 14:30-21:00 UTC)
5. Compute statistics (mean, std, min, max, percentiles)

### Output Format (CSV)
```
output/backtest/{asset}_basis.csv
```

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp |
| `tradfi_mid` | TradFi mid price |
| `defi_mid` | DeFi mid price |
| `basis_absolute` | Absolute spread ($) |
| `basis_bps` | Spread in basis points |
| `market_open` | Boolean: within NYSE hours |

---

## Stage 3: Visualization

**Script:** `3_visualization.py`

### Purpose
Generate charts and summary tables for quant team analysis.

### Input
```
output/backtest/{asset}_basis.csv
```

### Output
```
output/charts/{asset}_price_comparison.png
output/charts/{asset}_basis_timeseries.png
output/charts/{asset}_basis_distribution.png
output/charts/summary_table.png
```

### Charts Generated
1. **Price Comparison** — TradFi vs DeFi overlay with market hours shading
2. **Basis Timeseries** — Spread over time with threshold bands
3. **Basis Distribution** — Histogram of basis values
4. **Summary Table** — Statistics across all assets

### Statistics Table
| Metric | Description |
|--------|-------------|
| Mean Basis (bps) | Average spread |
| Std Dev (bps) | Volatility of spread |
| Min / Max (bps) | Range extremes |
| % > 20 bps | Tradeable opportunity frequency |
| Half-life (min) | Mean reversion speed (if applicable) |

---

## Assets Configured

| Asset | Yahoo | Aster | Hyperliquid |
|-------|-------|-------|-------------|
| Tesla | `TSLA` | `TSLAUSDT` ✓ | ✗ |
| Apple | `AAPL` | `AAPLUSDT` ✓ | ✗ |
| Nvidia | `NVDA` | `NVDAUSDT` ✓ | ✗ |
| Gold | `GC=F` | `XAUUSDT` ✓ | ✗ |
| Silver | `SI=F` | `XAGUSDT` ✓ | ✗ |
| Nasdaq 100 | `QQQ` | `QQQUSDT` ✓ | ✗ |

---

## Market Hours Reference

| Market | Local Time | UTC |
|--------|------------|-----|
| NYSE Open | 9:30 AM ET | 14:30 UTC |
| NYSE Close | 4:00 PM ET | 21:00 UTC |
| DeFi | 24/7 | 24/7 |

---

## Next Steps

If basis shows promise (mean reversion, exceeds friction):
1. Add mean reversion statistical tests (ADF, half-life)
2. Build real-time collection pipeline with websockets
3. Consider IBKR integration for execution-quality data
4. Implement signal generation and backtesting
