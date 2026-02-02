# Backtesting Framework

A modular Python framework for developing and backtesting trading strategies.

## Features

- **Single-asset strategies** - Long/short on one asset
- **Multi-legged strategies** - Pairs, spreads, arbitrage
- **Flexible data layer** - Parquet storage with multiple data sources
- **Cost modeling** - Commission, slippage, funding rates
- **Pre-computed indicators** - pandas-ta integration

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run a backtest
PYTHONPATH=. python -c "
from core.strategy import BacktestEngine, StrategyConfig
from core.strategy.position import CostModel

# Define your strategy (see docs/PLATFORM_OVERVIEW.md)
# result = BacktestEngine().run(strategy=my_strategy, capital=100_000)
# result.print_report()
"
```

## Architecture

```
core/
├── data/           # Data infrastructure
│   ├── storage.py      # Parquet I/O (yearly/monthly files)
│   ├── yahoo.py        # Yahoo Finance downloader
│   └── hyperliquid.py  # Hyperliquid API downloader
├── indicators/     # Technical indicators (pandas-ta)
└── strategy/       # Backtesting engine
    ├── base.py         # SingleAssetStrategy, MultiLeggedStrategy
    ├── position.py     # Position, Trade, Signal, CostModel
    └── engine.py       # BacktestEngine, BacktestResult

strategies/         # Strategy implementations
scripts/            # Utilities (param sweeps, report generation)
```

## Strategy Types

| Type | Use Case |
|------|----------|
| `SingleAssetStrategy` | Trend following, mean reversion |
| `MultiLeggedStrategy` | Pairs trading, basis arbitrage, spreads |

## Data Storage

```
data/{venue}/{market}/{ticker}/{interval}/
├── 2024.parquet       # Yearly files
├── 2025-01.parquet    # Monthly files
└── ...
```

## Cost Model

```python
CostModel(
    commission_bps=5.0,     # Per-trade commission
    slippage_bps=2.0,       # Slippage estimate
    funding_daily_bps=5.0,  # Daily funding rate
    bars_per_day=24,        # For funding calculation
)
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

See `docs/PLATFORM_OVERVIEW.md` for complete framework documentation:
- Data layer architecture
- Strategy base classes
- Backtest engine flow
- Cost calculations
- BacktestResult metrics

## Requirements

- Python 3.10+
- See `requirements.txt`

## License

MIT
