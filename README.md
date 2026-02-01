# ArbSeeker

**Modular Backtesting Framework for TradFi-DeFi Basis Arbitrage**

A Python framework for developing and backtesting arbitrage strategies, with a fully optimized Gold Basis Arbitrage strategy trading CME Gold Futures vs Hyperliquid perpetuals.

## Performance (Optimized Strategy)

| Metric | Value |
|--------|-------|
| **Annual Return** | 72% |
| **Sharpe Ratio** | 2.14 |
| **Win Rate** | 69% |
| **Max Drawdown** | 1.4% |

*With $500k/leg (2x leverage) on $1M capital*

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run BasisArb backtest
PYTHONPATH=. python -c "
from strategies.basis_arb import BasisArbitrage, BasisArbConfig
from core.strategy import BacktestEngine, StrategyConfig, DataSpec
from core.strategy.position import CostModel

strategy = BasisArbitrage(
    config=StrategyConfig(
        name='BasisArb',
        fixed_size=True,
        fixed_size_amount=500000,
        costs=CostModel(commission_bps=5.1, slippage_bps=4.0, funding_daily_bps=5.0),
    ),
    arb_config=BasisArbConfig(),
    tradfi_spec=DataSpec('test', 'futures', 'GC', '15m'),
    defi_spec=DataSpec('test', 'perp', 'PAXG', '15m'),
)

result = BacktestEngine().run(strategy=strategy, capital=1_000_000)
print(f'Net P&L: \${sum(t.net_pnl for t in result.trades):,.0f}')
print(f'Sharpe: {result.sharpe_ratio:.2f}')
"
```

## Framework Architecture

```
core/
├── data/           # Data infrastructure
│   ├── storage.py      # Parquet I/O (yearly/monthly)
│   ├── yahoo.py        # Yahoo Finance downloader
│   ├── hyperliquid.py  # Hyperliquid API
│   └── market_hours.py # CME hours + auto-close
├── indicators/     # Technical indicators (pandas-ta)
└── strategy/       # Backtesting engine
    ├── base.py         # SingleAssetStrategy, MultiLeggedStrategy
    ├── position.py     # Position, Trade, CostModel
    └── engine.py       # BacktestEngine

strategies/
└── basis_arb.py    # BasisArbitrage(MultiLeggedStrategy)
```

## BasisArb Strategy

**Entry:** `|basis| > 80 bps`

**Exit:**
1. Take-profit: Captured 55 bps
2. Time-expiry: ~5 hours max hold
3. Market-close: Auto-close 15 min before CME close

**No stop-loss** - spread is locked at entry; widening just means wait longer.

### Optimal Parameters
```python
threshold_bps = 80.0
take_profit_captured_bps = 55.0
half_life_bars = 2.5
max_half_lives = 8.0
```

## Leverage Scaling

| Position/Leg | Leverage | Annual Return |
|--------------|----------|---------------|
| $250k | 0.5x | 36% |
| $500k | 1.0x | 72% |
| $1.75M | 3.5x | 203% |

## Cost Model (~18 bps round-trip)

- CME commission: 0.1 bps
- DeFi taker fee: 5.0 bps
- Slippage: 4.0 bps
- Funding: 5.0 bps/day

## Data Sources

| Source | Symbol | Type |
|--------|--------|------|
| Yahoo Finance | GC=F | CME Gold Futures |
| Hyperliquid | PAXG | DeFi Perpetual |

## Output

```
output/
├── basis_arb_quantstats.html  # QuantStats tearsheet
├── basis_arb_returns.csv      # Daily returns
└── param_sweep_results.csv    # Parameter optimization
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- `PLAN.md` - Project overview and results
- `docs/STRATEGY_FRAMEWORK.md` - Framework design document

## Requirements

- Python 3.10+
- See `requirements.txt`

## License

MIT
