# Project Roadmap

## Status: ✅ Framework Complete

The backtesting framework is complete with:
- Data layer (storage, downloaders)
- Strategy layer (single-asset + multi-legged)
- Engine layer (backtest execution, cost model)
- BasisArb strategy (optimized via 1000-iteration sweep)

See `README.md` for quick start and `docs/PLATFORM_OVERVIEW.md` for architecture details.

---

## Completed

- [x] Core framework (`core/data`, `core/strategy`, `core/indicators`)
- [x] BasisArbitrage strategy with parameter optimization
- [x] HTML report generator (`scripts/generate_basis_report.py`)
- [x] Hyperliquid S3 trade downloader (`scripts/download_hl_trades.py`)

## Next Steps

### Research (Paused)
- [ ] Minute-level data acquisition (need CME minute data source)
- [ ] Get real Hyperliquid synthetic gold data (replace PAXG proxy)

### Production
- [ ] Paper trade to measure execution quality
- [ ] Monitor live funding rates
- [ ] Live trading system (CME + Hyperliquid APIs)

### Framework
- [ ] Web application for interactive backtesting
- [ ] Additional strategies (funding arb, cross-exchange)
