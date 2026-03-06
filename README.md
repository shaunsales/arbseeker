# Antlyr

Crypto strategy backtesting platform with a React web UI. Download OHLCV data, build multi-interval strategy datasets with indicators, run backtests, and visualise results.

## Setup

```bash
# Python backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# React frontend
cd frontend && npm install && cd ..
```

## Run

```bash
# Terminal 1 — API server
python run_app.py                    # http://localhost:8000

# Terminal 2 — React dev server
cd frontend && npm run dev           # http://localhost:5173
```

## Project Structure

```
core/                   # Python framework (data, indicators, engine)
strategies/             # Strategy implementations (Python classes)
app/                    # FastAPI JSON API
frontend/               # React SPA (Vite + TypeScript + shadcn/ui)
output/                 # Generated strategy data + backtest results
data/                   # Downloaded OHLCV parquet files
docs/                   # Documentation
tests/                  # Test suite
```

## Documentation

- **[Platform Overview](docs/PLATFORM_OVERVIEW.md)** — architecture, data layer, strategy framework, engine
- **[Single Asset Strategy Plan](docs/SINGLE_ASSET_STRATEGY_PLAN.md)** — implementation plan + status tracker

## Requirements

- Python 3.10+
- Node.js 18+
