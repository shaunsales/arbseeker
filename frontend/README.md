# Antlyr — Frontend

React SPA frontend for Antlyr.

## Stack

| Layer | Choice |
|-------|--------|
| Framework | React 19 + Vite |
| Language | TypeScript (strict) |
| Styling | TailwindCSS v4 |
| Components | shadcn/ui (sidebar, collapsible, tooltip, badge, button, etc.) |
| API State | TanStack Query v5 |
| Tables | TanStack Table v8 |
| Charts | lightweight-charts v4 (TradingView) |
| Routing | React Router v7 |
| Icons | Lucide React |

## Development

```bash
npm install
npm run dev          # Dev server on http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://localhost:8000` (the FastAPI backend). Run the backend first:

```bash
# From project root
python run_app.py
```

## Type Checking

```bash
npx tsc --noEmit
```

## Build

```bash
npm run build        # Output to dist/
```

## Project Structure

```
src/
├── api/                    # Typed API client
│   ├── client.ts               # Base get/post wrappers
│   ├── data.ts                 # Data endpoints
│   └── strategy.ts             # Strategy endpoints
├── components/
│   ├── layout/
│   │   └── AppLayout.tsx       # Sidebar nav (shadcn Sidebar + Collapsible)
│   ├── data/
│   │   ├── OhlcvChart.tsx      # Candlestick + volume chart
│   │   ├── DataTable.tsx       # TanStack Table with sorting + pagination
│   │   └── DataPreview.tsx     # Chart/table preview composite
│   ├── strategy/
│   │   ├── StrategyChart.tsx   # Multi-pane chart (price + overlays + indicators + volume)
│   │   ├── MonthRangePicker.tsx # Calendar grid picker (warmup, availability)
│   │   ├── BuildControls.tsx   # Date range + build button
│   │   ├── StrategySpec.tsx    # Read-only data spec display
│   │   ├── CurrentData.tsx     # Manifest info + file rows + preview
│   │   └── DataPreview.tsx     # Strategy data chart/table preview
│   └── ui/                     # shadcn/ui primitives (auto-generated)
├── pages/
│   ├── DataPage.tsx            # Data browser (tree + preview)
│   ├── DownloadPage.tsx        # Binance data downloader
│   ├── StrategyPage.tsx        # Strategy list + detail
│   ├── BasisPage.tsx           # Basis file builder
│   └── BacktestPage.tsx        # Backtest runner
├── types/
│   └── api.ts                  # TypeScript interfaces for API responses
├── hooks/
│   └── use-mobile.ts           # Responsive hook (shadcn)
├── lib/
│   └── utils.ts                # cn() utility
├── App.tsx                     # Root routes + providers
├── main.tsx                    # Entry point
└── index.css                   # Tailwind + shadcn CSS variables

## Routes

| Path | Page | Description |
|------|------|-------------|
| `/data` | DataPage | Browse on-disk data, chart/table preview |
| `/download` | DownloadPage | Download data from Binance |
| `/strategies/single-asset` | StrategyPage | Single asset strategy builder |
| `/strategies/basis` | BasisPage | Basis strategy builder |
| `/strategies/multi-leg` | Placeholder | Multi-leg strategy (future) |
| `/backtest` | BacktestPage | View backtest results (metrics, charts, trades) |
```
