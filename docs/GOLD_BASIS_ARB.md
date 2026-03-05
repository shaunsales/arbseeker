# Cross-Venue Gold Basis Arbitrage

**Goal:** Find persistent basis spread opportunities across gold-linked perpetual contracts on multiple venues. Gold is interesting because it trades across both crypto-native (tokenised gold) and TradFi (CME futures) venues, creating structural dislocations due to differing liquidity profiles, funding rates, and market hours.

## Thesis
- Gold perps on newer DEX venues (Aster, Gravity, Lighter) are likely less efficient than Binance/Hyperliquid
- Tokenised gold (PAXG, XAUT) may trade at premiums/discounts to spot gold (XAU) due to redemption friction
- Cross-venue spreads may be wide enough to arb even after gas/fees on-chain
- Even 1 opportunity per day at >20bps could be viable if depth supports it

## Instruments
| Instrument | Type | Tracking |
|------------|------|----------|
| **PAXG** | Pax Gold — gold-backed ERC-20, 1:1 with fine troy oz | perp price on each venue |
| **XAUT** | Tether Gold — gold-backed ERC-20, 1:1 with fine troy oz | perp price on each venue |
| **XAU** | Gold spot / CME gold futures (GC) | spot or front-month futures price |

## Venues
| Venue | Type | CCXT | Gold Tickers | Notes |
|-------|------|------|-------------|-------|
| Binance | CEX futures | ✅ | PAXGUSDT | Most liquid crypto gold market |
| Hyperliquid | DEX perp | ✅ | PAXG-USD | Good liquidity, S3 backfill available |
| Aster | DEX perp | ✅ | TBD | Check for PAXG/XAUT/GOLD markets |
| Gravity | DEX perp | ❌ | TBD | No CCXT — needs custom API integration |
| Lighter | DEX perp | ❌ | TBD | No CCXT — needs custom API integration |
| CME | TradFi futures | ❌ | GC (gold futures) | Need separate data source (e.g. Databento, Polygon) |

## Data Collection Plan
1. **Identify tickers** — enumerate all gold-linked perp/futures symbols on each venue
2. **Poll 1m OHLCV** — use CCXT where available (Binance, Hyperliquid, Aster), custom REST/WS for Gravity & Lighter
3. **Log to Parquet** — store 1m close prices per venue/ticker, monthly files
4. **Build basis files** — feed into basisarb-research platform's basis builder for cross-venue spread analysis
5. **Continuous monitoring** — run as a persistent service, alert on spread dislocations

## Output Format
Feed into basisarb-research basis pipeline:
```
Path: data/basis/{base_ticker}/{interval}/{period}.parquet
Columns: base_price, {venue}_price, {venue}_basis_abs, {venue}_basis_bps, data_quality
```

## Key Questions to Answer
- Which venue pairs have the widest and most persistent spreads?
- What is the typical spread duration and mean-reversion profile?
- How does liquidity depth compare across venues at spread extremes?
- Are spreads correlated with funding rate differentials?
- What is the realistic PnL after fees, slippage, and gas?
