"""Parameter sweep for BasisArb strategy."""

import itertools
from strategies.basis_arb import BasisArbitrage, BasisArbConfig
from core.strategy import BacktestEngine, StrategyConfig, DataSpec
from core.strategy.position import CostModel
import pandas as pd

# Fixed parameters
CAPITAL = 1_000_000
POSITION_SIZE = 250_000  # $250k per leg
MAX_TRADES_PER_DAY = 100

# Cost model (unchanged)
cost_model = CostModel(
    commission_bps=5.1,
    slippage_bps=4.0,
    funding_daily_bps=5.0,
    bars_per_day=96,
)

# Parameter ranges (designed for ~1000 iterations)
# 8 × 5 × 5 × 5 = 1000
threshold_bps_range = [50, 60, 70, 80, 90, 100, 110, 120]  # 8 values
take_profit_range = [15, 25, 35, 45, 55]  # 5 values
max_half_lives_range = [4, 6, 8, 10, 12]  # 5 values
half_life_bars_range = [1.5, 2.0, 2.5, 3.0, 3.5]  # 5 values

# Generate all combinations
combinations = list(itertools.product(
    threshold_bps_range,
    take_profit_range,
    max_half_lives_range,
    half_life_bars_range,
))

print(f"Running {len(combinations)} parameter combinations...")
print("=" * 80)

results = []

for i, (thresh, tp, max_hl, hl_bars) in enumerate(combinations):
    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(combinations)}")
    
    arb_config = BasisArbConfig(
        threshold_bps=float(thresh),
        take_profit_captured_bps=float(tp),
        half_life_bars=float(hl_bars),
        max_half_lives=float(max_hl),
        max_trades_per_day=MAX_TRADES_PER_DAY,
    )
    
    strategy = BasisArbitrage(
        config=StrategyConfig(
            name=f"sweep_{i}",
            fixed_size=True,
            fixed_size_amount=float(POSITION_SIZE),
            costs=cost_model,
        ),
        arb_config=arb_config,
        tradfi_spec=DataSpec('test', 'futures', 'GC', '15m'),
        defi_spec=DataSpec('test', 'perp', 'PAXG', '15m'),
    )
    
    try:
        result = BacktestEngine(verbose=False).run(strategy=strategy, capital=CAPITAL)
        
        if result.trades:
            total_net = sum(t.net_pnl for t in result.trades)
            total_gross = sum(t.gross_pnl for t in result.trades)
            trade_pairs = len(result.trades) // 2
            days = (result.end_time - result.start_time).days
            ann_ret = (total_net / CAPITAL) * 365 / days * 100 if days > 0 else 0
            
            # Win rate
            wins = sum(1 for t in result.trades if t.net_pnl > 0)
            win_rate = wins / len(result.trades) * 100
            
            results.append({
                'threshold_bps': thresh,
                'take_profit_bps': tp,
                'max_half_lives': max_hl,
                'half_life_bars': hl_bars,
                'max_hold_bars': int(hl_bars * max_hl),
                'trade_pairs': trade_pairs,
                'win_rate': win_rate,
                'gross_pnl': total_gross,
                'net_pnl': total_net,
                'annual_return': ann_ret,
                'sharpe': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown_pct,
            })
    except Exception as e:
        print(f"Error at {thresh}/{tp}/{max_hl}/{hl_bars}: {e}")

# Convert to DataFrame
df = pd.DataFrame(results)

# Sort by annual return
df = df.sort_values('annual_return', ascending=False)

# Save results
df.to_csv('output/param_sweep_results.csv', index=False)

# Print top 20 results
print("\n" + "=" * 100)
print("TOP 20 CONFIGURATIONS BY ANNUAL RETURN")
print("=" * 100)
print(f"{'Thresh':>7} {'TP':>5} {'MaxHL':>6} {'HLBars':>7} {'Hold':>5} {'Pairs':>6} {'Win%':>6} {'Net':>10} {'Ann%':>7} {'Sharpe':>7}")
print("-" * 100)

for _, row in df.head(20).iterrows():
    print(f"{row['threshold_bps']:>7.0f} {row['take_profit_bps']:>5.0f} {row['max_half_lives']:>6.0f} "
          f"{row['half_life_bars']:>7.1f} {row['max_hold_bars']:>5.0f} {row['trade_pairs']:>6.0f} "
          f"{row['win_rate']:>5.0f}% {row['net_pnl']:>10,.0f} {row['annual_return']:>6.0f}% {row['sharpe']:>7.2f}")

# Also show best by Sharpe ratio
print("\n" + "=" * 100)
print("TOP 10 CONFIGURATIONS BY SHARPE RATIO")
print("=" * 100)
df_sharpe = df.sort_values('sharpe', ascending=False)
print(f"{'Thresh':>7} {'TP':>5} {'MaxHL':>6} {'HLBars':>7} {'Hold':>5} {'Pairs':>6} {'Win%':>6} {'Net':>10} {'Ann%':>7} {'Sharpe':>7}")
print("-" * 100)

for _, row in df_sharpe.head(10).iterrows():
    print(f"{row['threshold_bps']:>7.0f} {row['take_profit_bps']:>5.0f} {row['max_half_lives']:>6.0f} "
          f"{row['half_life_bars']:>7.1f} {row['max_hold_bars']:>5.0f} {row['trade_pairs']:>6.0f} "
          f"{row['win_rate']:>5.0f}% {row['net_pnl']:>10,.0f} {row['annual_return']:>6.0f}% {row['sharpe']:>7.2f}")

print(f"\nResults saved to output/param_sweep_results.csv")
