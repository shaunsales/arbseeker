#!/usr/bin/env python3
"""
Generate HTML Report for Basis Arbitrage Strategy.

Creates a printable HTML report with:
- Strategy overview and configuration
- Performance metrics summary
- Equity curve chart (embedded base64)
- Trade analysis
- Trade log table

Usage:
    python scripts/generate_basis_report.py
    
    # Or with custom output path:
    python scripts/generate_basis_report.py --output my_report.html
"""

import argparse
import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from core.strategy import BacktestEngine, CostModel
from strategies.basis_arb import BasisArbitrage, BasisArbConfig, StrategyConfig


BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "output" / "reports"


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_equity_chart(equity_curve: pd.Series) -> str:
    """Create equity curve chart and return as base64."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), height_ratios=[3, 1])
    
    idx = equity_curve.index.tz_localize(None) if equity_curve.index.tz else equity_curve.index
    ax1.plot(idx, equity_curve.values, color="#2563eb", linewidth=1.2)
    ax1.fill_between(idx, equity_curve.iloc[0], equity_curve.values, alpha=0.1, color="#2563eb")
    ax1.set_ylabel("Portfolio Value ($)", fontsize=10)
    ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    ax2.fill_between(idx, 0, drawdown.values, color="#dc2626", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(min(drawdown.min() * 1.1, -0.5), 0.5)
    
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def create_trade_chart(trades: list) -> str:
    """Create trade P&L distribution chart and return as base64."""
    if not trades:
        return ""
    
    pnls = [t.net_pnl for t in trades]
    bars_held = [t.bars_held for t in trades]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    
    colors = ['#22c55e' if p > 0 else '#dc2626' for p in pnls]
    ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=np.mean(pnls), color='#2563eb', linewidth=1.5, linestyle='--', label=f'Avg: ${np.mean(pnls):,.0f}')
    ax1.set_xlabel("Trade #", fontsize=10)
    ax1.set_ylabel("Net P&L ($)", fontsize=10)
    ax1.set_title("Trade P&L Distribution", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(bars_held, bins=15, color='#8b5cf6', alpha=0.7, edgecolor='white')
    ax2.axvline(x=np.mean(bars_held), color='#2563eb', linewidth=1.5, linestyle='--', label=f'Avg: {np.mean(bars_held):.1f}')
    ax2.set_xlabel("Bars Held", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Holding Time Distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_html_report(
    result,
    arb_config: BasisArbConfig,
    cost_model: CostModel,
    output_path: Path,
    capital: float = 1_000_000,
) -> Path:
    """Generate the full HTML report."""
    
    equity_chart_b64 = create_equity_chart(result.equity_curve)
    trade_chart_b64 = create_trade_chart(result.trades)
    
    # Calculate simple annualized return (matching param sweep method)
    total_net = sum(t.net_pnl for t in result.trades)
    days = (result.end_time - result.start_time).days
    simple_annual_return = (total_net / capital) * 365 / days * 100 if days > 0 else 0
    
    # Trade statistics
    winning = [t for t in result.trades if t.net_pnl > 0]
    losing = [t for t in result.trades if t.net_pnl <= 0]
    
    avg_win = f"${np.mean([t.net_pnl for t in winning]):,.0f}" if winning else "N/A"
    avg_loss = f"${np.mean([t.net_pnl for t in losing]):,.0f}" if losing else "N/A"
    largest_win = f"${max(t.net_pnl for t in winning):,.0f}" if winning else "N/A"
    largest_loss = f"${min(t.net_pnl for t in losing):,.0f}" if losing else "N/A"
    avg_bars = f"{np.mean([t.bars_held for t in result.trades]):.1f}" if result.trades else "N/A"
    total_costs = f"${sum(t.costs for t in result.trades):,.0f}" if result.trades else "$0"
    
    # Generate trade log rows
    trade_rows = ""
    for trade in result.trades[-30:]:
        entry_str = trade.entry_time.strftime("%m-%d %H:%M") if trade.entry_time else "N/A"
        exit_str = trade.exit_time.strftime("%m-%d %H:%M") if trade.exit_time else "N/A"
        side_str = trade.side.value if trade.side else "N/A"
        pnl_class = "positive" if trade.net_pnl > 0 else "negative"
        
        trade_rows += f"""
        <tr>
            <td>{entry_str}</td>
            <td>{exit_str}</td>
            <td>{side_str}</td>
            <td>{trade.bars_held}</td>
            <td>${trade.gross_pnl:,.0f}</td>
            <td>${trade.costs:,.0f}</td>
            <td class="{pnl_class}">${trade.net_pnl:,.0f}</td>
            <td>{trade.exit_reason or ''}</td>
        </tr>"""
    
    # Return color
    return_class = "positive" if result.total_return_pct > 0 else "negative"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold Basis Arbitrage Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.5;
            color: #1f2937;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{ font-size: 24px; margin-bottom: 5px; }}
        h2 {{ 
            font-size: 16px; 
            background: #f3f4f6; 
            padding: 8px 12px; 
            margin: 20px 0 10px 0;
            border-left: 4px solid #2563eb;
        }}
        .subtitle {{ color: #6b7280; font-size: 12px; margin-bottom: 20px; }}
        .overview {{ color: #4b5563; font-size: 13px; margin-bottom: 15px; }}
        
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: #f9fafb; border-radius: 8px; padding: 15px; }}
        .card h3 {{ font-size: 13px; color: #6b7280; margin-bottom: 10px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th, td {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        td.right {{ text-align: right; }}
        
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0; }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 22px; font-weight: 700; }}
        .metric-label {{ font-size: 11px; color: #6b7280; }}
        
        .positive {{ color: #16a34a; }}
        .negative {{ color: #dc2626; }}
        
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        
        .trade-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; }}
        .trade-stat {{ text-align: center; padding: 10px; background: #f9fafb; border-radius: 6px; }}
        .trade-stat-value {{ font-size: 16px; font-weight: 600; }}
        .trade-stat-label {{ font-size: 10px; color: #6b7280; }}
        
        @media print {{
            body {{ padding: 10px; font-size: 11px; }}
            h1 {{ font-size: 20px; }}
            h2 {{ font-size: 14px; padding: 6px 10px; }}
            .metric-value {{ font-size: 18px; }}
            .no-print {{ display: none; }}
            .page-break {{ page-break-before: always; }}
        }}
    </style>
</head>
<body>
    <h1>Gold Basis Arbitrage Report</h1>
    <p class="subtitle">Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
    
    <h2>Strategy Overview</h2>
    <p class="overview">
        Trades the spread between CME Gold Futures (TradFi) and Hyperliquid PAXG Perpetuals (DeFi).
        Entry when |basis| > threshold. Exit on spread convergence or time expiry.
    </p>
    
    <div class="grid">
        <div class="card">
            <h3>Strategy Parameters</h3>
            <table>
                <tr><td>Entry Threshold</td><td class="right">{arb_config.threshold_bps:.0f} bps</td></tr>
                <tr><td>Take Profit (captured)</td><td class="right">{arb_config.take_profit_captured_bps:.0f} bps</td></tr>
                <tr><td>Max Hold Time</td><td class="right">{arb_config.max_hold_bars} bars (~{arb_config.max_hold_bars * 15 / 60:.1f}h)</td></tr>
                <tr><td>Half-Life</td><td class="right">{arb_config.half_life_bars:.1f} bars</td></tr>
                <tr><td>Max Trades/Day</td><td class="right">{arb_config.max_trades_per_day}</td></tr>
            </table>
        </div>
        <div class="card">
            <h3>Cost Model</h3>
            <table>
                <tr><td>Commission (round-trip)</td><td class="right">{cost_model.commission_bps:.1f} bps</td></tr>
                <tr><td>Slippage</td><td class="right">{cost_model.slippage_bps:.1f} bps</td></tr>
                <tr><td>Funding Rate</td><td class="right">{cost_model.funding_daily_bps:.1f} bps/day</td></tr>
            </table>
        </div>
    </div>
    
    <h2>Performance Summary</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value {return_class}">{simple_annual_return:+.1f}%</div>
            <div class="metric-label">Annual Return</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.sharpe_ratio:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.max_drawdown_pct:.2f}%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(result.trades)//2}</div>
            <div class="metric-label">Trade Pairs</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.win_rate:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.profit_factor:.2f}</div>
            <div class="metric-label">Profit Factor</div>
        </div>
    </div>
    
    <table>
        <tr>
            <td>Initial Capital</td><td class="right">${result.initial_capital:,.0f}</td>
            <td>Final Capital</td><td class="right">${result.final_capital:,.0f}</td>
            <td>Net P&L</td><td class="right {return_class}">${total_net:,.0f}</td>
        </tr>
    </table>
    
    <h2>Equity Curve & Drawdown</h2>
    <img src="data:image/png;base64,{equity_chart_b64}" alt="Equity Curve">
    
    <h2>Trade Analysis</h2>
    <img src="data:image/png;base64,{trade_chart_b64}" alt="Trade Distribution">
    
    <div class="trade-stats">
        <div class="trade-stat">
            <div class="trade-stat-value positive">{len(winning)}</div>
            <div class="trade-stat-label">Winning Trades</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value negative">{len(losing)}</div>
            <div class="trade-stat-label">Losing Trades</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value positive">{avg_win}</div>
            <div class="trade-stat-label">Avg Win</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value negative">{avg_loss}</div>
            <div class="trade-stat-label">Avg Loss</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value">{largest_win}</div>
            <div class="trade-stat-label">Largest Win</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value">{largest_loss}</div>
            <div class="trade-stat-label">Largest Loss</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value">{avg_bars}</div>
            <div class="trade-stat-label">Avg Bars Held</div>
        </div>
        <div class="trade-stat">
            <div class="trade-stat-value">{total_costs}</div>
            <div class="trade-stat-label">Total Costs</div>
        </div>
    </div>
    
    <h2>Trade Log (Last 30)</h2>
    <table>
        <thead>
            <tr>
                <th>Entry</th>
                <th>Exit</th>
                <th>Side</th>
                <th>Bars</th>
                <th>Gross P&L</th>
                <th>Costs</th>
                <th>Net P&L</th>
                <th>Exit Reason</th>
            </tr>
        </thead>
        <tbody>
            {trade_rows}
        </tbody>
    </table>
    
    <h2>Data Period</h2>
    <p class="overview">
        <strong>Start:</strong> {result.start_time}<br>
        <strong>End:</strong> {result.end_time}<br>
        <strong>Total Bars:</strong> {result.total_bars:,}
    </p>
</body>
</html>"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def run_backtest_and_report(output_path: Path = None) -> Path:
    """Run the basis arb backtest and generate a report."""
    
    # ========================================
    # CANONICAL BACKTEST CONFIGURATION
    # Produces: 35.8% annual, 2.13 Sharpe, 0.74% DD, 69% win rate
    # ========================================
    CAPITAL = 1_000_000
    POSITION_SIZE = 250_000  # $250k per leg (0.5x leverage)
    
    arb_config = BasisArbConfig(
        threshold_bps=80.0,
        take_profit_captured_bps=55.0,
        half_life_bars=2.5,
        max_half_lives=8.0,
        max_trades_per_day=100,
    )
    
    strategy_config = StrategyConfig(
        name="BasisArb Strategy",
        fixed_size=True,
        fixed_size_amount=float(POSITION_SIZE),
    )
    
    # Cost model matching param sweep (DEFAULT_COSTS)
    # Note: These are per-leg costs; spread trades pay 2x
    cost_model = CostModel(
        commission_bps=3.5,       # Taker fee per side
        slippage_bps=2.0,         # Conservative slippage
        funding_daily_bps=5.0,    # Daily funding rate
        bars_per_day=24,          # Hourly equivalent
    )
    
    # Use test data paths
    from core.strategy import DataSpec
    tradfi_spec = DataSpec(venue="test", market="futures", ticker="GC", interval="15m")
    defi_spec = DataSpec(venue="test", market="perp", ticker="PAXG", interval="15m")
    
    strategy = BasisArbitrage(
        config=strategy_config,
        arb_config=arb_config,
        tradfi_spec=tradfi_spec,
        defi_spec=defi_spec,
    )
    
    print("Running basis arbitrage backtest...")
    engine = BacktestEngine(verbose=True)
    result = engine.run(strategy=strategy, capital=CAPITAL, costs=cost_model)
    result.print_report()
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"basis_arb_report_{timestamp}.html"
    
    print(f"\nGenerating HTML report...")
    report_path = generate_html_report(result, arb_config, cost_model, output_path, capital=CAPITAL)
    print(f"Report saved: {report_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate Basis Arb HTML Report")
    parser.add_argument("--output", "-o", type=Path, help="Output HTML path")
    args = parser.parse_args()
    
    run_backtest_and_report(args.output)


if __name__ == "__main__":
    main()
