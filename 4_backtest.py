#!/usr/bin/env python3
"""
Stage 4: Backtest Simulation

Simulates threshold-based basis arbitrage strategy on historical data.
Supports scenario analysis with different capital, thresholds, and venues.

Usage:
    python 4_backtest.py                          # Default: Hyperliquid, $100K, >80 bps
    python 4_backtest.py --capital 1000000        # $1M capital
    python 4_backtest.py --threshold 50           # >50 bps entry
    python 4_backtest.py --venue aster            # Use Aster instead
    python 4_backtest.py --capture-rate 0.3       # Conservative 30% capture
    python 4_backtest.py --quantstats             # Generate QuantStats tearsheet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional
import json

# ============================================
# Configuration
# ============================================

DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("output/backtest")

# Default strategy parameters
DEFAULT_CONFIG = {
    "capital": 100_000,           # Starting capital ($)
    "threshold_bps": 80,          # Entry threshold (bps)
    "capture_rate": 0.50,         # % of basis captured
    "max_trades_per_day": 10,     # Cap on daily trades
    "volume_limit_pct": 0.02,     # Max 2% of daily volume
    "venue": "hyperliquid",       # Primary venue
}

# Cost structure (in bps)
COSTS = {
    "cme_commission_bps": 0.5,    # ~$2.50 per contract
    "defi_taker_fee_bps": 3.5,    # Taker fee
    "slippage_per_exec_bps": 2.0, # Per execution
    "funding_daily_bps": 5.0,     # Daily funding cost
}

# Margin requirements
MARGINS = {
    "cme_margin_pct": 0.10,       # 10% for CME
    "defi_margin_pct": 0.05,      # 5% for DeFi (20x leverage)
}

# Venue configurations
VENUES = {
    "hyperliquid": {
        "file": "gold_hl_merged_15m.parquet",
        "name": "Hyperliquid PAXG",
        "symbol": "PAXG",
    },
    "aster": {
        "file": "gold_merged_15m.parquet", 
        "name": "Aster XAUUSDT",
        "symbol": "XAUUSDT",
    },
}


# ============================================
# Data Classes
# ============================================

@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_basis_bps: float
    exit_basis_bps: Optional[float]
    position_size: float
    direction: str  # "long_defi" or "short_defi"
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    status: str = "open"  # "open", "closed", "stopped"


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: dict
    trades: List[Trade]
    daily_pnl: pd.Series
    equity_curve: pd.Series
    
    # Summary stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_gross_pnl: float = 0.0
    total_costs: float = 0.0
    total_net_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Annualized
    annual_return_pct: float = 0.0
    annual_volatility_pct: float = 0.0


# ============================================
# Backtester Class
# ============================================

class BasisBacktester:
    """
    Threshold-based basis arbitrage backtester.
    
    Strategy:
    1. Enter when |basis| > threshold
    2. Exit when basis reverts toward mean (capture_rate * entry_basis)
    3. Apply costs and funding
    """
    
    def __init__(
        self,
        capital: float = DEFAULT_CONFIG["capital"],
        threshold_bps: float = DEFAULT_CONFIG["threshold_bps"],
        capture_rate: float = DEFAULT_CONFIG["capture_rate"],
        max_trades_per_day: int = DEFAULT_CONFIG["max_trades_per_day"],
        volume_limit_pct: float = DEFAULT_CONFIG["volume_limit_pct"],
        venue: str = DEFAULT_CONFIG["venue"],
    ):
        self.capital = capital
        self.threshold_bps = threshold_bps
        self.capture_rate = capture_rate
        self.max_trades_per_day = max_trades_per_day
        self.volume_limit_pct = volume_limit_pct
        self.venue = venue
        
        # Calculate round-trip cost
        self.round_trip_cost_bps = (
            COSTS["cme_commission_bps"] * 2 +  # Open + close
            COSTS["defi_taker_fee_bps"] * 2 +  # Open + close
            COSTS["slippage_per_exec_bps"] * 4  # 4 executions total
        )
        
        # Total margin requirement
        self.total_margin_pct = MARGINS["cme_margin_pct"] + MARGINS["defi_margin_pct"]
        
    def load_data(self) -> pd.DataFrame:
        """Load merged data for the configured venue."""
        venue_config = VENUES.get(self.venue)
        if not venue_config:
            raise ValueError(f"Unknown venue: {self.venue}")
        
        filepath = DATA_DIR / venue_config["file"]
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        print(f"Loaded {len(df):,} bars from {venue_config['name']}")
        return df
    
    def calculate_position_size(self, df: pd.DataFrame) -> float:
        """
        Calculate position size based on:
        1. Capital / margin requirement (margin-constrained)
        2. 2% of daily volume (volume-constrained)
        
        Returns the smaller of the two.
        """
        # Margin-constrained position
        margin_position = self.capital / self.total_margin_pct
        
        # Volume-constrained position
        days = (df.index.max() - df.index.min()).days
        if days <= 0:
            days = 1
            
        # Calculate daily volume in USD
        if "defi_dollar_volume" in df.columns:
            daily_volume = df["defi_dollar_volume"].sum() / days
        else:
            avg_price = df["defi_close"].mean()
            daily_volume = (df["defi_volume"].sum() * avg_price) / days
        
        volume_position = daily_volume * self.volume_limit_pct
        
        # Use the smaller constraint
        position_size = min(margin_position, volume_position)
        limiting_factor = "margin" if margin_position < volume_position else "volume"
        
        print(f"Position sizing:")
        print(f"  Margin-constrained: ${margin_position:,.0f}")
        print(f"  Volume-constrained: ${volume_position:,.0f}")
        print(f"  Effective position: ${position_size:,.0f} ({limiting_factor}-limited)")
        
        return position_size
    
    def run(self) -> BacktestResult:
        """Execute the backtest."""
        print("=" * 60)
        print("Basis Arbitrage Backtest")
        print("=" * 60)
        print(f"\nVenue: {VENUES[self.venue]['name']}")
        print(f"Capital: ${self.capital:,.0f}")
        print(f"Threshold: >{self.threshold_bps} bps")
        print(f"Capture rate: {self.capture_rate*100:.0f}%")
        print(f"Round-trip cost: {self.round_trip_cost_bps:.1f} bps")
        print()
        
        # Load data
        df = self.load_data()
        
        # Filter to market hours only
        market_col = "tradfi_market_open" if "tradfi_market_open" in df.columns else "market_open"
        if market_col in df.columns:
            df = df[df[market_col]].copy()
            print(f"Filtered to market hours: {len(df):,} bars")
        
        # Calculate position size
        position_size = self.calculate_position_size(df)
        
        # Get data period info
        days = (df.index.max() - df.index.min()).days
        print(f"Data period: {days} days")
        print()
        
        # Run simulation
        trades = self._simulate_trades(df, position_size)
        
        # Calculate daily PnL and equity curve
        daily_pnl, equity_curve = self._calculate_equity_curve(trades, df)
        
        # Compile results
        result = self._compile_results(trades, daily_pnl, equity_curve, days)
        
        return result
    
    def _simulate_trades(self, df: pd.DataFrame, position_size: float) -> List[Trade]:
        """
        Simulate trades using threshold-based entry/exit.
        
        Entry: |basis| > threshold
        Exit: After capturing capture_rate of entry basis OR next bar (simplified)
        """
        trades = []
        current_trade: Optional[Trade] = None
        daily_trade_count = {}
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_date = timestamp.date()
            basis = row["basis_bps"]
            abs_basis = abs(basis)
            
            # Initialize daily counter
            if current_date not in daily_trade_count:
                daily_trade_count[current_date] = 0
            
            # Check for exit if in trade
            if current_trade is not None:
                # Simplified exit: assume we capture capture_rate of entry basis
                # In reality, this would check for mean reversion
                captured_bps = abs(current_trade.entry_basis_bps) * self.capture_rate
                
                # Close trade
                current_trade.exit_time = timestamp
                current_trade.exit_basis_bps = basis
                current_trade.gross_pnl = position_size * (captured_bps / 10000)
                current_trade.costs = position_size * (self.round_trip_cost_bps / 10000)
                current_trade.net_pnl = current_trade.gross_pnl - current_trade.costs
                current_trade.status = "closed"
                
                trades.append(current_trade)
                current_trade = None
                continue
            
            # Check for entry
            if abs_basis > self.threshold_bps:
                # Check daily trade limit
                if daily_trade_count[current_date] >= self.max_trades_per_day:
                    continue
                
                # Determine direction
                direction = "short_defi" if basis > 0 else "long_defi"
                
                # Open trade
                current_trade = Trade(
                    entry_time=timestamp,
                    exit_time=None,
                    entry_basis_bps=basis,
                    exit_basis_bps=None,
                    position_size=position_size,
                    direction=direction,
                )
                
                daily_trade_count[current_date] += 1
        
        # Close any remaining open trade
        if current_trade is not None:
            current_trade.status = "stopped"
            current_trade.exit_time = df.index[-1]
            current_trade.net_pnl = 0  # No profit on stopped trades
            trades.append(current_trade)
        
        return trades
    
    def _calculate_equity_curve(
        self, 
        trades: List[Trade], 
        df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate daily PnL and cumulative equity curve."""
        # Group trades by date
        daily_pnl = {}
        
        for trade in trades:
            if trade.status == "closed" and trade.exit_time:
                date = trade.exit_time.date()
                if date not in daily_pnl:
                    daily_pnl[date] = 0.0
                daily_pnl[date] += trade.net_pnl
        
        # Add funding costs for days with positions
        # (Simplified: assume always in position)
        trading_days = set()
        for trade in trades:
            if trade.entry_time and trade.exit_time:
                current = trade.entry_time.date()
                end = trade.exit_time.date()
                while current <= end:
                    trading_days.add(current)
                    current += timedelta(days=1)
        
        position_size = trades[0].position_size if trades else 0
        daily_funding = position_size * (COSTS["funding_daily_bps"] / 10000)
        
        for date in trading_days:
            if date not in daily_pnl:
                daily_pnl[date] = 0.0
            daily_pnl[date] -= daily_funding
        
        # Convert to Series
        daily_pnl_series = pd.Series(daily_pnl).sort_index()
        
        # Calculate equity curve
        equity_curve = self.capital + daily_pnl_series.cumsum()
        
        return daily_pnl_series, equity_curve
    
    def _compile_results(
        self,
        trades: List[Trade],
        daily_pnl: pd.Series,
        equity_curve: pd.Series,
        days: int,
    ) -> BacktestResult:
        """Compile all results into BacktestResult."""
        closed_trades = [t for t in trades if t.status == "closed"]
        
        # Basic stats
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.net_pnl > 0])
        losing_trades = len([t for t in closed_trades if t.net_pnl <= 0])
        
        total_gross = sum(t.gross_pnl for t in closed_trades)
        total_costs = sum(t.costs for t in closed_trades)
        total_net = sum(t.net_pnl for t in closed_trades)
        
        # Max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (annualized)
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            daily_returns = daily_pnl / self.capital
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Annualized returns
        if days > 0:
            annual_return = (total_net / self.capital) * (365 / days) * 100
            annual_vol = daily_pnl.std() * np.sqrt(252) / self.capital * 100 if len(daily_pnl) > 1 else 0
        else:
            annual_return = 0
            annual_vol = 0
        
        result = BacktestResult(
            config={
                "capital": self.capital,
                "threshold_bps": self.threshold_bps,
                "capture_rate": self.capture_rate,
                "venue": self.venue,
                "position_size": trades[0].position_size if trades else 0,
                "days": days,
            },
            trades=trades,
            daily_pnl=daily_pnl,
            equity_curve=equity_curve,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_gross_pnl=total_gross,
            total_costs=total_costs,
            total_net_pnl=total_net,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            annual_return_pct=annual_return,
            annual_volatility_pct=annual_vol,
        )
        
        return result
    
    def print_report(self, result: BacktestResult):
        """Print backtest summary report."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"\n--- Configuration ---")
        print(f"Venue:           {VENUES[self.venue]['name']}")
        print(f"Capital:         ${result.config['capital']:,.0f}")
        print(f"Position Size:   ${result.config['position_size']:,.0f}")
        print(f"Entry Threshold: >{result.config['threshold_bps']} bps")
        print(f"Capture Rate:    {result.config['capture_rate']*100:.0f}%")
        print(f"Data Period:     {result.config['days']} days")
        
        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:    {result.total_trades}")
        print(f"Winning:         {result.winning_trades} ({result.winning_trades/max(result.total_trades,1)*100:.1f}%)")
        print(f"Losing:          {result.losing_trades} ({result.losing_trades/max(result.total_trades,1)*100:.1f}%)")
        print(f"Trades/Day:      {result.total_trades/max(result.config['days'],1):.1f}")
        
        print(f"\n--- P&L Summary ---")
        print(f"Gross PnL:       ${result.total_gross_pnl:,.0f}")
        print(f"Total Costs:     ${result.total_costs:,.0f}")
        print(f"Net PnL:         ${result.total_net_pnl:,.0f}")
        print(f"Avg Trade:       ${result.total_net_pnl/max(result.total_trades,1):,.0f}")
        
        print(f"\n--- Performance Metrics ---")
        print(f"Return (period): {result.total_net_pnl/result.config['capital']*100:.1f}%")
        print(f"Return (annual): {result.annual_return_pct:.1f}%")
        print(f"Volatility:      {result.annual_volatility_pct:.1f}%")
        print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:    {result.max_drawdown*100:.1f}%")
        
        # Final equity
        final_equity = result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else self.capital
        print(f"\n--- Final Position ---")
        print(f"Starting:        ${result.config['capital']:,.0f}")
        print(f"Ending:          ${final_equity:,.0f}")
        print(f"Profit:          ${final_equity - result.config['capital']:,.0f}")
        
        print("\n" + "=" * 60)
    
    def save_results(self, result: BacktestResult, output_dir: Path = OUTPUT_DIR):
        """Save backtest results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades to CSV
        trades_data = []
        for t in result.trades:
            trades_data.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_basis_bps": t.entry_basis_bps,
                "exit_basis_bps": t.exit_basis_bps,
                "position_size": t.position_size,
                "direction": t.direction,
                "gross_pnl": t.gross_pnl,
                "costs": t.costs,
                "net_pnl": t.net_pnl,
                "status": t.status,
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_file = output_dir / f"backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Saved trades to: {trades_file}")
        
        # Save equity curve
        equity_file = output_dir / f"backtest_equity_{timestamp}.csv"
        result.equity_curve.to_csv(equity_file, header=["equity"])
        print(f"Saved equity curve to: {equity_file}")
        
        # Save summary as JSON
        summary = {
            "config": result.config,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "total_gross_pnl": result.total_gross_pnl,
            "total_costs": result.total_costs,
            "total_net_pnl": result.total_net_pnl,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "annual_return_pct": result.annual_return_pct,
            "annual_volatility_pct": result.annual_volatility_pct,
        }
        
        summary_file = output_dir / f"backtest_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary to: {summary_file}")
    
    def generate_quantstats_report(
        self, 
        result: BacktestResult, 
        output_dir: Path = OUTPUT_DIR,
        benchmark: str = "GLD",
        report_type: str = "basic"
    ):
        """
        Generate QuantStats report from backtest results.
        
        Args:
            result: BacktestResult from run()
            output_dir: Directory to save HTML report
            benchmark: Ticker for benchmark comparison (default: GLD)
            report_type: "basic" (console only), "metrics" (detailed console), "full" (HTML)
        """
        try:
            import quantstats as qs
        except ImportError:
            print("ERROR: quantstats not installed. Run: pip install quantstats")
            return None
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert equity curve to returns
        equity = result.equity_curve.copy()
        equity.index = pd.to_datetime(equity.index)
        returns = equity.pct_change().dropna()
        returns.name = "Basis Arb Strategy"
        
        # Extend pandas with quantstats methods
        qs.extend_pandas()
        
        print(f"\n{'='*60}")
        print(f"QUANTSTATS ANALYSIS")
        print(f"{'='*60}")
        print(f"Benchmark: {benchmark} (SPDR Gold Shares ETF)")
        print(f"Period: {returns.index.min().date()} to {returns.index.max().date()}")
        print(f"Trading days: {len(returns)}")
        
        if report_type == "basic":
            # Simple console output - key metrics only
            print(f"\n--- Key Metrics ---")
            print(f"CAGR:           {qs.stats.cagr(returns)*100:.1f}%")
            print(f"Sharpe:         {qs.stats.sharpe(returns):.2f}")
            print(f"Max Drawdown:   {qs.stats.max_drawdown(returns)*100:.1f}%")
            print(f"Win Rate:       {qs.stats.win_rate(returns)*100:.1f}%")
            print(f"Volatility:     {qs.stats.volatility(returns)*100:.1f}%")
            print(f"Avg Return:     {returns.mean()*100:.3f}%/day")
            return None
            
        elif report_type == "metrics":
            # Detailed console metrics table
            print(f"\n--- Detailed Metrics ---")
            qs.reports.metrics(returns, mode="full", display=True)
            return None
        
        elif report_type == "html":
            # Simple HTML report (basic metrics + plots)
            print(f"\nGenerating simple HTML report...")
            report_file = output_dir / f"quantstats_basic_{timestamp}.html"
            
            try:
                qs.reports.basic(
                    returns,
                    benchmark=benchmark,
                    output=str(report_file),
                    title=f"Basis Arbitrage - {VENUES[self.venue]['name']}",
                )
                print(f"Report saved: {report_file}")
            except Exception as e:
                print(f"Note: Benchmark fetch failed, generating without benchmark")
                qs.reports.basic(
                    returns,
                    output=str(report_file),
                    title=f"Basis Arbitrage - {VENUES[self.venue]['name']}",
                )
                print(f"Report saved: {report_file}")
            
            return report_file
            
        else:  # "full" - complete HTML tearsheet
            print(f"\nGenerating full HTML tearsheet...")
            report_file = output_dir / f"quantstats_full_{timestamp}.html"
            
            try:
                qs.reports.html(
                    returns,
                    benchmark=benchmark,
                    output=str(report_file),
                    title=f"Basis Arbitrage - {VENUES[self.venue]['name']}",
                )
                print(f"Report saved: {report_file}")
            except Exception as e:
                print(f"Note: Benchmark fetch failed, generating without benchmark")
                qs.reports.html(
                    returns,
                    output=str(report_file),
                    title=f"Basis Arbitrage - {VENUES[self.venue]['name']}",
                )
                print(f"Report saved: {report_file}")
            
            return report_file


# ============================================
# Scenario Analysis
# ============================================

def run_scenario_analysis(
    capital: float = 1_000_000,
    venue: str = "hyperliquid",
):
    """
    Run multiple scenarios with different thresholds and capture rates.
    """
    print("=" * 60)
    print("SCENARIO ANALYSIS")
    print("=" * 60)
    print(f"\nCapital: ${capital:,.0f}")
    print(f"Venue: {VENUES[venue]['name']}")
    print()
    
    results = []
    
    # Test different thresholds
    thresholds = [50, 80, 100, 150]
    capture_rates = [0.30, 0.50, 0.70]
    
    print(f"{'Threshold':<12} {'Capture':<10} {'Trades':<8} {'Net PnL':<12} {'Annual %':<10} {'Sharpe':<8}")
    print("-" * 60)
    
    for threshold in thresholds:
        for capture in capture_rates:
            bt = BasisBacktester(
                capital=capital,
                threshold_bps=threshold,
                capture_rate=capture,
                venue=venue,
            )
            
            # Suppress individual run output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                result = bt.run()
            finally:
                sys.stdout = old_stdout
            
            results.append({
                "threshold": threshold,
                "capture": capture,
                "trades": result.total_trades,
                "net_pnl": result.total_net_pnl,
                "annual_return": result.annual_return_pct,
                "sharpe": result.sharpe_ratio,
            })
            
            print(f">{threshold:<11} {capture*100:<9.0f}% {result.total_trades:<8} "
                  f"${result.total_net_pnl:<11,.0f} {result.annual_return_pct:<9.1f}% "
                  f"{result.sharpe_ratio:<.2f}")
    
    print()
    
    # Find best scenario
    best = max(results, key=lambda x: x["sharpe"] if x["sharpe"] > 0 else -999)
    print(f"Best Sharpe: >{best['threshold']} bps @ {best['capture']*100:.0f}% capture")
    print(f"  Annual Return: {best['annual_return']:.1f}%")
    print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
    
    return results


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Basis Arbitrage Backtester")
    parser.add_argument("--capital", type=float, default=DEFAULT_CONFIG["capital"],
                        help="Starting capital ($)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold_bps"],
                        help="Entry threshold (bps)")
    parser.add_argument("--capture-rate", type=float, default=DEFAULT_CONFIG["capture_rate"],
                        help="Capture rate (0.0-1.0)")
    parser.add_argument("--venue", type=str, default=DEFAULT_CONFIG["venue"],
                        choices=list(VENUES.keys()),
                        help="Trading venue")
    parser.add_argument("--scenario", action="store_true",
                        help="Run scenario analysis instead of single backtest")
    parser.add_argument("--save", action="store_true",
                        help="Save results to files")
    parser.add_argument("--quantstats", type=str, nargs="?", const="basic",
                        choices=["basic", "metrics", "html", "full"],
                        help="Generate QuantStats report: basic (console), metrics (table), html (simple HTML), full (complete tearsheet)")
    parser.add_argument("--benchmark", type=str, default="GLD",
                        help="Benchmark ticker for QuantStats (default: GLD = SPDR Gold ETF)")
    
    args = parser.parse_args()
    
    if args.scenario:
        run_scenario_analysis(capital=args.capital, venue=args.venue)
    else:
        bt = BasisBacktester(
            capital=args.capital,
            threshold_bps=args.threshold,
            capture_rate=args.capture_rate,
            venue=args.venue,
        )
        
        result = bt.run()
        bt.print_report(result)
        
        if args.save:
            bt.save_results(result)
        
        if args.quantstats:
            bt.generate_quantstats_report(result, benchmark=args.benchmark, report_type=args.quantstats)


if __name__ == "__main__":
    main()
