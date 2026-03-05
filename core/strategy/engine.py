"""
Backtest engine for running strategies.

Handles:
- Data loading and indicator computation
- Bar-by-bar iteration with strategy callbacks
- Position management and P&L tracking
- Result aggregation and metrics calculation
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from core.data.storage import load_ohlcv
from core.indicators import compute_indicators
from core.strategy.base import SingleAssetStrategy, MultiLeggedStrategy, DataSpec
from core.strategy.basis_strategy import BasisStrategy, BasisPosition, BasisSignal
from core.strategy.data import (
    StrategyData,
    StrategyDataValidator,
    strategy_folder,
    _serialise_value,
)
from core.strategy.position import (
    Position, Trade, Signal, CostModel, Side, PositionStatus,
    DEFAULT_COSTS,
)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    # Configuration
    strategy_name: str = ""
    config: dict = field(default_factory=dict)
    
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_bars: int = 0
    
    # Capital
    initial_capital: float = 0.0
    final_capital: float = 0.0
    
    # Trades
    trades: list[Trade] = field(default_factory=list)
    
    # Time series
    equity_curve: Optional[pd.Series] = None
    drawdown_curve: Optional[pd.Series] = None
    
    # Metrics (computed)
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    
    def compute_metrics(self):
        """Calculate performance metrics from equity curve and trades."""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return
        
        # Basic returns
        self.total_return_pct = (self.final_capital / self.initial_capital - 1) * 100
        
        # Annualized return (assuming hourly bars)
        days = self.total_bars / 24
        if days > 0:
            self.annual_return_pct = ((1 + self.total_return_pct / 100) ** (365 / days) - 1) * 100
        
        # Sharpe ratio (annualized, assuming hourly bars)
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            hourly_sharpe = returns.mean() / returns.std()
            self.sharpe_ratio = hourly_sharpe * np.sqrt(24 * 365)  # Annualize
        
        # Drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        self.max_drawdown_pct = abs(drawdown.min()) * 100
        self.drawdown_curve = drawdown
        
        # Trade metrics
        self.total_trades = len(self.trades)
        if self.total_trades > 0:
            winning = [t for t in self.trades if t.net_pnl > 0]
            losing = [t for t in self.trades if t.net_pnl <= 0]
            
            self.win_rate = len(winning) / self.total_trades * 100
            
            total_profit = sum(t.net_pnl for t in winning)
            total_loss = abs(sum(t.net_pnl for t in losing))
            
            if total_loss > 0:
                self.profit_factor = total_profit / total_loss
            elif total_profit > 0:
                self.profit_factor = float('inf')
    
    def summary(self) -> dict:
        """Return summary dict of key metrics."""
        return {
            "strategy": self.strategy_name,
            "period": f"{self.start_time} to {self.end_time}",
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return_pct": round(self.total_return_pct, 2),
            "annual_return_pct": round(self.annual_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float('inf') else "∞",
        }
    
    def print_report(self):
        """Print a formatted report."""
        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS: {self.strategy_name}")
        print("=" * 60)
        
        print(f"\n--- Configuration ---")
        print(f"Period:          {self.start_time.date()} to {self.end_time.date()}")
        print(f"Bars:            {self.total_bars:,}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        
        print(f"\n--- Performance ---")
        print(f"Final Capital:   ${self.final_capital:,.0f}")
        print(f"Total Return:    {self.total_return_pct:+.1f}%")
        print(f"Annual Return:   {self.annual_return_pct:+.1f}%")
        print(f"Sharpe Ratio:    {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown:    {self.max_drawdown_pct:.1f}%")
        
        print(f"\n--- Trades ---")
        print(f"Total Trades:    {self.total_trades}")
        print(f"Win Rate:        {self.win_rate:.1f}%")
        pf = f"{self.profit_factor:.2f}" if self.profit_factor != float('inf') else "∞"
        print(f"Profit Factor:   {pf}")
        
        if self.trades:
            avg_trade = sum(t.net_pnl for t in self.trades) / len(self.trades)
            print(f"Avg Trade P&L:   ${avg_trade:,.0f}")
        
        print("=" * 60)


class BacktestEngine:
    """
    Engine for running backtests.
    
    Supports both SingleAssetStrategy and MultiLeggedStrategy.
    
    Usage:
        engine = BacktestEngine()
        result = engine.run(
            strategy=MyStrategy(),
            data=df,  # or load from storage
            capital=100_000,
            costs=DEFAULT_COSTS,
        )
        result.print_report()
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def run(
        self,
        strategy: Union[SingleAssetStrategy, MultiLeggedStrategy],
        data: Optional[pd.DataFrame] = None,
        capital: float = 100_000,
        costs: Optional[CostModel] = None,
        # For loading data automatically
        venue: Optional[str] = None,
        market: Optional[str] = None,
        ticker: Optional[str] = None,
        interval: Optional[str] = None,
        years: Optional[list[int]] = None,
    ) -> BacktestResult:
        """
        Run a backtest.
        
        Args:
            strategy: Strategy instance to backtest
            data: DataFrame with OHLCV data (optional if loading from storage)
            capital: Starting capital
            costs: Cost model (default: DEFAULT_COSTS)
            venue, market, ticker, interval, years: For loading data from storage
            
        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        costs = costs or DEFAULT_COSTS
        
        if isinstance(strategy, BasisStrategy):
            return self._run_basis(strategy, capital, costs)
        elif isinstance(strategy, SingleAssetStrategy):
            # Use v2 path if strategy defines data_spec()
            if strategy.data_spec() is not None:
                return self._run_single_asset_v2(strategy, capital, costs)
            return self._run_single_asset(
                strategy, data, capital, costs,
                venue, market, ticker, interval, years
            )
        elif isinstance(strategy, MultiLeggedStrategy):
            return self._run_multi_legged(strategy, capital, costs)
        else:
            raise TypeError(f"Unknown strategy type: {type(strategy)}")
    
    def _run_single_asset(
        self,
        strategy: SingleAssetStrategy,
        data: Optional[pd.DataFrame],
        capital: float,
        costs: CostModel,
        venue: Optional[str],
        market: Optional[str],
        ticker: Optional[str],
        interval: Optional[str],
        years: Optional[list[int]],
    ) -> BacktestResult:
        """Run backtest for single-asset strategy."""
        
        # Load data if not provided
        if data is None:
            if not all([venue, market, ticker, interval]):
                raise ValueError("Must provide data or venue/market/ticker/interval")
            data = load_ohlcv(venue, market, ticker, interval, years=years)
            if self.verbose:
                print(f"Loaded {len(data):,} bars from {venue}/{market}/{ticker}/{interval}")
        
        # Compute indicators
        indicators = strategy.required_indicators()
        if indicators:
            data = compute_indicators(data, indicators)
            if self.verbose:
                print(f"Computed {len(indicators)} indicator(s)")
        
        # Initialize result
        result = BacktestResult(
            strategy_name=strategy.name,
            config={"capital": capital, "costs": costs.__dict__},
            start_time=data.index[0],
            end_time=data.index[-1],
            total_bars=len(data),
            initial_capital=capital,
        )
        
        # Initialize state
        current_capital = capital
        position: Optional[Position] = None
        trades: list[Trade] = []
        equity_curve = []
        entry_bar_idx = 0
        
        # Call strategy start hook
        strategy.on_start(data)
        
        # Main backtest loop
        for idx in range(len(data)):
            bar = data.iloc[idx]
            timestamp = data.index[idx]
            price = bar["close"]
            
            # Update position mark-to-market
            if position:
                position.update_price(price)
            
            # Get signal from strategy
            signal = strategy.on_bar(idx, data, current_capital, position)
            
            # Process signal
            if signal.action == "buy" and position is None:
                # Calculate position size
                if strategy.config.fixed_size and strategy.config.fixed_size_amount > 0:
                    size = strategy.config.fixed_size_amount * signal.size
                else:
                    size = current_capital * signal.size * strategy.config.max_position_pct
                
                position = Position(
                    symbol=ticker or "UNKNOWN",
                    side=Side.LONG,
                    entry_time=timestamp,
                    entry_price=price,
                    size=size,
                    entry_reason=signal.reason,
                )
                entry_bar_idx = idx
                current_capital -= costs.round_trip_cost(size) / 2  # Entry cost
                
            elif signal.action == "sell" and position is None:
                # Calculate position size
                if strategy.config.fixed_size and strategy.config.fixed_size_amount > 0:
                    size = strategy.config.fixed_size_amount * signal.size
                else:
                    size = current_capital * signal.size * strategy.config.max_position_pct
                
                position = Position(
                    symbol=ticker or "UNKNOWN",
                    side=Side.SHORT,
                    entry_time=timestamp,
                    entry_price=price,
                    size=size,
                    entry_reason=signal.reason,
                )
                entry_bar_idx = idx
                current_capital -= costs.round_trip_cost(size) / 2
                
            elif signal.action == "close" and position is not None:
                # Close position
                bars_held = idx - entry_bar_idx
                trade = position.close(price, timestamp)
                trade.bars_held = bars_held
                trade.exit_reason = signal.reason
                trade.costs = costs.total_cost(position.size, bars_held)
                trade.net_pnl = trade.gross_pnl - trade.costs
                
                current_capital += trade.net_pnl
                trades.append(trade)
                position = None
            
            # Record equity
            equity = current_capital
            if position:
                equity += position.unrealized_pnl
            equity_curve.append(equity)
        
        # Close any open position at end
        if position:
            idx = len(data) - 1
            bars_held = idx - entry_bar_idx
            trade = position.close(data.iloc[-1]["close"], data.index[-1])
            trade.bars_held = bars_held
            trade.exit_reason = "end_of_data"
            trade.costs = costs.total_cost(position.size, bars_held)
            trade.net_pnl = trade.gross_pnl - trade.costs
            
            current_capital += trade.net_pnl
            trades.append(trade)
            equity_curve[-1] = current_capital
        
        # Call strategy end hook
        strategy.on_end(data)
        
        # Build result
        result.trades = trades
        result.final_capital = current_capital
        result.equity_curve = pd.Series(equity_curve, index=data.index)
        result.compute_metrics()
        
        return result
    
    # ------------------------------------------------------------------
    # V2: Multi-interval single-asset engine
    # ------------------------------------------------------------------
    
    def _run_single_asset_v2(
        self,
        strategy: SingleAssetStrategy,
        capital: float,
        costs: CostModel,
    ) -> BacktestResult:
        """
        Run backtest for a strategy that defines data_spec().
        
        Iterates 1m bars with look-ahead-safe multi-interval data access.
        Records bar-level state, captures decision context on trades.
        Saves results to the strategy output folder.
        """
        spec = strategy.data_spec()
        folder = strategy_folder(strategy.name)
        
        # Validate data exists
        errors = StrategyDataValidator.validate(strategy.name, spec)
        if errors:
            raise ValueError(
                f"Strategy data validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
        
        # Load multi-interval data
        data = StrategyData.from_strategy_folder(folder, spec)
        if self.verbose:
            for iv in spec.intervals:
                n = len(data._frames[iv])
                print(f"Loaded {iv}: {n:,} bars")
        
        # Get the 1m DataFrame — this drives the iteration
        df_1m = data._frames["1m"]
        ticker = spec.ticker
        
        # Costs: override bars_per_day for 1m bars
        costs_1m = CostModel(
            commission_bps=costs.commission_bps,
            slippage_bps=costs.slippage_bps,
            funding_daily_bps=costs.funding_daily_bps,
            bars_per_day=1440,
        )
        
        # Initialize result
        result = BacktestResult(
            strategy_name=strategy.name,
            config={
                "capital": capital,
                "costs": costs_1m.__dict__,
                "spec": spec.to_dict(),
            },
            start_time=df_1m.index[0],
            end_time=df_1m.index[-1],
            total_bars=len(df_1m),
            initial_capital=capital,
        )
        
        # Initialize state
        balance = capital  # Realised capital — changes only on trade close
        position: Optional[Position] = None
        trades: list[Trade] = []
        entry_bar_idx = 0
        
        # Bar-level recording
        bar_records: list[dict] = []
        
        if self.verbose:
            print(f"Running {strategy.name} on {len(df_1m):,} 1m bars...")
        
        # Main backtest loop — iterate 1m bars
        for idx in range(len(df_1m)):
            timestamp = df_1m.index[idx]
            price = df_1m.iloc[idx]["close"]
            
            # Update position mark-to-market
            if position:
                position.update_price(price)
            
            # Get signal from strategy (new signature: timestamp + StrategyData)
            signal = strategy.on_bar(timestamp, data, balance, position)
            
            # Process signal
            if signal.action == "buy" and position is None:
                # Position size
                if strategy.config.fixed_size and strategy.config.fixed_size_amount > 0:
                    size = strategy.config.fixed_size_amount * signal.size
                else:
                    size = balance * signal.size * strategy.config.max_position_pct
                
                position = Position(
                    symbol=ticker,
                    side=Side.LONG,
                    entry_time=timestamp,
                    entry_price=price,
                    size=size,
                    entry_reason=signal.reason,
                    metadata={"entry_context": data.snapshot(timestamp)},
                )
                entry_bar_idx = idx
                
            elif signal.action == "sell" and position is None:
                if strategy.config.fixed_size and strategy.config.fixed_size_amount > 0:
                    size = strategy.config.fixed_size_amount * signal.size
                else:
                    size = balance * signal.size * strategy.config.max_position_pct
                
                position = Position(
                    symbol=ticker,
                    side=Side.SHORT,
                    entry_time=timestamp,
                    entry_price=price,
                    size=size,
                    entry_reason=signal.reason,
                    metadata={"entry_context": data.snapshot(timestamp)},
                )
                entry_bar_idx = idx
                
            elif signal.action == "close" and position is not None:
                bars_held = idx - entry_bar_idx
                trade = position.close(price, timestamp)
                trade.bars_held = bars_held
                trade.exit_reason = signal.reason
                trade.costs = costs_1m.total_cost(position.size, bars_held)
                trade.net_pnl = trade.gross_pnl - trade.costs
                
                # Decision context: merge entry + exit snapshots
                trade.metadata = {
                    "entry_context": position.metadata.get("entry_context", {}),
                    "exit_context": data.snapshot(timestamp),
                }
                
                balance += trade.net_pnl
                trades.append(trade)
                position = None
            
            # Compute NAV and drawdown
            unrealized = position.unrealized_pnl if position else 0.0
            nav = balance + unrealized
            
            # Record bar state
            bar_records.append({
                "timestamp": timestamp,
                "close": price,
                "balance": balance,
                "nav": nav,
                "position_side": position.side.value if position else "flat",
                "position_size": position.size if position else 0.0,
                "position_pnl": unrealized,
                "position_pnl_pct": (unrealized / position.size * 100) if position and position.size > 0 else 0.0,
                "signal": signal.action if signal.action != "hold" else "",
            })
        
        # Close any open position at end
        if position:
            idx = len(df_1m) - 1
            price = df_1m.iloc[-1]["close"]
            bars_held = idx - entry_bar_idx
            trade = position.close(price, df_1m.index[-1])
            trade.bars_held = bars_held
            trade.exit_reason = "end_of_data"
            trade.costs = costs_1m.total_cost(position.size, bars_held)
            trade.net_pnl = trade.gross_pnl - trade.costs
            trade.metadata = {
                "entry_context": position.metadata.get("entry_context", {}),
                "exit_context": data.snapshot(df_1m.index[-1]),
            }
            
            balance += trade.net_pnl
            trades.append(trade)
            
            # Update last bar record
            bar_records[-1]["balance"] = balance
            bar_records[-1]["nav"] = balance
            bar_records[-1]["position_side"] = "flat"
            bar_records[-1]["position_size"] = 0.0
            bar_records[-1]["position_pnl"] = 0.0
            bar_records[-1]["position_pnl_pct"] = 0.0
        
        # Build bar-level DataFrame
        bars_df = pd.DataFrame(bar_records)
        bars_df = bars_df.set_index("timestamp")
        
        # Compute drawdown from peak NAV
        peak_nav = bars_df["nav"].cummax()
        bars_df["drawdown_pct"] = ((bars_df["nav"] - peak_nav) / peak_nav * 100)
        
        # Build equity curve for BacktestResult compatibility
        equity_curve = bars_df["nav"]
        
        # Build result
        result.trades = trades
        result.final_capital = balance
        result.equity_curve = equity_curve
        result.compute_metrics()
        
        # Save results to strategy folder
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = folder / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save bar-level parquet
        bars_path = results_dir / f"{run_id}_bars.parquet"
        bars_df.to_parquet(bars_path)
        
        # Save trades parquet
        if trades:
            trades_data = []
            for t in trades:
                trades_data.append({
                    "symbol": t.symbol,
                    "side": t.side.value if t.side else "",
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "gross_pnl": t.gross_pnl,
                    "costs": t.costs,
                    "net_pnl": t.net_pnl,
                    "bars_held": t.bars_held,
                    "entry_reason": t.entry_reason,
                    "exit_reason": t.exit_reason,
                    "context": json.dumps(t.metadata, default=str),
                })
            trades_df = pd.DataFrame(trades_data)
            trades_path = results_dir / f"{run_id}_trades.parquet"
            trades_df.to_parquet(trades_path)
        
        # Save metadata JSON
        meta = {
            "run_id": run_id,
            "strategy_name": strategy.name,
            "spec": spec.to_dict(),
            "config": result.config,
            "start_time": str(result.start_time),
            "end_time": str(result.end_time),
            "total_bars": result.total_bars,
            "initial_capital": capital,
            "final_capital": balance,
            "metrics": result.summary(),
            "total_trades": len(trades),
        }
        meta_path = results_dir / f"{run_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\nResults saved to {results_dir}/")
            print(f"  {run_id}_bars.parquet ({len(bars_df):,} bars)")
            if trades:
                print(f"  {run_id}_trades.parquet ({len(trades)} trades)")
            print(f"  {run_id}_meta.json")
            result.print_report()
        
        return result
    
    def _run_multi_legged(
        self,
        strategy: MultiLeggedStrategy,
        capital: float,
        costs: CostModel,
    ) -> BacktestResult:
        """Run backtest for multi-legged strategy."""
        
        # Load data for each leg
        data_specs = strategy.required_data()
        data: dict[str, pd.DataFrame] = {}
        
        for leg_name, spec in data_specs.items():
            df = load_ohlcv(spec.venue, spec.market, spec.ticker, spec.interval)
            if self.verbose:
                print(f"Loaded {len(df):,} bars for {leg_name} ({spec})")
            data[leg_name] = df
        
        # Compute indicators for each leg
        indicators = strategy.required_indicators()
        for leg_name, ind_list in indicators.items():
            if ind_list and leg_name in data:
                data[leg_name] = compute_indicators(data[leg_name], ind_list)
        
        # Align data to common timestamps
        common_index = data[list(data.keys())[0]].index
        for leg_name, df in data.items():
            common_index = common_index.intersection(df.index)
        
        for leg_name in data:
            data[leg_name] = data[leg_name].loc[common_index]
        
        if self.verbose:
            print(f"Aligned to {len(common_index):,} common bars")
        
        # Initialize result
        first_leg = list(data.keys())[0]
        result = BacktestResult(
            strategy_name=strategy.name,
            config={"capital": capital, "costs": costs.__dict__},
            start_time=common_index[0],
            end_time=common_index[-1],
            total_bars=len(common_index),
            initial_capital=capital,
        )
        
        # Initialize state
        current_capital = capital
        positions: dict[str, Optional[Position]] = {leg: None for leg in data_specs}
        entry_bar_idx: dict[str, int] = {}
        trades: list[Trade] = []
        equity_curve = []
        
        # Call strategy start hook
        strategy.on_start(data)
        
        # Main backtest loop
        for idx in range(len(common_index)):
            timestamp = common_index[idx]
            
            # Check if all markets are open (skip if any closed)
            all_markets_open = all(
                data[leg].iloc[idx].get("market_open", True)
                for leg in data_specs
            )
            
            # Check if any market is near close (force close positions)
            any_near_close = any(
                data[leg].iloc[idx].get("near_close", False)
                for leg in data_specs
            )
            
            # Update positions mark-to-market
            for leg_name, pos in positions.items():
                if pos:
                    price = data[leg_name].iloc[idx]["close"]
                    pos.update_price(price)
            
            # Force close all positions if market closing soon
            if any_near_close and any(positions.values()):
                signals = {
                    leg: Signal.close(reason="market_closing")
                    for leg, pos in positions.items() if pos
                }
            # Skip trading if any market is closed
            elif not all_markets_open:
                signals = {}  # Hold, don't trade
            else:
                # Get signals from strategy
                signals = strategy.on_bar(idx, data, current_capital, positions)
            
            # Process signals for each leg
            for leg_name, signal in signals.items():
                pos = positions.get(leg_name)
                price = data[leg_name].iloc[idx]["close"]
                spec = data_specs[leg_name]
                
                if signal.action == "buy" and pos is None:
                    # Calculate position size
                    if strategy.config.fixed_size and strategy.config.fixed_size_amount > 0:
                        size = strategy.config.fixed_size_amount * signal.size
                    else:
                        size = current_capital * signal.size * strategy.config.max_position_pct
                    
                    positions[leg_name] = Position(
                        symbol=spec.ticker,
                        side=Side.LONG,
                        leg=leg_name,
                        entry_time=timestamp,
                        entry_price=price,
                        size=size,
                        entry_reason=signal.reason,
                    )
                    entry_bar_idx[leg_name] = idx
                    # Note: costs are accounted for in trade.net_pnl on close
                    
                elif signal.action == "sell" and pos is None:
                    # Calculate position size
                    if strategy.config.fixed_size and strategy.config.fixed_size_amount > 0:
                        size = strategy.config.fixed_size_amount * signal.size
                    else:
                        size = current_capital * signal.size * strategy.config.max_position_pct
                    
                    positions[leg_name] = Position(
                        symbol=spec.ticker,
                        side=Side.SHORT,
                        leg=leg_name,
                        entry_time=timestamp,
                        entry_price=price,
                        size=size,
                        entry_reason=signal.reason,
                    )
                    entry_bar_idx[leg_name] = idx
                    # Note: costs are accounted for in trade.net_pnl on close
                    
                elif signal.action == "close" and pos is not None:
                    bars_held = idx - entry_bar_idx.get(leg_name, idx)
                    trade = pos.close(price, timestamp)
                    trade.bars_held = bars_held
                    trade.exit_reason = signal.reason
                    
                    # Spread P&L mode: calculate based on basis convergence
                    if strategy.config.spread_pnl_mode:
                        # Get entry and exit basis from strategy
                        entry_basis = strategy.get_entry_basis()
                        exit_basis = strategy.calculate_basis(data, idx, "tradfi", "defi")
                        
                        # Calculate spread P&L (split between legs)
                        spread_gross_pnl = strategy.calculate_spread_pnl(
                            entry_basis, exit_basis, pos.size
                        )
                        # Each leg gets half the spread P&L
                        trade.gross_pnl = spread_gross_pnl / 2
                        # Costs also split between legs (it's one spread trade, not two)
                        trade.costs = costs.total_cost(pos.size, bars_held) / 2
                    else:
                        trade.costs = costs.total_cost(pos.size, bars_held)
                    
                    trade.net_pnl = trade.gross_pnl - trade.costs
                    
                    current_capital += trade.net_pnl
                    trades.append(trade)
                    positions[leg_name] = None
            
            # Record equity
            equity = current_capital
            for pos in positions.values():
                if pos:
                    equity += pos.unrealized_pnl
            equity_curve.append(equity)
        
        # Close any open positions at end
        for leg_name, pos in positions.items():
            if pos:
                idx = len(common_index) - 1
                price = data[leg_name].iloc[-1]["close"]
                bars_held = idx - entry_bar_idx.get(leg_name, idx)
                trade = pos.close(price, common_index[-1])
                trade.bars_held = bars_held
                trade.exit_reason = "end_of_data"
                trade.costs = costs.total_cost(pos.size, bars_held)
                trade.net_pnl = trade.gross_pnl - trade.costs
                
                current_capital += trade.net_pnl
                trades.append(trade)
        
        # Update final equity
        equity_curve[-1] = current_capital
        
        # Call strategy end hook
        strategy.on_end(data)
        
        # Build result
        result.trades = trades
        result.final_capital = current_capital
        result.equity_curve = pd.Series(equity_curve, index=common_index)
        result.compute_metrics()
        
        return result
    
    def _run_basis(
        self,
        strategy: BasisStrategy,
        capital: float,
        costs: CostModel,
    ) -> BacktestResult:
        """Run backtest for basis strategy using pre-computed basis files."""
        
        # Load basis data
        data = strategy.load_data()
        if self.verbose:
            print(f"Loaded {len(data):,} bars from basis file")
            print(f"Quote venue: {strategy.quote_venue}")
        
        # Compute additional indicators if needed
        indicators = strategy.required_indicators()
        if indicators:
            data = compute_indicators(data, indicators)
            if self.verbose:
                print(f"Computed {len(indicators)} indicator(s)")
        
        # Initialize result
        result = BacktestResult(
            strategy_name=strategy.name,
            config={
                "capital": capital,
                "costs": costs.__dict__,
                "base_ticker": strategy.config.base_ticker,
                "quote_venue": strategy.quote_venue,
            },
            start_time=data.index[0],
            end_time=data.index[-1],
            total_bars=len(data),
            initial_capital=capital,
        )
        
        # Initialize state
        current_capital = capital
        position: Optional[BasisPosition] = None
        trades: list[Trade] = []
        equity_curve = []
        
        # Call strategy start hook
        strategy.on_start(data)
        
        # Main backtest loop
        for idx in range(len(data)):
            timestamp = data.index[idx]
            base_price = data["base_price"].iloc[idx]
            quote_price = data[f"{strategy.quote_venue}_price"].iloc[idx]
            basis_bps = data[f"{strategy.quote_venue}_basis_bps"].iloc[idx]
            
            # Skip if data quality is not ok
            data_quality = data["data_quality"].iloc[idx]
            if data_quality != "ok" and position is None:
                equity_curve.append(current_capital)
                continue
            
            # Get signal from strategy
            signal = strategy.on_bar(idx, data, current_capital, position)
            
            # Process signal
            if signal.action == "open_long" and position is None:
                size = current_capital * signal.size * strategy.config.position_size
                position = BasisPosition(
                    direction=1,
                    entry_bar=idx,
                    entry_basis_bps=basis_bps,
                    entry_base_price=base_price,
                    entry_quote_price=quote_price,
                    size=size,
                    reason=signal.reason,
                )
                if self.verbose:
                    print(f"[{timestamp}] OPEN LONG @ {basis_bps:.0f} bps, size=${size:,.0f}")
                    
            elif signal.action == "open_short" and position is None:
                size = current_capital * signal.size * strategy.config.position_size
                position = BasisPosition(
                    direction=-1,
                    entry_bar=idx,
                    entry_basis_bps=basis_bps,
                    entry_base_price=base_price,
                    entry_quote_price=quote_price,
                    size=size,
                    reason=signal.reason,
                )
                if self.verbose:
                    print(f"[{timestamp}] OPEN SHORT @ {basis_bps:.0f} bps, size=${size:,.0f}")
                    
            elif signal.action == "close" and position is not None:
                # Calculate P&L from basis convergence
                gross_pnl = position.unrealized_pnl(basis_bps)
                bars_held = position.bars_held(idx)
                trade_costs = costs.total_cost(position.size, bars_held)
                net_pnl = gross_pnl - trade_costs
                
                # Create trade record
                trade = Trade(
                    symbol=f"{strategy.config.base_ticker}/{strategy.quote_venue}",
                    side=Side.LONG if position.direction == 1 else Side.SHORT,
                    entry_time=data.index[position.entry_bar],
                    entry_price=position.entry_basis_bps,
                    exit_time=timestamp,
                    exit_price=basis_bps,
                    size=position.size,
                    gross_pnl=gross_pnl,
                    costs=trade_costs,
                    net_pnl=net_pnl,
                    bars_held=bars_held,
                    entry_reason=position.reason,
                    exit_reason=signal.reason,
                )
                
                current_capital += net_pnl
                trades.append(trade)
                
                if self.verbose:
                    print(f"[{timestamp}] CLOSE @ {basis_bps:.0f} bps, P&L=${net_pnl:,.0f} ({bars_held} bars)")
                
                position = None
            
            # Record equity
            equity = current_capital
            if position:
                equity += position.unrealized_pnl(basis_bps)
            equity_curve.append(equity)
        
        # Close any open position at end
        if position:
            idx = len(data) - 1
            basis_bps = data[f"{strategy.quote_venue}_basis_bps"].iloc[-1]
            gross_pnl = position.unrealized_pnl(basis_bps)
            bars_held = position.bars_held(idx)
            trade_costs = costs.total_cost(position.size, bars_held)
            net_pnl = gross_pnl - trade_costs
            
            trade = Trade(
                symbol=f"{strategy.config.base_ticker}/{strategy.quote_venue}",
                side=Side.LONG if position.direction == 1 else Side.SHORT,
                entry_time=data.index[position.entry_bar],
                entry_price=position.entry_basis_bps,
                exit_time=data.index[-1],
                exit_price=basis_bps,
                size=position.size,
                gross_pnl=gross_pnl,
                costs=trade_costs,
                net_pnl=net_pnl,
                bars_held=bars_held,
                entry_reason=position.reason,
                exit_reason="end_of_data",
            )
            
            current_capital += net_pnl
            trades.append(trade)
            equity_curve[-1] = current_capital
        
        # Call strategy end hook
        strategy.on_end(data)
        
        # Build result
        result.trades = trades
        result.final_capital = current_capital
        result.equity_curve = pd.Series(equity_curve, index=data.index)
        result.compute_metrics()
        
        return result
