"""
Backtest engine for running strategies.

Handles:
- Data loading and indicator computation
- Bar-by-bar iteration with strategy callbacks
- Position management and P&L tracking
- Result aggregation and metrics calculation
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union
import pandas as pd
import numpy as np

from core.data.storage import load_ohlcv
from core.indicators import compute_indicators
from core.strategy.base import SingleAssetStrategy, MultiLeggedStrategy, DataSpec
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
        
        if isinstance(strategy, SingleAssetStrategy):
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
