"""
Basis Arbitrage Strategy - MultiLeggedStrategy Implementation.

Trades the spread between TradFi (CME Gold) and DeFi (Hyperliquid PAXG).
Uses standard OHLCV data sources with automatic basis calculation.

Entry: |basis| > threshold_bps
Exit: Mean reversion (|basis| < take_profit_bps) or configurable conditions
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from core.strategy import (
    MultiLeggedStrategy,
    DataSpec,
    Signal,
    Position,
    StrategyConfig,
    CostModel,
)


@dataclass
class BasisArbConfig:
    """Configuration for Basis Arbitrage strategy.
    
    Basis arb locks in the spread at entry - there's no directional risk.
    Exit is based on:
    1. Convergence: spread narrowed (take profit)
    2. Time: held too long (opportunity cost / funding)
    
    No traditional stop-loss needed - spread widening just means wait longer.
    """
    # Entry threshold
    threshold_bps: float = 80.0
    
    # Exit: convergence threshold (captured bps from entry)
    take_profit_captured_bps: float = 40.0
    
    # Exit: time-based (half-life multiples)
    half_life_bars: float = 2.3      # Mean reversion half-life (35 min / 15 min)
    max_half_lives: float = 4.0      # Exit after this many half-lives
    
    # Trade limits
    max_trades_per_day: int = 16
    
    # Position sizing (as fraction of capital per leg)
    position_size_per_leg: float = 0.5
    
    @property
    def max_hold_bars(self) -> int:
        """Maximum bars to hold based on half-life."""
        return int(self.half_life_bars * self.max_half_lives)


class BasisArbitrage(MultiLeggedStrategy):
    """
    Gold Basis Arbitrage using MultiLeggedStrategy.
    
    Trades the spread between:
    - TradFi: CME Gold Futures (Yahoo GC=F)
    - DeFi: Hyperliquid PAXG Perpetual
    
    Usage:
        config = StrategyConfig(name="BasisArb", capital=1_000_000)
        arb_config = BasisArbConfig(threshold_bps=50)
        strategy = BasisArbitrage(config=config, arb_config=arb_config)
        
        engine = BacktestEngine()
        result = engine.run(strategy=strategy, capital=1_000_000)
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        arb_config: Optional[BasisArbConfig] = None,
        tradfi_spec: Optional[DataSpec] = None,
        defi_spec: Optional[DataSpec] = None,
        random_seed: Optional[int] = None,
    ):
        config = config or StrategyConfig(name="BasisArbitrageV2")
        super().__init__(config)
        
        self.arb_config = arb_config or BasisArbConfig()
        
        # Data specifications (configurable)
        self.tradfi_spec = tradfi_spec or DataSpec(
            venue="yahoo",
            market="futures",
            ticker="GC=F",
            interval="15m",
        )
        self.defi_spec = defi_spec or DataSpec(
            venue="hyperliquid",
            market="perp",
            ticker="PAXG",
            interval="15m",
        )
        
        # Random state for stochastic exits
        self.rng = np.random.RandomState(random_seed)
        
        # State tracking
        self.entry_basis_bps: float = 0.0
        self.entry_bar_idx: int = 0
        self.bars_in_trade: int = 0  # Count actual trading bars held
        self.daily_trade_count: dict = {}
        self.in_position: bool = False
        
        # Enable spread P&L mode
        self.config.spread_pnl_mode = True
    
    def required_data(self) -> dict[str, DataSpec]:
        """Return data specifications for each leg."""
        return {
            "tradfi": self.tradfi_spec,
            "defi": self.defi_spec,
        }
    
    def required_indicators(self) -> dict[str, list[tuple[str, dict]]]:
        """No indicators needed - basis calculated from prices."""
        return {
            "tradfi": [],
            "defi": [],
        }
    
    def on_start(self, data: dict[str, pd.DataFrame]) -> None:
        """Reset state at start of backtest."""
        self.daily_trade_count = {}
        self.entry_basis_bps = 0.0
        self.entry_bar_idx = 0
        self.bars_in_trade = 0
        self.in_position = False
    
    def on_bar(
        self,
        idx: int,
        data: dict[str, pd.DataFrame],
        capital: float,
        positions: dict[str, Optional[Position]],
    ) -> dict[str, Signal]:
        """
        Process each bar and return trading signals for both legs.
        """
        cfg = self.arb_config
        
        # Get current prices
        tradfi_price = data["tradfi"].iloc[idx]["close"]
        defi_price = data["defi"].iloc[idx]["close"]
        
        # Calculate basis (defi - tradfi) / tradfi * 10000
        basis_bps = self.calculate_basis(data, idx, "tradfi", "defi")
        abs_basis = abs(basis_bps)
        
        # Get timestamp for trade limiting
        timestamp = data["tradfi"].index[idx]
        current_date = timestamp.date()
        if current_date not in self.daily_trade_count:
            self.daily_trade_count[current_date] = 0
        
        # Check if we're in a position
        self.in_position = any(pos is not None for pos in positions.values())
        
        # ===== EXIT LOGIC =====
        if self.in_position:
            self.bars_in_trade += 1  # Count this trading bar
            exit_signal = self._check_exit(idx, basis_bps, abs_basis, self.bars_in_trade)
            
            if exit_signal is not None:
                self.in_position = False
                self.bars_in_trade = 0
                return {
                    "tradfi": exit_signal,
                    "defi": exit_signal,
                }
            return {}  # Hold
        
        # ===== ENTRY LOGIC =====
        if abs_basis > cfg.threshold_bps:
            # Check daily trade limit
            if self.daily_trade_count[current_date] >= cfg.max_trades_per_day:
                return {}
            
            # Record entry state
            self.entry_basis_bps = basis_bps
            self.entry_bar_idx = idx
            self.daily_trade_count[current_date] += 1
            self.in_position = True
            
            # Determine direction based on basis sign
            # Positive basis = defi premium → short defi, long tradfi
            # Negative basis = tradfi premium → long defi, short tradfi
            size = cfg.position_size_per_leg
            
            if basis_bps > 0:
                # Defi is expensive, TradFi is cheap
                # Long TradFi (buy), Short DeFi (sell)
                return {
                    "tradfi": Signal.buy(size=size, reason=f"basis={basis_bps:.1f}bps, long_tradfi"),
                    "defi": Signal.sell(size=size, reason=f"basis={basis_bps:.1f}bps, short_defi"),
                }
            else:
                # TradFi is expensive, DeFi is cheap
                # Short TradFi (sell), Long DeFi (buy)
                return {
                    "tradfi": Signal.sell(size=size, reason=f"basis={basis_bps:.1f}bps, short_tradfi"),
                    "defi": Signal.buy(size=size, reason=f"basis={basis_bps:.1f}bps, long_defi"),
                }
        
        return {}  # No signal
    
    def get_entry_basis(self) -> float:
        """Return entry basis for spread P&L calculation."""
        return self.entry_basis_bps
    
    def _check_exit(
        self,
        idx: int,
        basis_bps: float,
        abs_basis: float,
        bars_held: int,
    ) -> Optional[Signal]:
        """Check exit conditions.
        
        Basis arb has no traditional stop-loss - spread is locked at entry.
        Exit only on:
        1. Convergence (captured enough bps)
        2. Time (held too long - opportunity cost)
        """
        cfg = self.arb_config
        entry_abs = abs(self.entry_basis_bps)
        
        # Calculate captured bps (how much spread has converged)
        captured_bps = entry_abs - abs_basis
        
        # 1. Take-profit: spread converged enough
        if captured_bps >= cfg.take_profit_captured_bps:
            return Signal.close(reason="take_profit")
        
        # 2. Time-based exit: held too long (opportunity cost / funding accumulation)
        if bars_held >= cfg.max_hold_bars:
            return Signal.close(reason="time_expiry")
        
        return None  # Continue holding - spread will converge
