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
    """Configuration for Basis Arbitrage strategy."""
    # Entry threshold
    threshold_bps: float = 50.0
    
    # Exit conditions (two modes: absolute or captured)
    # Mode 1: Absolute - exit when |basis| < take_profit_bps
    # Mode 2: Captured - exit when we've captured take_profit_captured_bps from entry
    use_captured_exit: bool = True  # Use captured bps mode (recommended)
    take_profit_bps: float = 15.0   # For absolute mode
    take_profit_captured_bps: float = 20.0  # Exit when captured this many bps
    stop_loss_bps: float = 100.0    # Exit if basis widens by this much from entry
    max_hold_bars: int = 10         # Maximum bars to hold (time-stop)
    
    # Stochastic exit (optional mean-reversion modeling)
    enable_stochastic_exit: bool = False
    half_life_bars: float = 2.3    # Expected mean-reversion half-life
    stochastic_factor: float = 0.3  # Probability scaling factor
    
    # Trade limits
    max_trades_per_day: int = 10
    
    # Position sizing (as fraction of capital per leg)
    position_size_per_leg: float = 0.5  # 50% of capital per leg


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
            bars_held = idx - self.entry_bar_idx
            exit_signal = self._check_exit(idx, basis_bps, abs_basis, bars_held)
            
            if exit_signal is not None:
                self.in_position = False
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
        """Check exit conditions and return signal if should exit."""
        cfg = self.arb_config
        entry_abs = abs(self.entry_basis_bps)
        
        # Calculate captured bps (positive = profit, negative = loss)
        captured_bps = entry_abs - abs_basis
        
        # 1. Take-profit check
        if cfg.use_captured_exit:
            # Captured mode: exit when we've captured enough bps
            if captured_bps >= cfg.take_profit_captured_bps:
                return Signal.close(reason="take_profit")
        else:
            # Absolute mode: exit when basis drops below threshold
            if abs_basis < cfg.take_profit_bps:
                return Signal.close(reason="take_profit")
        
        # 2. Stop-loss: basis widened from entry
        if captured_bps < -cfg.stop_loss_bps:
            return Signal.close(reason="stop_loss")
        
        # 3. Time-stop: held too long
        if bars_held >= cfg.max_hold_bars:
            return Signal.close(reason="time_stop")
        
        # 4. Stochastic exit (optional mean-reversion model)
        if cfg.enable_stochastic_exit and bars_held >= cfg.half_life_bars:
            base_prob = 1 - np.exp(-1 / cfg.half_life_bars)
            exit_prob = base_prob * cfg.stochastic_factor
            if self.rng.random() < exit_prob:
                return Signal.close(reason="stochastic")
        
        return None  # Continue holding
