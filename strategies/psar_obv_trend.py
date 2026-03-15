"""
PSAR-OBV Trend Strategy

Uses Parabolic SAR for trend direction / stop-and-reverse levels,
and On-Balance Volume (OBV) for volume confirmation.

Trades BTCUSDT on Binance USD-M Futures.
1h bars for indicators (PSAR, OBV), 1m bars for execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.strategy.base import SingleAssetStrategy, StrategyConfig
from core.strategy.data import StrategyDataSpec, StrategyData
from core.strategy.position import Signal, Position


# ─── Signal module framework ────────────────────────────────────────────────


@dataclass
class ModuleSignal:
    """Output from a signal module."""
    favor_long: bool = False
    favor_short: bool = False

    @property
    def neutral(self) -> bool:
        return not self.favor_long and not self.favor_short

    @staticmethod
    def long() -> ModuleSignal:
        return ModuleSignal(favor_long=True)

    @staticmethod
    def short() -> ModuleSignal:
        return ModuleSignal(favor_short=True)

    @staticmethod
    def hold() -> ModuleSignal:
        return ModuleSignal()


class SignalModule(ABC):
    """
    Base class for signal modules.

    Each module reads indicator data, maintains its own internal state,
    and outputs a simple (favor_long, favor_short) signal.

    All complexity (vector math, smoothing, conviction logic) lives
    inside the module — the strategy only sees the bool output.
    """

    @abstractmethod
    def evaluate(self, bar: pd.Series, price: float) -> ModuleSignal:
        """
        Evaluate the module on the latest 1h bar.

        Args:
            bar: The latest closed 1h bar (includes indicator columns).
            price: Current execution price (1m close).

        Returns:
            ModuleSignal with favor_long / favor_short bools.
        """
        ...

    def reset(self) -> None:
        """Reset internal state (called at start of backtest)."""
        pass


# ─── PSAR module ─────────────────────────────────────────────────────────────


class PSARModule(SignalModule):
    """
    Parabolic SAR direction module.

    Favor long when price is above PSAR (dots below).
    Favor short when price is below PSAR (dots above).
    """

    def evaluate(self, bar: pd.Series, price: float) -> ModuleSignal:
        psar = bar.get("PSAR")
        if psar is None or pd.isna(psar):
            return ModuleSignal.hold()

        close = bar["close"]
        if close > psar:
            return ModuleSignal.long()
        elif close < psar:
            return ModuleSignal.short()
        return ModuleSignal.hold()


# ─── SOBV module ─────────────────────────────────────────────────────────────


class SOBVModule(SignalModule):
    """
    Smoothed On-Balance Volume (SOBV) trend module.

    Uses EMA-smoothed OBV which filters hourly noise better than raw OBV.
    Tracks SOBV direction over a lookback window.
    Only signals when velocity (slope as % of SOBV) exceeds a threshold.
    """

    def __init__(self, ema_length: int = 14, lookback: int = 24, min_velocity_pct: float = 2.0):
        self.ema_length = ema_length
        self.lookback = lookback
        self.min_velocity_pct = min_velocity_pct
        self._sobv_history: list[float] = []

    @property
    def column_name(self) -> str:
        return f"SOBV_{self.ema_length}"

    def evaluate(self, bar: pd.Series, price: float) -> ModuleSignal:
        sobv = bar.get(self.column_name)
        if sobv is None or pd.isna(sobv):
            return ModuleSignal.hold()

        self._sobv_history.append(float(sobv))
        if len(self._sobv_history) > self.lookback:
            self._sobv_history = self._sobv_history[-self.lookback:]

        if len(self._sobv_history) < self.lookback:
            return ModuleSignal.hold()

        current = self._sobv_history[-1]
        oldest = self._sobv_history[0]

        # Velocity: slope normalized as % of SOBV value
        if abs(oldest) < 1e-9:
            return ModuleSignal.hold()
        velocity_pct = ((current - oldest) / abs(oldest)) * 100

        # Only signal if velocity exceeds threshold
        if velocity_pct > self.min_velocity_pct:
            return ModuleSignal.long()
        elif velocity_pct < -self.min_velocity_pct:
            return ModuleSignal.short()
        return ModuleSignal.hold()

    def reset(self) -> None:
        self._sobv_history.clear()


# ─── Strategy ────────────────────────────────────────────────────────────────


class PSAROBVTrend(SingleAssetStrategy):
    """
    PSAR + OBV trend following on Binance BTCUSDT Futures.

    Entry: all modules agree on direction.
    Exit: TBD — will be defined in next iteration.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        config = config or StrategyConfig(name="PSAROBVTrend")
        super().__init__(config)

        self.modules: list[SignalModule] = [
            PSARModule(),
            SOBVModule(ema_length=14, lookback=24, min_velocity_pct=2.0),
        ]
        self._last_h1_time = None

    def data_spec(self) -> StrategyDataSpec:
        return StrategyDataSpec(
            venue="binance",
            market="futures",
            ticker="BTCUSDT",
            intervals={
                "1m": [],
                "1h": [
                    ("psar", {"af": 0.02, "af_step": 0.02, "max_af": 0.2}),
                    ("sobv", {"length": 14}),
                ],
            },
        )

    def required_indicators(self) -> list[tuple[str, dict]]:
        return [
            ("psar", {"af": 0.02, "af_step": 0.02, "max_af": 0.2}),
            ("sobv", {"length": 14}),
        ]

    def on_bar(self, timestamp, data: StrategyData, balance: float, position: Optional[Position]):
        """Called every 1m bar. Evaluates modules on new 1h bars."""
        # Get latest closed 1h bar
        try:
            h1 = data.bar("1h", timestamp)
        except (ValueError, KeyError):
            return Signal.hold()

        # Only re-evaluate modules when a new 1h bar arrives
        h1_time = h1.name if hasattr(h1, "name") else None
        if h1_time is not None and h1_time == self._last_h1_time:
            return Signal.hold()
        self._last_h1_time = h1_time

        # Current execution price
        m1 = data.bar("1m", timestamp)
        price = m1["close"]

        # Evaluate all modules
        signals = [m.evaluate(h1, price) for m in self.modules]

        # --- Entry: all modules must agree, use 5% trailing stop ---
        if position is None:
            if all(s.favor_long for s in signals):
                return Signal.buy(
                    size=1.0,
                    reason="all_modules_long",
                    trailing_stop_pct=5.0,
                )
            if all(s.favor_short for s in signals):
                return Signal.sell(
                    size=1.0,
                    reason="all_modules_short",
                    trailing_stop_pct=5.0,
                )

        return Signal.hold()
