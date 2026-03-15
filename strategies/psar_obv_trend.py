"""
PSAR-OBV Trend Strategy

Uses Parabolic SAR for trend direction / stop-and-reverse levels,
and On-Balance Volume (OBV) for volume confirmation.

Trades BTCUSDT on Binance USD-M Futures.
1h bars for indicators (PSAR, OBV), 1m bars for execution.
"""

from typing import Optional

import pandas as pd

from core.strategy.base import SingleAssetStrategy, StrategyConfig
from core.strategy.data import StrategyDataSpec, StrategyData
from core.strategy.position import Signal, Position


class PSAROBVTrend(SingleAssetStrategy):
    """
    PSAR + OBV trend following on Binance BTCUSDT Futures.

    Entry/exit logic: TBD — scaffold only.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        config = config or StrategyConfig(name="PSAROBVTrend")
        super().__init__(config)

    def data_spec(self) -> StrategyDataSpec:
        return StrategyDataSpec(
            venue="binance",
            market="futures",
            ticker="BTCUSDT",
            intervals={
                "1m": [],
                "1h": [
                    ("psar", {"af": 0.02, "af_step": 0.02, "max_af": 0.2}),
                    ("obv", {}),
                ],
            },
        )

    def required_indicators(self) -> list[tuple[str, dict]]:
        return [
            ("psar", {"af": 0.02, "af_step": 0.02, "max_af": 0.2}),
            ("obv", {}),
        ]

    def on_bar(self, timestamp, data: StrategyData, balance: float, position: Optional[Position]):
        """Called every 1m bar. Entry/exit rules to be implemented."""
        return Signal.hold()
