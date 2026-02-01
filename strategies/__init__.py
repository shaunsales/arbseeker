"""Strategy library.

Example strategies (prefixed with _example_) are for testing and demonstration.
Production strategies are named without prefix.
"""

from strategies.basis_arb import BasisArbitrage, BasisArbConfig
from strategies._example_strategies import (
    ExampleMACrossover,
    ExampleRSIMeanReversion,
    ExampleBollingerBreakout,
    _TestBuyAtBarN,
    _TestBuyAndSell,
    _TestRecording,
    _TestHoldOnly,
    _TestCapitalTracking,
)

__all__ = [
    # Production strategies
    "BasisArbitrage",
    "BasisArbConfig",
    # Example strategies
    "ExampleMACrossover",
    "ExampleRSIMeanReversion", 
    "ExampleBollingerBreakout",
    # Test utilities
    "_TestBuyAtBarN",
    "_TestBuyAndSell",
    "_TestRecording",
    "_TestHoldOnly",
    "_TestCapitalTracking",
]
