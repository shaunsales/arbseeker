"""Strategy library.

Example strategies (prefixed with _example_) are for testing and demonstration.
"""

from strategies._example_strategies import (
    # Public example strategies
    ExampleMACrossover,
    ExampleRSIMeanReversion,
    ExampleBollingerBreakout,
    # Test utilities (prefixed with _Test)
    _TestBuyAtBarN,
    _TestBuyAndSell,
    _TestRecording,
    _TestHoldOnly,
    _TestCapitalTracking,
)

__all__ = [
    "ExampleMACrossover",
    "ExampleRSIMeanReversion", 
    "ExampleBollingerBreakout",
    "_TestBuyAtBarN",
    "_TestBuyAndSell",
    "_TestRecording",
    "_TestHoldOnly",
    "_TestCapitalTracking",
]
