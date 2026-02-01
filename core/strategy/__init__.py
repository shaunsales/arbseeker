"""Strategy framework for backtesting."""

from core.strategy.position import (
    Position,
    Trade,
    Signal,
    CostModel,
    Side,
    PositionStatus,
    DEFAULT_COSTS,
    ZERO_COSTS,
)
from core.strategy.base import (
    SingleAssetStrategy,
    MultiLeggedStrategy,
    DataSpec,
    StrategyConfig,
)
from core.strategy.engine import (
    BacktestEngine,
    BacktestResult,
)

__all__ = [
    # Position tracking
    "Position",
    "Trade",
    "Signal",
    "CostModel",
    "Side",
    "PositionStatus",
    "DEFAULT_COSTS",
    "ZERO_COSTS",
    # Strategy base classes
    "SingleAssetStrategy",
    "MultiLeggedStrategy",
    "DataSpec",
    "StrategyConfig",
    # Engine
    "BacktestEngine",
    "BacktestResult",
]
