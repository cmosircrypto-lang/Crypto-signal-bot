"""Genetic Strategy Engine.

This package is an isolated research layer over the existing SwingEngine.
It does not place orders or modify the live trading system.
"""

from .genome import StrategyGenome, GenomeValidationError

__all__ = ["StrategyGenome", "GenomeValidationError"]
