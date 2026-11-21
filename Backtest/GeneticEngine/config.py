"""Immutable research constraints and initial search domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DATA_TIMEFRAMES: Final = ("15m", "1h", "4h")
EXECUTION_TIMEFRAME: Final = "15m"
ANALYSIS_TIMEFRAMES: Final = ("1h", "4h", "1d")
EXPECTED_SYMBOLS: Final = 76
INITIAL_EQUITY: Final = 100.0

MAX_OPEN_POSITIONS: Final = 6
MAX_OPEN_PER_DIRECTION: Final = 3
MAX_OPEN_PER_SYMBOL: Final = 1

BASELINE_FEE_RATE: Final = 0.0
BASELINE_SLIPPAGE_RATE: Final = 0.0
SAME_CANDLE_SL_TP_POLICY: Final = "SL_FIRST"
LOOKAHEAD_POLICY: Final = "CLOSED_CANDLES_ONLY"
EXECUTION_MODE: Final = "SIMULATION_ONLY"

ENTRY_METHODS: Final = (
    "DIRECT",
    "ENGULFING",
    "PIN_BAR",
    "M15_STRUCTURE_BREAK",
)
ZONE_TYPES: Final = ("body", "wick")
TOUCH_RULES: Final = ("first", "max_two", "progressive")
BTC_FILTER_MODES: Final = (None, "1d_4h", "1d_1h", "1d_4h_1h")

# Initial bounded search domains; these are not optimized results.
SWING_LEFT_VALUES: Final = (2, 3, 5, 8, 10)
SWING_RIGHT_VALUES: Final = (2, 3, 5, 8, 10)
OB_THRESHOLD_VALUES: Final = (0.75, 1.0, 1.25, 1.5, 2.0)
ATR_PERIOD_VALUES: Final = (7, 10, 14, 20, 28)
OB_LOOKBACK_VALUES: Final = (5, 8, 10, 15, 20)
RISK_PERCENT_VALUES: Final = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5)
ATR_STOP_MULTIPLIER_VALUES: Final = (1.0, 1.5, 2.0, 2.5)
STOP_MODES: Final = ("OB_WICK", "ATR")


@dataclass(frozen=True)
class EvolutionConfig:
    """Reproducible defaults for the two planned evolution stages."""

    random_seed: int = 20261121
    smoke_population: int = 40
    smoke_generations: int = 10
    main_population: int = 100
    main_generations: int = 30
    tournament_size: int = 2
    crossover_probability: float = 0.8
    mutation_probability: float = 0.2
    max_rule_depth: int = 4
    max_rule_nodes: int = 31

    def validate(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        for name in (
            "smoke_population",
            "smoke_generations",
            "main_population",
            "main_generations",
            "tournament_size",
            "max_rule_depth",
            "max_rule_nodes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("crossover_probability", "mutation_probability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
