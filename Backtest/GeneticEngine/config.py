"""Research constraints and bounded genetic search domains."""
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

ENTRY_METHODS: Final = ("DIRECT", "ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK")
ZONE_TYPES: Final = ("body", "wick")
TOUCH_RULES: Final = ("first", "max_two", "progressive")
BTC_FILTER_MODES: Final = (None, "1d_4h", "1d_1h", "1d_4h_1h")

# Strategy-family genes: source strategies are ingredients, not fixed symbol assignments.
STRATEGY_FAMILIES: Final = (
    "SWING_OB", "EMA_CROSS_PRO", "BTC_ALIGNED_PULLBACK",
    "SMC_LIQUIDITY_SWEEP", "MOMENTUM_BREAKOUT",
    "MEAN_REVERSION", "VOLATILITY_SQUEEZE",
)
FAMILY_COMBINATION_MODES: Final = ("SINGLE", "ANY", "ALL", "VOTE")

SWING_LEFT_VALUES: Final = (2, 3, 5, 8, 10)
SWING_RIGHT_VALUES: Final = (2, 3, 5, 8, 10)
OB_THRESHOLD_VALUES: Final = (0.75, 1.0, 1.25, 1.5, 2.0)
ATR_PERIOD_VALUES: Final = (7, 10, 14, 20, 28)
OB_LOOKBACK_VALUES: Final = (5, 8, 10, 15, 20)
RISK_PERCENT_VALUES: Final = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5)
REWARD_RATIO_VALUES: Final = (1.5, 2.0, 2.5, 3.0)
ATR_STOP_MULTIPLIER_VALUES: Final = (1.0, 1.5, 2.0, 2.5)
STOP_MODES: Final = ("OB_WICK", "ATR")

# EMA Cross PRO: preserve main.py concepts while letting evolution tune thresholds.
EMA_CROSS_LOOKBACK_VALUES: Final = (3, 4, 6, 9, 12)
EMA_FAST_PERIOD_VALUES: Final = (5, 7, 9, 12)
EMA_SLOW_PERIOD_VALUES: Final = (18, 21, 26, 34, 50)
EMA_ADX_VALUES: Final = (12.0, 14.0, 16.0, 18.0, 22.0, 26.0)
EMA_VOLUME_MULT_VALUES: Final = (1.0, 1.2, 1.4, 1.5, 1.8, 2.0)
EMA_SLOPE_ATR_VALUES: Final = (0.04, 0.06, 0.08, 0.12, 0.16)
EMA_SL_ATR_VALUES: Final = (1.0, 1.3, 1.6, 2.0, 2.4)
EMA_TP_ATR_VALUES: Final = (2.0, 2.5, 3.0, 3.4, 4.0, 5.0)

# BTC-aligned pullback.
BTC_GAP_ATR_VALUES: Final = (0.12, 0.18, 0.22, 0.28, 0.35, 0.45)
BTC_TOUCH_ATR_VALUES: Final = (0.10, 0.16, 0.22, 0.30, 0.40)
BTC_MAX_EXTENSION_ATR_VALUES: Final = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0)
BTC_PULLBACK_VOLUME_VALUES: Final = (0.8, 1.0, 1.2, 1.4, 1.8, 2.0)
BTC_PULLBACK_SL_ATR_VALUES: Final = (1.0, 1.2, 1.4, 1.6, 2.0)
BTC_PULLBACK_TP_ATR_VALUES: Final = (2.0, 2.5, 2.95, 3.5, 4.0)

# SMC liquidity sweep.
SMC_SWEEP_LOOKBACK_VALUES: Final = (5, 7, 10, 14, 20, 30)
SMC_SL_BUFFER_ATR_VALUES: Final = (0.05, 0.1, 0.2, 0.3, 0.5)
SMC_MIN_SL_ATR_VALUES: Final = (0.3, 0.4, 0.6, 0.8, 1.0)
SMC_MAX_SL_ATR_VALUES: Final = (2.0, 2.5, 3.0, 4.0, 5.0)
SMC_TP_ATR_VALUES: Final = (1.5, 2.0, 2.5, 3.0, 4.0)
SMC_MIN_RR_VALUES: Final = (1.0, 1.2, 1.5, 1.8, 2.0)
SMC_MIN_VOLUME_VALUES: Final = (0.6, 0.8, 0.9, 1.0, 1.2, 1.5)

# Additional research families proposed for broader discovery.
MOMENTUM_LOOKBACK_VALUES: Final = (10, 15, 20, 30, 40)
MOMENTUM_BREAKOUT_BUFFER_ATR_VALUES: Final = (0.0, 0.05, 0.1, 0.2, 0.3)
MOMENTUM_ADX_VALUES: Final = (14.0, 18.0, 22.0, 26.0, 30.0)
MOMENTUM_VOLUME_VALUES: Final = (0.8, 1.0, 1.2, 1.5, 1.8)
MEAN_REVERSION_RSI_LOW_VALUES: Final = (20.0, 25.0, 30.0, 35.0)
MEAN_REVERSION_RSI_HIGH_VALUES: Final = (65.0, 70.0, 75.0, 80.0)
MEAN_REVERSION_ZSCORE_VALUES: Final = (1.5, 1.8, 2.0, 2.2, 2.5, 3.0)
SQUEEZE_BB_PERIOD_VALUES: Final = (15, 20, 24, 30)
SQUEEZE_WIDTH_PERCENTILE_VALUES: Final = (10.0, 15.0, 20.0, 25.0, 30.0)
SQUEEZE_EXPANSION_ATR_VALUES: Final = (0.8, 1.0, 1.2, 1.5, 2.0)

@dataclass(frozen=True)
class EvolutionConfig:
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
        for name in ("smoke_population", "smoke_generations", "main_population",
                     "main_generations", "tournament_size", "max_rule_depth", "max_rule_nodes"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("crossover_probability", "mutation_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
