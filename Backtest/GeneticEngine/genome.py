"""Versioned, validated and JSON-serializable strategy genome."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
from . import config

class GenomeValidationError(ValueError):
    """Raised when a candidate genome is outside the allowed search space."""

@dataclass(frozen=True)
class StrategyGenome:
    version: int = 1
    entry_methods: tuple[str, ...] = ("DIRECT", "ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK")
    zone_type: str = "body"
    touch_rule: str = "first"
    swing_left: int = 5
    swing_right: int = 5
    ob_threshold: float = 1.0
    atr_period: int = 14
    ob_lookback: int = 10
    btc_filter_mode: str | None = None
    require_4h_alignment: bool = False
    risk_percent: float = 0.5
    reward_ratio: float = 2.0
    stop_mode: str = "OB_WICK"
    atr_stop_multiplier: float = 1.5
    symbols: tuple[str, ...] = field(default_factory=tuple)

    # Composable signal-family genes. Direction is evaluated separately by the runner.
    signal_families: tuple[str, ...] = ("SWING_OB",)
    family_combination: str = "SINGLE"
    min_family_agreement: int = 1

    # EMA Cross PRO genes.
    ema_cross_lookback: int = 6
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    ema_min_adx_4h: float = 18.0
    ema_min_adx_1h: float = 16.0
    ema_volume_mult: float = 1.5
    ema_slope_atr: float = 0.08
    ema_sl_atr: float = 1.6
    ema_tp_atr: float = 3.4

    # BTC-aligned pullback genes.
    btc_gap_atr: float = 0.22
    btc_touch_atr: float = 0.22
    btc_max_extension_atr: float = 1.0
    btc_pullback_volume: float = 1.4
    btc_pullback_sl_atr: float = 1.4
    btc_pullback_tp_atr: float = 2.95

    # SMC liquidity-sweep genes.
    smc_sweep_lookback: int = 10
    smc_sl_buffer_atr: float = 0.2
    smc_min_sl_atr: float = 0.6
    smc_max_sl_atr: float = 3.0
    smc_tp_atr: float = 2.0
    smc_min_rr: float = 1.2
    smc_min_volume: float = 0.9

    # Additional independently evolvable research families.
    momentum_lookback: int = 20
    momentum_breakout_buffer_atr: float = 0.1
    momentum_min_adx: float = 18.0
    momentum_volume_mult: float = 1.2
    mean_reversion_rsi_low: float = 30.0
    mean_reversion_rsi_high: float = 70.0
    mean_reversion_zscore: float = 2.0
    squeeze_bb_period: int = 20
    squeeze_width_percentile: float = 20.0
    squeeze_expansion_atr: float = 1.2

    def validate(self) -> None:
        if self.version != 1:
            raise GenomeValidationError(f"Unsupported genome version: {self.version}")
        if not self.entry_methods or len(set(self.entry_methods)) != len(self.entry_methods):
            raise GenomeValidationError("entry_methods must be non-empty and unique")
        unknown_methods = set(self.entry_methods) - set(config.ENTRY_METHODS)
        if unknown_methods:
            raise GenomeValidationError(f"Unsupported entry methods: {sorted(unknown_methods)}")
        if self.zone_type not in config.ZONE_TYPES:
            raise GenomeValidationError(f"Unsupported zone_type: {self.zone_type}")
        if self.touch_rule not in config.TOUCH_RULES:
            raise GenomeValidationError(f"Unsupported touch_rule: {self.touch_rule}")
        domains = {
            "swing_left": config.SWING_LEFT_VALUES, "swing_right": config.SWING_RIGHT_VALUES,
            "ob_threshold": config.OB_THRESHOLD_VALUES, "atr_period": config.ATR_PERIOD_VALUES,
            "ob_lookback": config.OB_LOOKBACK_VALUES, "btc_filter_mode": config.BTC_FILTER_MODES,
            "risk_percent": config.RISK_PERCENT_VALUES, "reward_ratio": config.REWARD_RATIO_VALUES,
            "stop_mode": config.STOP_MODES, "atr_stop_multiplier": config.ATR_STOP_MULTIPLIER_VALUES,
            "ema_cross_lookback": config.EMA_CROSS_LOOKBACK_VALUES,
            "ema_fast_period": config.EMA_FAST_PERIOD_VALUES,
            "ema_slow_period": config.EMA_SLOW_PERIOD_VALUES,
            "ema_min_adx_4h": config.EMA_ADX_VALUES, "ema_min_adx_1h": config.EMA_ADX_VALUES,
            "ema_volume_mult": config.EMA_VOLUME_MULT_VALUES, "ema_slope_atr": config.EMA_SLOPE_ATR_VALUES,
            "ema_sl_atr": config.EMA_SL_ATR_VALUES, "ema_tp_atr": config.EMA_TP_ATR_VALUES,
            "btc_gap_atr": config.BTC_GAP_ATR_VALUES, "btc_touch_atr": config.BTC_TOUCH_ATR_VALUES,
            "btc_max_extension_atr": config.BTC_MAX_EXTENSION_ATR_VALUES,
            "btc_pullback_volume": config.BTC_PULLBACK_VOLUME_VALUES,
            "btc_pullback_sl_atr": config.BTC_PULLBACK_SL_ATR_VALUES,
            "btc_pullback_tp_atr": config.BTC_PULLBACK_TP_ATR_VALUES,
            "smc_sweep_lookback": config.SMC_SWEEP_LOOKBACK_VALUES,
            "smc_sl_buffer_atr": config.SMC_SL_BUFFER_ATR_VALUES,
            "smc_min_sl_atr": config.SMC_MIN_SL_ATR_VALUES,
            "smc_max_sl_atr": config.SMC_MAX_SL_ATR_VALUES,
            "smc_tp_atr": config.SMC_TP_ATR_VALUES, "smc_min_rr": config.SMC_MIN_RR_VALUES,
            "smc_min_volume": config.SMC_MIN_VOLUME_VALUES,
            "momentum_lookback": config.MOMENTUM_LOOKBACK_VALUES,
            "momentum_breakout_buffer_atr": config.MOMENTUM_BREAKOUT_BUFFER_ATR_VALUES,
            "momentum_min_adx": config.MOMENTUM_ADX_VALUES,
            "momentum_volume_mult": config.MOMENTUM_VOLUME_VALUES,
            "mean_reversion_rsi_low": config.MEAN_REVERSION_RSI_LOW_VALUES,
            "mean_reversion_rsi_high": config.MEAN_REVERSION_RSI_HIGH_VALUES,
            "mean_reversion_zscore": config.MEAN_REVERSION_ZSCORE_VALUES,
            "squeeze_bb_period": config.SQUEEZE_BB_PERIOD_VALUES,
            "squeeze_width_percentile": config.SQUEEZE_WIDTH_PERCENTILE_VALUES,
            "squeeze_expansion_atr": config.SQUEEZE_EXPANSION_ATR_VALUES,
        }
        for name, allowed in domains.items():
            value = getattr(self, name)
            if value not in allowed:
                raise GenomeValidationError(f"{name}={value!r} is outside its configured search domain")
        if self.ema_fast_period >= self.ema_slow_period:
            raise GenomeValidationError("ema_fast_period must be less than ema_slow_period")
        if self.smc_min_sl_atr >= self.smc_max_sl_atr:
            raise GenomeValidationError("smc_min_sl_atr must be less than smc_max_sl_atr")
        if self.mean_reversion_rsi_low >= self.mean_reversion_rsi_high:
            raise GenomeValidationError("mean_reversion_rsi_low must be below mean_reversion_rsi_high")
        if self.family_combination not in config.FAMILY_COMBINATION_MODES:
            raise GenomeValidationError(f"Unsupported family_combination: {self.family_combination}")
        if not self.signal_families or len(set(self.signal_families)) != len(self.signal_families):
            raise GenomeValidationError("signal_families must be non-empty and unique")
        unknown_families = set(self.signal_families) - set(config.STRATEGY_FAMILIES)
        if unknown_families:
            raise GenomeValidationError(f"Unsupported signal_families: {sorted(unknown_families)}")
        if self.family_combination == "SINGLE" and len(self.signal_families) != 1:
            raise GenomeValidationError("SINGLE combination requires exactly one signal family")
        if not 1 <= self.min_family_agreement <= len(self.signal_families):
            raise GenomeValidationError("min_family_agreement must be within the selected family count")
        normalized = tuple(s.strip().upper() for s in self.symbols)
        if any(not s for s in normalized) or len(set(normalized)) != len(normalized):
            raise GenomeValidationError("symbols must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in ("entry_methods", "symbols", "signal_families"):
            payload[key] = list(getattr(self, key))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyGenome":
        if not isinstance(payload, dict):
            raise GenomeValidationError("Genome payload must be a dictionary")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise GenomeValidationError(f"Unknown genome fields: {sorted(unknown)}")
        values = dict(payload)
        for key in ("entry_methods", "symbols", "signal_families"):
            if key in values:
                values[key] = tuple(values[key])
        genome = cls(**values)
        genome.validate()
        return genome

    @classmethod
    def baseline(cls) -> "StrategyGenome":
        genome = cls()
        genome.validate()
        return genome
