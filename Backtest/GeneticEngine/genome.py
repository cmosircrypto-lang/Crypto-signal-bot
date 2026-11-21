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
    entry_methods: tuple[str, ...] = (
        "DIRECT",
        "ENGULFING",
        "PIN_BAR",
        "M15_STRUCTURE_BREAK",
    )
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

    def validate(self) -> None:
        if self.version != 1:
            raise GenomeValidationError(f"Unsupported genome version: {self.version}")
        if not self.entry_methods:
            raise GenomeValidationError("At least one entry method is required")
        if len(set(self.entry_methods)) != len(self.entry_methods):
            raise GenomeValidationError("entry_methods must not contain duplicates")
        unknown_methods = set(self.entry_methods) - set(config.ENTRY_METHODS)
        if unknown_methods:
            raise GenomeValidationError(f"Unsupported entry methods: {sorted(unknown_methods)}")
        if self.zone_type not in config.ZONE_TYPES:
            raise GenomeValidationError(f"Unsupported zone_type: {self.zone_type}")
        if self.touch_rule not in config.TOUCH_RULES:
            raise GenomeValidationError(f"Unsupported touch_rule: {self.touch_rule}")
        if self.swing_left not in config.SWING_LEFT_VALUES:
            raise GenomeValidationError("swing_left is outside the configured search domain")
        if self.swing_right not in config.SWING_RIGHT_VALUES:
            raise GenomeValidationError("swing_right is outside the configured search domain")
        if self.ob_threshold not in config.OB_THRESHOLD_VALUES:
            raise GenomeValidationError("ob_threshold is outside the configured search domain")
        if self.atr_period not in config.ATR_PERIOD_VALUES:
            raise GenomeValidationError("atr_period is outside the configured search domain")
        if self.ob_lookback not in config.OB_LOOKBACK_VALUES:
            raise GenomeValidationError("ob_lookback is outside the configured search domain")
        if self.btc_filter_mode not in config.BTC_FILTER_MODES:
            raise GenomeValidationError(f"Unsupported BTC filter mode: {self.btc_filter_mode}")
        if self.risk_percent not in config.RISK_PERCENT_VALUES:
            raise GenomeValidationError("risk_percent is outside the configured search domain")
        if self.reward_ratio not in config.REWARD_RATIO_VALUES:
            raise GenomeValidationError("reward_ratio is outside the configured search domain")
        if self.stop_mode not in config.STOP_MODES:
            raise GenomeValidationError(f"Unsupported stop_mode: {self.stop_mode}")
        if self.atr_stop_multiplier not in config.ATR_STOP_MULTIPLIER_VALUES:
            raise GenomeValidationError("atr_stop_multiplier is outside the configured search domain")
        normalized = tuple(symbol.strip().upper() for symbol in self.symbols)
        if any(not symbol for symbol in normalized):
            raise GenomeValidationError("symbols cannot contain blank values")
        if len(set(normalized)) != len(normalized):
            raise GenomeValidationError("symbols must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["entry_methods"] = list(self.entry_methods)
        payload["symbols"] = list(self.symbols)
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
        if "entry_methods" in values:
            values["entry_methods"] = tuple(values["entry_methods"])
        if "symbols" in values:
            values["symbols"] = tuple(values["symbols"])
        genome = cls(**values)
        genome.validate()
        return genome

    @classmethod
    def baseline(cls) -> "StrategyGenome":
        """Current SwingEngine defaults, used for parity checks."""
        genome = cls()
        genome.validate()
        return genome
