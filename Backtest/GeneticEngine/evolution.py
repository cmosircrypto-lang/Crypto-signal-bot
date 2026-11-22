"""Deterministic genetic operators for the composable StrategyGenome.

This module evolves parameter combinations only. It does not evaluate trades or
claim that a family has a working signal evaluator.
"""
from __future__ import annotations
import random
from dataclasses import fields
from typing import Iterable
from . import config
from .genome import StrategyGenome

_DOMAINS = {
    "zone_type": config.ZONE_TYPES, "touch_rule": config.TOUCH_RULES,
    "swing_left": config.SWING_LEFT_VALUES, "swing_right": config.SWING_RIGHT_VALUES,
    "ob_threshold": config.OB_THRESHOLD_VALUES, "atr_period": config.ATR_PERIOD_VALUES,
    "ob_lookback": config.OB_LOOKBACK_VALUES, "btc_filter_mode": config.BTC_FILTER_MODES,
    "require_4h_alignment": (False, True), "risk_percent": config.RISK_PERCENT_VALUES,
    "reward_ratio": config.REWARD_RATIO_VALUES, "stop_mode": config.STOP_MODES,
    "atr_stop_multiplier": config.ATR_STOP_MULTIPLIER_VALUES,
    "ema_cross_lookback": config.EMA_CROSS_LOOKBACK_VALUES,
    "ema_fast_period": config.EMA_FAST_PERIOD_VALUES, "ema_slow_period": config.EMA_SLOW_PERIOD_VALUES,
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
    "smc_min_sl_atr": config.SMC_MIN_SL_ATR_VALUES, "smc_max_sl_atr": config.SMC_MAX_SL_ATR_VALUES,
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

def _repair(values: dict, rng: random.Random) -> StrategyGenome:
    # Keep related parameters coherent after independent crossover/mutation.
    if values["ema_fast_period"] >= values["ema_slow_period"]:
        pairs = [(f, s) for f in config.EMA_FAST_PERIOD_VALUES
                 for s in config.EMA_SLOW_PERIOD_VALUES if f < s]
        values["ema_fast_period"], values["ema_slow_period"] = rng.choice(pairs)
    if values["smc_min_sl_atr"] >= values["smc_max_sl_atr"]:
        pairs = [(lo, hi) for lo in config.SMC_MIN_SL_ATR_VALUES
                 for hi in config.SMC_MAX_SL_ATR_VALUES if lo < hi]
        values["smc_min_sl_atr"], values["smc_max_sl_atr"] = rng.choice(pairs)
    if values["mean_reversion_rsi_low"] >= values["mean_reversion_rsi_high"]:
        pairs = [(lo, hi) for lo in config.MEAN_REVERSION_RSI_LOW_VALUES
                 for hi in config.MEAN_REVERSION_RSI_HIGH_VALUES if lo < hi]
        values["mean_reversion_rsi_low"], values["mean_reversion_rsi_high"] = rng.choice(pairs)
    families = tuple(dict.fromkeys(values["signal_families"]))
    if not families:
        families = (rng.choice(config.STRATEGY_FAMILIES),)
    values["signal_families"] = families
    if len(families) == 1:
        values["family_combination"] = "SINGLE"
        values["min_family_agreement"] = 1
    else:
        if values["family_combination"] == "SINGLE":
            values["family_combination"] = rng.choice(("ANY", "ALL", "VOTE"))
        if values["family_combination"] == "ALL":
            values["min_family_agreement"] = len(families)
        else:
            values["min_family_agreement"] = min(max(1, int(values["min_family_agreement"])), len(families))
    methods = tuple(dict.fromkeys(values["entry_methods"]))
    values["entry_methods"] = methods or (rng.choice(config.ENTRY_METHODS),)
    values["symbols"] = tuple(values.get("symbols", ()))
    return StrategyGenome.from_dict(values)

def random_genome(rng: random.Random | None = None, *, symbols: Iterable[str] = ()) -> StrategyGenome:
    rng = rng or random.Random()
    values = StrategyGenome.baseline().to_dict()
    for name, domain in _DOMAINS.items():
        values[name] = rng.choice(domain)
    family_count = rng.randint(1, min(4, len(config.STRATEGY_FAMILIES)))
    values["signal_families"] = rng.sample(config.STRATEGY_FAMILIES, family_count)
    values["family_combination"] = "SINGLE" if family_count == 1 else rng.choice(config.FAMILY_COMBINATION_MODES[1:])
    values["min_family_agreement"] = family_count if values["family_combination"] == "ALL" else rng.randint(1, family_count)
    method_count = rng.randint(1, len(config.ENTRY_METHODS))
    values["entry_methods"] = rng.sample(config.ENTRY_METHODS, method_count)
    values["symbols"] = tuple(s.strip().upper() for s in symbols)
    return _repair(values, rng)

def mutate_genome(parent: StrategyGenome, rng: random.Random | None = None,
                  mutation_probability: float = 0.15) -> StrategyGenome:
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("mutation_probability must be in [0, 1]")
    rng = rng or random.Random()
    parent.validate()
    values = parent.to_dict()
    for name, domain in _DOMAINS.items():
        if rng.random() < mutation_probability:
            alternatives = [x for x in domain if x != values[name]]
            if alternatives:
                values[name] = rng.choice(alternatives)
    if rng.random() < mutation_probability:
        choices = [f for f in config.STRATEGY_FAMILIES if f not in values["signal_families"]]
        if choices and len(values["signal_families"]) < min(4, len(config.STRATEGY_FAMILIES)) and rng.random() < 0.5:
            values["signal_families"].append(rng.choice(choices))
        elif len(values["signal_families"]) > 1:
            values["signal_families"].remove(rng.choice(values["signal_families"]))
        else:
            values["signal_families"] = [rng.choice(config.STRATEGY_FAMILIES)]
    if rng.random() < mutation_probability:
        values["family_combination"] = rng.choice(config.FAMILY_COMBINATION_MODES)
    if rng.random() < mutation_probability:
        values["entry_methods"] = rng.sample(config.ENTRY_METHODS, rng.randint(1, len(config.ENTRY_METHODS)))
    return _repair(values, rng)

def crossover_genomes(parent_a: StrategyGenome, parent_b: StrategyGenome,
                      rng: random.Random | None = None) -> StrategyGenome:
    rng = rng or random.Random()
    parent_a.validate(); parent_b.validate()
    a, b = parent_a.to_dict(), parent_b.to_dict()
    values = {}
    for field in fields(StrategyGenome):
        name = field.name
        if name in ("version", "symbols"):
            values[name] = a[name] if rng.random() < 0.5 else b[name]
        elif name in ("signal_families", "entry_methods"):
            source = a[name] if rng.random() < 0.5 else b[name]
            other = b[name] if source is a[name] else a[name]
            merged = list(dict.fromkeys(source + other))
            rng.shuffle(merged)
            values[name] = merged[:rng.randint(1, min(len(merged), 4 if name == "signal_families" else len(config.ENTRY_METHODS)))]
        else:
            values[name] = a[name] if rng.random() < 0.5 else b[name]
    return _repair(values, rng)
