"""Compatibility adapter that delegates to the existing SwingEngine.

This first adapter deliberately supports only the baseline parameter values
that the current SwingEngine exposes without modification. Parameter injection
for genetic genes is a separate, test-gated change to the legacy engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .genome import StrategyGenome, GenomeValidationError


def _legacy_runner():
    """Import the script-oriented SwingEngine modules from a package context."""
    swing_engine_dir = str(Path(__file__).resolve().parents[1] / "SwingEngine")
    if swing_engine_dir not in sys.path:
        sys.path.insert(0, swing_engine_dir)
    from backtest_runner import prepare_signals, simulate_scenario, build_btc_trend_frames
    return prepare_signals, simulate_scenario, build_btc_trend_frames


def assert_legacy_compatible(genome: StrategyGenome) -> None:
    """Reject genes that the current SwingEngine cannot yet honor."""
    genome.validate()
    fixed = {
        "swing_left": 5,
        "swing_right": 5,
        "ob_threshold": 1.0,
        "atr_period": 14,
        "ob_lookback": 10,
        "stop_mode": "OB_WICK",
        "atr_stop_multiplier": 1.5,
    }
    for field, expected in fixed.items():
        actual = getattr(genome, field)
        if actual != expected:
            raise GenomeValidationError(
                f"{field}={actual!r} is not wired into SwingEngine yet; "
                f"the compatibility adapter currently requires {expected!r}"
            )


def _select_symbols(symbol_data: dict, genome: StrategyGenome) -> dict:
    if not genome.symbols:
        return symbol_data
    requested = {symbol.strip().upper() for symbol in genome.symbols}
    available = {symbol.strip().upper(): symbol for symbol in symbol_data}
    missing = sorted(requested - set(available))
    if missing:
        raise ValueError(f"Genome references symbols absent from dataset: {missing}")
    return {available[symbol]: symbol_data[available[symbol]] for symbol in sorted(requested)}


def prepare_genome_inputs(
    symbol_data: dict,
    genome: StrategyGenome,
    progress_monitor: Any = None,
    checkpoint_dir: str | Path | None = None,
):
    """Prepare signals using the existing runner with baseline-compatible genes."""
    assert_legacy_compatible(genome)
    prepare_signals, _, _ = _legacy_runner()
    selected_data = _select_symbols(symbol_data, genome)
    return prepare_signals(
        selected_data,
        methods=genome.entry_methods,
        zone_type=genome.zone_type,
        touch_rule=genome.touch_rule,
        require_4h_alignment=genome.require_4h_alignment,
        progress_monitor=progress_monitor,
        checkpoint_dir=checkpoint_dir,
    )


def evaluate_prepared_genome(
    symbol_data: dict,
    signals_by_symbol: dict,
    order_blocks_by_symbol: dict,
    genome: StrategyGenome,
    contract_specs=None,
    initial_capital: float = 100.0,
    progress_context=None,
    btc_trend_frames=None,
):
    """Evaluate a prepared candidate through the original portfolio simulator."""
    assert_legacy_compatible(genome)
    _, simulate_scenario, build_btc_trend_frames = _legacy_runner()
    selected_data = _select_symbols(symbol_data, genome)
    if genome.symbols:
        selected_symbols = set(selected_data)
        signals_by_symbol = {key: value for key, value in signals_by_symbol.items() if key in selected_symbols}
        order_blocks_by_symbol = {key: value for key, value in order_blocks_by_symbol.items() if key in selected_symbols}
    btc_frames = btc_trend_frames
    if btc_frames is None and genome.btc_filter_mode is not None:
        btc_frames = build_btc_trend_frames(selected_data)
    return simulate_scenario(
        selected_data,
        signals_by_symbol,
        order_blocks_by_symbol,
        risk_percent=genome.risk_percent,
        reward_ratio=genome.reward_ratio,
        contract_specs=contract_specs,
        initial_capital=initial_capital,
        progress_context=progress_context,
        filter_mode=genome.btc_filter_mode,
        btc_trend_frames=btc_frames,
    )
