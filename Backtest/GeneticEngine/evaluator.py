"""Compatibility adapter that delegates to the existing SwingEngine.

This first adapter deliberately supports only the baseline parameter values
that the current SwingEngine exposes without modification. Parameter injection
for genetic genes is a separate, test-gated change to the legacy engine.
"""

from __future__ import annotations

from typing import Any

from .genome import StrategyGenome, GenomeValidationError


def assert_legacy_compatible(genome: StrategyGenome) -> None:
    """Reject genes that the current SwingEngine cannot yet honor."""
    genome.validate()
    fixed = {
        "swing_left": 5,
        "swing_right": 5,
        "ob_threshold": 1.0,
        "atr_period": 14,
        "ob_lookback": 10,
    }
    for field, expected in fixed.items():
        actual = getattr(genome, field)
        if actual != expected:
            raise GenomeValidationError(
                f"{field}={actual!r} is not wired into SwingEngine yet; "
                f"the compatibility adapter currently requires {expected!r}"
            )


def prepare_genome_inputs(
    symbol_data: dict,
    genome: StrategyGenome,
    progress_monitor: Any = None,
):
    """Prepare signals using the existing runner with baseline-compatible genes."""
    assert_legacy_compatible(genome)

    # Import lazily because the legacy runner uses script-style local imports.
    from backtest_runner import prepare_signals

    return prepare_signals(
        symbol_data,
        methods=genome.entry_methods,
        zone_type=genome.zone_type,
        touch_rule=genome.touch_rule,
        require_4h_alignment=genome.require_4h_alignment,
        progress_monitor=progress_monitor,
    )


def evaluate_prepared_genome(
    symbol_data: dict,
    signals_by_symbol: dict,
    order_blocks_by_symbol: dict,
    genome: StrategyGenome,
    contract_specs=None,
    initial_capital: float = 100.0,
    progress_context=None,
):
    """Evaluate a prepared candidate through the original portfolio simulator."""
    assert_legacy_compatible(genome)
    from backtest_runner import simulate_scenario, build_btc_trend_frames

    btc_frames = (
        build_btc_trend_frames(symbol_data)
        if genome.btc_filter_mode is not None
        else None
    )
    return simulate_scenario(
        symbol_data,
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
