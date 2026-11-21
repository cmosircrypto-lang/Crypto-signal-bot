"""Full cached-OKX-dataset parity gate for the Genetic Engine adapter.

Run only after the workflow has downloaded Backtest/data/okx and network access
to the public OKX instruments endpoint is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[3]
SWING_DIR = ROOT / "Backtest" / "SwingEngine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SWING_DIR) not in sys.path:
    sys.path.insert(0, str(SWING_DIR))

from backtest_runner import (  # noqa: E402
    ProgressMonitor,
    prepare_signals,
    simulate_scenario,
)
from data_loader import load_okx_data  # noqa: E402
from okx_contract_specs import OKXContractSpecs  # noqa: E402
from Backtest.GeneticEngine.evaluator import (  # noqa: E402
    evaluate_prepared_genome,
    prepare_genome_inputs,
)
from Backtest.GeneticEngine.genome import StrategyGenome  # noqa: E402
from Backtest.GeneticEngine.config import (  # noqa: E402
    EXPECTED_SYMBOLS,
    INITIAL_EQUITY,
)


def compare_frame_maps(label: str, expected: dict, actual: dict) -> int:
    if set(expected) != set(actual):
        raise AssertionError(
            f"{label} symbol keys differ: "
            f"expected_only={sorted(set(expected) - set(actual))}, "
            f"actual_only={sorted(set(actual) - set(expected))}"
        )
    rows = 0
    for symbol in sorted(expected):
        assert_frame_equal(
            expected[symbol],
            actual[symbol],
            check_dtype=True,
            check_exact=True,
            obj=f"{label}[{symbol}]",
        )
        rows += len(expected[symbol])
    return rows


def main() -> None:
    data_root = ROOT / "Backtest" / "data" / "okx"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Cached OKX dataset is missing: {data_root}. "
            "Enable run_genetic_data_parity in the workflow."
        )

    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    try:
        monitor.set_stage("genetic_data_parity_load", "loading and validating all cached OKX symbols")
        data = load_okx_data(
            data_root,
            expected_symbols=EXPECTED_SYMBOLS,
            progress_callback=lambda tf, symbol, index, total: (
                monitor.progress("genetic_dataset_loading", index, total, f"{tf}: {symbol}")
                if tf == "4h" else None
            ),
        )
        symbols = sorted(data)
        if len(symbols) != EXPECTED_SYMBOLS:
            raise AssertionError(f"Expected {EXPECTED_SYMBOLS} symbols, got {len(symbols)}")
        print(f"[PARITY] Loaded {len(symbols)} symbols across 15m/1h/4h.", flush=True)

        monitor.set_stage("genetic_contract_specs", "loading current public OKX USDT linear SWAP specifications")
        contract_specs = OKXContractSpecs()
        specs = contract_specs.load()
        missing_specs = sorted(set(symbols) - set(specs))
        if missing_specs:
            raise AssertionError(
                "Missing live OKX contract specifications for cached symbols: "
                + ", ".join(missing_specs)
            )
        print(
            f"[PARITY] Contract specifications cover all {len(symbols)} cached symbols "
            f"({len(specs)} live USDT linear SWAP contracts available).",
            flush=True,
        )

        genome = StrategyGenome.baseline()
        monitor.set_stage("genetic_direct_prepare", "preparing baseline signals with direct SwingEngine calls")
        direct_signals, direct_obs = prepare_signals(
            data,
            methods=genome.entry_methods,
            zone_type=genome.zone_type,
            touch_rule=genome.touch_rule,
            require_4h_alignment=genome.require_4h_alignment,
            progress_monitor=monitor,
        )

        monitor.set_stage("genetic_adapter_prepare", "preparing the same baseline through the Genetic Engine adapter")
        adapter_signals, adapter_obs = prepare_genome_inputs(
            data,
            genome,
            progress_monitor=monitor,
        )

        signal_rows = compare_frame_maps(
            "entry_signals", direct_signals, adapter_signals
        )
        ob_rows = compare_frame_maps(
            "order_blocks", direct_obs, adapter_obs
        )
        signal_count = sum(len(frame) for frame in direct_signals.values())
        ob_count = sum(len(frame) for frame in direct_obs.values())
        print(
            f"[PARITY] Exact signal/order-block equality: "
            f"signals={signal_count} ({signal_rows} rows), "
            f"order_blocks={ob_count} ({ob_rows} rows).",
            flush=True,
        )

        monitor.set_stage("genetic_direct_simulation", "simulating direct SwingEngine baseline on all 76 symbols")
        direct_summary, direct_trades, _ = simulate_scenario(
            data,
            direct_signals,
            direct_obs,
            risk_percent=genome.risk_percent,
            reward_ratio=genome.reward_ratio,
            contract_specs=contract_specs,
            initial_capital=INITIAL_EQUITY,
        )

        monitor.set_stage("genetic_adapter_simulation", "simulating baseline through Genetic Engine adapter on all 76 symbols")
        adapter_summary, adapter_trades, _ = evaluate_prepared_genome(
            data,
            adapter_signals,
            adapter_obs,
            genome,
            contract_specs=contract_specs,
            initial_capital=INITIAL_EQUITY,
        )

        if direct_summary != adapter_summary:
            raise AssertionError(
                "Backtest summary mismatch between direct SwingEngine and Genetic Engine adapter:\n"
                f"direct={direct_summary!r}\nadapter={adapter_summary!r}"
            )
        assert_frame_equal(
            direct_trades,
            adapter_trades,
            check_dtype=True,
            check_exact=True,
            obj="complete_trade_ledger",
        )
        print(
            "[PARITY] PASS: exact equality across all 76 symbols, "
            f"summary={direct_summary!r}, trades={len(direct_trades)}.",
            flush=True,
        )
        monitor.set_stage("genetic_data_parity_complete", "full cached-data parity gate passed")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
