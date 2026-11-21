"""Exact old-vs-optimized signal-engine parity on the cached 76-symbol OKX data.

This deliberately compares the frozen pre-optimization signal generator with
the current SwingEngine generator. Order Blocks are detected once and shared,
so any mismatch is isolated to entry-signal generation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[3]
SWING_DIR = ROOT / "Backtest" / "SwingEngine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SWING_DIR) not in sys.path:
    sys.path.insert(0, str(SWING_DIR))

from data_loader import load_okx_data  # noqa: E402
from order_block import detect_order_blocks  # noqa: E402
from signal_engine import generate_entry_signals as optimized_generate  # noqa: E402
from Backtest.GeneticEngine.genome import StrategyGenome  # noqa: E402
from Backtest.GeneticEngine.tests.signal_engine_reference import (  # noqa: E402
    generate_entry_signals as reference_generate,
)


def _progress(symbol: str, engine: str):
    last = {"stage": None, "percent": -1}

    def report(stage: str, completed: int, total: int) -> None:
        percent = int(completed * 100 / total) if total else 100
        if stage != last["stage"] or percent >= last["percent"] + 10 or percent == 100:
            print(
                f"[PROGRESS] {symbol} | {engine} | {stage} "
                f"{completed}/{total} ({percent}%)",
                flush=True,
            )
            last["stage"] = stage
            last["percent"] = percent

    return report


def main() -> None:
    data_root = ROOT / "Backtest" / "data" / "okx"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Cached OKX dataset is missing: {data_root}. "
            "Enable run_signal_engine_reference_parity in the workflow."
        )

    started = time.monotonic()
    print("[PARITY] Loading and validating cached OKX 15m/1h/4h data.", flush=True)

    def load_progress(timeframe: str, symbol: str, index: int, total: int) -> None:
        if timeframe == "4h":
            print(
                f"[DATA] {timeframe}: {index}/{total} ({index * 100 // total}%) {symbol}",
                flush=True,
            )

    data = load_okx_data(
        data_root,
        expected_symbols=76,
        progress_callback=load_progress,
    )
    symbols = sorted(data)
    if len(symbols) != 76:
        raise AssertionError(f"Expected 76 cached symbols, received {len(symbols)}")

    genome = StrategyGenome.baseline()
    print(
        "[CONFIG] baseline "
        f"methods={genome.entry_methods}, zone={genome.zone_type}, "
        f"touch={genome.touch_rule}, 4h_alignment={genome.require_4h_alignment}",
        flush=True,
    )

    total_signals = 0
    total_obs = 0
    symbol_started = time.monotonic()

    for index, symbol in enumerate(symbols, start=1):
        item = data[symbol]
        one_hour = item.frames["1h"]
        fifteen = item.frames["15m"]
        four_hour = item.frames["4h"]

        print(
            f"[SYMBOL] {index}/76 ({index * 100 // 76}%) {symbol} "
            f"| candles 15m={len(fifteen)} 1h={len(one_hour)} 4h={len(four_hour)}",
            flush=True,
        )

        # Shared OB input prevents unrelated OB-generation differences from
        # contaminating this signal-engine-only comparison.
        obs = detect_order_blocks(
            one_hour,
            threshold=genome.ob_threshold,
            atr_period=genome.atr_period,
            lookback=genome.ob_lookback,
        )

        reference_signals = reference_generate(
            symbol=symbol,
            candles_15m=fifteen,
            order_blocks_1h=obs,
            candles_4h=four_hour,
            methods=genome.entry_methods,
            zone_type=genome.zone_type,
            touch_rule=genome.touch_rule,
            require_4h_alignment=genome.require_4h_alignment,
            progress_callback=_progress(symbol, "REFERENCE"),
        )
        optimized_signals = optimized_generate(
            symbol=symbol,
            candles_15m=fifteen,
            order_blocks_1h=obs,
            candles_4h=four_hour,
            methods=genome.entry_methods,
            zone_type=genome.zone_type,
            touch_rule=genome.touch_rule,
            require_4h_alignment=genome.require_4h_alignment,
            progress_callback=_progress(symbol, "OPTIMIZED"),
        )

        try:
            assert_frame_equal(
                reference_signals,
                optimized_signals,
                check_dtype=True,
                check_exact=True,
                check_like=False,
                obj=f"signal_engine[{symbol}]",
            )
        except AssertionError as exc:
            print(
                f"[MISMATCH] {symbol}: reference_rows={len(reference_signals)} "
                f"optimized_rows={len(optimized_signals)}",
                flush=True,
            )
            raise AssertionError(
                f"Old-vs-optimized signal parity failed for {symbol}:\n{exc}"
            ) from exc

        total_obs += len(obs)
        total_signals += len(reference_signals)
        elapsed = time.monotonic() - symbol_started
        print(
            f"[PASS] {symbol} | OBs={len(obs)} | exact_signals={len(reference_signals)} "
            f"| completed={index}/76 ({index * 100 // 76}%) "
            f"| elapsed={elapsed / 60:.1f}m",
            flush=True,
        )
        symbol_started = time.monotonic()

    elapsed = time.monotonic() - started
    print(
        "[PARITY] PASS: frozen reference and optimized signal engine are "
        "exactly equal across all 76 cached OKX symbols. "
        f"order_blocks={total_obs}, signals={total_signals}, "
        f"elapsed={elapsed / 60:.1f}m.",
        flush=True,
    )


if __name__ == "__main__":
    main()
