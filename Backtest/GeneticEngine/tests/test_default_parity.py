from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

REPO_ROOT = Path(__file__).resolve().parents[3]
SWING_ENGINE_DIR = REPO_ROOT / "Backtest" / "SwingEngine"
if str(SWING_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(SWING_ENGINE_DIR))

from backtest_runner import prepare_signals, simulate_scenario
from Backtest.GeneticEngine.evaluator import (
    prepare_genome_inputs,
    evaluate_prepared_genome,
)
from Backtest.GeneticEngine.genome import StrategyGenome


def make_candles(count: int, interval_ms: int) -> pd.DataFrame:
    index = np.arange(count, dtype=float)
    close = 100.0 + 0.025 * index + 2.5 * np.sin(index / 4.7) + 0.8 * np.sin(index / 1.9)
    open_price = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_price, close) + 0.25 + 0.1 * (1 + np.sin(index / 3.0))
    low = np.minimum(open_price, close) - 0.25 - 0.1 * (1 + np.cos(index / 3.0))
    return pd.DataFrame({
        "ts": (index.astype("int64") * interval_ms),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(count, 100.0),
    })


def make_symbol_data():
    frames = {
        "15m": make_candles(800, 15 * 60 * 1000),
        "1h": make_candles(200, 60 * 60 * 1000),
        "4h": make_candles(50, 4 * 60 * 60 * 1000),
    }
    return {
        "ETH-USDT-SWAP": SimpleNamespace(
            symbol="ETH-USDT-SWAP",
            frames=frames,
        )
    }


class DefaultGenomeParityTests(unittest.TestCase):
    def test_default_genome_matches_direct_swing_engine_pipeline(self):
        data = make_symbol_data()
        genome = StrategyGenome.baseline()

        direct_signals, direct_obs = prepare_signals(
            data,
            methods=(
                "DIRECT", "ENGULFING", "PIN_BAR",
                "M15_STRUCTURE_BREAK",
            ),
            zone_type="body",
            touch_rule="first",
            require_4h_alignment=False,
        )
        adapter_signals, adapter_obs = prepare_genome_inputs(data, genome)

        self.assertEqual(set(direct_signals), set(adapter_signals))
        self.assertEqual(set(direct_obs), set(adapter_obs))
        for symbol in direct_signals:
            assert_frame_equal(direct_signals[symbol], adapter_signals[symbol])
            assert_frame_equal(direct_obs[symbol], adapter_obs[symbol])

        direct_summary, direct_trades, _ = simulate_scenario(
            data,
            direct_signals,
            direct_obs,
            risk_percent=genome.risk_percent,
            reward_ratio=genome.reward_ratio,
            contract_specs=None,
            initial_capital=100.0,
        )
        adapter_summary, adapter_trades, _ = evaluate_prepared_genome(
            data,
            adapter_signals,
            adapter_obs,
            genome,
            contract_specs=None,
            initial_capital=100.0,
        )

        self.assertEqual(direct_summary, adapter_summary)
        assert_frame_equal(direct_trades, adapter_trades)


if __name__ == "__main__":
    unittest.main()
