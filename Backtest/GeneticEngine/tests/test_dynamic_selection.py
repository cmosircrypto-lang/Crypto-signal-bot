from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Backtest.GeneticEngine.dynamic_selector import (
    HistoricalMetrics, ScoringPolicy, SignalCandidate,
    historical_quality_score, score_candidate, select_cycle_signals,
)
from Backtest.GeneticEngine.checkpoint import (
    CheckpointCompatibilityError, CheckpointIntegrityError,
    load_checkpoint, save_checkpoint,
)


def metrics(**overrides):
    values = dict(average_r=0.35, max_drawdown_pct=8.0, profit_factor=1.6,
                  positive_window_ratio=0.75, trade_count=80,
                  complexity_penalty=0.1)
    values.update(overrides)
    return HistoricalMetrics(**values)


def candidate(symbol, candidate_id, method, setup=0.8, **kwargs):
    values = dict(direction="LONG", signal_timestamp="2026-11-22T12:00:00Z",
                  historical=metrics(), current_setup_quality=setup)
    values.update(kwargs)
    return SignalCandidate(symbol, candidate_id, method, **values)


class DynamicSelectorTests(unittest.TestCase):
    def test_historical_score_is_bounded(self):
        self.assertGreaterEqual(historical_quality_score(metrics()), 0.0)
        self.assertLessEqual(historical_quality_score(metrics()), 1.0)

    def test_best_current_candidate_wins_per_symbol(self):
        decisions = select_cycle_signals([
            candidate("bnb", "a", "PIN_BAR", setup=0.70),
            candidate("BNB", "b", "M15_STRUCTURE_BREAK", setup=0.95),
            candidate("XRP", "c", "ENGULFING", setup=0.90),
        ])
        self.assertEqual(decisions["BNB"].winner.candidate_id, "b")
        self.assertEqual(decisions["XRP"].winner.candidate_id, "c")

    def test_no_signal_if_candidate_has_no_current_setup(self):
        decision = select_cycle_signals([
            candidate("BTC", "a", "DIRECT", has_current_signal=False)
        ])["BTC"]
        self.assertIsNone(decision.winner)
        self.assertEqual(decision.status, "NO_SIGNAL")
        self.assertIn("NO_CURRENT_SIGNAL",
                      decision.ranked_candidates[0].rejection_reasons)

    def test_open_candle_is_rejected(self):
        scored = score_candidate(candidate("ETH", "a", "DIRECT", candle_closed=False))
        self.assertFalse(scored.eligible)
        self.assertIn("CANDLE_NOT_CLOSED", scored.rejection_reasons)

    def test_low_sample_and_bad_drawdown_are_rejected(self):
        scored = score_candidate(candidate(
            "SOL", "a", "DIRECT",
            historical=metrics(trade_count=5, max_drawdown_pct=35.0),
        ))
        self.assertFalse(scored.eligible)
        self.assertIn("INSUFFICIENT_TRADES", scored.rejection_reasons)
        self.assertIn("DRAWDOWN_LIMIT", scored.rejection_reasons)

    def test_ties_are_deterministic(self):
        decision = select_cycle_signals([
            candidate("ADA", "z", "DIRECT"),
            candidate("ADA", "a", "PIN_BAR"),
        ])["ADA"]
        self.assertEqual(decision.winner.candidate_id, "a")


class CheckpointTests(unittest.TestCase):
    def test_round_trip_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "symbol.json"
            metadata = {"code_hash": "abc", "dataset_hash": "data1", "symbol": "BTC"}
            save_checkpoint(path, payload={"next_candidate": 12}, metadata=metadata)
            loaded = load_checkpoint(path, expected_metadata=metadata)
            self.assertEqual(loaded["payload"]["next_candidate"], 12)
            with self.assertRaises(CheckpointCompatibilityError):
                load_checkpoint(path, expected_metadata={"dataset_hash": "data2"})

    def test_checksum_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "symbol.json"
            save_checkpoint(path, payload={"done": [1, 2]}, metadata={"symbol": "ETH"})
            envelope = json.loads(path.read_text())
            envelope["payload"]["done"].append(3)
            path.write_text(json.dumps(envelope))
            with self.assertRaises(CheckpointIntegrityError):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
