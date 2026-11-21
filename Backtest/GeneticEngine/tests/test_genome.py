from __future__ import annotations

import unittest

from Backtest.GeneticEngine.config import EvolutionConfig
from Backtest.GeneticEngine.evaluator import assert_legacy_compatible
from Backtest.GeneticEngine.genome import (
    GenomeValidationError,
    StrategyGenome,
)


class StrategyGenomeTests(unittest.TestCase):
    def test_baseline_round_trip(self):
        genome = StrategyGenome.baseline()
        restored = StrategyGenome.from_dict(genome.to_dict())
        self.assertEqual(genome, restored)

    def test_baseline_matches_current_engine_defaults(self):
        genome = StrategyGenome.baseline()
        self.assertEqual(genome.entry_methods, (
            "DIRECT", "ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK"
        ))
        self.assertEqual(genome.zone_type, "body")
        self.assertEqual(genome.touch_rule, "first")
        self.assertEqual((genome.swing_left, genome.swing_right), (5, 5))
        self.assertEqual(
            (genome.ob_threshold, genome.atr_period, genome.ob_lookback),
            (1.0, 14, 10),
        )
        self.assertFalse(genome.require_4h_alignment)
        self.assertIsNone(genome.btc_filter_mode)
        self.assertEqual(genome.stop_mode, "OB_WICK")
        self.assertEqual(genome.atr_stop_multiplier, 1.5)

    def test_legacy_adapter_blocks_unwired_genes(self):
        assert_legacy_compatible(StrategyGenome.baseline())
        with self.assertRaises(GenomeValidationError):
            assert_legacy_compatible(StrategyGenome(swing_left=3))
        with self.assertRaises(GenomeValidationError):
            assert_legacy_compatible(StrategyGenome(stop_mode="ATR"))

    def test_rejects_unknown_method(self):
        with self.assertRaises(GenomeValidationError):
            StrategyGenome(entry_methods=("MAGIC",)).validate()

    def test_rejects_duplicate_methods(self):
        with self.assertRaises(GenomeValidationError):
            StrategyGenome(entry_methods=("DIRECT", "DIRECT")).validate()

    def test_rejects_unknown_payload_field(self):
        with self.assertRaises(GenomeValidationError):
            StrategyGenome.from_dict({"unexpected": True})

    def test_evolution_config_validation(self):
        EvolutionConfig().validate()
        with self.assertRaises(ValueError):
            EvolutionConfig(smoke_population=0).validate()


if __name__ == "__main__":
    unittest.main()
