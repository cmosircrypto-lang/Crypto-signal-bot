import random
import unittest

from Backtest.GeneticEngine.free_evolution_runner import (
    _random_genome, _mate, _mutate, _signal_key, METHOD_SETS,
)
from Backtest.GeneticEngine.genome import StrategyGenome


class FreeEvolutionGenomeTests(unittest.TestCase):
    def test_random_genomes_are_valid_and_explore_signal_genes(self):
        rng = random.Random(1234)
        genomes = [_random_genome(rng) for _ in range(40)]
        for genome in genomes:
            genome.validate()
            self.assertIn(genome.entry_methods, METHOD_SETS)
            self.assertIn(genome.zone_type, ("body", "wick"))
            self.assertIn(genome.touch_rule, ("first", "max_two", "progressive"))
        self.assertGreater(len({_signal_key(g) for g in genomes}), 1)

    def test_crossover_and_mutation_preserve_genome_contract(self):
        rng = random.Random(77)
        a, b = _random_genome(rng), _random_genome(rng)
        child = _mate(a, b, rng)
        child.validate()
        mutated = _mutate(child, rng)
        mutated.validate()
        self.assertEqual(mutated.swing_left, StrategyGenome().swing_left)
        self.assertEqual(mutated.stop_mode, StrategyGenome().stop_mode)

    def test_signal_cache_key_ignores_execution_only_genes(self):
        a = StrategyGenome()
        b = StrategyGenome(risk_percent=0.25, reward_ratio=3.0, btc_filter_mode="1d_4h")
        self.assertEqual(_signal_key(a), _signal_key(b))


if __name__ == "__main__":
    unittest.main()
