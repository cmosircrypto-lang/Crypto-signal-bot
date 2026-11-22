from __future__ import annotations
import random
import unittest
from Backtest.GeneticEngine.evolution import random_genome, mutate_genome, crossover_genomes
from Backtest.GeneticEngine.genome import StrategyGenome

class GeneticOperatorsTests(unittest.TestCase):
    def test_random_genome_is_reproducible_and_valid(self):
        a=random_genome(random.Random(20261122),symbols=("bnb",))
        b=random_genome(random.Random(20261122),symbols=("bnb",))
        self.assertEqual(a,b)
        self.assertEqual(a.symbols,("BNB",))
        a.validate()

    def test_mutation_is_reproducible_and_stays_inside_domains(self):
        p=StrategyGenome.baseline()
        a=mutate_genome(p,random.Random(17),1.0)
        b=mutate_genome(p,random.Random(17),1.0)
        self.assertEqual(a,b)
        a.validate()
        self.assertNotEqual(a,p)

    def test_crossover_inherits_genes_from_both_parents_and_validates(self):
        a=StrategyGenome(signal_families=("EMA_CROSS_PRO",),ema_cross_lookback=12,
                         ema_volume_mult=2.0)
        b=StrategyGenome(signal_families=("SMC_LIQUIDITY_SWEEP",),smc_sweep_lookback=30,
                         smc_min_rr=2.0)
        child=crossover_genomes(a,b,random.Random(91))
        child.validate()
        self.assertTrue(set(child.signal_families).issubset(
            {"EMA_CROSS_PRO","SMC_LIQUIDITY_SWEEP"}))
        self.assertTrue(child.signal_families)

    def test_invalid_mutation_probability_rejected(self):
        with self.assertRaises(ValueError):
            mutate_genome(StrategyGenome.baseline(),random.Random(1),1.1)

if __name__ == "__main__": unittest.main()
