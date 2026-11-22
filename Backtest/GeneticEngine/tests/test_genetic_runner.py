from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Backtest.GeneticEngine.config import EvolutionConfig
from Backtest.GeneticEngine.genetic_runner import run_direction_evolution, run_genetic_evolution
from Backtest.GeneticEngine.genome import StrategyGenome


METRICS={
    "average_r":0.35,"max_drawdown_pct":6.0,"profit_factor":1.55,
    "positive_window_ratio":0.75,"trade_count":60,
    "validation_score":0.72,"complexity_penalty":0.08,
}
META={"code_hash":"code-v1","config_hash":"config-v1","dataset_hash":"data-v1"}


class GeneticRunnerTests(unittest.TestCase):
    def config(self):
        return EvolutionConfig(random_seed=12345,smoke_population=8,
            smoke_generations=3,tournament_size=2,
            crossover_probability=0.8,mutation_probability=0.25)

    def test_long_and_short_populations_evolve_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen=set()
            def evaluate(spec):
                seen.add(spec["direction"])
                return METRICS
            long=run_direction_evolution(symbol="BNB-USDT-SWAP",direction="LONG",
                evaluator=evaluate,checkpoint_path=Path(tmp)/"long.json",
                checkpoint_metadata=META,evolution_config=self.config())
            short=run_direction_evolution(symbol="BNB-USDT-SWAP",direction="SHORT",
                evaluator=evaluate,checkpoint_path=Path(tmp)/"short.json",
                checkpoint_metadata=META,evolution_config=self.config())
            self.assertEqual(seen,{"LONG","SHORT"})
            self.assertEqual(long["direction"],"LONG")
            self.assertEqual(short["direction"],"SHORT")
            self.assertIsNotNone(long["champion"])
            self.assertIsNotNone(short["champion"])
            self.assertFalse(long["oos_used_for_selection"])
            self.assertFalse(short["oos_used_for_selection"])
            self.assertEqual(long["evaluated_phenotypes"],short["evaluated_phenotypes"])

    def test_resume_skips_candidates_already_checkpointed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"resume.json"
            calls={"first":0,"second":0}
            def interrupted(spec):
                calls["first"]+=1
                if calls["first"]==3:
                    raise RuntimeError("simulated interruption")
                return METRICS
            with self.assertRaisesRegex(RuntimeError,"simulated interruption"):
                run_direction_evolution(symbol="BNB-USDT-SWAP",direction="LONG",
                    evaluator=interrupted,checkpoint_path=path,
                    checkpoint_metadata=META,evolution_config=self.config())
            self.assertTrue(path.exists())
            def resumed(spec):
                calls["second"]+=1
                return METRICS
            result=run_direction_evolution(symbol="BNB-USDT-SWAP",direction="LONG",
                evaluator=resumed,checkpoint_path=path,
                checkpoint_metadata=META,evolution_config=self.config())
            self.assertEqual(result["eligible_phenotypes"],result["evaluated_phenotypes"])
            self.assertEqual(calls["first"],3)
            self.assertEqual(calls["second"],result["evaluated_phenotypes"]-2)


    def test_symbol_orchestrator_invokes_independent_direction_runs(self):
        dummy_data={"BNB-USDT-SWAP":object()}
        seen=[]
        def fake_walk(data,symbol,genome,direction,**kwargs):
            seen.append((symbol,direction))
            return METRICS
        with tempfile.TemporaryDirectory() as tmp:
            with patch("Backtest.GeneticEngine.historical_evaluator.evaluate_candidate_walk_forward",
                       side_effect=fake_walk):
                result=run_genetic_evolution(
                    all_symbol_data=dummy_data,symbols=("BNB-USDT-SWAP",),
                    checkpoint_dir=tmp,checkpoint_metadata=META,
                    evolution_config=self.config(),profile="smoke",
                )
        self.assertIn(("BNB-USDT-SWAP","LONG"),seen)
        self.assertIn(("BNB-USDT-SWAP","SHORT"),seen)
        self.assertEqual(set(result["BNB-USDT-SWAP"]),{"LONG","SHORT"})

    def test_stale_dataset_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"stale.json"
            run_direction_evolution(symbol="BNB-USDT-SWAP",direction="LONG",
                evaluator=lambda spec:METRICS,checkpoint_path=path,
                checkpoint_metadata=META,evolution_config=self.config())
            changed={**META,"dataset_hash":"different-data"}
            with self.assertRaises(Exception):
                run_direction_evolution(symbol="BNB-USDT-SWAP",direction="LONG",
                    evaluator=lambda spec:METRICS,checkpoint_path=path,
                    checkpoint_metadata=changed,evolution_config=self.config())


if __name__=="__main__":
    unittest.main()
