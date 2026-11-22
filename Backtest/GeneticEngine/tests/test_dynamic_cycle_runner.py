from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from Backtest.GeneticEngine.dynamic_cycle_runner import build_symbol_candidate_pool, run_symbol_cycle
from Backtest.GeneticEngine.genome import StrategyGenome
from Backtest.GeneticEngine.checkpoint import CheckpointCompatibilityError
from Backtest.GeneticEngine import config

def valid_result(setup=0.85, direction="LONG"):
    return {
        "direction": direction, "signal_timestamp": "2026-11-22T12:00:00Z",
        "historical": {"average_r": 0.4, "max_drawdown_pct": 8.0,
            "profit_factor": 1.6, "positive_window_ratio": 0.8,
            "trade_count": 80, "validation_score": 0.7, "complexity_penalty": 0.1},
        "current_setup_quality": setup, "has_current_signal": True, "candle_closed": True,
    }

class DynamicCycleRunnerTests(unittest.TestCase):
    def test_candidate_pool_covers_all_families_hybrids_and_directions(self):
        genome = StrategyGenome(
            entry_methods=("DIRECT", "ENGULFING", "PIN_BAR"),
            zone_type="wick", touch_rule="max_two", btc_filter_mode="1d_4h",
            require_4h_alignment=True,
        )
        pool = build_symbol_candidate_pool("bnb", [genome])
        self.assertEqual({x["direction"] for x in pool}, {"LONG", "SHORT"})
        self.assertEqual({tuple(x["signal_families"]) for x in pool},
                         {tuple([f]) for f in config.STRATEGY_FAMILIES} |
                         set(__import__("itertools").combinations(config.STRATEGY_FAMILIES, 2)))
        self.assertEqual(len({x["candidate_id"] for x in pool}), len(pool))
        self.assertTrue(all(x["genome"]["symbols"] == ["BNB"] for x in pool))
        self.assertTrue(all(len(x["genome"]["entry_methods"]) == 1 for x in pool))
        swing = [x for x in pool if x["signal_families"] == ["SWING_OB"]]
        self.assertEqual({x["method"] for x in swing}, set(config.ENTRY_METHODS))
        self.assertEqual(len(pool), 104)

    def test_evolved_hybrid_genome_and_its_parameters_survive_candidate_expansion(self):
        hybrid = StrategyGenome(
            signal_families=("EMA_CROSS_PRO", "SMC_LIQUIDITY_SWEEP"),
            family_combination="VOTE", min_family_agreement=2,
            ema_cross_lookback=9, ema_volume_mult=1.8,
            smc_sweep_lookback=20, smc_min_rr=1.5,
        )
        pool = build_symbol_candidate_pool("sol", [hybrid])
        matches = [x for x in pool if x["signal_families"] == ["EMA_CROSS_PRO", "SMC_LIQUIDITY_SWEEP"]]
        self.assertTrue(matches)
        self.assertTrue(any(x["genome"]["ema_cross_lookback"] == 9 for x in matches))
        self.assertTrue(any(x["genome"]["smc_sweep_lookback"] == 20 for x in matches))
        self.assertTrue(any(x["genome"]["family_combination"] == "VOTE" for x in matches))

    def test_resumes_after_interruption_without_repeating_finished_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BNB-cycle.json"
            candidates = [{"candidate_id":"direct","method":"DIRECT"},
                          {"candidate_id":"engulf","method":"ENGULFING"},
                          {"candidate_id":"pin","method":"PIN_BAR"}]
            calls = []
            def interrupted(spec):
                calls.append(spec["candidate_id"])
                if spec["candidate_id"] == "engulf": raise RuntimeError("simulated interruption")
                return valid_result()
            metadata = {"code_hash":"code-v1","config_hash":"cfg-v1","dataset_hash":"ohlcv-v1"}
            with self.assertRaises(RuntimeError):
                run_symbol_cycle(symbol="BNB",cycle_id="cycle-001",candidates=candidates,
                    evaluator=interrupted,checkpoint_path=path,checkpoint_metadata=metadata)
            self.assertEqual(calls,["direct","engulf"])
            resumed_calls=[]
            def resumed(spec):
                resumed_calls.append(spec["candidate_id"])
                return valid_result(0.95 if spec["candidate_id"]=="pin" else 0.8)
            decision=run_symbol_cycle(symbol="BNB",cycle_id="cycle-001",candidates=candidates,
                evaluator=resumed,checkpoint_path=path,checkpoint_metadata=metadata)
            self.assertEqual(resumed_calls,["engulf","pin"])
            self.assertEqual(decision.status,"SELECTED")
            self.assertEqual(decision.winner.candidate_id,"pin")

    def test_long_and_short_are_ranked_independently_before_symbol_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            candidates=[{"candidate_id":"long","method":"EMA_CROSS","direction":"LONG"},
                        {"candidate_id":"short","method":"EMA_CROSS","direction":"SHORT"}]
            results={"long":valid_result(0.75,"LONG"),"short":valid_result(0.95,"SHORT")}
            decision=run_symbol_cycle(symbol="BNB",cycle_id="direction-cycle",candidates=candidates,
                evaluator=lambda spec:results[spec["candidate_id"]],
                checkpoint_path=Path(directory)/"BNB.json",
                checkpoint_metadata={"code_hash":"c1","config_hash":"cfg1","dataset_hash":"d1"})
            self.assertEqual(decision.winner.direction,"SHORT")
            self.assertEqual(decision.winner.candidate_id,"short")

    def test_direction_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"direction-specific evaluator mismatch"):
                run_symbol_cycle(symbol="ADA",cycle_id="bad-direction",
                    candidates=[{"candidate_id":"short","method":"DIRECT","direction":"SHORT"}],
                    evaluator=lambda _:valid_result(direction="LONG"),
                    checkpoint_path=Path(directory)/"ADA.json",
                    checkpoint_metadata={"code_hash":"c1","config_hash":"cfg1","dataset_hash":"d1"})

    def test_different_cycle_or_dataset_rejects_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"XRP-cycle.json"
            kwargs=dict(symbol="XRP",cycle_id="cycle-1",
                candidates=[{"candidate_id":"one","method":"DIRECT"}],
                evaluator=lambda _:valid_result(),checkpoint_path=path,
                checkpoint_metadata={"code_hash":"code-v1","config_hash":"cfg-v1","dataset_hash":"data-v1"})
            run_symbol_cycle(**kwargs)
            with self.assertRaises(CheckpointCompatibilityError):
                run_symbol_cycle(**{**kwargs,"cycle_id":"cycle-2"})

    def test_no_current_setup_means_no_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            result=valid_result(); result["has_current_signal"]=False
            decision=run_symbol_cycle(symbol="ADA",cycle_id="cycle-1",
                candidates=[{"candidate_id":"direct","method":"DIRECT"}],
                evaluator=lambda _:result,checkpoint_path=Path(directory)/"ADA.json",
                checkpoint_metadata={"code_hash":"c1","config_hash":"cfg1","dataset_hash":"d1"})
            self.assertEqual(decision.status,"NO_SIGNAL")
            self.assertIsNone(decision.winner)

if __name__ == "__main__": unittest.main()
