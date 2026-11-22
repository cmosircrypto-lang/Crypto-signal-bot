from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from Backtest.GeneticEngine.historical_evaluator import (
    build_complete_btc_daily, evaluate_candidate_walk_forward,
)
from Backtest.GeneticEngine.genome import StrategyGenome


def make_frames(count=1200):
    start=1_700_000_000_000
    start=(start//86_400_000)*86_400_000
    def make(tf, step, n):
        rows=[]
        for i in range(n):
            price=100.0
            rows.append({"ts":start+i*step,"open":price,"high":102.0,
                         "low":98.0,"close":101.0,"volume":10.0})
        return pd.DataFrame(rows)
    return {
        "15m":make("15m",900_000,count),
        "1h":make("1h",3_600_000,count//4),
        "4h":make("4h",14_400_000,count//16),
    }


class HistoricalEvaluatorTests(unittest.TestCase):
    def test_daily_aggregation_uses_only_complete_contiguous_days(self):
        start=1_700_000_000_000
        start=(start//86_400_000)*86_400_000
        rows=[]
        for i in range(47):
            rows.append({"timestamp":start+i*3_600_000,"open":100+i,
                "high":102+i,"low":99+i,"close":101+i,"volume":1})
        daily=build_complete_btc_daily(pd.DataFrame(rows))
        self.assertEqual(len(daily),1)
        self.assertEqual(daily.iloc[0]["timestamp"],start)
        self.assertEqual(daily.iloc[0]["volume"],24)

    def test_walk_forward_uses_closed_candles_and_sl_first(self):
        frames=make_frames()
        data={"TEST-USDT-SWAP":frames}
        seen=[]
        def fake(frames_at_time, genome, direction, btc_frames=None, swing_signal=None):
            latest=int(frames_at_time["1h"].iloc[-1]["timestamp"])
            seen.append(latest)
            return {
                "method":"MOMENTUM_BREAKOUT","signal_timestamp":str(latest),
                "entry":100.0,"stop_loss":99.0,"take_profit":101.0,
                "risk_distance":1.0,"reward_distance":1.0,"rr":1.0,
                "setup_quality":0.8,"candle_closed":True,
                "families_agreed":["MOMENTUM_BREAKOUT"],
                "family_signal_timestamps":{"MOMENTUM_BREAKOUT":str(latest)},
            }
        genome=StrategyGenome(signal_families=("MOMENTUM_BREAKOUT",),
            family_combination="SINGLE",symbols=("TEST-USDT-SWAP",))
        with patch("Backtest.GeneticEngine.historical_evaluator.evaluate_genome_signals",
                   side_effect=fake):
            metrics=evaluate_candidate_walk_forward(data,"TEST-USDT-SWAP",genome,"LONG")
        self.assertGreater(metrics["trade_count"],0)
        self.assertAlmostEqual(metrics["average_r"],-1.0,places=6)
        self.assertEqual(metrics["profit_factor"],0.0)
        # 85% development cutoff; no 1h candle from the final 15% can be read.
        development_end=int(len(frames["15m"])*0.85)
        oos_start=int(frames["15m"].iloc[development_end]["ts"])
        self.assertTrue(seen)
        self.assertLess(max(seen),oos_start)
        self.assertLessEqual(metrics["positive_window_ratio"],1.0)

    def test_short_trade_direction_uses_short_stop_and_target(self):
        frames=make_frames()
        data={"TEST-USDT-SWAP":frames}
        def fake(frames_at_time, genome, direction, btc_frames=None, swing_signal=None):
            latest=int(frames_at_time["1h"].iloc[-1]["timestamp"])
            return {
                "method":"MOMENTUM_BREAKOUT","signal_timestamp":str(latest),
                "entry":100.0,"stop_loss":101.0,"take_profit":99.0,
                "risk_distance":1.0,"reward_distance":1.0,"rr":1.0,
                "setup_quality":0.8,"candle_closed":True,
                "families_agreed":["MOMENTUM_BREAKOUT"],
                "family_signal_timestamps":{"MOMENTUM_BREAKOUT":str(latest)},
            }
        genome=StrategyGenome(signal_families=("MOMENTUM_BREAKOUT",),
            family_combination="SINGLE",symbols=("TEST-USDT-SWAP",))
        with patch("Backtest.GeneticEngine.historical_evaluator.evaluate_genome_signals",
                   side_effect=fake):
            metrics=evaluate_candidate_walk_forward(data,"TEST-USDT-SWAP",genome,"SHORT")
        self.assertGreater(metrics["trade_count"],0)
        self.assertAlmostEqual(metrics["average_r"],-1.0,places=6)


if __name__=="__main__":
    unittest.main()
