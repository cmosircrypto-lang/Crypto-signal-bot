from __future__ import annotations

import unittest
import pandas as pd

from Backtest.GeneticEngine.family_signals import evaluate_genome_signals
from Backtest.GeneticEngine.genome import StrategyGenome


def trending_frame(n=90):
    rows=[]
    price=100.0
    for i in range(n):
        open_=price
        close=price+0.25
        high=close+0.15
        low=open_-0.15
        if i==n-1:
            close=price+5.0
            high=close+0.2
        rows.append({"timestamp":i*3600000,"open":open_,"high":high,"low":low,
                     "close":close,"volume":1.0})
        price=close
    return pd.DataFrame(rows)


def squeeze_breakout_frame(n=90):
    rows=[]
    for i in range(n):
        center=100.0
        amplitude=0.25*(1-i/n)
        close=center+amplitude*(1 if i%2 else -1)
        open_=center-amplitude*(1 if i%2 else -1)
        high=max(open_,close)+0.08
        low=min(open_,close)-0.08
        if i==n-1:
            open_=100.0
            close=105.0
            high=105.2
            low=99.8
        rows.append({"timestamp":i*3600000,"open":open_,"high":high,"low":low,
                     "close":close,"volume":1.0})
    return pd.DataFrame(rows)


class FamilySignalTests(unittest.TestCase):
    def test_momentum_breakout_emits_only_requested_direction(self):
        frames={"1h":trending_frame()}
        genome=StrategyGenome(signal_families=("MOMENTUM_BREAKOUT",),
            family_combination="SINGLE",momentum_lookback=20,
            momentum_breakout_buffer_atr=0.0,momentum_min_adx=14.0,
            momentum_volume_mult=0.8)
        long_signal=evaluate_genome_signals(frames,genome,"LONG")
        short_signal=evaluate_genome_signals(frames,genome,"SHORT")
        self.assertIsNotNone(long_signal)
        self.assertIsNone(short_signal)
        self.assertEqual(long_signal["direction"],"LONG")
        self.assertTrue(long_signal["candle_closed"])
        self.assertGreater(long_signal["take_profit"],long_signal["entry"])
        self.assertLess(long_signal["stop_loss"],long_signal["entry"])

    def test_swing_adapter_rejects_unclosed_or_wrong_direction_signal(self):
        genome=StrategyGenome(signal_families=("SWING_OB",),family_combination="SINGLE")
        swing={"direction":"LONG","entry":100,"stop_loss":98,"take_profit":104,
               "setup_quality":0.8,"signal_timestamp":"t1","candle_closed":False}
        self.assertIsNone(evaluate_genome_signals({},genome,"LONG",swing_signal=swing))
        swing["candle_closed"]=True
        self.assertIsNone(evaluate_genome_signals({},genome,"SHORT",swing_signal=swing))
        result=evaluate_genome_signals({},genome,"LONG",swing_signal=swing)
        self.assertEqual(result["method"],"SWING_OB")

    def test_all_combination_requires_every_family(self):
        genome=StrategyGenome(signal_families=("SWING_OB","EMA_CROSS_PRO"),
            family_combination="ALL",min_family_agreement=2)
        swing={"direction":"LONG","entry":100,"stop_loss":98,"take_profit":104,
               "setup_quality":0.8,"signal_timestamp":"t1","candle_closed":True}
        self.assertIsNone(evaluate_genome_signals({},genome,"LONG",swing_signal=swing))

    def test_vote_combines_two_real_families_and_tracks_agreement(self):
        frames={"1h":squeeze_breakout_frame()}
        genome=StrategyGenome(signal_families=("MOMENTUM_BREAKOUT","VOLATILITY_SQUEEZE"),
            family_combination="VOTE",min_family_agreement=2,
            momentum_lookback=20,momentum_breakout_buffer_atr=0.0,
            momentum_min_adx=14.0,momentum_volume_mult=0.8,
            squeeze_bb_period=20,squeeze_width_percentile=30.0,
            squeeze_expansion_atr=1.2)
        result=evaluate_genome_signals(frames,genome,"LONG")
        self.assertIsNotNone(result)
        self.assertEqual(result["family_votes"],2)
        self.assertEqual(set(result["families_agreed"]),
                         {"MOMENTUM_BREAKOUT","VOLATILITY_SQUEEZE"})
        self.assertAlmostEqual(result["rr"],2.0,places=6)
        self.assertTrue(result["candle_closed"])

    def test_future_rows_are_not_needed_for_current_signal(self):
        frames={"1h":trending_frame()}
        genome=StrategyGenome(signal_families=("MOMENTUM_BREAKOUT",),
            family_combination="SINGLE",momentum_lookback=20,
            momentum_breakout_buffer_atr=0.0,momentum_min_adx=14.0,
            momentum_volume_mult=0.8)
        result=evaluate_genome_signals(frames,genome,"LONG")
        self.assertEqual(result["signal_timestamp"],str(frames["1h"].iloc[-1]["timestamp"]))


if __name__ == "__main__":
    unittest.main()
