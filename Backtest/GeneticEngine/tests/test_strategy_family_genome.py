from __future__ import annotations
import unittest
from Backtest.GeneticEngine import config
from Backtest.GeneticEngine.genome import StrategyGenome, GenomeValidationError

class StrategyFamilyGenomeTests(unittest.TestCase):
    def test_baseline_defaults_remain_legacy_swing_only(self):
        g=StrategyGenome.baseline()
        self.assertEqual(g.signal_families,("SWING_OB",))
        self.assertEqual(g.family_combination,"SINGLE")
        self.assertEqual(g.ema_cross_lookback,6)
        self.assertEqual(g.btc_pullback_tp_atr,2.95)
        self.assertEqual(g.smc_sweep_lookback,10)

    def test_all_main_py_family_parameters_round_trip(self):
        g=StrategyGenome(signal_families=("EMA_CROSS_PRO","BTC_ALIGNED_PULLBACK","SMC_LIQUIDITY_SWEEP"),
            family_combination="VOTE",min_family_agreement=2,
            ema_cross_lookback=9,ema_fast_period=7,ema_slow_period=26,
            ema_min_adx_4h=22.0,ema_volume_mult=1.8,
            btc_gap_atr=0.35,btc_touch_atr=0.3,btc_pullback_volume=1.8,
            smc_sweep_lookback=20,smc_sl_buffer_atr=0.3,smc_min_rr=1.5)
        restored=StrategyGenome.from_dict(g.to_dict())
        self.assertEqual(restored,g)

    def test_custom_families_are_available_as_genes(self):
        g=StrategyGenome(signal_families=("MOMENTUM_BREAKOUT","MEAN_REVERSION","VOLATILITY_SQUEEZE"),
            family_combination="ANY",min_family_agreement=1,
            momentum_lookback=30,mean_reversion_zscore=2.5,
            squeeze_bb_period=24,squeeze_width_percentile=15.0)
        g.validate()
        self.assertTrue(set(g.signal_families).issubset(set(config.STRATEGY_FAMILIES)))

    def test_invalid_fast_slow_order_and_hybrid_agreement_are_rejected(self):
        with self.assertRaises(GenomeValidationError):
            StrategyGenome(ema_fast_period=21,ema_slow_period=9).validate()
        with self.assertRaises(GenomeValidationError):
            StrategyGenome(signal_families=("EMA_CROSS_PRO","SMC_LIQUIDITY_SWEEP"),
                family_combination="VOTE",min_family_agreement=3).validate()

    def test_every_family_has_a_bounded_domain(self):
        self.assertEqual(len(config.STRATEGY_FAMILIES),7)
        self.assertEqual(len(config.EMA_CROSS_LOOKBACK_VALUES),5)
        self.assertEqual(len(config.SMC_SWEEP_LOOKBACK_VALUES),6)

if __name__ == "__main__": unittest.main()
