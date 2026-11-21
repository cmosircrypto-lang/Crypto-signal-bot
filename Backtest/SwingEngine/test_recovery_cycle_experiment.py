import unittest

from recovery_cycle_experiment import _levels, recovery_risk_amount


class RecoveryCycleExperimentTests(unittest.TestCase):
    def test_exact_recovery_includes_initial_trade_target_profit(self):
        # Prior losses 3, initial target profit 2.5, RR 2.5 => risk 2.2.
        risk = recovery_risk_amount(3.0, 1.0, 2.5, 2.5,
                                    "initial_target_profit", "exact_recovery", 1)
        self.assertAlmostEqual(risk, 2.2)

    def test_breakeven_recovery_only_covers_prior_losses(self):
        risk = recovery_risk_amount(3.0, 1.0, 2.5, 2.5,
                                    "breakeven", "exact_recovery", 1)
        self.assertAlmostEqual(risk, 1.2)

    def test_double_size_progression(self):
        # With a 0.5 base quantity and a 1.0 stop distance, the
        # first recovery is 1.0 quantity (2x) and the third is 2.0? 
        # The requested risk equals target quantity times stop distance.
        first_recovery_risk = recovery_risk_amount(
            1, 0.5, 1.25, 2.5, "initial_target_profit", "double_size", 0,
            stop_distance=1.0, base_quantity=0.5,
        )
        third_recovery_risk = recovery_risk_amount(
            1, 0.5, 1.25, 2.5, "breakeven", "double_size", 2,
            stop_distance=1.0, base_quantity=0.5,
        )
        self.assertEqual(first_recovery_risk, 1.0)  # 0.5 * 2
        self.assertEqual(third_recovery_risk, 2.0)  # 0.5 * 4

    def test_atr_stop_and_rr_levels(self):
        signal = {"entry_price": 100.0, "atr14": 2.0, "direction": "BUY"}
        self.assertEqual(_levels(signal, 1.5, 2.0), (97.0, 106.0))
        signal["direction"] = "SELL"
        self.assertEqual(_levels(signal, 1.5, 2.0), (103.0, 94.0))


if __name__ == "__main__":
    unittest.main()
