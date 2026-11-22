import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from recovery_cycle_resumable import (
    _cycle_close_status, _levels, _update_open_risk_peaks,
    recovery_risk_amount, _btc_filter_matches, _simulate,
)


class RecoveryCycleExperimentTests(unittest.TestCase):
    def test_fixed_one_dollar_cycle_target_after_losses(self):
        # The recovery cycle always targets $1 net profit, regardless of
        # the initial trade's own TP amount: (losses + $1) / RR.
        risk = recovery_risk_amount(3.0, 1.0, 2.5, 2.5,
                                    "initial_target_profit", "exact_recovery", 1,
                                    target_profit_override=1.0)
        self.assertAlmostEqual(risk, 1.6)

    def test_recovery_formula_uses_total_accumulated_losses(self):
        risk = recovery_risk_amount(2.0, 1.0, 2.5, 2.0,
                                    "initial_target_profit", "exact_recovery", 2,
                                    target_profit_override=1.0)
        self.assertAlmostEqual(risk, 1.5)

    def test_breakeven_recovery_only_covers_prior_losses(self):
        risk = recovery_risk_amount(3.0, 1.0, 2.5, 2.5,
                                    "breakeven", "exact_recovery", 1)
        self.assertAlmostEqual(risk, 1.2)

    def test_exact_recovery_uses_tick_rounded_reward_risk(self):
        risk = recovery_risk_amount(
            5.0, 1.0, 2.0, 2.0,
            "initial_target_profit", "exact_recovery", 1,
            stop_distance=1.1, target_distance=2.09,
            target_profit_override=1.0,
        )
        self.assertAlmostEqual(risk, 6.0 * 1.1 / 2.09)

    def test_double_size_progression(self):
        # With a 0.5 base quantity and a 1.0 stop distance, recovery
        # entries target 1.0 (2x), 2.0 (4x), then 4.0 (8x) quantity.
        # Requested risk equals target quantity times stop distance.
        first_recovery_risk = recovery_risk_amount(
            1, 0.5, 1.25, 2.5, "initial_target_profit", "double_size", 0,
            stop_distance=1.0, base_quantity=0.5,
        )
        third_recovery_risk = recovery_risk_amount(
            1, 0.5, 1.25, 2.5, "breakeven", "double_size", 2,
            stop_distance=1.0, base_quantity=0.5,
        )
        self.assertEqual(first_recovery_risk, 1.0)  # 0.5 * 2
        self.assertEqual(third_recovery_risk, 4.0)  # 0.5 * 8

    def test_cycle_status_requires_actual_breakeven_or_target(self):
        self.assertEqual(_cycle_close_status(2, 2.50, 2.50), "RECOVERED_TARGET")
        self.assertEqual(_cycle_close_status(2, 0.00, 2.50), "RECOVERED_BREAKEVEN")
        self.assertEqual(_cycle_close_status(2, -0.02, 2.50), "CLOSED_SHORTFALL")
        self.assertEqual(_cycle_close_status(0, 2.50, 2.50), "BASE_TP")

    def test_open_risk_peak_uses_same_timestamp_equity_snapshot(self):
        system = SimpleNamespace(execution=SimpleNamespace(
            equity=100.0,
            open_positions={"BTC": SimpleNamespace(risk_amount=2.5)},
        ))
        peak_amount, peak_percent = _update_open_risk_peaks(system, 0.0, 0.0)
        self.assertAlmostEqual(peak_amount, 2.5)
        self.assertAlmostEqual(peak_percent, 2.5)

        # Another position closes at a loss while BTC remains open.
        # The open-risk amount is unchanged, but its percentage must use
        # the reduced equity from this same portfolio snapshot.
        system.execution.equity = 80.0
        peak_amount, peak_percent = _update_open_risk_peaks(
            system, peak_amount, peak_percent
        )
        self.assertAlmostEqual(peak_amount, 2.5)
        self.assertAlmostEqual(peak_percent, 3.125)


    def test_btc_filter_requires_1d_and_4h(self):
        trend = {"1d": "BUY", "4h": "BUY", "1h": "SELL"}
        self.assertTrue(_btc_filter_matches(trend, "BUY", "1d_4h"))
        self.assertFalse(_btc_filter_matches(trend, "BUY", "1d_4h_1h"))

    def test_btc_three_timeframe_filter_requires_all_closed_trends(self):
        trend = {"1d": "SELL", "4h": "SELL", "1h": "SELL"}
        self.assertTrue(_btc_filter_matches(trend, "SELL", "1d_4h_1h"))
        trend["1h"] = "BUY"
        self.assertFalse(_btc_filter_matches(trend, "SELL", "1d_4h_1h"))

    def test_unknown_btc_filter_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _btc_filter_matches({"1d": "BUY", "4h": "BUY"}, "BUY", "1d_4h_15m")


    def test_simulation_checkpoint_resumes_from_last_saved_event(self):
        symbol = "BTC-USDT-SWAP"
        data = {
            symbol: SimpleNamespace(
                symbol=symbol,
                frames={"15m": pd.DataFrame({"ts": [1_000, 2_000, 3_000]})},
            )
        }
        signals = {symbol: pd.DataFrame()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pkl"

            def interrupt_after_first_checkpoint(done, total, timestamp):
                if done == 1:
                    raise RuntimeError("intentional interruption")

            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                _simulate(
                    data, signals, {}, {}, {}, 2.0, 2.0, "checkpoint_test",
                    progress=interrupt_after_first_checkpoint,
                    filter_mode="1d_4h",
                    checkpoint_path=checkpoint,
                    checkpoint_interval=1,
                )

            self.assertTrue(checkpoint.exists())
            resumed = _simulate(
                data, signals, {}, {}, {}, 2.0, 2.0, "checkpoint_test",
                filter_mode="1d_4h",
                checkpoint_path=checkpoint,
                checkpoint_interval=1,
            )
            uninterrupted = _simulate(
                data, signals, {}, {}, {}, 2.0, 2.0, "checkpoint_test",
                filter_mode="1d_4h",
            )

            self.assertEqual(resumed[0], uninterrupted[0])
            self.assertTrue(resumed[1].equals(uninterrupted[1]))
            self.assertTrue(resumed[2].equals(uninterrupted[2]))
            self.assertTrue(resumed[3].equals(uninterrupted[3]))
            self.assertFalse(checkpoint.exists())

    def test_atr_stop_and_rr_levels(self):
        signal = {"entry_price": 100.0, "atr14": 2.0, "direction": "BUY"}
        self.assertEqual(_levels(signal, 1.5, 2.0), (97.0, 106.0))
        signal["direction"] = "SELL"
        self.assertEqual(_levels(signal, 1.5, 2.0), (103.0, 94.0))


if __name__ == "__main__":
    unittest.main()
