import unittest
from types import SimpleNamespace

import pandas as pd

from backtest_runner import (
    simulate_scenario,
    run_matrix,
)


def fixture_data(direction="BUY", second_candle=None):
    candles = pd.DataFrame([
        {"ts": 0, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
        {"ts": 900_000, **(second_candle or {
            "open": 100.0, "high": 108.0, "low": 94.0, "close": 101.0, "volume": 1.0
        })},
        {"ts": 1_800_000, "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1.0},
    ])
    item = SimpleNamespace(frames={"15m": candles})
    signals = pd.DataFrame([{
        "symbol": "BTC-USDT-SWAP",
        "method": "DIRECT",
        "direction": direction,
        "ob_id": 0,
        "ob_ts": 0,
        "bos_ts": 0,
        "touch_number": 1,
        "touch_score": 1.0,
        "signal_index": 0,
        "signal_ts": 900_000,
        "entry_price": 100.0,
        "htf_trend": "NOT_USED",
        "confirmation": "DIRECT",
    }])
    ob = pd.DataFrame([{
        "direction": "BULLISH" if direction == "BUY" else "BEARISH",
        "wick_low": 95.0,
        "wick_high": 105.0,
        "body_low": 97.0,
        "body_high": 103.0,
    }])
    return {"BTC-USDT-SWAP": item}, {"BTC-USDT-SWAP": signals}, {"BTC-USDT-SWAP": ob}


class BacktestRunnerTests(unittest.TestCase):
    def test_entry_candle_is_not_used_for_exit_and_sl_wins_tie(self):
        data, signals, obs = fixture_data()
        summary, trades, system = simulate_scenario(
            data, signals, obs, risk_percent=1.0, reward_ratio=1.5
        )
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "SL")
        self.assertAlmostEqual(trades.iloc[0]["pnl"], -1.0)
        self.assertEqual(trades.iloc[0]["opened_at"], 900_000)
        self.assertEqual(trades.iloc[0]["closed_at"], 1_800_000)

    def test_sell_stop_uses_far_upper_wick(self):
        data, signals, obs = fixture_data(
            direction="SELL",
            second_candle={"open": 100.0, "high": 106.0, "low": 92.0, "close": 99.0, "volume": 1.0},
        )
        summary, trades, _ = simulate_scenario(
            data, signals, obs, risk_percent=1.0, reward_ratio=2.0
        )
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "SL")
        self.assertAlmostEqual(trades.iloc[0]["exit_price"], 105.0)

    def test_progress_reporting_preserves_results(self):
        data, signals, obs = fixture_data()
        baseline, baseline_trades, _ = simulate_scenario(
            data, signals, obs, risk_percent=1.0, reward_ratio=1.5
        )
        observed, observed_trades, _ = simulate_scenario(
            data, signals, obs, risk_percent=1.0, reward_ratio=1.5
        )
        self.assertEqual(baseline, observed)
        pd.testing.assert_frame_equal(baseline_trades, observed_trades)

    def test_matrix_contains_all_nine_risk_rr_pairs(self):
        data, signals, obs = fixture_data()
        summary, ledgers = run_matrix(data, signals, obs)
        self.assertEqual(len(summary), 9)
        self.assertEqual(len(ledgers), 9)
        self.assertEqual(
            set(summary["risk_percent"]),
            {0.05, 1.0, 1.5},
        )
        self.assertEqual(
            set(summary["reward_ratio"]),
            {1.5, 2.0, 2.5},
        )


if __name__ == "__main__":
    unittest.main()
