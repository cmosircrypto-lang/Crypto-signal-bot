from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from signal_engine import generate_entry_signals


class SignalEngineStructureBreakTests(unittest.TestCase):
    def setUp(self):
        self.candles = pd.DataFrame([
            {"ts": 3_600_000 + i * 900_000,
             "open": 105.0, "high": 106.0,
             "low": 104.0, "close": 105.0}
            for i in range(6)
        ])

        # The second 15m candle touches the bullish OB zone.
        self.candles.loc[1, ["open", "high", "low", "close"]] = [
            102.0, 103.0, 100.0, 101.0
        ]

        self.order_blocks = pd.DataFrame([{
            "direction": "BULLISH",
            "bos_ts": 0,
            "body_low": 99.0,
            "body_high": 101.0,
            "wick_low": 98.0,
            "wick_high": 102.0,
            "ob_ts": 0,
        }])

    def _structure(self, event_index=None, event=None):
        events = [None] * len(self.candles)
        if event_index is not None:
            events[event_index] = event
        return pd.DataFrame({"structure_event": events})

    def _run(self, structure):
        with patch(
            "signal_engine.analyze_market_structure",
            return_value=structure,
        ):
            return generate_entry_signals(
                symbol="BTC-USDT-SWAP",
                candles_15m=self.candles,
                order_blocks_1h=self.order_blocks,
                methods=("M15_STRUCTURE_BREAK",),
            )

    def test_scan_starts_at_first_candle_closed_after_ob_bos(self):
        order_blocks = self.order_blocks.copy()
        # BOS candle closes at 4,500,000. Candle 0 closes exactly then
        # and must be skipped; candle 1 is the first eligible touch.
        order_blocks.loc[0, "bos_ts"] = 900_000
        signals = generate_entry_signals(
            symbol="BTC-USDT-SWAP",
            candles_15m=self.candles,
            order_blocks_1h=order_blocks,
            methods=("DIRECT",),
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals.iloc[0]["signal_index"], 1)
        self.assertEqual(signals.iloc[0]["signal_ts"], 5_400_000)

    def test_delayed_structure_break_confirms_after_touch(self):
        signals = self._run(
            self._structure(3, "BOS_BULLISH")
        )

        self.assertEqual(len(signals), 1)
        signal = signals.iloc[0]
        self.assertEqual(signal["method"], "M15_STRUCTURE_BREAK")
        self.assertEqual(signal["signal_index"], 3)
        self.assertEqual(signal["touch_number"], 1)
        self.assertEqual(signal["entry_price"], 105.0)
        self.assertEqual(signal["confirmation"], "M15_STRUCTURE_BREAK")

    def test_progress_callback_reports_order_blocks_and_candles(self):
        calls = []
        structure = self._structure()
        with patch(
            "signal_engine.analyze_market_structure",
            return_value=structure,
        ):
            generate_entry_signals(
                symbol="BTC-USDT-SWAP",
                candles_15m=self.candles,
                order_blocks_1h=self.order_blocks,
                methods=("M15_STRUCTURE_BREAK",),
                progress_callback=lambda stage, done, total: calls.append((stage, done, total)),
            )
        self.assertIn(("order_blocks", 1, 1), calls)
        self.assertIn(("candles", len(self.candles), len(self.candles)), calls)

    def test_structure_break_on_touch_candle_is_ignored(self):
        signals = self._run(
            self._structure(1, "BOS_BULLISH")
        )
        self.assertTrue(signals.empty)

    def test_structure_break_before_touch_is_ignored(self):
        signals = self._run(
            self._structure(0, "BOS_BULLISH")
        )
        self.assertTrue(signals.empty)

    def test_no_structure_break_produces_no_signal(self):
        signals = self._run(self._structure())
        self.assertTrue(signals.empty)

    def test_opposite_direction_break_is_ignored(self):
        signals = self._run(
            self._structure(3, "BOS_BEARISH")
        )
        self.assertTrue(signals.empty)


if __name__ == "__main__":
    unittest.main()
