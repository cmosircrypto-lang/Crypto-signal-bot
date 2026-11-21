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


# Differential regression tests for the signal-scan optimization.
import numpy as np
import signal_engine as optimized_signal_engine
from Backtest.GeneticEngine.tests import signal_engine_reference


class SignalEngineOptimizationParityTests(unittest.TestCase):
    @staticmethod
    def _fixtures():
        interval_15m = 15 * 60 * 1000
        interval_4h = 4 * 60 * 60 * 1000
        count = 240
        ts = np.arange(count, dtype=np.int64) * interval_15m

        # Most candles stay outside the OB. Selected candles touch it,
        # including gaps that create distinct touches and confirmation shapes.
        close = np.full(count, 104.0, dtype=float)
        open_price = np.full(count, 104.0, dtype=float)
        high = np.full(count, 104.5, dtype=float)
        low = np.full(count, 103.5, dtype=float)

        def set_candle(i, o, h, l, c):
            open_price[i], high[i], low[i], close[i] = o, h, l, c

        # Bullish engulfing touch after a prior candle outside the zone.
        set_candle(41, 103.0, 103.2, 101.5, 102.0)
        set_candle(42, 102.0, 104.2, 99.5, 104.0)
        # Gap out, then bullish pin-bar touch.
        set_candle(50, 104.0, 104.5, 103.5, 104.0)
        set_candle(51, 103.8, 104.3, 99.0, 104.0)
        # Gap out, then bearish pin-bar touch.
        set_candle(60, 104.0, 104.5, 103.5, 104.0)
        set_candle(61, 100.2, 104.0, 99.8, 100.0)
        # Bearish engulfing touch after a prior candle outside the zone.
        set_candle(70, 102.0, 103.5, 101.5, 103.0)
        set_candle(71, 104.0, 104.2, 99.5, 101.0)

        # Additional separated touches exercise touch counts and decay.
        for i in (85, 100, 115, 130, 145, 160, 175, 190, 205, 220):
            set_candle(i, 100.5, 101.2, 99.0, 100.0)

        candles_15m = pd.DataFrame({
            "ts": ts, "open": open_price, "high": high,
            "low": low, "close": close,
            "volume": np.full(count, 100.0),
        })

        candles_4h = pd.DataFrame({
            "ts": np.arange(60, dtype=np.int64) * interval_4h,
            "open": np.full(60, 100.0),
            "high": np.full(60, 102.0),
            "low": np.full(60, 98.0),
            "close": np.full(60, 100.0),
            "volume": np.full(60, 100.0),
        })

        order_blocks = pd.DataFrame([
            {
                "direction": "BULLISH", "bos_ts": 9 * 60 * 60 * 1000,
                "ob_ts": 8 * 60 * 60 * 1000,
                "body_low": 99.0, "body_high": 101.0,
                "wick_low": 98.0, "wick_high": 102.0,
            },
            {
                "direction": "BEARISH", "bos_ts": 12 * 60 * 60 * 1000,
                "ob_ts": 11 * 60 * 60 * 1000,
                "body_low": 99.0, "body_high": 101.0,
                "wick_low": 98.0, "wick_high": 102.0,
            },
            {
                "direction": "BULLISH", "bos_ts": 18 * 60 * 60 * 1000,
                "ob_ts": 17 * 60 * 60 * 1000,
                "body_low": 99.0, "body_high": 101.0,
                "wick_low": 98.0, "wick_high": 102.0,
            },
        ])

        events = [None] * count
        for i, event in (
            (55, "BOS_BULLISH"),
            (75, "CHOCH_BULLISH"),
            (105, "BOS_BEARISH"),
            (135, "CHOCH_BEARISH"),
            (180, "BOS_BULLISH"),
            (215, "BOS_BEARISH"),
        ):
            events[i] = event

        trends = [
            "BULLISH" if i % 3 else "BEARISH"
            for i in range(len(candles_4h))
        ]
        return candles_15m, candles_4h, order_blocks, events, trends

    def test_optimized_scan_matches_frozen_legacy_for_entry_modes_and_filters(self):
        candles_15m, candles_4h, order_blocks, events, trends = self._fixtures()

        def structure_factory(frame, left=5, right=5, progress_callback=None):
            if len(frame) == len(candles_4h):
                return pd.DataFrame({"structure_trend": trends})
            return pd.DataFrame({"structure_event": events})

        configurations = [
            (("DIRECT",), "body", "first", False),
            (("DIRECT", "ENGULFING", "PIN_BAR"), "body", "max_two", False),
            (("DIRECT", "ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK"),
             "wick", "progressive", False),
            (("ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK"),
             "body", "progressive", True),
            (("DIRECT", "M15_STRUCTURE_BREAK"), "wick", "max_two", True),
            (("DIRECT", "ENGULFING", "PIN_BAR", "M15_STRUCTURE_BREAK"),
             "body", "first", True),
        ]

        for methods, zone_type, touch_rule, require_4h in configurations:
            with self.subTest(
                methods=methods, zone_type=zone_type,
                touch_rule=touch_rule, require_4h=require_4h,
            ):
                kwargs = dict(
                    symbol="ETH-USDT-SWAP",
                    candles_15m=candles_15m,
                    order_blocks_1h=order_blocks,
                    candles_4h=candles_4h if require_4h else None,
                    methods=methods,
                    zone_type=zone_type,
                    touch_rule=touch_rule,
                    decay_per_touch=0.4,
                    require_4h_alignment=require_4h,
                )
                with patch.object(
                    signal_engine_reference, "analyze_market_structure",
                    side_effect=structure_factory,
                ), patch.object(
                    optimized_signal_engine, "analyze_market_structure",
                    side_effect=structure_factory,
                ):
                    legacy = signal_engine_reference.generate_entry_signals(**kwargs)
                    optimized = optimized_signal_engine.generate_entry_signals(**kwargs)

                # Compare every output field, row order, dtype, and index.
                assert_frame_equal(legacy, optimized, check_exact=True)

        # Guard against a vacuous fixture that produces no entries at all.
        with patch.object(
            signal_engine_reference, "analyze_market_structure",
            side_effect=structure_factory,
        ):
            smoke = signal_engine_reference.generate_entry_signals(
                symbol="ETH-USDT-SWAP",
                candles_15m=candles_15m,
                order_blocks_1h=order_blocks,
                methods=("DIRECT",),
                zone_type="body",
                touch_rule="progressive",
            )
        self.assertGreater(len(smoke), 0)
