
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from order_block import (
    detect_order_blocks,
    compare_order_block_thresholds,
)


def make_test_data():
    """
    Create synthetic OHLC data with:
    - One bearish candle before bullish BOS
    - One bullish candle before bearish BOS
    - Controlled ATR and displacement
    """

    n = 50

    data = {
        "ts": list(range(n)),
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
    }

    # Create an initial range for ATR stabilization.
    for i in range(14):
        data["open"][i] = 100.0
        data["high"][i] = 101.0
        data["low"][i] = 99.0
        data["close"][i] = 100.0

    # Bearish OB candle
    data["open"][18] = 100.0
    data["high"][18] = 101.0
    data["low"][18] = 98.0
    data["close"][18] = 99.0

    # Bullish BOS candle
    data["open"][20] = 100.0
    data["high"][20] = 104.0
    data["low"][20] = 100.0
    data["close"][20] = 104.0

    # Bullish OB candle
    data["open"][28] = 100.0
    data["high"][28] = 102.0
    data["low"][28] = 99.0
    data["close"][28] = 101.0

    # Bearish BOS candle
    data["open"][30] = 100.0
    data["high"][30] = 100.0
    data["low"][30] = 96.0
    data["close"][30] = 96.0

    return pd.DataFrame(data)


def make_test_structure():
    """
    Supply controlled BOS events for isolated
    Order Block tests.
    """

    structure = pd.DataFrame(
        {
            "structure_event": [None] * 50,
        }
    )

    structure.loc[20, "structure_event"] = "BOS_BULLISH"
    structure.loc[30, "structure_event"] = "BOS_BEARISH"

    return structure


def test_bullish_order_block():
    df = make_test_data()
    structure = make_test_structure()

    result = detect_order_blocks(
        df,
        threshold=1.0,
        atr_period=14,
        lookback=10,
        structure=structure,
    )

    bullish = result[
        result["direction"] == "BULLISH"
    ]

    assert len(bullish) == 1

    ob = bullish.iloc[0]

    assert ob["ob_index"] == 18
    assert ob["bos_index"] == 20
    assert ob["ob_ts"] == 18
    assert ob["bos_ts"] == 20

    print("PASS: bullish Order Block detection")


def test_bearish_order_block():
    df = make_test_data()
    structure = make_test_structure()

    result = detect_order_blocks(
        df,
        threshold=1.0,
        atr_period=14,
        lookback=10,
        structure=structure,
    )

    bearish = result[
        result["direction"] == "BEARISH"
    ]

    assert len(bearish) == 1

    ob = bearish.iloc[0]

    assert ob["ob_index"] == 28
    assert ob["bos_index"] == 30
    assert ob["ob_ts"] == 28
    assert ob["bos_ts"] == 30

    print("PASS: bearish Order Block detection")


def test_threshold_comparison():
    df = make_test_data()
    structure = make_test_structure()

    results = compare_order_block_thresholds(
        df,
        thresholds=(1.0, 1.25, 1.5),
        atr_period=14,
        lookback=10,
        structure=structure,
    )

    assert set(results.keys()) == {
        1.0,
        1.25,
        1.5,
    }

    counts = {
        threshold: len(result)
        for threshold, result in results.items()
    }

    assert counts[1.0] >= counts[1.25]
    assert counts[1.25] >= counts[1.5]

    print("PASS: threshold comparison")

    for threshold, count in counts.items():
        print(
            f"  ATR threshold {threshold}: "
            f"{count} Order Blocks"
        )


def test_no_ob_before_bos():
    df = make_test_data()

    structure = make_test_structure()

    # Remove BOS confirmation.
    structure.loc[20, "structure_event"] = None
    structure.loc[30, "structure_event"] = None

    result = detect_order_blocks(
        df,
        threshold=1.0,
        atr_period=14,
        lookback=10,
        structure=structure,
    )

    assert result.empty

    print("PASS: no Order Block without BOS")


def test_order_block_price_ranges():
    df = make_test_data()
    structure = make_test_structure()

    result = detect_order_blocks(
        df,
        threshold=1.0,
        atr_period=14,
        lookback=10,
        structure=structure,
    )

    bullish = result[
        result["direction"] == "BULLISH"
    ].iloc[0]

    assert bullish["body_low"] == 99.0
    assert bullish["body_high"] == 100.0
    assert bullish["wick_low"] == 98.0
    assert bullish["wick_high"] == 101.0

    bearish = result[
        result["direction"] == "BEARISH"
    ].iloc[0]

    assert bearish["body_low"] == 100.0
    assert bearish["body_high"] == 101.0
    assert bearish["wick_low"] == 99.0
    assert bearish["wick_high"] == 102.0

    print("PASS: Order Block price ranges")


def run_all_tests():
    test_bullish_order_block()
    test_bearish_order_block()
    test_threshold_comparison()
    test_no_ob_before_bos()
    test_order_block_price_ranges()

    print("\nALL ORDER BLOCK TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
