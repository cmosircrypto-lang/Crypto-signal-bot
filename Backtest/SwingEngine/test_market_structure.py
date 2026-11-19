
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the current folder importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_structure import analyze_market_structure


def make_test_data():
    """
    Synthetic candles for testing:
    - Swing High at index 5
    - Swing Low at index 15
    - Bullish breakout at index 11
    - Bearish reversal at index 21
    """

    n = 30

    data = {
        "ts": list(range(n)),
        "open": [7.0] * n,
        "high": [10.0] * n,
        "low": [5.0] * n,
        "close": [7.0] * n,
        "volume": [100.0] * n,
    }

    # Swing High
    data["high"][5] = 20.0

    # Bullish breakout above Swing High
    data["open"][11] = 20.5
    data["high"][11] = 22.0
    data["low"][11] = 20.0
    data["close"][11] = 21.0

    # Swing Low
    data["low"][15] = 1.0

    # Bearish break below Swing Low
    data["open"][21] = 0.8
    data["high"][21] = 1.2
    data["low"][21] = 0.4
    data["close"][21] = 0.5

    return pd.DataFrame(data)


def test_swing_confirmation():
    df = make_test_data()

    result = analyze_market_structure(
        df,
        left=5,
        right=5,
    )

    # Swing High at candle 5 is confirmed at candle 10
    assert not result.loc[5, "pivot_high_confirmed"]
    assert result.loc[10, "pivot_high_confirmed"]
    assert result.loc[10, "pivot_high_price"] == 20.0
    assert result.loc[10, "pivot_high_ts"] == 5

    # Swing Low at candle 15 is confirmed at candle 20
    assert not result.loc[15, "pivot_low_confirmed"]
    assert result.loc[20, "pivot_low_confirmed"]
    assert result.loc[20, "pivot_low_price"] == 1.0
    assert result.loc[20, "pivot_low_ts"] == 15

    print("PASS: swing confirmation timing")


def test_no_lookahead():
    df = make_test_data()

    full = analyze_market_structure(
        df,
        left=5,
        right=5,
    )

    # At index 9, the Swing High is not yet confirmed
    partial = analyze_market_structure(
        df.iloc[:10].copy(),
        left=5,
        right=5,
    )

    assert not partial["pivot_high_confirmed"].any()

    # At index 10, the Swing High becomes confirmed
    partial = analyze_market_structure(
        df.iloc[:11].copy(),
        left=5,
        right=5,
    )

    assert partial.loc[10, "pivot_high_confirmed"]

    # All generated columns must remain consistent
    # when historical data is processed in shorter prefixes.
    columns = [
        "pivot_high_confirmed",
        "pivot_low_confirmed",
        "pivot_high_price",
        "pivot_low_price",
        "pivot_high_ts",
        "pivot_low_ts",
        "high_label",
        "low_label",
        "last_swing_high",
        "last_swing_low",
        "structure_event",
        "break_level",
        "structure_trend",
    ]

    missing_value = "__MISSING__"

    for end in [9, 10, 11, 19, 20, 21, 25]:

        prefix = analyze_market_structure(
            df.iloc[:end + 1].copy(),
            left=5,
            right=5,
        )

        expected = full.loc[
            :end,
            columns,
        ].reset_index(drop=True)

        actual = prefix[
            columns
        ].reset_index(drop=True)

        # Normalize all missing values and data types
        # before comparing the two DataFrames.
        expected = expected.astype(object).where(
            pd.notna(expected),
            missing_value,
        )

        actual = actual.astype(object).where(
            pd.notna(actual),
            missing_value,
        )

        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=False,
            check_exact=True,
        )

    print("PASS: no-lookahead prefix consistency")


def test_bos_and_choch():
    df = make_test_data()

    result = analyze_market_structure(
        df,
        left=5,
        right=5,
    )

    # Break above the confirmed Swing High
    assert result.loc[11, "structure_event"] == "BOS_BULLISH"

    # Break below the confirmed Swing Low
    # after a bullish trend
    assert result.loc[21, "structure_event"] == "CHOCH_BEARISH"

    assert result.loc[11, "structure_trend"] == "BULLISH"
    assert result.loc[21, "structure_trend"] == "BEARISH"

    print("PASS: BOS and CHoCH detection")


def test_close_based_breakout():
    df = make_test_data()

    # Wick crosses the Swing High,
    # but candle close remains below it.
    df.loc[11, "open"] = 19.0
    df.loc[11, "high"] = 22.0
    df.loc[11, "low"] = 18.0
    df.loc[11, "close"] = 19.5

    result = analyze_market_structure(
        df,
        left=5,
        right=5,
    )

    assert result.loc[11, "structure_event"] != "BOS_BULLISH"

    print("PASS: wick-only breakout is ignored")


def run_all_tests():
    test_swing_confirmation()
    test_no_lookahead()
    test_bos_and_choch()
    test_close_based_breakout()

    print("\nALL MARKET STRUCTURE TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
