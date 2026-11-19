
import pandas as pd

from ob_zone_rules import (
    calculate_zone_touches,
    compare_zone_rules,
)


def make_candles():
    rows = []

    for i in range(10):
        rows.append({
            "ts": i * 900000,
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.0,
        })

    # Before OB creation: must not count.
    rows[0]["low"] = 99.0
    rows[0]["high"] = 101.0

    # Wick-only touch.
    rows[4]["low"] = 100.0
    rows[4]["high"] = 100.5

    # Body touch.
    rows[6]["low"] = 101.2
    rows[6]["high"] = 101.8

    # Another wick-only touch.
    rows[8]["low"] = 102.5
    rows[8]["high"] = 103.0

    return pd.DataFrame(rows)


def make_order_blocks():
    return pd.DataFrame([{
        "direction": "BULLISH",
        "ob_index": 2,
        "body_low": 101.0,
        "body_high": 102.0,
        "wick_low": 100.0,
        "wick_high": 103.0,
    }])


def test_zone_touch_detection():
    touches = calculate_zone_touches(
        make_order_blocks(),
        make_candles(),
    )

    assert len(touches) == 3

    assert touches["body_touched"].tolist() == [
        False,
        True,
        False,
    ]

    assert touches["wick_touched"].tolist() == [
        True,
        True,
        True,
    ]

    print("PASS: body and wick touches detected")


def test_touch_counts():
    touches = calculate_zone_touches(
        make_order_blocks(),
        make_candles(),
    )

    assert touches["body_touch_number"].tolist() == [
        0,
        1,
        1,
    ]

    assert touches["wick_touch_number"].tolist() == [
        1,
        2,
        3,
    ]

    print("PASS: independent touch counters")


def test_zone_comparison():
    comparison = compare_zone_rules(
        make_order_blocks(),
        make_candles(),
    )

    counts = comparison.set_index("zone_rule")[
        "total_touches"
    ]

    assert counts["body"] == 1
    assert counts["full_wick"] == 3

    print("PASS: body versus full wick comparison")


def test_no_lookback_before_ob():
    candles = make_candles()
    order_blocks = make_order_blocks()

    touches = calculate_zone_touches(
        order_blocks,
        candles,
    )

    assert not (touches["candle_index"] <= 2).any()

    print("PASS: candles before OB creation ignored")


def test_invalid_zone():
    order_blocks = make_order_blocks()

    order_blocks.loc[0, "wick_low"] = 101.5

    try:
        calculate_zone_touches(
            order_blocks,
            make_candles(),
        )
    except ValueError:
        print("PASS: invalid zone rejected")
        return

    raise AssertionError("Invalid zone was accepted")


def run_all_tests():
    test_zone_touch_detection()
    test_touch_counts()
    test_zone_comparison()
    test_no_lookback_before_ob()
    test_invalid_zone()

    print("\nALL ORDER BLOCK ZONE TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
