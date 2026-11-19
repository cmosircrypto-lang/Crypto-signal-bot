
import pandas as pd

from ob_zone_rules import (
    calculate_zone_touches,
    compare_zone_rules,
)


def make_order_blocks():
    return pd.DataFrame([
        {
            "direction": "BULLISH",
            "ob_index": 2,
            "bos_index": 3,
            "body_low": 101.0,
            "body_high": 102.0,
            "wick_low": 100.0,
            "wick_high": 103.0,
        }
    ])


def make_candles():
    rows = [
        # 0-2: before BOS
        (104.5, 105.0, 104.0, 104.5),
        (104.5, 105.0, 104.0, 104.5),
        (104.5, 105.0, 104.0, 104.5),

        # 3: BOS candle - ignored
        (101.0, 102.0, 100.5, 101.5),

        # 4: wick-only touch
        (103.5, 104.0, 102.5, 103.5),

        # 5: still wick zone - no new touch
        (103.5, 104.0, 102.2, 103.5),

        # 6: completely outside both zones
        (104.5, 105.0, 104.0, 104.5),

        # 7: body and wick touch
        (101.0, 102.0, 100.5, 101.5),

        # 8: still inside both zones
        (101.2, 102.0, 100.5, 101.8),

        # 9: completely outside both zones
        (104.5, 105.0, 104.0, 104.5),

        # 10: wick-only re-entry
        (103.5, 104.0, 102.5, 103.5),
    ]

    return pd.DataFrame([
        {
            "ts": i * 900000,
            "open": row[0],
            "high": row[1],
            "low": row[2],
            "close": row[3],
        }
        for i, row in enumerate(rows)
    ])


def test_bos_and_reentry():
    touches = calculate_zone_touches(
        make_order_blocks(),
        make_candles(),
    )

    assert 3 not in touches["candle_index"].tolist()
    assert touches["candle_index"].tolist() == [4, 7, 10]

    print("PASS: BOS candle and earlier candles ignored")
    print("PASS: consecutive overlaps count once")
    print("PASS: exit and re-entry create new events")


def test_independent_zone_counts():
    touches = calculate_zone_touches(
        make_order_blocks(),
        make_candles(),
    )

    assert touches["body_touched"].tolist() == [
        False, True, False
    ]

    assert touches["wick_touched"].tolist() == [
        True, True, True
    ]

    assert touches["body_touch_number"].tolist() == [
        0, 1, 0
    ]

    assert touches["wick_touch_number"].tolist() == [
        1, 2, 3
    ]

    print("PASS: independent body and wick touch events")


def test_comparison():
    touches = calculate_zone_touches(
        make_order_blocks(),
        make_candles(),
    )

    comparison = compare_zone_rules(touches)

    result = comparison.iloc[0]

    assert result["body_touch_count"] == 1
    assert result["wick_touch_count"] == 3

    print("PASS: body and wick comparison")


def test_invalid_inputs():
    ob = make_order_blocks()
    candles = make_candles()

    missing_bos = ob.drop(columns=["bos_index"])

    try:
        calculate_zone_touches(missing_bos, candles)
        raise AssertionError("Missing bos_index was accepted")
    except ValueError:
        pass

    invalid_range = ob.copy()
    invalid_range.loc[0, "wick_low"] = 104.0

    try:
        calculate_zone_touches(invalid_range, candles)
        raise AssertionError("Invalid zone was accepted")
    except ValueError:
        pass

    print("PASS: invalid inputs rejected")


def main():
    test_bos_and_reentry()
    test_independent_zone_counts()
    test_comparison()
    test_invalid_inputs()

    print("ALL ORDER BLOCK ZONE TESTS PASSED")


if __name__ == "__main__":
    main()
