
import pandas as pd

from ob_touch_rules import (
    evaluate_ob_touches,
    compare_touch_rules,
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

        # 3: BOS candle - must be ignored
        (101.0, 102.0, 100.5, 101.5),

        # 4: first touch
        (101.0, 102.0, 100.5, 101.5),

        # 5: still inside - not a new touch
        (101.2, 102.5, 100.5, 101.8),

        # 6: completely outside - reset
        (104.5, 105.0, 104.0, 104.5),

        # 7: second touch
        (101.0, 102.0, 100.5, 101.5),

        # 8: still inside - not a new touch
        (101.2, 102.5, 100.5, 101.8),

        # 9: completely outside - reset
        (104.5, 105.0, 104.0, 104.5),

        # 10: third touch
        (101.0, 102.0, 100.5, 101.5),
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


def test_touch_count():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
    )

    assert len(touches) == 3
    assert touches["candle_index"].tolist() == [4, 7, 10]
    assert touches["touch_number"].tolist() == [1, 2, 3]

    print("PASS: BOS candle ignored; counting starts after BOS")
    print("PASS: consecutive candles count as one touch")
    print("PASS: full exit and re-entry create a new touch")


def test_touch_rules():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
        decay_per_touch=0.25,
    )

    assert touches["first_touch_only_valid"].tolist() == [
        True, False, False
    ]

    assert touches["max_two_touches_valid"].tolist() == [
        True, True, False
    ]

    assert touches["progressive_score"].tolist() == [
        1.0, 0.25, 0.0625
    ]

    print("PASS: first-touch-only rule")
    print("PASS: maximum-two-touches rule")
    print("PASS: progressive decay scores")


def test_comparison():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
    )

    comparison = compare_touch_rules(touches)

    assert len(comparison) == 1

    result = comparison.iloc[0]

    assert result["total_touches"] == 3
    assert result["first_touch_only_count"] == 1
    assert result["max_two_touches_count"] == 2
    assert abs(result["progressive_total_score"] - 1.3125) < 1e-9

    print("PASS: all touch rules compared")


def test_invalid_inputs():
    ob = make_order_blocks()
    candles = make_candles()

    missing_bos = ob.drop(columns=["bos_index"])

    try:
        evaluate_ob_touches(missing_bos, candles)
        raise AssertionError("Missing bos_index was accepted")
    except ValueError:
        pass

    try:
        evaluate_ob_touches(ob, candles, decay_per_touch=0)
        raise AssertionError("Invalid decay was accepted")
    except ValueError:
        pass

    invalid_range = ob.copy()
    invalid_range.loc[0, "wick_low"] = 104

    try:
        evaluate_ob_touches(invalid_range, candles)
        raise AssertionError("Invalid zone range was accepted")
    except ValueError:
        pass

    print("PASS: invalid inputs rejected")


def main():
    test_touch_count()
    test_touch_rules()
    test_comparison()
    test_invalid_inputs()

    print("ALL ORDER BLOCK TOUCH TESTS PASSED")


if __name__ == "__main__":
    main()
