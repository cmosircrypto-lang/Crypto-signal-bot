
import pandas as pd

from ob_touch_rules import (
    evaluate_ob_touches,
    compare_touch_rules,
)


def make_candles():
    rows = []

    # OB is created at index 2.
    # Touches happen at indexes 4, 6 and 8.
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

    # First touch.
    rows[4]["low"] = 101.0
    rows[4]["high"] = 103.0

    # Second touch.
    rows[6]["low"] = 100.5
    rows[6]["high"] = 102.5

    # Third touch.
    rows[8]["low"] = 101.5
    rows[8]["high"] = 103.5

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


def test_touch_detection():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
    )

    assert len(touches) == 3
    assert touches["touch_number"].tolist() == [1, 2, 3]

    print("PASS: three Order Block touches detected")


def test_first_touch_only():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
    )

    valid = touches["first_touch_only_valid"].tolist()

    assert valid == [True, False, False]

    print("PASS: first touch only rule")


def test_max_two_touches():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
    )

    valid = touches["max_two_touches_valid"].tolist()

    assert valid == [True, True, False]

    print("PASS: maximum two touches rule")


def test_progressive_decay():
    touches = evaluate_ob_touches(
        make_order_blocks(),
        make_candles(),
        decay_per_touch=0.25,
    )

    scores = touches["progressive_score"].tolist()

    assert scores == [1.0, 0.25, 0.0625]
    assert scores[0] > scores[1] > scores[2]

    print("PASS: progressive score decay")


def test_comparison():
    comparison = compare_touch_rules(
        make_order_blocks(),
        make_candles(),
    )

    assert len(comparison) == 3

    valid_counts = comparison.set_index("rule")[
        "valid_touches"
    ]

    assert valid_counts["first_touch_only"] == 1
    assert valid_counts["max_two_touches"] == 2
    assert valid_counts["progressive_decay"] == 3

    print("PASS: all three rules compared")


def test_invalid_decay():
    try:
        evaluate_ob_touches(
            make_order_blocks(),
            make_candles(),
            decay_per_touch=1.0,
        )
    except ValueError:
        print("PASS: invalid decay rejected")
        return

    raise AssertionError("Invalid decay was accepted")


def run_all_tests():
    test_touch_detection()
    test_first_touch_only()
    test_max_two_touches()
    test_progressive_decay()
    test_comparison()
    test_invalid_decay()

    print("\nALL ORDER BLOCK TOUCH TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
