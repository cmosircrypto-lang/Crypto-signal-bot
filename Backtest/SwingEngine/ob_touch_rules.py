
import pandas as pd


TOUCH_RULES = (
    "first_touch_only",
    "max_two_touches",
    "progressive_decay",
)


def evaluate_ob_touches(
    order_blocks,
    candles,
    decay_per_touch=0.25,
):
    if not 0 < decay_per_touch < 1:
        raise ValueError("decay_per_touch must be between 0 and 1")

    required_ob = {
        "direction",
        "ob_index",
        "body_low",
        "body_high",
        "wick_low",
        "wick_high",
    }

    required_candles = {"ts", "high", "low"}

    if not isinstance(order_blocks, pd.DataFrame):
        raise TypeError("order_blocks must be a DataFrame")

    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles must be a DataFrame")

    if not required_ob.issubset(order_blocks.columns):
        raise ValueError("Missing Order Block columns")

    if not required_candles.issubset(candles.columns):
        raise ValueError("Missing candle columns")

    if candles.empty:
        raise ValueError("candles cannot be empty")

    if not candles["ts"].is_monotonic_increasing:
        raise ValueError("Candles must be sorted by timestamp")

    if candles["ts"].duplicated().any():
        raise ValueError("Duplicate candle timestamps")

    results = []

    for ob_id, ob in order_blocks.iterrows():
        direction = ob["direction"]

        if direction not in ("BULLISH", "BEARISH"):
            raise ValueError(f"Invalid direction: {direction}")

        ob_index = int(ob["ob_index"])

        if ob_index < 0 or ob_index >= len(candles):
            raise ValueError("Invalid ob_index")

        if direction == "BULLISH":
            body_low = float(ob["body_low"])
            body_high = float(ob["body_high"])
            wick_low = float(ob["wick_low"])
            wick_high = float(ob["wick_high"])
        else:
            body_low = float(ob["body_low"])
            body_high = float(ob["body_high"])
            wick_low = float(ob["wick_low"])
            wick_high = float(ob["wick_high"])

        if not body_low <= body_high:
            raise ValueError("Invalid body range")

        if not wick_low <= wick_high:
            raise ValueError("Invalid wick range")

        touch_count = 0

        for candle_index in range(ob_index + 1, len(candles)):
            candle = candles.iloc[candle_index]

            candle_high = float(candle["high"])
            candle_low = float(candle["low"])

            if candle_low > candle_high:
                raise ValueError("Invalid candle range")

            touched_body = (
                candle_low <= body_high
                and candle_high >= body_low
            )

            touched_wick = (
                candle_low <= wick_high
                and candle_high >= wick_low
            )

            if not touched_wick:
                continue

            touch_count += 1

            results.append({
                "ob_id": ob_id,
                "direction": direction,
                "candle_index": candle_index,
                "ts": candle["ts"],
                "touch_number": touch_count,
                "body_touched": touched_body,
                "wick_touched": True,
                "first_touch_only_valid": touch_count == 1,
                "max_two_touches_valid": touch_count <= 2,
                "progressive_score": (
                    decay_per_touch ** (touch_count - 1)
                ),
            })

    return pd.DataFrame(results)


def compare_touch_rules(
    order_blocks,
    candles,
    decay_per_touch=0.25,
):
    touches = evaluate_ob_touches(
        order_blocks,
        candles,
        decay_per_touch,
    )

    if touches.empty:
        return pd.DataFrame(columns=[
            "rule",
            "valid_touches",
            "total_touches",
            "average_score",
        ])

    total = len(touches)

    progressive = touches["progressive_score"]

    return pd.DataFrame([
        {
            "rule": "first_touch_only",
            "valid_touches": int(
                touches["first_touch_only_valid"].sum()
            ),
            "total_touches": total,
            "average_score": float(
                touches.loc[
                    touches["first_touch_only_valid"],
                    "progressive_score",
                ].mean()
            ),
        },
        {
            "rule": "max_two_touches",
            "valid_touches": int(
                touches["max_two_touches_valid"].sum()
            ),
            "total_touches": total,
            "average_score": float(
                touches.loc[
                    touches["max_two_touches_valid"],
                    "progressive_score",
                ].mean()
            ),
        },
        {
            "rule": "progressive_decay",
            "valid_touches": total,
            "total_touches": total,
            "average_score": float(progressive.mean()),
        },
    ])
