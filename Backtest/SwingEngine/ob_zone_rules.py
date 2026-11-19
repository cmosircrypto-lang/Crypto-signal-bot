
import pandas as pd


ZONE_RULES = (
    "body",
    "full_wick",
)


def validate_inputs(order_blocks, candles):
    if not isinstance(order_blocks, pd.DataFrame):
        raise TypeError("order_blocks must be a DataFrame")

    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles must be a DataFrame")

    required_ob = {
        "direction",
        "ob_index",
        "body_low",
        "body_high",
        "wick_low",
        "wick_high",
    }

    required_candles = {"ts", "high", "low"}

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

    for column in ("high", "low"):
        if not pd.api.types.is_numeric_dtype(candles[column]):
            raise ValueError(f"{column} must be numeric")

        if not candles[column].map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        ).all():
            raise ValueError(f"Invalid values in {column}")

    if (candles["low"] > candles["high"]).any():
        raise ValueError("Invalid candle high/low range")


def calculate_zone_touches(
    order_blocks,
    candles,
):
    validate_inputs(order_blocks, candles)

    results = []

    for ob_id, ob in order_blocks.iterrows():
        ob_index = int(ob["ob_index"])

        if ob_index < 0 or ob_index >= len(candles):
            raise ValueError("Invalid ob_index")

        if ob["direction"] not in ("BULLISH", "BEARISH"):
            raise ValueError("Invalid Order Block direction")

        body_low = float(ob["body_low"])
        body_high = float(ob["body_high"])
        wick_low = float(ob["wick_low"])
        wick_high = float(ob["wick_high"])

        if not body_low <= body_high:
            raise ValueError("Invalid body zone")

        if not wick_low <= wick_high:
            raise ValueError("Invalid wick zone")

        if wick_low > body_low or wick_high < body_high:
            raise ValueError("Wick zone must contain body zone")

        body_touch_count = 0
        wick_touch_count = 0

        for candle_index in range(ob_index + 1, len(candles)):
            candle = candles.iloc[candle_index]

            candle_low = float(candle["low"])
            candle_high = float(candle["high"])

            body_touched = (
                candle_low <= body_high
                and candle_high >= body_low
            )

            wick_touched = (
                candle_low <= wick_high
                and candle_high >= wick_low
            )

            if body_touched:
                body_touch_count += 1

            if wick_touched:
                wick_touch_count += 1

            if body_touched or wick_touched:
                results.append({
                    "ob_id": ob_id,
                    "direction": ob["direction"],
                    "candle_index": candle_index,
                    "ts": candle["ts"],
                    "body_touched": body_touched,
                    "wick_touched": wick_touched,
                    "body_touch_number": body_touch_count,
                    "wick_touch_number": wick_touch_count,
                })

    return pd.DataFrame(results)


def compare_zone_rules(
    order_blocks,
    candles,
):
    touches = calculate_zone_touches(
        order_blocks,
        candles,
    )

    if touches.empty:
        return pd.DataFrame([
            {
                "zone_rule": "body",
                "total_touches": 0,
            },
            {
                "zone_rule": "full_wick",
                "total_touches": 0,
            },
        ])

    return pd.DataFrame([
        {
            "zone_rule": "body",
            "total_touches": int(
                touches["body_touched"].sum()
            ),
        },
        {
            "zone_rule": "full_wick",
            "total_touches": int(
                touches["wick_touched"].sum()
            ),
        },
    ])
