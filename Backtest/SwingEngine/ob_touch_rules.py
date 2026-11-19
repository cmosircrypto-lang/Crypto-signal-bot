
from __future__ import annotations

import numpy as np
import pandas as pd


TOUCH_COLUMNS = [
    "ob_id",
    "direction",
    "candle_index",
    "ts",
    "touch_number",
    "body_touched",
    "wick_touched",
    "first_touch_only_valid",
    "max_two_touches_valid",
    "progressive_score",
]


def _validate_candles(candles: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "ts"}
    missing = required - set(candles.columns)

    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    if candles.empty:
        return

    prices = candles[["open", "high", "low", "close"]].to_numpy(
        dtype=float
    )

    if not np.isfinite(prices).all():
        raise ValueError("Candle OHLC values must be finite")

    if (prices[:, 1] < prices[:, 2]).any():
        raise ValueError("Candle high cannot be below low")

    if (prices[:, 0] < prices[:, 2]).any():
        raise ValueError("Candle open cannot be below low")

    if (prices[:, 0] > prices[:, 1]).any():
        raise ValueError("Candle open cannot be above high")

    if (prices[:, 3] < prices[:, 2]).any():
        raise ValueError("Candle close cannot be below low")

    if (prices[:, 3] > prices[:, 1]).any():
        raise ValueError("Candle close cannot be above high")

    timestamps = pd.to_numeric(candles["ts"], errors="coerce")

    if timestamps.isna().any():
        raise ValueError("Candle timestamps must be numeric")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Candle timestamps must be monotonic")

    if timestamps.duplicated().any():
        raise ValueError("Candle timestamps must be unique")


def _validate_order_blocks(order_blocks: pd.DataFrame) -> None:
    required = {
        "direction",
        "ob_index",
        "bos_index",
        "body_low",
        "body_high",
        "wick_low",
        "wick_high",
    }

    missing = required - set(order_blocks.columns)

    if missing:
        raise ValueError(f"Missing Order Block columns: {sorted(missing)}")

    for column in [
        "ob_index",
        "bos_index",
        "body_low",
        "body_high",
        "wick_low",
        "wick_high",
    ]:
        values = pd.to_numeric(order_blocks[column], errors="coerce")

        if values.isna().any() or not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Invalid Order Block values: {column}")

    for _, ob in order_blocks.iterrows():
        if ob["direction"] not in ("BULLISH", "BEARISH"):
            raise ValueError("Invalid Order Block direction")

        if int(ob["ob_index"]) < 0 or int(ob["bos_index"]) < 0:
            raise ValueError("Order Block indices cannot be negative")

        if int(ob["bos_index"]) < int(ob["ob_index"]):
            raise ValueError("BOS index cannot precede OB index")

        if ob["body_low"] > ob["body_high"]:
            raise ValueError("Invalid body zone range")

        if ob["wick_low"] > ob["wick_high"]:
            raise ValueError("Invalid wick zone range")

        if ob["wick_low"] > ob["body_low"]:
            raise ValueError("Wick low cannot exceed body low")

        if ob["wick_high"] < ob["body_high"]:
            raise ValueError("Wick high cannot be below body high")


def evaluate_ob_touches(
    order_blocks: pd.DataFrame,
    candles: pd.DataFrame,
    decay_per_touch: float = 0.25,
) -> pd.DataFrame:
    """
    Count independent Order Block touches.

    Rules:
    - Touch counting starts at bos_index + 1.
    - Consecutive overlapping candles count as one touch.
    - A new touch requires a complete candle outside the wick zone.
    - Touch events are recorded on the first overlapping candle.
    """

    if not 0 < decay_per_touch <= 1:
        raise ValueError("decay_per_touch must be in (0, 1]")

    _validate_order_blocks(order_blocks)
    _validate_candles(candles)

    events = []

    for ob_id, ob in order_blocks.reset_index(drop=True).iterrows():
        start_index = int(ob["bos_index"]) + 1

        wick_low = float(ob["wick_low"])
        wick_high = float(ob["wick_high"])

        inside = False
        touch_number = 0

        for candle_index in range(start_index, len(candles)):
            candle = candles.iloc[candle_index]

            candle_low = float(candle["low"])
            candle_high = float(candle["high"])

            overlaps = (
                candle_low <= wick_high
                and candle_high >= wick_low
            )

            if not overlaps:
                inside = False
                continue

            if inside:
                continue

            inside = True
            touch_number += 1

            events.append({
                "ob_id": ob_id,
                "direction": ob["direction"],
                "candle_index": candle_index,
                "ts": candle["ts"],
                "touch_number": touch_number,
                "body_touched": (
                    float(candle["low"]) <= float(ob["body_high"])
                    and float(candle["high"]) >= float(ob["body_low"])
                ),
                "wick_touched": True,
                "first_touch_only_valid": touch_number == 1,
                "max_two_touches_valid": touch_number <= 2,
                "progressive_score": decay_per_touch ** (touch_number - 1),
            })

    return pd.DataFrame(events, columns=TOUCH_COLUMNS)


def compare_touch_rules(
    touches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare first-touch, maximum-two-touch and progressive-decay rules.
    """

    required = {"ob_id", "touch_number", "progressive_score"}
    missing = required - set(touches.columns)

    if missing:
        raise ValueError(f"Missing touch columns: {sorted(missing)}")

    columns = [
        "ob_id",
        "total_touches",
        "first_touch_only_count",
        "max_two_touches_count",
        "progressive_total_score",
        "progressive_average_score",
    ]

    if touches.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for ob_id, group in touches.groupby("ob_id", sort=False):
        scores = group["progressive_score"].astype(float)

        rows.append({
            "ob_id": ob_id,
            "total_touches": len(group),
            "first_touch_only_count": int(
                group["first_touch_only_valid"].sum()
            ) if "first_touch_only_valid" in group else int(
                (group["touch_number"] == 1).sum()
            ),
            "max_two_touches_count": int(
                group["max_two_touches_valid"].sum()
            ) if "max_two_touches_valid" in group else int(
                (group["touch_number"] <= 2).sum()
            ),
            "progressive_total_score": float(scores.sum()),
            "progressive_average_score": float(scores.mean()),
        })

    return pd.DataFrame(rows, columns=columns)
