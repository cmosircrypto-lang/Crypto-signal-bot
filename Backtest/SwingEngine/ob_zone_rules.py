
from __future__ import annotations

import numpy as np
import pandas as pd


ZONE_COLUMNS = [
    "ob_id",
    "direction",
    "candle_index",
    "ts",
    "body_touched",
    "wick_touched",
    "body_touch_number",
    "wick_touch_number",
]


def _validate_candles(candles: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "ts"}
    missing = required - set(candles.columns)

    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    if candles.empty:
        return

    prices = candles[
        ["open", "high", "low", "close"]
    ].to_numpy(dtype=float)

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
        raise ValueError(
            f"Missing Order Block columns: {sorted(missing)}"
        )

    numeric_columns = [
        "ob_index",
        "bos_index",
        "body_low",
        "body_high",
        "wick_low",
        "wick_high",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            order_blocks[column], errors="coerce"
        )

        if values.isna().any() or not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Invalid Order Block values: {column}")

    for _, ob in order_blocks.iterrows():
        if ob["direction"] not in ("BULLISH", "BEARISH"):
            raise ValueError("Invalid Order Block direction")

        ob_index = float(ob["ob_index"])
        bos_index = float(ob["bos_index"])

        if not ob_index.is_integer() or not bos_index.is_integer():
            raise ValueError("Order Block indices must be integers")

        if ob_index < 0 or bos_index < 0:
            raise ValueError("Order Block indices cannot be negative")

        if bos_index < ob_index:
            raise ValueError("BOS index cannot precede OB index")

        if ob["body_low"] > ob["body_high"]:
            raise ValueError("Invalid body zone range")

        if ob["wick_low"] > ob["wick_high"]:
            raise ValueError("Invalid wick zone range")

        if ob["wick_low"] > ob["body_low"]:
            raise ValueError("Wick low cannot exceed body low")

        if ob["wick_high"] < ob["body_high"]:
            raise ValueError("Wick high cannot be below body high")


def _overlaps_zone(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
) -> bool:
    return (
        candle_low <= zone_high
        and candle_high >= zone_low
    )


def calculate_zone_touches(
    order_blocks: pd.DataFrame,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count independent body-zone and wick-zone touches.

    Rules:
    - Counting starts after the BOS candle closes.
    - Consecutive overlapping candles count as one touch.
    - A new touch requires a complete exit and re-entry.
    - Body and wick zones have independent touch states.
    - An event row is created only for a new touch.
    """

    _validate_order_blocks(order_blocks)
    _validate_candles(candles)

    events = []

    for ob_id, ob in order_blocks.reset_index(drop=True).iterrows():
        start_index = int(ob["bos_index"]) + 1

        body_low = float(ob["body_low"])
        body_high = float(ob["body_high"])
        wick_low = float(ob["wick_low"])
        wick_high = float(ob["wick_high"])

        body_inside = False
        wick_inside = False

        body_touch_number = 0
        wick_touch_number = 0

        for candle_index in range(start_index, len(candles)):
            candle = candles.iloc[candle_index]

            candle_low = float(candle["low"])
            candle_high = float(candle["high"])

            body_overlap = _overlaps_zone(
                candle_low,
                candle_high,
                body_low,
                body_high,
            )

            wick_overlap = _overlaps_zone(
                candle_low,
                candle_high,
                wick_low,
                wick_high,
            )

            new_body_touch = body_overlap and not body_inside
            new_wick_touch = wick_overlap and not wick_inside

            if new_body_touch:
                body_touch_number += 1

            if new_wick_touch:
                wick_touch_number += 1

            if new_body_touch or new_wick_touch:
                events.append({
                    "ob_id": ob_id,
                    "direction": ob["direction"],
                    "candle_index": candle_index,
                    "ts": candle["ts"],
                    "body_touched": new_body_touch,
                    "wick_touched": new_wick_touch,
                    "body_touch_number": (
                        body_touch_number
                        if new_body_touch else 0
                    ),
                    "wick_touch_number": (
                        wick_touch_number
                        if new_wick_touch else 0
                    ),
                })

            # Each zone has its own independent inside state.
            body_inside = body_overlap
            wick_inside = wick_overlap

    return pd.DataFrame(events, columns=ZONE_COLUMNS)


def compare_zone_rules(
    zone_touches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare independent body-zone and wick-zone touch counts.
    """

    required = {
        "ob_id",
        "body_touched",
        "wick_touched",
    }

    missing = required - set(zone_touches.columns)

    if missing:
        raise ValueError(
            f"Missing zone touch columns: {sorted(missing)}"
        )

    columns = [
        "ob_id",
        "body_touch_count",
        "wick_touch_count",
    ]

    if zone_touches.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for ob_id, group in zone_touches.groupby(
        "ob_id", sort=False
    ):
        rows.append({
            "ob_id": ob_id,
            "body_touch_count": int(
                group["body_touched"].sum()
            ),
            "wick_touch_count": int(
                group["wick_touched"].sum()
            ),
        })

    return pd.DataFrame(rows, columns=columns)
