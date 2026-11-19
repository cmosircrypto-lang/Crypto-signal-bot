
from __future__ import annotations

import math
from typing import Callable

import pandas as pd


def analyze_market_structure(
    df: pd.DataFrame,
    left: int = 5,
    right: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Detect confirmed swing points, market structure labels,
    BOS and CHoCH without lookahead bias.

    A pivot at candle p becomes available only after candle
    p + right has fully closed.

    Required columns:
        ts, open, high, low, close

    Returns a copy of the input DataFrame with additional columns.
    """

    # ---------------------------------------------------------
    # 1. Validate parameters and input
    # ---------------------------------------------------------

    if left < 1 or right < 1:
        raise ValueError("left and right must be >= 1")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = ["ts", "open", "high", "low", "close"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    data = df.copy().reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. Validate timestamps
    # ---------------------------------------------------------

    if data["ts"].isna().any():
        raise ValueError("Timestamp contains missing values")

    if data["ts"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    if not data["ts"].is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted ascending")

    # ---------------------------------------------------------
    # 3. Validate OHLC values
    # ---------------------------------------------------------

    for col in ["open", "high", "low", "close"]:
        data[col] = pd.to_numeric(data[col], errors="raise")

        values = data[col].tolist()

        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"Invalid numeric values in {col}")

    if (data["high"] < data["low"]).any():
        raise ValueError("High cannot be below Low")

    if (data["high"] < data["open"]).any():
        raise ValueError("High cannot be below Open")

    if (data["high"] < data["close"]).any():
        raise ValueError("High cannot be below Close")

    if (data["low"] > data["open"]).any():
        raise ValueError("Low cannot be above Open")

    if (data["low"] > data["close"]).any():
        raise ValueError("Low cannot be above Close")

    highs = data["high"].tolist()
    lows = data["low"].tolist()
    closes = data["close"].tolist()
    timestamps = data["ts"].tolist()

    n = len(data)

    # ---------------------------------------------------------
    # 4. Output columns
    # ---------------------------------------------------------

    pivot_high_confirmed = [False] * n
    pivot_low_confirmed = [False] * n

    pivot_high_price = [None] * n
    pivot_low_price = [None] * n

    pivot_high_ts = [None] * n
    pivot_low_ts = [None] * n

    high_label = [None] * n
    low_label = [None] * n

    last_swing_high = [None] * n
    last_swing_low = [None] * n

    structure_event = [None] * n
    break_level = [None] * n

    structure_trend = ["NEUTRAL"] * n

    # ---------------------------------------------------------
    # 5. Internal state
    # ---------------------------------------------------------

    previous_swing_high = None
    previous_swing_low = None

    active_swing_high = None
    active_swing_low = None

    active_swing_high_ts = None
    active_swing_low_ts = None

    high_broken = False
    low_broken = False

    #  1 = bullish
    # -1 = bearish
    #  0 = not established
    trend = 0

    # ---------------------------------------------------------
    # 6. Process candles chronologically
    # ---------------------------------------------------------

    for i in range(n):
        if progress_callback is not None and (i == 0 or (i + 1) % 50000 == 0 or i + 1 == n):
            progress_callback(i + 1, n)

        # A pivot is checked only after its right-side
        # confirmation candles have closed.

        if i >= left + right:

            p = i - right

            left_highs = highs[p - left:p]
            right_highs = highs[p + 1:p + right + 1]

            left_lows = lows[p - left:p]
            right_lows = lows[p + 1:p + right + 1]

            # -------------------------------------------------
            # Swing High
            # -------------------------------------------------

            is_swing_high = (
                highs[p] > max(left_highs)
                and highs[p] >= max(right_highs)
            )

            if is_swing_high:

                pivot_high_confirmed[i] = True
                pivot_high_price[i] = highs[p]
                pivot_high_ts[i] = timestamps[p]

                if previous_swing_high is not None:

                    if highs[p] > previous_swing_high:
                        high_label[i] = "HH"

                    elif highs[p] < previous_swing_high:
                        high_label[i] = "LH"

                    else:
                        high_label[i] = "EH"

                previous_swing_high = highs[p]

                active_swing_high = highs[p]
                active_swing_high_ts = timestamps[p]

                high_broken = False

            # -------------------------------------------------
            # Swing Low
            # -------------------------------------------------

            is_swing_low = (
                lows[p] < min(left_lows)
                and lows[p] <= min(right_lows)
            )

            if is_swing_low:

                pivot_low_confirmed[i] = True
                pivot_low_price[i] = lows[p]
                pivot_low_ts[i] = timestamps[p]

                if previous_swing_low is not None:

                    if lows[p] > previous_swing_low:
                        low_label[i] = "HL"

                    elif lows[p] < previous_swing_low:
                        low_label[i] = "LL"

                    else:
                        low_label[i] = "EL"

                previous_swing_low = lows[p]

                active_swing_low = lows[p]
                active_swing_low_ts = timestamps[p]

                low_broken = False

        # -----------------------------------------------------
        # 7. Detect BOS and CHoCH using candle CLOSE
        # -----------------------------------------------------

        if i > 0:

            previous_close = closes[i - 1]
            current_close = closes[i]

            bullish_break = (
                active_swing_high is not None
                and not high_broken
                and previous_close <= active_swing_high
                and current_close > active_swing_high
            )

            bearish_break = (
                active_swing_low is not None
                and not low_broken
                and previous_close >= active_swing_low
                and current_close < active_swing_low
            )

            if bullish_break:

                if trend == -1:
                    structure_event[i] = "CHOCH_BULLISH"
                else:
                    structure_event[i] = "BOS_BULLISH"

                break_level[i] = active_swing_high

                high_broken = True
                trend = 1

            elif bearish_break:

                if trend == 1:
                    structure_event[i] = "CHOCH_BEARISH"
                else:
                    structure_event[i] = "BOS_BEARISH"

                break_level[i] = active_swing_low

                low_broken = True
                trend = -1

        # -----------------------------------------------------
        # 8. Save current confirmed structure
        # -----------------------------------------------------

        last_swing_high[i] = active_swing_high
        last_swing_low[i] = active_swing_low

        if trend == 1:
            structure_trend[i] = "BULLISH"

        elif trend == -1:
            structure_trend[i] = "BEARISH"

        else:
            structure_trend[i] = "NEUTRAL"

    # ---------------------------------------------------------
    # 9. Attach results
    # ---------------------------------------------------------

    data["pivot_high_confirmed"] = pivot_high_confirmed
    data["pivot_low_confirmed"] = pivot_low_confirmed

    data["pivot_high_price"] = pivot_high_price
    data["pivot_low_price"] = pivot_low_price

    data["pivot_high_ts"] = pivot_high_ts
    data["pivot_low_ts"] = pivot_low_ts

    data["high_label"] = high_label
    data["low_label"] = low_label

    data["last_swing_high"] = last_swing_high
    data["last_swing_low"] = last_swing_low

    data["structure_event"] = structure_event
    data["break_level"] = break_level

    data["structure_trend"] = structure_trend

    return data
