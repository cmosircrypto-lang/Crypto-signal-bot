
from __future__ import annotations

from typing import Callable

import pandas as pd

from market_structure import analyze_market_structure


REQUIRED_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
]

ORDER_BLOCK_COLUMNS = [
    "direction",
    "ob_index",
    "ob_ts",
    "bos_index",
    "bos_ts",
    "atr",
    "impulse_body",
    "impulse_atr_ratio",
    "threshold",
    "body_low",
    "body_high",
    "wick_low",
    "wick_high",
]


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calculate ATR using True Range.

    ATR is calculated using candles available
    at the current candle close.
    """

    if period < 1:
        raise ValueError("ATR period must be positive")

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()


def validate_ohlc(df: pd.DataFrame) -> None:
    """
    Validate input OHLC data.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("DataFrame is empty")

    if df["ts"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    if not df["ts"].is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column {column} must be numeric"
            )

        if not pd.Series(df[column]).map(
            lambda value: pd.notna(value)
        ).all():
            raise ValueError(
                f"Column {column} contains missing values"
            )

    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid.any():
        raise ValueError("Invalid OHLC candle detected")


def detect_order_blocks(
    df: pd.DataFrame,
    threshold: float = 1.0,
    atr_period: int = 14,
    lookback: int = 10,
    structure: pd.DataFrame | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Detect bullish and bearish Order Blocks.

    Bullish OB:
        Last bearish candle before a bullish BOS.

    Bearish OB:
        Last bullish candle before a bearish BOS.

    Displacement requirement:
        BOS candle body / ATR >= threshold.

    The Order Block becomes available only after
    the BOS candle has closed.

    No future candles are used to create an OB.
    """

    validate_ohlc(df)

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero"
        )

    if lookback < 1:
        raise ValueError(
            "Lookback must be at least 1"
        )

    if atr_period < 1:
        raise ValueError(
            "ATR period must be positive"
        )

    data = df.reset_index(drop=True).copy()

    if structure is None:
        structure = analyze_market_structure(
            data,
            left=5,
            right=5,
        )
    else:
        structure = structure.reset_index(drop=True)

    if len(structure) != len(data):
        raise ValueError(
            "Structure and OHLC data lengths do not match"
        )

    if "structure_event" not in structure.columns:
        raise ValueError(
            "Missing structure_event column"
        )

    atr = calculate_atr(
        data,
        period=atr_period,
    )

    order_blocks = []

    for bos_index in range(len(data)):
        if progress_callback is not None and (bos_index == 0 or (bos_index + 1) % 50000 == 0 or bos_index + 1 == len(data)):
            progress_callback(bos_index + 1, len(data))

        event = structure.loc[
            bos_index,
            "structure_event",
        ]

        if event not in (
            "BOS_BULLISH",
            "BOS_BEARISH",
        ):
            continue

        current_atr = atr.iloc[bos_index]

        if pd.isna(current_atr) or current_atr <= 0:
            continue

        bos_candle = data.iloc[bos_index]

        impulse_body = abs(
            bos_candle["close"] - bos_candle["open"]
        )

        impulse_ratio = impulse_body / current_atr

        if impulse_ratio < threshold:
            continue

        if event == "BOS_BULLISH":
            direction = "BULLISH"

            # Search for the last bearish candle
            # before the bullish BOS.
            candidate_indices = range(
                bos_index - 1,
                max(-1, bos_index - lookback - 1),
                -1,
            )

            ob_index = None

            for index in candidate_indices:
                candle = data.iloc[index]

                if candle["close"] < candle["open"]:
                    ob_index = index
                    break

        else:
            direction = "BEARISH"

            # Search for the last bullish candle
            # before the bearish BOS.
            candidate_indices = range(
                bos_index - 1,
                max(-1, bos_index - lookback - 1),
                -1,
            )

            ob_index = None

            for index in candidate_indices:
                candle = data.iloc[index]

                if candle["close"] > candle["open"]:
                    ob_index = index
                    break

        if ob_index is None:
            continue

        ob_candle = data.iloc[ob_index]

        body_low = min(
            ob_candle["open"],
            ob_candle["close"],
        )

        body_high = max(
            ob_candle["open"],
            ob_candle["close"],
        )

        wick_low = ob_candle["low"]
        wick_high = ob_candle["high"]

        order_blocks.append(
            {
                "direction": direction,
                "ob_index": ob_index,
                "ob_ts": ob_candle["ts"],
                "bos_index": bos_index,
                "bos_ts": bos_candle["ts"],
                "atr": float(current_atr),
                "impulse_body": float(impulse_body),
                "impulse_atr_ratio": float(impulse_ratio),
                "threshold": float(threshold),
                "body_low": float(body_low),
                "body_high": float(body_high),
                "wick_low": float(wick_low),
                "wick_high": float(wick_high),
            }
        )

    return pd.DataFrame(
        order_blocks,
        columns=ORDER_BLOCK_COLUMNS,
    )


def compare_order_block_thresholds(
    df: pd.DataFrame,
    thresholds: tuple[float, ...] = (
        1.0,
        1.25,
        1.5,
    ),
    atr_period: int = 14,
    lookback: int = 10,
    structure: pd.DataFrame | None = None,
) -> dict[float, pd.DataFrame]:
    """
    Run Order Block detection with multiple
    displacement thresholds.

    Returns one DataFrame per threshold.
    """

    results = {}

    if not thresholds:
        raise ValueError("Threshold list is empty")

    for threshold in thresholds:

        results[float(threshold)] = detect_order_blocks(
            df=df,
            threshold=float(threshold),
            atr_period=atr_period,
            lookback=lookback,
            structure=structure,
        )

    return results
