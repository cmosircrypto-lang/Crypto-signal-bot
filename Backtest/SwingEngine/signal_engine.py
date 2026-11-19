
from __future__ import annotations

from typing import Callable

import pandas as pd

from market_structure import analyze_market_structure
from data_loader import latest_closed_index


TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}

SIGNAL_COLUMNS = [
    "symbol",
    "method",
    "direction",
    "ob_id",
    "ob_ts",
    "bos_ts",
    "touch_number",
    "touch_score",
    "signal_index",
    "signal_ts",
    "entry_price",
    "htf_trend",
    "confirmation",
]


def _validate_candles(df: pd.DataFrame, name: str) -> None:
    required = {
        "ts", "open", "high", "low", "close"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{name}: candles are empty")

    if df["ts"].isna().any():
        raise ValueError(f"{name}: missing timestamps")

    if df["ts"].duplicated().any():
        raise ValueError(f"{name}: duplicate timestamps")

    if not df["ts"].is_monotonic_increasing:
        raise ValueError(f"{name}: timestamps must be sorted")

    prices = df[
        ["open", "high", "low", "close"]
    ].apply(pd.to_numeric, errors="coerce")

    if prices.isna().any().any():
        raise ValueError(f"{name}: invalid OHLC values")

    if not (prices["high"] >= prices["low"]).all():
        raise ValueError(f"{name}: invalid high/low")

    if not (prices["high"] >= prices["open"]).all():
        raise ValueError(f"{name}: invalid high/open")

    if not (prices["high"] >= prices["close"]).all():
        raise ValueError(f"{name}: invalid high/close")

    if not (prices["low"] <= prices["open"]).all():
        raise ValueError(f"{name}: invalid low/open")

    if not (prices["low"] <= prices["close"]).all():
        raise ValueError(f"{name}: invalid low/close")


def is_bullish_engulfing(
    previous: pd.Series,
    current: pd.Series,
) -> bool:
    """A bullish candle body engulfs the previous bearish body."""

    return (
        previous["close"] < previous["open"]
        and current["close"] > current["open"]
        and current["open"] <= previous["close"]
        and current["close"] >= previous["open"]
    )


def is_bearish_engulfing(
    previous: pd.Series,
    current: pd.Series,
) -> bool:
    """A bearish candle body engulfs the previous bullish body."""

    return (
        previous["close"] > previous["open"]
        and current["close"] < current["open"]
        and current["open"] >= previous["close"]
        and current["close"] <= previous["open"]
    )


def is_bullish_pin_bar(
    candle: pd.Series,
    wick_ratio: float = 2.0,
) -> bool:
    """Long lower rejection wick with a bullish close."""

    body = abs(candle["close"] - candle["open"])
    total_range = candle["high"] - candle["low"]

    if total_range <= 0:
        return False

    lower_wick = (
        min(candle["open"], candle["close"])
        - candle["low"]
    )

    upper_wick = (
        candle["high"]
        - max(candle["open"], candle["close"])
    )

    return (
        candle["close"] > candle["open"]
        and lower_wick >= max(body * wick_ratio, total_range * 0.4)
        and upper_wick <= total_range * 0.3
        and candle["close"] >= candle["low"] + total_range * 0.6
    )


def is_bearish_pin_bar(
    candle: pd.Series,
    wick_ratio: float = 2.0,
) -> bool:
    """Long upper rejection wick with a bearish close."""

    body = abs(candle["close"] - candle["open"])
    total_range = candle["high"] - candle["low"]

    if total_range <= 0:
        return False

    upper_wick = (
        candle["high"]
        - max(candle["open"], candle["close"])
    )

    lower_wick = (
        min(candle["open"], candle["close"])
        - candle["low"]
    )

    return (
        candle["close"] < candle["open"]
        and upper_wick >= max(body * wick_ratio, total_range * 0.4)
        and lower_wick <= total_range * 0.3
        and candle["close"] <= candle["low"] + total_range * 0.4
    )


def _zone_for_ob(ob: pd.Series, zone_type: str):
    if zone_type == "body":
        return float(ob["body_low"]), float(ob["body_high"])

    if zone_type == "wick":
        return float(ob["wick_low"]), float(ob["wick_high"])

    raise ValueError("zone_type must be 'body' or 'wick'")


def _candle_overlaps_zone(
    candle: pd.Series,
    zone_low: float,
    zone_high: float,
) -> bool:
    return (
        float(candle["low"]) <= zone_high
        and float(candle["high"]) >= zone_low
    )


def _direction_matches_trend(
    direction: str,
    trend: str,
) -> bool:
    if direction == "BUY":
        return trend == "BULLISH"

    return trend == "BEARISH"


def _get_htf_trend(
    candles_4h: pd.DataFrame,
    decision_time: int,
    structure_4h: pd.DataFrame,
) -> str:
    index = latest_closed_index(
        candles_4h,
        "4h",
        decision_time,
    )

    if index is None:
        return "NEUTRAL"

    return str(
        structure_4h.loc[index, "structure_trend"]
    )


def generate_entry_signals(
    symbol: str,
    candles_15m: pd.DataFrame,
    order_blocks_1h: pd.DataFrame,
    candles_4h: pd.DataFrame | None = None,
    methods: tuple[str, ...] = (
        "DIRECT",
        "ENGULFING",
        "PIN_BAR",
        "M15_STRUCTURE_BREAK",
    ),
    zone_type: str = "body",
    touch_rule: str = "first",
    decay_per_touch: float = 0.25,
    require_4h_alignment: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Generate independent entry signals from 1h Order Blocks.

    Inputs:
        candles_15m:
            Lower timeframe candles used for entry.

        order_blocks_1h:
            Order Blocks detected from fully processed 1h candles.

        candles_4h:
            Optional higher timeframe trend filter.

    Entry methods:
        DIRECT
        ENGULFING
        PIN_BAR
        M15_STRUCTURE_BREAK

    Touch rules:
        first
        max_two
        progressive

    All signals are evaluated only after the 15m candle closes.
    No orders are opened by this function.
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("symbol cannot be empty")

    _validate_candles(candles_15m, "15m")

    if not 0 < decay_per_touch <= 1:
        raise ValueError("decay_per_touch must be in (0, 1]")

    if zone_type not in ("body", "wick"):
        raise ValueError("Invalid zone_type")

    if touch_rule not in (
        "first",
        "max_two",
        "progressive",
    ):
        raise ValueError("Invalid touch_rule")

    allowed_methods = {
        "DIRECT",
        "ENGULFING",
        "PIN_BAR",
        "M15_STRUCTURE_BREAK",
    }

    if not methods:
        raise ValueError("At least one entry method is required")

    if not set(methods).issubset(allowed_methods):
        raise ValueError("Unsupported entry method")

    required_ob_columns = {
        "direction",
        "bos_ts",
        "body_low",
        "body_high",
        "wick_low",
        "wick_high",
    }

    missing = required_ob_columns - set(
        order_blocks_1h.columns
    )

    if missing:
        raise ValueError(
            f"Order Blocks missing columns: {sorted(missing)}"
        )

    if require_4h_alignment:
        if candles_4h is None:
            raise ValueError(
                "candles_4h required for 4h trend filter"
            )

        _validate_candles(candles_4h, "4h")

        structure_4h = analyze_market_structure(
            candles_4h,
            left=5,
            right=5,
        )
    else:
        structure_4h = None

    candles = candles_15m.reset_index(drop=True).copy()

    structure_15m = analyze_market_structure(
        candles,
        left=5,
        right=5,
    )

    results = []

    total_obs = len(order_blocks_1h)
    for ob_position, (ob_id, ob) in enumerate(order_blocks_1h.reset_index(
        drop=True
    ).iterrows(), start=1):
        if progress_callback is not None:
            progress_callback("order_blocks", ob_position, total_obs)

        if ob["direction"] == "BULLISH":
            direction = "BUY"
            expected_event = {
                "BOS_BULLISH",
                "CHOCH_BULLISH",
            }

        elif ob["direction"] == "BEARISH":
            direction = "SELL"
            expected_event = {
                "BOS_BEARISH",
                "CHOCH_BEARISH",
            }

        else:
            raise ValueError("Invalid Order Block direction")

        zone_low, zone_high = _zone_for_ob(
            ob,
            zone_type,
        )

        bos_close_time = (
            int(ob["bos_ts"]) + TIMEFRAME_MS["1h"]
        )

        inside = False
        touch_number = 0
        triggered_methods = set()
        pending_structure_touch = None

        for i in range(len(candles)):
            if progress_callback is not None and (i == 0 or (i + 1) % 50000 == 0 or i + 1 == len(candles)):
                progress_callback("candles", i + 1, len(candles))

            candle = candles.iloc[i]

            candle_open_time = int(candle["ts"])
            decision_time = (
                candle_open_time + TIMEFRAME_MS["15m"]
            )

            # An OB is not usable until its 1h BOS candle closes.
            if decision_time <= bos_close_time:
                continue

            overlaps = _candle_overlaps_zone(
                candle,
                zone_low,
                zone_high,
            )

            is_new_touch = False

            if not overlaps:
                inside = False
            elif not inside:
                inside = True
                touch_number += 1
                is_new_touch = True

            # Direct/reversal entries are evaluated on a new,
            # eligible OB touch. Structure-break entries are armed
            # by that touch and may confirm on a later closed candle.
            eligible_touch = (
                is_new_touch
                and not (
                    touch_rule == "first"
                    and touch_number > 1
                )
                and not (
                    touch_rule == "max_two"
                    and touch_number > 2
                )
            )

            if eligible_touch:
                touch_score = decay_per_touch ** (
                    touch_number - 1
                )

                if (
                    "M15_STRUCTURE_BREAK" in methods
                    and pending_structure_touch is None
                    and "M15_STRUCTURE_BREAK"
                    not in triggered_methods
                ):
                    pending_structure_touch = {
                        "index": i,
                        "touch_number": touch_number,
                        "touch_score": float(touch_score),
                    }

                touch_htf_trend = "NOT_USED"

                if require_4h_alignment:
                    touch_htf_trend = _get_htf_trend(
                        candles_4h,
                        decision_time,
                        structure_4h,
                    )

                touch_alignment_ok = (
                    not require_4h_alignment
                    or _direction_matches_trend(
                        direction,
                        touch_htf_trend,
                    )
                )

                if touch_alignment_ok:
                    confirmations = {}

                    if "DIRECT" in methods:
                        confirmations["DIRECT"] = True

                    if i > 0 and "ENGULFING" in methods:
                        previous = candles.iloc[i - 1]

                        if direction == "BUY":
                            confirmations["ENGULFING"] = (
                                is_bullish_engulfing(
                                    previous,
                                    candle,
                                )
                            )
                        else:
                            confirmations["ENGULFING"] = (
                                is_bearish_engulfing(
                                    previous,
                                    candle,
                                )
                            )

                    if "PIN_BAR" in methods:
                        if direction == "BUY":
                            confirmations["PIN_BAR"] = (
                                is_bullish_pin_bar(candle)
                            )
                        else:
                            confirmations["PIN_BAR"] = (
                                is_bearish_pin_bar(candle)
                            )

                    for method in methods:
                        if method == "M15_STRUCTURE_BREAK":
                            continue

                        if method in triggered_methods:
                            continue

                        if not confirmations.get(method, False):
                            continue

                        triggered_methods.add(method)

                        results.append({
                            "symbol": symbol,
                            "method": method,
                            "direction": direction,
                            "ob_id": ob_id,
                            "ob_ts": ob.get("ob_ts", None),
                            "bos_ts": ob["bos_ts"],
                            "touch_number": touch_number,
                            "touch_score": float(touch_score),
                            "signal_index": i,
                            "signal_ts": decision_time,
                            "entry_price": float(candle["close"]),
                            "htf_trend": touch_htf_trend,
                            "confirmation": method,
                        })

            # A structure break on the touch candle itself is not
            # accepted. Wait for a later fully closed 15m candle.
            if (
                pending_structure_touch is not None
                and "M15_STRUCTURE_BREAK" in methods
                and "M15_STRUCTURE_BREAK" not in triggered_methods
                and i > pending_structure_touch["index"]
            ):
                event = structure_15m.loc[
                    i,
                    "structure_event",
                ]

                if event in expected_event:
                    signal_htf_trend = "NOT_USED"

                    if require_4h_alignment:
                        signal_htf_trend = _get_htf_trend(
                            candles_4h,
                            decision_time,
                            structure_4h,
                        )

                    if (
                        not require_4h_alignment
                        or _direction_matches_trend(
                            direction,
                            signal_htf_trend,
                        )
                    ):
                        triggered_methods.add(
                            "M15_STRUCTURE_BREAK"
                        )

                        results.append({
                            "symbol": symbol,
                            "method": "M15_STRUCTURE_BREAK",
                            "direction": direction,
                            "ob_id": ob_id,
                            "ob_ts": ob.get("ob_ts", None),
                            "bos_ts": ob["bos_ts"],
                            "touch_number": pending_structure_touch[
                                "touch_number"
                            ],
                            "touch_score": pending_structure_touch[
                                "touch_score"
                            ],
                            "signal_index": i,
                            "signal_ts": decision_time,
                            "entry_price": float(candle["close"]),
                            "htf_trend": signal_htf_trend,
                            "confirmation": "M15_STRUCTURE_BREAK",
                        })

            if len(triggered_methods) == len(methods):
                break

    return pd.DataFrame(
        results,
        columns=SIGNAL_COLUMNS,
    )
