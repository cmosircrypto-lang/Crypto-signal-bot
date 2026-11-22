"""Audit BTC 1d+4h filter decisions for OOS period 6 and its neighbors."""
from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest_runner import btc_trends_at_time
from recovery_cycle_resumable import _btc_filter_matches
from recovery_robustness_validation import (
    PERIODS, _dataset_time_bounds, _segment_ranges, _slice_period,
)

CACHE = Path(os.environ.get(
    "RECOVERY_PREPARED_CACHE",
    "Backtest/results/recovery_adaptive_matrix/prepared_oos_cache.pkl",
))
OUT = Path(os.environ.get(
    "RECOVERY_BTC_AUDIT_OUTPUT_DIR",
    "Backtest/results/recovery_btc_filter_audit",
))
TARGET_PERIODS = (5, 6, 7)


def _utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError(f"Prepared OOS cache not found: {CACHE}")
    OUT.mkdir(parents=True, exist_ok=True)
    with CACHE.open("rb") as handle:
        prepared = pickle.load(handle)

    data = prepared["test_data"]
    signals = prepared["test_signals"]
    btc_frames = prepared["btc_frames"]
    symbols = prepared["sets"].get("all_76", [])
    if len(symbols) != 76:
        raise ValueError(f"Expected 76 OOS symbols, got {len(symbols)}")
    data = {symbol: data[symbol] for symbol in symbols}
    signals = {symbol: signals[symbol] for symbol in symbols}

    start_ms, end_ms = _dataset_time_bounds(data)
    ranges = _segment_ranges(start_ms, end_ms, PERIODS)
    details = []
    summaries = []

    for period_index in TARGET_PERIODS:
        start, end = ranges[period_index - 1]
        _, period_signals, counts = _slice_period(data, signals, start, end)
        direction_counts = Counter()
        trend_combo_counts = Counter()
        timeframe_counts = {
            "1d_matches": 0, "4h_matches": 0, "both_match": 0,
            "one_matches": 0, "neither_matches": 0,
            "unavailable_1d": 0, "unavailable_4h": 0,
            "neutral_1d": 0, "neutral_4h": 0,
        }
        matched_symbols = Counter()
        rejected_symbols = Counter()

        for symbol in sorted(period_signals):
            frame = period_signals[symbol]
            if frame.empty:
                continue
            for row in frame.to_dict("records"):
                ts = int(row["signal_ts"])
                direction = str(row["direction"]).upper()
                trend = btc_trends_at_time(btc_frames, ts)
                t1d, t4h = trend.get("1d", "UNAVAILABLE"), trend.get("4h", "UNAVAILABLE")
                match_1d, match_4h = t1d == direction, t4h == direction
                passed = _btc_filter_matches(trend, direction, "1d_4h")

                direction_counts[direction] += 1
                trend_combo_counts[(direction, t1d, t4h)] += 1
                timeframe_counts["1d_matches"] += int(match_1d)
                timeframe_counts["4h_matches"] += int(match_4h)
                timeframe_counts["both_match"] += int(match_1d and match_4h)
                timeframe_counts["one_matches"] += int(match_1d != match_4h)
                timeframe_counts["neither_matches"] += int(not match_1d and not match_4h)
                timeframe_counts["unavailable_1d"] += int(t1d == "UNAVAILABLE")
                timeframe_counts["unavailable_4h"] += int(t4h == "UNAVAILABLE")
                timeframe_counts["neutral_1d"] += int(t1d == "NEUTRAL")
                timeframe_counts["neutral_4h"] += int(t4h == "NEUTRAL")
                (matched_symbols if passed else rejected_symbols)[symbol] += 1

                details.append({
                    "period_index": period_index,
                    "symbol": symbol,
                    "signal_ts_utc": _utc(ts),
                    "direction": direction,
                    "btc_1d": t1d,
                    "btc_4h": t4h,
                    "btc_1d_matches": match_1d,
                    "btc_4h_matches": match_4h,
                    "btc_filter_pass": passed,
                })

        total = sum(direction_counts.values())
        passed_total = timeframe_counts["both_match"]
        summaries.append({
            "period_index": period_index,
            "period_start_utc": _utc(start),
            "period_end_exclusive_utc": _utc(end),
            **counts,
            "signals_audited": total,
            "signals_passed_btc_1d_4h": passed_total,
            "signals_rejected_by_btc_gate": total - passed_total,
            "pass_rate_pct": (passed_total / total * 100) if total else 0.0,
            "signal_directions": dict(direction_counts),
            **timeframe_counts,
            "symbols_with_passed_signals": len(matched_symbols),
            "symbols_with_rejected_signals": len(rejected_symbols),
            "top_rejected_symbols": rejected_symbols.most_common(15),
        })
        print(
            f"[BTC AUDIT] period={period_index} raw={total} "
            f"passed={passed_total} rejected={total-passed_total} "
            f"1d_match={timeframe_counts['1d_matches']} "
            f"4h_match={timeframe_counts['4h_matches']} "
            f"unavailable_1d={timeframe_counts['unavailable_1d']} "
            f"unavailable_4h={timeframe_counts['unavailable_4h']}",
            flush=True,
        )

    pd.DataFrame(details).to_csv(OUT / "btc_filter_signal_details.csv", index=False)
    (OUT / "btc_filter_audit_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
