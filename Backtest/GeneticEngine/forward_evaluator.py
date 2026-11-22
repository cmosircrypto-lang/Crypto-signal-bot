"""Forward-only evaluation for an immutable set of genetic champions.

Historical candles are available only as indicator warm-up. Trades are simulated
strictly inside the fixed forward window; no genetic evolution or reselection runs.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .genome import StrategyGenome
from .historical_evaluator import (
    SYMBOL_TIMEFRAMES, TF_MS, _find_btc_frames, _normalize, _simulate_fold,
)

EXPECTED_SYMBOLS = 76
EXPECTED_15M_ROWS = 568
WINDOW_START = "2026-11-17T00:00:00Z"
WINDOW_END_EXCLUSIVE = "2026-11-22T22:00:00Z"


def _utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"Timestamp must be explicitly UTC: {value}")
    return int(parsed.timestamp() * 1000)


def validate_quality_report(report: Mapping[str, Any]) -> tuple[int, int]:
    if report.get("status") != "PASS" or report.get("errors"):
        raise ValueError("Forward data quality report is not PASS or contains errors")
    if report.get("forward_window_start_utc") != "2026-11-17T00:00:00+00:00":
        raise ValueError("Unexpected forward-window start in data quality report")
    if report.get("validated_through_exclusive_utc") != "2026-11-22T22:00:00+00:00":
        raise ValueError("Unexpected forward-window end in data quality report")
    if int(report.get("bar_interval_minutes", 0)) != 15:
        raise ValueError("Forward data must use 15-minute candles")
    rows = report.get("symbols")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SYMBOLS:
        raise ValueError(f"Expected quality records for {EXPECTED_SYMBOLS} symbols")
    names: set[str] = set()
    for item in rows:
        symbol = str(item.get("symbol", ""))
        if not symbol or symbol in names:
            raise ValueError(f"Missing or duplicate symbol in quality report: {symbol}")
        names.add(symbol)
        if (
            item.get("status") != "OK"
            or int(item.get("new_rows", -1)) != EXPECTED_15M_ROWS
            or int(item.get("new_window_missing", -1)) != 0
            or int(item.get("new_window_extra", -1)) != 0
            or item.get("new_first_utc") != "2026-11-17T00:00:00+00:00"
            or item.get("new_last_utc") != "2026-11-22T21:45:00+00:00"
        ):
            raise ValueError(f"Forward 15m data quality failed for {symbol}")
    return _utc_ms(WINDOW_START), _utc_ms(WINDOW_END_EXCLUSIVE)


def _validate_frame_window(frame, timeframe: str, start_ms: int, end_ms: int) -> None:
    interval = TF_MS[timeframe]
    expected = np.arange(start_ms, end_ms, interval, dtype=np.int64)
    stamps = frame["ts"].to_numpy(dtype=np.int64)
    actual = stamps[(stamps >= start_ms) & (stamps < end_ms)]
    if not np.array_equal(actual, expected):
        raise ValueError(
            f"{timeframe} forward window is incomplete: expected {len(expected)} "
            f"bars, found {len(actual)}"
        )


def load_frozen_manifest(path: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or manifest.get("frozen") is not True:
        raise ValueError("Candidate manifest is not a supported frozen manifest")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 11:
        raise ValueError("Forward validation requires exactly 11 frozen candidates")
    ids: set[str] = set()
    for candidate in candidates:
        symbol = str(candidate["symbol"]).upper()
        direction = str(candidate["direction"]).upper()
        candidate_id = f"{symbol}__{direction}"
        if candidate.get("candidate_id") != candidate_id or candidate_id in ids:
            raise ValueError(f"Invalid or duplicate candidate ID: {candidate_id}")
        ids.add(candidate_id)
        genome = StrategyGenome.from_dict(candidate["genome"])
        if genome.symbols != (symbol,):
            raise ValueError(f"Candidate genome is not scoped to {symbol}")
        if direction not in ("LONG", "SHORT"):
            raise ValueError(f"Invalid candidate direction: {direction}")
    excluded = manifest.get("excluded", [])
    if len(excluded) != 1 or excluded[0].get("candidate_id") != "BNB-USDT-SWAP__LONG":
        raise ValueError("Frozen manifest must preserve the single BNB LONG exclusion")
    return manifest, raw


def run_forward_validation(
    all_symbol_data: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    start_ms, end_ms = validate_quality_report(quality_report)
    manifest, manifest_bytes = load_frozen_manifest(manifest_path)
    if manifest.get("window", {}).get("start_utc") != WINDOW_START:
        raise ValueError("Frozen candidate manifest window start does not match")
    if manifest.get("window", {}).get("end_exclusive_utc") != WINDOW_END_EXCLUSIVE:
        raise ValueError("Frozen candidate manifest window end does not match")
    if len(all_symbol_data) != EXPECTED_SYMBOLS:
        raise ValueError(f"Expected {EXPECTED_SYMBOLS} loaded symbols, found {len(all_symbol_data)}")

    # Verify the actual loaded 15m candles, not only the sidecar quality report.
    for symbol, item in all_symbol_data.items():
        frames = item.frames if hasattr(item, "frames") else item
        if not all(tf in frames for tf in SYMBOL_TIMEFRAMES):
            raise ValueError(f"{symbol}: missing required timeframe")
        _validate_frame_window(frames["15m"], "15m", start_ms, end_ms)

    needs_btc = any(
        "BTC_ALIGNED_PULLBACK" in candidate["genome"]["signal_families"]
        for candidate in manifest["candidates"]
    )
    btc_frames = _find_btc_frames(all_symbol_data) if needs_btc else None
    if needs_btc and btc_frames is None:
        raise ValueError("BTC-aligned candidates require complete BTC 1h/4h/1d history")

    results: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        symbol = candidate["symbol"]
        direction = candidate["direction"]
        genome = StrategyGenome.from_dict(candidate["genome"])
        item = all_symbol_data[symbol]
        raw = item.frames if hasattr(item, "frames") else item
        normalized = {tf: _normalize(raw[tf]) for tf in SYMBOL_TIMEFRAMES}

        # The 1h and 4h frames must also cover the same forward period without gaps.
        _validate_frame_window(raw["1h"], "1h", start_ms, end_ms)
        _validate_frame_window(raw["4h"], "4h", start_ms, end_ms)

        metrics = _simulate_fold(
            symbol, normalized, btc_frames, genome, direction,
            start_ms, end_ms, initial_equity=100.0, swing_signal_provider=None,
        )
        results.append({
            "candidate_id": candidate["candidate_id"],
            "symbol": symbol,
            "direction": direction,
            "genome_sha256": hashlib.sha256(
                json.dumps(candidate["genome"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "historical_reference_metrics": candidate["historical_reference_metrics"],
            "forward_metrics": asdict(metrics),
        })

    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "source": manifest["source"],
        "forward_window": {
            "start_utc": WINDOW_START,
            "end_exclusive_utc": WINDOW_END_EXCLUSIVE,
            "bar_interval_minutes": 15,
            "bars_per_symbol": EXPECTED_15M_ROWS,
        },
        "data_quality": {
            "status": quality_report["status"],
            "symbol_count": len(quality_report["symbols"]),
            "all_symbols_15m_complete": True,
        },
        "candidate_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "candidate_count": len(results),
        "excluded_candidates": manifest["excluded"],
        "evolution_ran": False,
        "candidate_parameters_changed": False,
        "portfolio_simulation": False,
        "results": results,
        "methodology_notes": [
            "Each frozen symbol-direction candidate is simulated independently from $100 initial equity.",
            "Pre-window candles are used only to warm up indicators; no pre-window trade is carried into the test.",
            "Only decisions and 15m execution candles inside [2026-11-17 00:00 UTC, 2026-11-22 22:00 UTC) are simulated.",
            "The existing causal signal and trade engine is reused: closed candles, next 15m open entries, and SL-first same-candle resolution.",
            "An open trade at the end is valued at the final in-window 15m close by the existing simulator.",
            "Reported max_drawdown_pct follows the existing closed-trade equity calculation; it is not intrabar or 15m-close mark-to-market drawdown.",
            "Fees and slippage remain zero, matching the source genetic research run.",
            "Results are per-candidate, not a combined portfolio; no candidate is re-ranked or selected from forward results.",
        ],
    }
