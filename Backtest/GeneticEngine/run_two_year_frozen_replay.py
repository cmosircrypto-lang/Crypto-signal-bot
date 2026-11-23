"""Two-year replay for immutable genetic champions with a separate final holdout.

The 85% development replay is descriptive/in-sample because champions were
selected using development history. The final 15% is reported separately and
must not be used to alter or rank the frozen genomes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from .forward_evaluator import load_frozen_manifest
from .historical_evaluator import (
    DAY_MS, TF_MS, _find_btc_frames, _normalize, _simulate_fold,
)

MIN_HISTORY_DAYS = 700
EXPECTED_SYMBOLS = 76


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _metrics_dict(result) -> dict:
    return {
        "start_utc": _iso(result.start_ms),
        "end_exclusive_utc": _iso(result.end_ms),
        "trade_count": int(result.trade_count),
        "net_pnl": float(result.net_pnl),
        "average_r": float(result.average_r),
        "profit_factor": float(result.profit_factor),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "gross_profit": float(result.gross_profit),
        "gross_loss": float(result.gross_loss),
    }


def run_two_year_replay(data, manifest_path: str | Path) -> dict:
    manifest, manifest_bytes = load_frozen_manifest(manifest_path)
    if len(data) != EXPECTED_SYMBOLS:
        raise ValueError(f"Expected {EXPECTED_SYMBOLS} symbols; got {len(data)}")

    candidate_symbols = sorted({item["symbol"] for item in manifest["candidates"]})
    required_symbols = set(candidate_symbols)
    needs_btc = any(
        "BTC_ALIGNED_PULLBACK" in item["genome"]["signal_families"]
        for item in manifest["candidates"]
    )
    if needs_btc:
        required_symbols.add("BTC-USDT-SWAP")

    # Use the common timestamp intersection so every frozen candidate is tested
    # over the same market window. 15m timestamps are candle OPEN times.
    starts, ends = [], []
    normalized = {}
    for symbol in required_symbols:
        if symbol not in data:
            raise ValueError(f"Missing required symbol data: {symbol}")
        frames = data[symbol].frames
        current = {tf: _normalize(frames[tf]) for tf in ("15m", "1h", "4h")}
        normalized[symbol] = current
        starts.append(int(current["15m"]["timestamp"].iloc[0]))
        ends.append(int(current["15m"]["timestamp"].iloc[-1]) + TF_MS["15m"])
        for tf in ("1h", "4h"):
            starts.append(int(current[tf]["timestamp"].iloc[0]))
            ends.append(int(current[tf]["timestamp"].iloc[-1]) + TF_MS[tf])

    common_start = max(starts)
    common_end = min(ends)
    # Reserve a small common warm-up buffer before counting the measured period.
    measured_start = common_start + 500 * TF_MS["15m"]
    measured_start = ((measured_start + TF_MS["15m"] - 1) // TF_MS["15m"]) * TF_MS["15m"]
    measured_end = (common_end // TF_MS["15m"]) * TF_MS["15m"]
    span_days = (measured_end - measured_start) / DAY_MS
    if span_days < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Cached OKX data covers only {span_days:.1f} common days; "
            f"at least {MIN_HISTORY_DAYS} days are required for a two-year test. "
            "Do not label a shorter history as two years."
        )

    split = measured_start + int((measured_end - measured_start) * 0.85)
    split = (split // TF_MS["15m"]) * TF_MS["15m"]
    if split <= measured_start or split >= measured_end:
        raise ValueError("Invalid 85/15 chronological split")

    btc_frames = _find_btc_frames(data) if needs_btc else None
    if needs_btc and btc_frames is None:
        raise ValueError("BTC-aligned candidate(s) require BTC 1h/4h/derived 1d history")

    results = []
    for index, candidate in enumerate(manifest["candidates"], start=1):
        symbol, direction = candidate["symbol"], candidate["direction"]
        genome_payload = candidate["genome"]
        from .genome import StrategyGenome
        genome = StrategyGenome.from_dict(genome_payload)
        frames = normalized[symbol]
        print(f"[CANDIDATE {index}/11] {symbol} {direction}", flush=True)

        development = _simulate_fold(
            symbol, frames, btc_frames, genome, direction,
            measured_start, split, initial_equity=100.0,
        )
        holdout = _simulate_fold(
            symbol, frames, btc_frames, genome, direction,
            split, measured_end, initial_equity=100.0,
        )
        combined = _simulate_fold(
            symbol, frames, btc_frames, genome, direction,
            measured_start, measured_end, initial_equity=100.0,
        )

        results.append({
            "candidate_id": candidate["candidate_id"],
            "symbol": symbol,
            "direction": direction,
            "genome_sha256": hashlib.sha256(
                json.dumps(genome_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_historical_reference_metrics": candidate["historical_reference_metrics"],
            "development_85pct": _metrics_dict(development),
            "untouched_holdout_15pct": _metrics_dict(holdout),
            "combined_two_year_descriptive_only": _metrics_dict(combined),
        })

    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "source": manifest["source"],
        "dataset": {
            "exchange": "OKX",
            "symbols_loaded": len(data),
            "candidate_symbols": candidate_symbols,
            "common_measured_start_utc": _iso(measured_start),
            "common_end_exclusive_utc": _iso(measured_end),
            "common_history_days": round(span_days, 3),
            "two_year_minimum_days": MIN_HISTORY_DAYS,
            "split_rule": "chronological first 85% development / final 15% untouched holdout",
            "development_end_exclusive_utc": _iso(split),
            "holdout_start_utc": _iso(split),
        },
        "candidate_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "candidate_count": len(results),
        "excluded_candidates": manifest["excluded"],
        "evolution_ran": False,
        "candidate_parameters_changed": False,
        "holdout_used_for_selection": False,
        "portfolio_simulation": False,
        "results": results,
        "methodology_notes": [
            "The 11 serialized champion genomes are frozen byte-for-byte; no evolution, mutation, retraining, or reselection is performed.",
            "All candidates use the exact same common chronological history window.",
            "The first 85% is a development replay and is not independent because source champions were selected using development history.",
            "The final 15% is evaluated separately as the untouched holdout and is not used for ranking, tuning, or selection.",
            "The combined two-year metric is descriptive only and must not be treated as independent out-of-sample performance.",
            "Each symbol-direction candidate is simulated independently from $100; this is not a combined multi-symbol portfolio simulation.",
            "Existing causal simulator semantics are preserved: closed candles, next 15m open entries, SL-first same-candle resolution.",
            "Fees and slippage are zero, matching the source genetic research baseline; live profitability may be lower.",
            "Max drawdown is the existing closed-trade equity drawdown, not intrabar mark-to-market drawdown.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Backtest/data/okx")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="Backtest/results/genetic_two_year_replay")
    args = parser.parse_args(argv)

    root = Path.cwd()
    sys.path.insert(0, str(root / "Backtest/SwingEngine"))
    from data_loader import load_okx_data

    print("[LOAD] loading validated cached OKX data (76 symbols)", flush=True)
    data = load_okx_data(args.data_root, expected_symbols=EXPECTED_SYMBOLS)
    print("[RUN] frozen 11-candidate two-year replay", flush=True)
    report = run_two_year_replay(data, args.manifest)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "genetic_two_year_replay.json"
    csv_path = output / "genetic_two_year_replay.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    columns = [
        "candidate_id", "symbol", "direction", "segment",
        "start_utc", "end_exclusive_utc", "trade_count", "net_pnl",
        "average_r", "profit_factor", "max_drawdown_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in report["results"]:
            for segment in ("development_85pct", "untouched_holdout_15pct", "combined_two_year_descriptive_only"):
                writer.writerow({
                    "candidate_id": item["candidate_id"],
                    "symbol": item["symbol"],
                    "direction": item["direction"],
                    "segment": segment,
                    **item[segment],
                })

    print(
        f"[COMPLETE] candidates={report['candidate_count']}; "
        f"common_days={report['dataset']['common_history_days']}; "
        f"holdout_start={report['dataset']['holdout_start_utc']}",
        flush=True,
    )
    print(f"[REPORT] {json_path}", flush=True)
    print(f"[REPORT] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
