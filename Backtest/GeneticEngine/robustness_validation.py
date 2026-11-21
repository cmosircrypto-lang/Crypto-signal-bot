"""Robustness validation for the Run #6 winning genetic candidate.

Simulation only. Fees, slippage and funding remain zero to match Run #6.
The script does not mutate signal-generation logic or submit exchange orders.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SWING_ENGINE_DIR = str(Path(__file__).resolve().parents[1] / "SwingEngine")
if SWING_ENGINE_DIR not in sys.path:
    sys.path.insert(0, SWING_ENGINE_DIR)

from Backtest.GeneticEngine.evaluator import evaluate_prepared_genome, prepare_genome_inputs
from Backtest.GeneticEngine.genome import StrategyGenome
from data_loader import TIMEFRAME_MS, load_okx_data
from backtest_runner import ProgressMonitor, build_btc_trend_frames
from okx_contract_specs import OKXContractSpecs


SELECTED_26 = (
    "ADA-USDT-SWAP", "APT-USDT-SWAP", "AVAX-USDT-SWAP", "BNB-USDT-SWAP",
    "BRETT-USDT-SWAP", "BTC-USDT-SWAP", "CHZ-USDT-SWAP", "COMP-USDT-SWAP",
    "DOGE-USDT-SWAP", "EGLD-USDT-SWAP", "ENJ-USDT-SWAP", "IOTA-USDT-SWAP",
    "JTO-USDT-SWAP", "JUP-USDT-SWAP", "MEME-USDT-SWAP", "ONDO-USDT-SWAP",
    "OP-USDT-SWAP", "POPCAT-USDT-SWAP", "SNX-USDT-SWAP", "SUSHI-USDT-SWAP",
    "THETA-USDT-SWAP", "TIA-USDT-SWAP", "TRX-USDT-SWAP", "XLM-USDT-SWAP",
    "XRP-USDT-SWAP", "ZIL-USDT-SWAP",
)
INITIAL_EQUITY = 100.0
WINNER = {"btc_filter_mode": "1d_4h_1h", "risk_percent": 0.05, "reward_ratio": 2.5}


def split_oos_windows(close_times: list[int], count: int = 3) -> list[tuple[int, int]]:
    """Return disjoint, exhaustive inclusive windows over ordered OOS candle closes."""
    values = np.asarray(close_times, dtype=np.int64)
    if count < 1 or len(values) < count:
        raise ValueError("OOS timestamps must contain at least one value per window")
    if np.any(np.diff(values) <= 0):
        raise ValueError("OOS timestamps must be strictly increasing")
    chunks = np.array_split(values, count)
    windows = [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]
    for left, right in zip(windows, windows[1:]):
        if left[1] >= right[0]:
            raise AssertionError("OOS windows overlap")
    return windows


def slice_data(data: dict, start_ms: int, end_ms: int) -> dict:
    sliced = {}
    for symbol, item in data.items():
        frames = dict(item.frames)
        frame = item.frames["15m"]
        close_ts = frame["ts"].astype("int64") + TIMEFRAME_MS["15m"]
        frames["15m"] = frame.loc[(close_ts >= start_ms) & (close_ts <= end_ms)].reset_index(drop=True)
        if frames["15m"].empty:
            continue
        sliced[symbol] = SimpleNamespace(symbol=symbol, frames=frames)
    return sliced


def slice_signals(signals: dict, start_ms: int, end_ms: int) -> dict:
    result = {}
    for symbol, frame in signals.items():
        if frame.empty:
            result[symbol] = frame.copy()
        else:
            ts = frame["signal_ts"].astype("int64")
            result[symbol] = frame.loc[(ts >= start_ms) & (ts <= end_ms)].reset_index(drop=True)
    return result


def run_candidate(label, genome, data, signals, obs, specs, btc_frames):
    summary, trades, system = evaluate_prepared_genome(
        data, signals, obs, genome,
        contract_specs=specs,
        initial_capital=INITIAL_EQUITY,
        btc_trend_frames=btc_frames,
    )
    row = {"experiment": label, **summary, "trade_rows": int(len(trades))}
    row["diagnostic_skip_rows"] = len(getattr(system, "_diagnostic_skips", []))
    if not trades.empty:
        trades = trades.copy()
        trades["experiment"] = label
    return row, trades


def main():
    out = Path(os.environ.get("ROBUSTNESS_OUTPUT_DIR", "Backtest/results/genetic_robustness"))
    out.mkdir(parents=True, exist_ok=True)
    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    try:
        monitor.set_stage("load_data", "loading the same cached OKX 76-symbol dataset as Run #6")
        all_data = load_okx_data("Backtest/data/okx", expected_symbols=76)
        missing = sorted(set(SELECTED_26) - set(all_data))
        if missing:
            raise ValueError(f"Run #60 selected symbols missing from cached dataset: {missing}")
        data = {symbol: all_data[symbol] for symbol in SELECTED_26}

        monitor.set_stage("contract_specs", "loading current OKX linear SWAP contract specifications")
        specs = OKXContractSpecs()
        specs.load()

        base = StrategyGenome.baseline()
        monitor.set_stage("restore_signals", "restoring Run #6 per-symbol signal checkpoints where signatures match")
        signals, obs = prepare_genome_inputs(
            data, base, progress_monitor=monitor,
            checkpoint_dir="Backtest/results/genetic_limited/signal_symbol_checkpoints",
        )
        btc_frames = build_btc_trend_frames(data)

        btc_15m = data["BTC-USDT-SWAP"].frames["15m"]
        all_close_times = (btc_15m["ts"].astype("int64") + TIMEFRAME_MS["15m"]).to_numpy(dtype=np.int64)
        cutoff = int(os.environ.get("RUN6_TRAIN_CUTOFF_MS", "1774397700000"))
        oos_times = all_close_times[all_close_times > cutoff]
        windows = split_oos_windows(oos_times.tolist(), 3)
        oos_start, oos_end = windows[0][0], windows[-1][1]

        winner = replace(
            base,
            btc_filter_mode=WINNER["btc_filter_mode"],
            risk_percent=WINNER["risk_percent"],
            reward_ratio=WINNER["reward_ratio"],
        )
        candidates = [
            ("winner", winner),
            ("risk_0.10_rr_2.5_triple", replace(winner, risk_percent=0.10)),
            ("risk_0.25_rr_2.5_triple", replace(winner, risk_percent=0.25)),
            ("risk_0.05_rr_2.0_triple", replace(winner, reward_ratio=2.0)),
            ("risk_0.05_rr_3.0_triple", replace(winner, reward_ratio=3.0)),
            ("filter_1d_4h_risk_0.05_rr_2.5", replace(winner, btc_filter_mode="1d_4h")),
            ("filter_1d_1h_risk_0.05_rr_2.5", replace(winner, btc_filter_mode="1d_1h")),
            ("no_filter_risk_0.05_rr_2.5", replace(winner, btc_filter_mode=None)),
        ]
        for _, genome in candidates:
            genome.validate()

        summary_rows = []
        trade_frames = []
        monitor.set_stage("full_oos_sensitivity", f"evaluating {len(candidates)} candidates on the complete OOS holdout")
        for label, genome in candidates:
            row, trades = run_candidate(label, genome, data, signals, obs, specs, btc_frames)
            row["btc_filter_mode"] = genome.btc_filter_mode or "none"
            row["risk_percent"] = genome.risk_percent
            row["reward_ratio"] = genome.reward_ratio
            summary_rows.append(row)
            if not trades.empty:
                trade_frames.append(trades)
            print(f"[ROBUSTNESS CANDIDATE] {label}: pnl={row.get('net_pnl')} pf={row.get('profit_factor')} trades={row.get('closed_trades')}", flush=True)

        monitor.set_stage("time_windows", "testing the winning candidate independently in three chronological OOS windows")
        for index, (start_ms, end_ms) in enumerate(windows, 1):
            segment_data = slice_data(data, start_ms, end_ms)
            segment_signals = slice_signals(signals, start_ms, end_ms)
            row, trades = run_candidate(
                f"winner_oos_period_{index}", winner,
                segment_data, segment_signals, obs, specs, btc_frames,
            )
            row.update({"window": index, "start_ms": start_ms, "end_ms": end_ms})
            summary_rows.append(row)
            if not trades.empty:
                trade_frames.append(trades)
            print(f"[ROBUSTNESS WINDOW] {index}/3: pnl={row.get('net_pnl')} pf={row.get('profit_factor')} trades={row.get('closed_trades')}", flush=True)

        monitor.set_stage("symbol_contribution", "evaluating the winner independently on each of the 26 selected symbols")
        symbol_rows = []
        for index, symbol in enumerate(SELECTED_26, 1):
            one_data = {symbol: data[symbol]}
            one_signals = {symbol: signals[symbol]}
            one_obs = {symbol: obs[symbol]}
            row, trades = run_candidate(
                f"symbol_{symbol}", winner, one_data, one_signals, one_obs, specs, btc_frames,
            )
            row["symbol"] = symbol
            symbol_rows.append(row)
            if not trades.empty:
                trade_frames.append(trades)
            monitor.progress("symbol_contribution", index, len(SELECTED_26), f"{symbol}: trades={row.get('closed_trades')} pnl={row.get('net_pnl')}")

        summary_frame = pd.DataFrame(summary_rows)
        symbol_frame = pd.DataFrame(symbol_rows)
        summary_frame.to_csv(out / "robustness_summary.csv", index=False)
        symbol_frame.to_csv(out / "winner_per_symbol.csv", index=False)
        if trade_frames:
            pd.concat(trade_frames, ignore_index=True, sort=False).to_csv(out / "robustness_trades.csv", index=False)

        winner_trades = next((frame for frame in trade_frames if not frame.empty and frame["experiment"].iloc[0] == "winner"), pd.DataFrame())
        if not winner_trades.empty and "closed_at" in winner_trades:
            monthly = winner_trades.copy()
            monthly["month_utc"] = pd.to_datetime(monthly["closed_at"], unit="ms", utc=True).dt.strftime("%Y-%m")
            monthly_report = monthly.groupby("month_utc", as_index=False).agg(
                trades=("pnl", "size"), net_pnl=("pnl", "sum"),
                wins=("pnl", lambda values: int((values > 0).sum())),
                losses=("pnl", lambda values: int((values < 0).sum())),
            )
            monthly_report.to_csv(out / "winner_monthly.csv", index=False)

        payload = {
            "run6_source_run_id": 44959250472,
            "initial_equity_usd": INITIAL_EQUITY,
            "training_cutoff_ms": cutoff,
            "oos_start_ms": oos_start,
            "oos_end_ms": oos_end,
            "oos_windows": [{"index": i, "start_ms": a, "end_ms": b} for i, (a, b) in enumerate(windows, 1)],
            "winner_genome": winner.to_dict(),
            "candidate_count": len(candidates),
            "symbol_count": len(symbol_rows),
            "fees_slippage_funding": "zero; same frictionless assumptions as Run #6",
            "execution": "simulation only; no orders",
        }
        (out / "robustness_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (out / "README.md").write_text(
            "# Run #6 Robustness Validation\n\n"
            "- Source: Genetic Engine Run #6 (44959250472).\n"
            "- Universe: same 26 Run #60 selected symbols; same cached 76-symbol OKX dataset.\n"
            "- Winning candidate: BTC filter 1d_4h_1h, risk 0.05%, RR 2.5.\n"
            "- Tests: full OOS parameter sensitivity, three disjoint chronological OOS windows, and independent per-symbol contribution.\n"
            "- Every window and single-symbol test starts from USD 100; do not sum these independent equity curves.\n"
            "- Fees, slippage and funding remain zero to match Run #6; this is not a live-trading cost estimate.\n"
            "- Signal-generation parameters remain at the baseline; no signal logic was changed.\n"
            "- Simulation only; no exchange orders.\n\n"
            "See robustness_summary.csv, winner_per_symbol.csv, winner_monthly.csv (if trades exist), and robustness_trades.csv.\n",
            encoding="utf-8",
        )
        print(f"[ROBUSTNESS COMPLETE] candidates={len(candidates)} windows={len(windows)} symbols={len(symbol_rows)}", flush=True)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
