"""Chronological train/OOS experiment for BTC-filtered symbol selection."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtest_runner import FILTER_MODES, INITIAL_CAPITAL, build_btc_trend_frames, prepare_signals, simulate_scenario
from data_loader import TIMEFRAME_MS, load_okx_data
from okx_contract_specs import OKXContractSpecs

RISK_PERCENT = 0.5
REWARD_RATIO = 2.5
TRAIN_FRACTION = 0.70
MIN_TRAIN_TRADES_PER_FILTER = 5


def _subset_data(data, cutoff, training):
    result = {}
    for symbol, item in data.items():
        frame = item.frames["15m"]
        close_ts = frame["ts"].astype("int64") + TIMEFRAME_MS["15m"]
        mask = close_ts <= cutoff if training else close_ts > cutoff
        result[symbol] = SimpleNamespace(symbol=symbol, frames={"15m": frame.loc[mask].reset_index(drop=True)})
    return result


def _subset_signals(signals, cutoff, training):
    result = {}
    for symbol, frame in signals.items():
        if frame is None or frame.empty:
            result[symbol] = frame.copy()
        else:
            mask = frame["signal_ts"].astype("int64") <= cutoff if training else frame["signal_ts"].astype("int64") > cutoff
            result[symbol] = frame.loc[mask].reset_index(drop=True)
    return result


def _symbol_stats(trades):
    columns = ["symbol", "closed_trades", "wins", "losses", "net_pnl", "gross_profit", "gross_loss", "profit_factor"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for symbol, group in trades.groupby("symbol", sort=True):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
        rows.append({
            "symbol": symbol, "closed_trades": int(len(group)),
            "wins": int((pnl > 0).sum()), "losses": int((pnl < 0).sum()),
            "net_pnl": float(pnl.sum()), "gross_profit": gp, "gross_loss": gl,
            "profit_factor": gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0),
        })
    return pd.DataFrame(rows)


def classify_symbols(training_by_filter, min_trades=MIN_TRAIN_TRADES_PER_FILTER):
    tables, universe = {}, set()
    for mode, trades in training_by_filter.items():
        table = _symbol_stats(trades).set_index("symbol")
        tables[mode] = table
        universe.update(table.index)
    strong, weak, rows = [], [], []
    for symbol in sorted(universe):
        per_mode = {}
        for mode in FILTER_MODES:
            table = tables[mode]
            if symbol not in table.index or int(table.loc[symbol, "closed_trades"]) < min_trades:
                break
            per_mode[mode] = table.loc[symbol]
        if len(per_mode) != len(FILTER_MODES):
            continue
        pfs = [float(per_mode[mode]["profit_factor"]) for mode in FILTER_MODES]
        classification = "mixed"
        if all(value > 1.0 for value in pfs):
            strong.append(symbol)
            classification = "strong"
        elif all(value < 1.0 for value in pfs):
            weak.append(symbol)
            classification = "weak"
        row = {"symbol": symbol, "classification": classification}
        for mode in FILTER_MODES:
            row[f"{mode}_trades"] = int(per_mode[mode]["closed_trades"])
            row[f"{mode}_pf"] = float(per_mode[mode]["profit_factor"])
        rows.append(row)
    return strong, weak, pd.DataFrame(rows)


def _run_one(data, signals, obs, specs, btc_frames, mode):
    return simulate_scenario(
        data, signals, obs, risk_percent=RISK_PERCENT, reward_ratio=REWARD_RATIO,
        contract_specs=specs, initial_capital=INITIAL_CAPITAL,
        filter_mode=mode, btc_trend_frames=btc_frames,
    )


def run_experiment(data_root="Backtest/data/okx", output_dir="Backtest/results/symbol_filter_experiment"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("[STAGE] loading and validating OKX dataset", flush=True)
    data = load_okx_data(data_root, expected_symbols=76)

    timeline = sorted({
        int(ts) + TIMEFRAME_MS["15m"]
        for item in data.values() for ts in item.frames["15m"]["ts"].tolist()
    })
    if len(timeline) < 10:
        raise ValueError("Insufficient 15m timeline for chronological split")
    cutoff = timeline[max(1, min(len(timeline) - 2, int(len(timeline) * TRAIN_FRACTION) - 1))]
    cutoff_iso = pd.to_datetime(cutoff, unit="ms", utc=True).isoformat()

    print("[STAGE] loading actual OKX contract specifications", flush=True)
    specs = OKXContractSpecs()
    specs.load()
    print("[STAGE] preparing causal signals and BTC trends", flush=True)
    signals, obs = prepare_signals(data)
    btc_frames = build_btc_trend_frames(data)

    train_data = _subset_data(data, cutoff, True)
    test_data = _subset_data(data, cutoff, False)
    train_signals = _subset_signals(signals, cutoff, True)
    test_signals = _subset_signals(signals, cutoff, False)

    train_trades, train_summaries = {}, []
    for index, mode in enumerate(FILTER_MODES, 1):
        print(f"[TRAIN] {index}/{len(FILTER_MODES)} {mode}", flush=True)
        summary, trades, _ = _run_one(train_data, train_signals, obs, specs, btc_frames, mode)
        train_trades[mode] = trades
        train_summaries.append({"phase": "train", "symbol_set": "all_76", "scenario": mode, **summary})
        trades.to_csv(output / f"train_trades_{mode}.csv", index=False)

    strong, weak, classification = classify_symbols(train_trades)
    classification.to_csv(output / "training_symbol_classification.csv", index=False)
    symbol_sets = {
        "all_76": sorted(data),
        "exclude_training_weak": sorted(set(data) - set(weak)),
        "training_strong_only": sorted(strong),
    }
    (output / "symbol_sets.json").write_text(json.dumps({
        "cutoff_utc": cutoff_iso, "train_fraction": TRAIN_FRACTION,
        "minimum_trades_per_filter": MIN_TRAIN_TRADES_PER_FILTER,
        "risk_percent": RISK_PERCENT, "reward_ratio": REWARD_RATIO,
        "symbol_sets": symbol_sets,
    }, indent=2), encoding="utf-8")

    test_summaries = []
    for set_name, symbols in symbol_sets.items():
        if not symbols:
            print(f"[WARN] {set_name} empty; skipping", flush=True)
            continue
        selected_data = {symbol: test_data[symbol] for symbol in symbols}
        selected_signals = {symbol: test_signals[symbol] for symbol in symbols}
        for mode in FILTER_MODES:
            print(f"[OOS] set={set_name} filter={mode} symbols={len(symbols)}", flush=True)
            summary, trades, system = _run_one(selected_data, selected_signals, obs, specs, btc_frames, mode)
            scenario = f"{set_name}_{mode}"
            test_summaries.append({
                "phase": "out_of_sample", "symbol_set": set_name,
                "scenario": scenario, "symbols_in_universe": len(symbols), **summary,
            })
            trades.to_csv(output / f"oos_trades_{scenario}.csv", index=False)
            _symbol_stats(trades).to_csv(output / f"oos_symbol_stats_{scenario}.csv", index=False)
            pd.DataFrame(getattr(system, "_diagnostic_skips", [])).to_csv(
                output / f"oos_skips_{scenario}.csv", index=False
            )

    summary = pd.DataFrame(train_summaries + test_summaries)
    summary.to_csv(output / "experiment_summary.csv", index=False)
    lines = [
        "# Chronological symbol-filter experiment", "",
        f"- Train/OOS cutoff UTC: {cutoff_iso}",
        f"- Train share: {TRAIN_FRACTION:.0%}",
        f"- Risk: {RISK_PERCENT}% of current equity; RR: {REWARD_RATIO}",
        "- Actual OKX contract lot/minimum rules enabled.",
        "- Fees, slippage and funding are zero, matching prior runs.",
        f"- Strong symbols: {len(strong)}; weak symbols: {len(weak)}.",
        f"- All-symbol universe: {len(symbol_sets['all_76'])}.",
        f"- Excluding training-weak: {len(symbol_sets['exclude_training_weak'])}.",
        f"- Training-strong only: {len(symbol_sets['training_strong_only'])}.",
        "", "Detailed metrics are in experiment_summary.csv; symbol-level OOS files are also included.",
        "Research only; results are not evidence of live profitability."
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[COMPLETE] saved experiment to {output}", flush=True)
    return summary


if __name__ == "__main__":
    run_experiment()
