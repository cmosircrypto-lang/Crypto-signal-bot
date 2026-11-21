"""Robustness validation for the Run #6 limited-genetic champion.

Keeps signal-generation logic fixed at the parity-tested baseline. Compares
BTC filters, risk and reward/risk on the same 26-symbol universe, then reports
the champion's chronological subperiod and per-symbol performance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import replace

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWING = ROOT / "Backtest" / "SwingEngine"
if str(SWING) not in sys.path:
    sys.path.insert(0, str(SWING))

from Backtest.GeneticEngine.evaluator import prepare_genome_inputs, evaluate_prepared_genome
from Backtest.GeneticEngine.genome import StrategyGenome
from Backtest.SwingEngine.data_loader import load_okx_data, TIMEFRAME_MS
from Backtest.SwingEngine.okx_contract_specs import OKXContractSpecs
from Backtest.SwingEngine.backtest_runner import ProgressMonitor, build_btc_trend_frames


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
CHAMPION = {"risk_percent": 0.05, "reward_ratio": 2.5, "btc_filter_mode": "1d_4h_1h"}
FILTERS = (None, "1d_4h", "1d_1h", "1d_4h_1h")
RISKS = (0.05, 0.1, 0.25, 0.5)
REWARDS = (1.5, 2.0, 2.5)


def _make_genome(base, risk, reward, filter_mode):
    return replace(base, risk_percent=float(risk), reward_ratio=float(reward),
                   btc_filter_mode=filter_mode)


def _trade_stats(trades):
    if trades.empty:
        return {"closed_trades": 0, "wins": 0, "losses": 0,
                "win_rate_pct": 0.0, "net_pnl": 0.0}
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    return {
        "closed_trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins * 100.0 / len(trades), 6),
        "net_pnl": round(float(pnl.sum()), 8),
    }


def _period_data(data, signals, lower_exclusive, upper_inclusive):
    sliced = {}
    for symbol, item in data.items():
        frame = item.frames["15m"]
        close_ts = frame["ts"].astype("int64") + TIMEFRAME_MS["15m"]
        selected = frame.loc[
            (close_ts > lower_exclusive) & (close_ts <= upper_inclusive)
        ].reset_index(drop=True)
        sliced[symbol] = type(item)(symbol=symbol, frames={
            **item.frames, "15m": selected
        })
    sliced_signals = {}
    for symbol, frame in signals.items():
        if frame.empty:
            sliced_signals[symbol] = frame.copy()
        else:
            ts = frame["signal_ts"].astype("int64")
            sliced_signals[symbol] = frame.loc[
                (ts > lower_exclusive) & (ts <= upper_inclusive)
            ].reset_index(drop=True)
    return sliced, sliced_signals


def run(data_root="Backtest/data/okx", output_dir="Backtest/results/genetic_robustness"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    try:
        monitor.set_stage("data_loading", "loading validated 76-symbol OKX data")
        all_data = load_okx_data(data_root, expected_symbols=76)
        missing = sorted(set(SELECTED_26) - set(all_data))
        if missing:
            raise ValueError(f"Missing selected symbols: {missing}")
        data = {s: all_data[s] for s in SELECTED_26}
        coverage = []
        for symbol, item in sorted(data.items()):
            f = item.frames["15m"]
            coverage.append({
                "symbol": symbol,
                "first_close_utc": pd.to_datetime(int(f["ts"].iloc[0]) + TIMEFRAME_MS["15m"], unit="ms", utc=True).isoformat(),
                "last_close_utc": pd.to_datetime(int(f["ts"].iloc[-1]) + TIMEFRAME_MS["15m"], unit="ms", utc=True).isoformat(),
                "candles_15m": len(f),
            })
        pd.DataFrame(coverage).to_csv(out / "dataset_coverage.csv", index=False)

        monitor.set_stage("contract_specs", "loading OKX linear swap contract specifications")
        specs = OKXContractSpecs()
        specs.load()

        base = StrategyGenome.baseline()
        monitor.set_stage("signal_preparation", "preparing baseline signals for the selected 26 symbols")
        signals, obs = prepare_genome_inputs(
            data, base, progress_monitor=monitor,
            checkpoint_dir=out / "signal_symbol_checkpoints",
        )
        btc_frames = build_btc_trend_frames(data)

        # Nine unique, controlled scenarios. Every comparison changes one
        # dimension at a time from the Run #6 champion.
        scenarios = []
        for filt in FILTERS:
            scenarios.append(("filter", CHAMPION["risk_percent"],
                              CHAMPION["reward_ratio"], filt))
        for risk in RISKS:
            scenarios.append(("risk", risk, CHAMPION["reward_ratio"],
                              CHAMPION["btc_filter_mode"]))
        for rr in REWARDS:
            scenarios.append(("reward", CHAMPION["risk_percent"], rr,
                              CHAMPION["btc_filter_mode"]))
        unique = {}
        for dimension, risk, rr, filt in scenarios:
            key = (float(risk), float(rr), filt)
            unique[key] = dimension

        rows = []
        champion_trades = None
        total = len(unique)
        for index, ((risk, rr, filt), dimension) in enumerate(unique.items(), 1):
            monitor.set_stage("scenario_backtests", f"scenario {index}/{total}: risk={risk:g}% RR={rr:g} filter={filt or 'none'}")
            genome = _make_genome(base, risk, rr, filt)
            summary, trades, system = evaluate_prepared_genome(
                data, signals, obs, genome, contract_specs=specs,
                initial_capital=INITIAL_EQUITY, btc_trend_frames=btc_frames,
            )
            row = {
                "scenario": index, "comparison_dimension": dimension,
                "risk_percent": risk, "reward_ratio": rr,
                "btc_filter_mode": filt or "no_filter",
                **summary,
                "trade_rows": len(trades),
                "skip_rows": len(getattr(system, "_diagnostic_skips", [])),
            }
            rows.append(row)
            tag = f"risk_{risk:g}_rr_{rr:g}_{filt or 'no_filter'}".replace(".", "p")
            trades.to_csv(out / f"trades_{tag}.csv", index=False)
            if (risk, rr, filt) == (
                CHAMPION["risk_percent"], CHAMPION["reward_ratio"],
                CHAMPION["btc_filter_mode"]
            ):
                champion_trades = trades.copy()
            print(f"[ROBUSTNESS SCENARIO] {index}/{total} {tag} final={summary.get('final_equity')} trades={len(trades)}", flush=True)

        pd.DataFrame(rows).to_csv(out / "scenario_comparison.csv", index=False)
        if champion_trades is None:
            raise RuntimeError("Champion scenario was not evaluated")

        # Per-symbol decomposition of the full-period champion ledger.
        symbol_rows = []
        for symbol in SELECTED_26:
            group = champion_trades.loc[champion_trades["symbol"] == symbol]
            symbol_rows.append({"symbol": symbol, **_trade_stats(group)})
        pd.DataFrame(symbol_rows).to_csv(out / "champion_by_symbol.csv", index=False)

        # Three contiguous chronological partitions. Trades are not carried
        # between partitions; any position still open at a partition boundary
        # is reported as open by the simulator, not force-closed.
        btc = data["BTC-USDT-SWAP"].frames["15m"]
        close_times = (btc["ts"].astype("int64") + TIMEFRAME_MS["15m"]).tolist()
        lower = int(close_times[0] - TIMEFRAME_MS["15m"])
        upper = int(close_times[-1])
        boundaries = [
            lower,
            int(close_times[len(close_times) // 3]),
            int(close_times[(2 * len(close_times)) // 3]),
            upper,
        ]
        period_rows = []
        for p in range(3):
            lo, hi = boundaries[p], boundaries[p + 1]
            p_data, p_signals = _period_data(data, signals, lo, hi)
            monitor.set_stage("period_validation", f"period {p + 1}/3")
            genome = _make_genome(base, **CHAMPION)
            summary, trades, system = evaluate_prepared_genome(
                p_data, p_signals, obs, genome, contract_specs=specs,
                initial_capital=INITIAL_EQUITY, btc_trend_frames=btc_frames,
            )
            period_rows.append({
                "period": p + 1,
                "start_utc": pd.to_datetime(lo, unit="ms", utc=True).isoformat(),
                "end_utc": pd.to_datetime(hi, unit="ms", utc=True).isoformat(),
                **summary,
                "trade_rows": len(trades),
                "skip_rows": len(getattr(system, "_diagnostic_skips", [])),
            })
            trades.to_csv(out / f"champion_period_{p + 1}_trades.csv", index=False)
        pd.DataFrame(period_rows).to_csv(out / "champion_by_period.csv", index=False)

        metadata = {
            "source_run": 6,
            "source_commit": "21a01ee4c2f8068a42e0ea9a8e374301cdb59f7e",
            "universe": list(SELECTED_26),
            "initial_equity_usd": INITIAL_EQUITY,
            "champion": CHAMPION,
            "comparison_scenarios": len(unique),
            "chronological_periods": 3,
            "fees": 0,
            "slippage": 0,
            "signal_generation_genes": "fixed at parity-tested baseline",
            "execution": "simulation only; no orders",
            "period_boundary_policy": "no positions carried between periods; positions remaining open at a boundary are reported, not force-closed",
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out / "README.md").write_text(
            "# Run 6 Robustness Validation\n\n"
            "- Same Run #60 selected 26-symbol universe.\n"
            "- Nine unique controlled comparisons: BTC filter, risk and reward/risk.\n"
            "- Champion also measured across three chronological periods and decomposed by symbol.\n"
            "- Signal-generation logic remains fixed at the parity-tested baseline.\n"
            "- Initial equity USD 100; fees and slippage zero; simulation only.\n"
            "- Periods are independent; positions are not carried across period boundaries.\n"
            "- See scenario_comparison.csv, champion_by_period.csv and champion_by_symbol.csv.\n",
            encoding="utf-8",
        )
        print("[ROBUSTNESS COMPLETE] scenarios=9 periods=3 symbols=26", flush=True)
    finally:
        monitor.stop()


if __name__ == "__main__":
    run()
