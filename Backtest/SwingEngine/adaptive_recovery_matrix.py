"""Six-scenario OOS experiment for uncapped and balance-adaptive recovery.

Uses the validated 76-symbol prepared OOS cache and unchanged signal preparation. Compares 1D+4H against 1D+4H+1H.
Zero fees, slippage and funding. Simulation only.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import pandas as pd

from recovery_cycle_resumable import CUTOFF_UTC, INITIAL_CAPITAL, _simulate
from okx_contract_specs import OKXContractSpecs


RISK_CAPS = (3.0, 5.0, None)
BTC_FILTER_MODES = ("1d_4h", "1d_4h_1h")
METHODS = ("unlimited_until_tp_or_equity_wipeout",)


def run():
    cache = Path(os.environ.get(
        "RECOVERY_PREPARED_CACHE",
        "Backtest/results/recovery_adaptive_matrix/prepared_oos_cache.pkl",
    ))
    if not cache.exists():
        raise FileNotFoundError(f"Prepared OOS cache not found: {cache}")
    out = Path(os.environ.get(
        "RECOVERY_ADAPTIVE_OUTPUT_DIR",
        "Backtest/results/recovery_adaptive_matrix",
    ))
    out.mkdir(parents=True, exist_ok=True)

    with cache.open("rb") as handle:
        prepared = pickle.load(handle)
    test_data = prepared["test_data"]
    test_signals = prepared["test_signals"]
    obs = prepared["obs"]
    btc_frames = prepared["btc_frames"]
    sets = prepared["sets"]
    selected = sets.get("all_76", [])
    if len(selected) != 76:
        raise ValueError(f"Expected 76 symbols in prepared cache, found {len(selected)}")

    specs = OKXContractSpecs()
    specs.load()
    subset_data = {symbol: test_data[symbol] for symbol in selected}
    subset_signals = {symbol: test_signals[symbol] for symbol in selected}

    summaries = []
    started = time.monotonic()
    total = len(METHODS) * len(BTC_FILTER_MODES) * len(RISK_CAPS)
    scenario = 0
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
      for filter_mode in BTC_FILTER_MODES:
        for cap in RISK_CAPS:
            scenario += 1
            cap_label = "uncapped" if cap is None else f"{int(cap)}pct"
            label = f"all76_target_profit_{method}_btc_{filter_mode}_portfolio_{cap_label}"
            cfg = {
                "objective": "initial_target_profit",
                "sizing_method": "exact_recovery",
                "max_recovery_trades": None,
                "portfolio_open_risk_cap_percent": cap,
                "max_total_positions": 1,
                "max_buy_positions": 1,
                "max_sell_positions": 1,
                "single_position_global": True,
                # If an entry cannot fit the aggregate open-risk ceiling,
                # retain the cycle and wait for a later eligible signal.
                "keep_cycle_when_risk_blocked": True,
                "smart_balance_cap": method == "smart_balance_adaptive_cap",
            }

            def report(done, count, ts):
                if done == count or done % max(1, count // 10) == 0:
                    print(
                        f"[PROGRESS] {scenario}/{total} {label}: "
                        f"{done}/{count} ({done * 100 // max(1, count)}%)",
                        flush=True,
                    )

            summary_path = out / f"summary_{label}.json"
            trades_path = out / f"trades_{label}.csv"
            cycles_path = out / f"cycles_{label}.csv"
            skips_path = out / f"skips_{label}.csv"
            if all(p.exists() for p in (summary_path, trades_path, cycles_path, skips_path)):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summaries.append(summary)
                print(f"[CHECKPOINT] Skipping completed scenario {scenario}/{total}: {label}", flush=True)
                continue
            summary, trades, cycles, skips = _simulate(
                subset_data, subset_signals, obs, specs, btc_frames,
                2.0, 2.0, "all_76", cfg, report,
                filter_mode=filter_mode,
                checkpoint_path=checkpoint_dir / f"{label}.pkl",
                checkpoint_interval=1000,
            )
            summary.update({
                "scenario": label,
                "btc_filter_mode": filter_mode,
                "btc_filter_timeframes": ["1d", "4h"] if filter_mode == "1d_4h" else ["1d", "4h", "1h"],
                "recovery_method": method,
                "recovery_objective": "initial_target_profit",
                "recovery_sizing": "exact_recovery",
                "portfolio_open_risk_cap_percent": cap,
                "max_simultaneous_positions_limit": 1,
                "single_position_global": True,
                "fee_rate": 0.0,
                "slippage_rate": 0.0,
                "funding_cost": 0.0,
                "smart_cap_rule": (
                    "5 if equity>80%; 4 if >60%; 3 if >40%; "
                    "2 if >20%; 1 if >10%; 0 otherwise"
                    if method == "smart_balance_adaptive_cap" else "none"
                ),
                "liquidation_definition": (
                    "conservative equity-wipeout proxy: open stop-risk plus "
                    "next recovery stop-risk >= current equity; not exchange "
                    "maintenance-margin liquidation"
                ),
            })
            trades.to_csv(trades_path, index=False)
            cycles.to_csv(cycles_path, index=False)
            skips.to_csv(skips_path, index=False)
            summary_path.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            summaries.append(summary)
            print(
                f"[SCENARIO COMPLETE] {scenario}/{total} {label} "
                f"equity={summary.get('final_equity')} "
                f"pnl={summary.get('net_pnl')} "
                f"dd={summary.get('max_drawdown_pct')} "
                f"cycles={len(cycles)} elapsed={(time.monotonic()-started)/60:.1f}m",
                flush=True,
            )

    frame = pd.DataFrame(summaries)
    frame.to_csv(out / "adaptive_recovery_summary.csv", index=False)
    (out / "README.md").write_text(
        "# Serialized unlimited recovery — 76-symbol OOS\n\n"
        f"- OOS cutoff: {CUTOFF_UTC}\n"
        "- Same 76-symbol universe, signals, ATR 2.0, RR 2.0 and base risk 0.5%.
- BTC filter comparison: 1D+4H versus 1D+4H+1H; all other inputs are held constant.\n"
        "- Recovery objective: initial-trade target profit; exact recovery sizing.\n"
        "- Three scenarios per BTC filter: unlimited recovery with aggregate open-risk caps of 3%, 5%, and no aggregate cap.\n"
        "- Exactly one open position across the entire portfolio. After an SL, the next eligible signal from any of the 76 symbols may continue the single active recovery cycle.\n"
        "- The recovery-cycle lock is portfolio-wide, not tied to the symbol that first received SL. No second position opens while a position is already active.\n"
        "- Under 3%/5% caps, an oversized recovery entry is skipped while the cycle remains active; the uncapped scenario has no aggregate open-risk ceiling.\n"
        "- Fees, slippage and funding are zero.\n"
        "- Equity wipeout is a conservative proxy based on aggregate open stop-risk plus the next recovery stop-risk reaching current equity; it is not OKX maintenance-margin liquidation.\n"
        "- Historical OOS research only; no live trading.\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)
    print(f"[COMPLETE] {len(summaries)}/{total} scenarios", flush=True)


if __name__ == "__main__":
    run()
