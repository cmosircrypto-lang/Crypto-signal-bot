"""Six-scenario OOS experiment for uncapped and balance-adaptive recovery.

Uses the validated 76-symbol prepared OOS cache and unchanged signal preparation.
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


RISK_CAPS = (None, 3.0, 5.0)
METHODS = ("unlimited_until_tp_or_equity_wipeout", "smart_balance_adaptive_cap")


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
    total = len(METHODS) * len(RISK_CAPS)
    scenario = 0
    for method in METHODS:
        for cap in RISK_CAPS:
            scenario += 1
            cap_label = "uncapped" if cap is None else f"{int(cap)}pct"
            label = f"all76_target_profit_{method}_portfolio_{cap_label}"
            cfg = {
                "objective": "initial_target_profit",
                "sizing_method": "exact_recovery",
                "max_recovery_trades": None,
                "portfolio_open_risk_cap_percent": cap,
                "max_total_positions": 76,
                "max_buy_positions": 76,
                "max_sell_positions": 76,
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

            summary, trades, cycles, skips = _simulate(
                subset_data, subset_signals, obs, specs, btc_frames,
                2.0, 2.0, "all_76", cfg, report,
            )
            summary.update({
                "scenario": label,
                "recovery_method": method,
                "recovery_objective": "initial_target_profit",
                "recovery_sizing": "exact_recovery",
                "portfolio_open_risk_cap_percent": cap,
                "max_simultaneous_positions": 76,
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
            trades.to_csv(out / f"trades_{label}.csv", index=False)
            cycles.to_csv(out / f"cycles_{label}.csv", index=False)
            skips.to_csv(out / f"skips_{label}.csv", index=False)
            (out / f"summary_{label}.json").write_text(
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
        "# Adaptive recovery matrix — 76-symbol OOS\n\n"
        f"- OOS cutoff: {CUTOFF_UTC}\n"
        "- Same 76-symbol universe, signals, BTC 1d+4h filter, ATR 2.0, RR 2.0 and base risk 0.5%.\n"
        "- Recovery objective: initial-trade target profit; exact recovery sizing.\n"
        "- Six scenarios: two recovery methods × aggregate open-risk caps of none, 3%, 5%.\n"
        "- Both methods remove the previous 6-position portfolio limit; max one open position per symbol remains.\n"
        "- Smart cap: 5/4/3/2/1/0 recovery entries as current equity falls through >80/>60/>40/>20/>10/≤10% of initial capital.\n"
        "- When a recovery entry does not fit the 3%/5% open-risk budget, the cycle is retained and waits for a later eligible signal.\n"
        "- Fees, slippage and funding are zero.\n"
        "- Equity wipeout is a conservative proxy based on aggregate open stop-risk plus the next recovery stop-risk reaching current equity; it is not OKX maintenance-margin liquidation.\n"
        "- Historical OOS research only; no live trading.\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)
    print(f"[COMPLETE] {len(summaries)}/{total} scenarios", flush=True)


if __name__ == "__main__":
    run()
