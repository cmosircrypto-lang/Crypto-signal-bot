"""Focused OOS recovery matrix: compare recovery caps and objectives at zero costs.

Reuses the prepared OOS cache from the validated resumable run. No fees,
slippage, funding, live orders, or production code are involved.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

from recovery_cycle_resumable import (
    CUTOFF_UTC, SELECTED_26, _simulate, _atomic_json,
)
from okx_contract_specs import OKXContractSpecs
from data_loader import load_okx_data
from recovery_cycle_resumable import _dataset_fingerprint

CAPS = (2, 3, 5, None)
OBJECTIVES = ("breakeven", "initial_target_profit")
UNIVERSES = ("all_76", "selected_26_training_strong")
ATR_MULTIPLE = 2.0
REWARD_RATIO = 2.0


def run():
    out = Path(os.environ.get(
        "RECOVERY_FOCUSED_OUTPUT_DIR",
        "Backtest/results/recovery_focused_matrix",
    ))
    out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(os.environ.get(
        "RECOVERY_PREPARED_CACHE",
        "Backtest/results/recovery_resumable/prepared_oos_cache.pkl",
    ))
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Prepared OOS cache not found: {cache_path}. "
            "Restore artifact from successful run #44987822498."
        )

    data_root = os.environ.get("RECOVERY_DATA_ROOT", "Backtest/data/okx")
    data = load_okx_data(data_root, expected_symbols=76)
    fingerprint = _dataset_fingerprint(data)
    with cache_path.open("rb") as handle:
        prepared = pickle.load(handle)
    test_data = prepared["test_data"]
    test_signals = prepared["test_signals"]
    obs = prepared["obs"]
    btc_frames = prepared["btc_frames"]
    sets = prepared["sets"]
    if len(sets.get("all_76", [])) != 76:
        raise ValueError("Prepared cache does not contain the expected 76-symbol universe")
    if len(sets.get("selected_26_training_strong", [])) != 26:
        raise ValueError("Prepared cache selected universe is not exactly 26 symbols")

    specs = OKXContractSpecs()
    specs.load()

    configs = []
    for universe in UNIVERSES:
        configs.append((universe, None))
        for objective in OBJECTIVES:
            for cap in CAPS:
                configs.append((universe, {
                    "objective": objective,
                    "sizing_method": "exact_recovery",
                    "max_recovery_trades": cap,
                }))

    manifest_path = out / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    signature = json.dumps({
        "dataset_fingerprint": fingerprint,
        "cutoff": CUTOFF_UTC,
        "atr": ATR_MULTIPLE,
        "rr": REWARD_RATIO,
        "risk_percent": 0.5,
        "portfolio_open_risk_cap_percent": 3.0,
        "configs": configs,
        "costs": {"fee_rate": 0.0, "slippage_rate": 0.0, "funding": 0.0},
    }, sort_keys=True, default=str)
    if manifest.get("signature") != signature:
        manifest = {"signature": signature, "completed_scenarios": {}}
    completed = manifest.setdefault("completed_scenarios", {})
    summaries = []
    started = time.monotonic()

    for index, (universe, cycle_cfg) in enumerate(configs, 1):
        cap = "none" if cycle_cfg is None else (
            "unlimited" if cycle_cfg["max_recovery_trades"] is None
            else str(cycle_cfg["max_recovery_trades"])
        )
        objective = "no_recovery" if cycle_cfg is None else cycle_cfg["objective"]
        label = f"{universe}_atr2_rr2_{objective}_cap{cap}"
        summary_path = out / f"summary_{label}.json"
        if label in completed and summary_path.exists():
            summaries.append(completed[label])
            print(f"[CHECKPOINT] skip {index}/{len(configs)} {label}", flush=True)
            continue

        selected = sets[universe]
        subset_data = {symbol: test_data[symbol] for symbol in selected}
        subset_signals = {symbol: test_signals[symbol] for symbol in selected}

        def report(done, total, ts):
            if done == total or done % max(1, total // 10) == 0:
                print(
                    f"[PROGRESS] {index}/{len(configs)} {label} "
                    f"{done}/{total} ({done * 100 // max(1, total)}%)",
                    flush=True,
                )

        summary, trades, cycles, skips = _simulate(
            subset_data, subset_signals, obs, specs, btc_frames,
            ATR_MULTIPLE, REWARD_RATIO, universe, cycle_cfg, report,
        )
        summary.update({
            "scenario": label,
            "recovery_objective": objective,
            "recovery_sizing": "exact_recovery" if cycle_cfg else "none",
            "max_recovery_trades": cycle_cfg["max_recovery_trades"] if cycle_cfg else 0,
            "fee_rate": 0.0,
            "slippage_rate": 0.0,
            "funding_cost": 0.0,
        })
        trades.to_csv(out / f"trades_{label}.csv", index=False)
        cycles.to_csv(out / f"cycles_{label}.csv", index=False)
        skips.to_csv(out / f"skips_{label}.csv", index=False)
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        completed[label] = summary
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(manifest_path, manifest)
        summaries.append(summary)
        print(
            f"[SCENARIO COMPLETE] {index}/{len(configs)} {label} "
            f"trades={len(trades)} cycles={len(cycles)} "
            f"final_equity={summary.get('final_equity')} "
            f"elapsed={(time.monotonic()-started)/60:.1f}m",
            flush=True,
        )

    import pandas as pd
    frame = pd.DataFrame(summaries)
    frame.to_csv(out / "focused_recovery_summary.csv", index=False)
    readme = [
        "# Focused recovery matrix — OOS",
        "",
        f"- OOS cutoff UTC: {CUTOFF_UTC}",
        "- Universes: all 76 symbols and selected 26 training-strong symbols.",
        "- ATR stop: 2.0 ATR; reward/risk: 2.0; base risk: 0.5%.",
        "- BTC trend filter: causal 1d + 4h alignment.",
        "- Recovery sizing: exact recovery; objectives: breakeven and initial-trade target profit.",
        "- Recovery caps: 2, 3, 5, and unlimited additional recovery entries.",
        "- Fees, slippage, and funding: zero.",
        "- Maximum one open position per symbol; aggregate open-risk ceiling: 3%.",
        "- Unresolved cycles and portfolio-risk rejections remain explicitly reported.",
        "- This is historical OOS research, not live-trading validation.",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    manifest["complete"] = len(completed) == len(configs)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    print(frame.to_string(index=False), flush=True)
    print(f"[COMPLETE] {len(completed)}/{len(configs)} scenarios in {out}", flush=True)


if __name__ == "__main__":
    run()
