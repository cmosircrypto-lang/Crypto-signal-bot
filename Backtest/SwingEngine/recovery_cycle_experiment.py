"""Run #48: per-symbol one-position, ATR stop/RR and recovery-cycle OOS matrix."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from backtest_runner import (
    INITIAL_CAPITAL, ProgressMonitor, build_btc_trend_frames,
    btc_trends_at_time, prepare_signals, _equity_metrics,
)
from data_loader import TIMEFRAME_MS, load_okx_data
from okx_contract_specs import OKXContractSpecs
from trading_system import TradingSystem

RISK_PERCENT = 0.5
FILTER_MODE = "1d_4h"
ATR_MULTIPLIERS = (1.0, 1.5, 2.0)
REWARD_RATIOS = (1.5, 2.0, 2.5)
RECOVERY_LIMITS = (2, 3, 5, None)  # recovery attempts after the initial trade
OBJECTIVES = ("initial_target_profit", "breakeven")
SIZING_METHODS = ("exact_recovery", "double_size")
CUTOFF_UTC = "2026-03-25T00:00:00+00:00"
SELECTED_26 = [
    "ADA-USDT-SWAP", "APT-USDT-SWAP", "AVAX-USDT-SWAP", "BNB-USDT-SWAP",
    "BRETT-USDT-SWAP", "BTC-USDT-SWAP", "CHZ-USDT-SWAP", "COMP-USDT-SWAP",
    "DOGE-USDT-SWAP", "EGLD-USDT-SWAP", "ENJ-USDT-SWAP", "IOTA-USDT-SWAP",
    "JTO-USDT-SWAP", "JUP-USDT-SWAP", "MEME-USDT-SWAP", "ONDO-USDT-SWAP",
    "OP-USDT-SWAP", "POPCAT-USDT-SWAP", "SNX-USDT-SWAP", "SUSHI-USDT-SWAP",
    "THETA-USDT-SWAP", "TIA-USDT-SWAP", "TRX-USDT-SWAP", "XLM-USDT-SWAP",
    "XRP-USDT-SWAP", "ZIL-USDT-SWAP",
]


def _atr14(frame):
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-previous).abs(),
                    (low-previous).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _attach_atr(signals, data):
    result = {}
    for symbol, frame in signals.items():
        if frame is None or frame.empty:
            result[symbol] = frame.copy()
            continue
        atr = _atr14(data[symbol].frames["15m"])
        copied = frame.copy()
        copied["atr14"] = copied["signal_index"].map(atr)
        result[symbol] = copied
    return result


def _split_oos(data, signals, cutoff):
    test_data, test_signals = {}, {}
    for symbol, item in data.items():
        frame = item.frames["15m"]
        close_ts = frame["ts"].astype("int64") + TIMEFRAME_MS["15m"]
        mask = close_ts > cutoff
        test_data[symbol] = SimpleNamespace(
            symbol=symbol,
            frames={"15m": frame.loc[mask].reset_index(drop=True)},
        )
        sig = signals[symbol]
        test_signals[symbol] = sig.loc[
            sig["signal_ts"].astype("int64") > cutoff
        ].reset_index(drop=True) if not sig.empty else sig.copy()
    return test_data, test_signals


def _levels(signal, atr_multiple, rr):
    entry = float(signal["entry_price"])
    atr = float(signal["atr14"])
    if not (pd.notna(atr) and atr > 0):
        return None
    distance = atr * atr_multiple
    if str(signal["direction"]).upper() == "BUY":
        stop, target = entry-distance, entry+distance*rr
    else:
        stop, target = entry+distance, entry-distance*rr
    if stop <= 0 or target <= 0:
        return None
    return stop, target


def recovery_risk_amount(losses, base_risk, base_target_profit, rr, objective, sizing_method, recovery_entries):
    """Return requested risk dollars for the next recovery entry."""
    if sizing_method == "exact_recovery":
        target = base_target_profit if objective == "initial_target_profit" else 0.0
        return (float(losses) + float(target)) / float(rr)
    if sizing_method == "double_size":
        return float(base_risk) * (2 ** (int(recovery_entries) + 1))
    raise ValueError("Unknown recovery sizing method")


def _simulate(data, signals, obs, specs, btc_frames, atr_multiple, rr,
              universe, cycle_config=None, progress=None):
    system = TradingSystem(
        initial_capital=INITIAL_CAPITAL, fee_rate=0.0, slippage_rate=0.0,
        max_total=6, max_buy=3, max_sell=3, contract_specs=specs,
    )
    candles_at_time, signals_at_time = {}, {}
    for symbol, item in sorted(data.items()):
        for row in item.frames["15m"].to_dict("records"):
            close_time = int(row["ts"]) + TIMEFRAME_MS["15m"]
            candles_at_time.setdefault(close_time, {})[symbol] = row
    for symbol, frame in signals.items():
        if not frame.empty:
            for row in frame.to_dict("records"):
                signals_at_time.setdefault(int(row["signal_ts"]), []).append((symbol, row))
    times = sorted(set(candles_at_time) | set(signals_at_time))
    active, cycles, completed_cycles = {}, {}, []
    cycle_serial = {}
    closed, skips = [], []
    cycle_counts = {"cycles_started": 0, "cycles_recovered": 0,
                    "cycles_failed_limit": 0, "cycles_unresolved": 0,
                    "cycles_sizing_failed": 0}
    for idx, ts in enumerate(times, 1):
        candles = candles_at_time.get(ts, {})
        # Close first; same-symbol capacity is available only after exit.
        for symbol in list(system.execution.open_positions):
            candle = candles.get(symbol)
            pos = system.execution.open_positions.get(symbol)
            if candle is None or pos is None or ts <= pos.opened_at:
                continue
            result = system.check_exit(symbol, candle, ts)
            if result is None:
                continue
            meta = active.pop(symbol, {})
            row = {
                **meta, "symbol": result.symbol, "direction": result.direction,
                "entry_price": result.entry_price, "exit_price": result.exit_price,
                "quantity": result.quantity, "contract_count": result.contract_count,
                "risk_percent": result.risk_percent, "risk_amount": pos.risk_amount,
                "pnl": result.pnl, "result": result.result,
                "exit_reason": result.exit_reason, "opened_at": result.opened_at,
                "closed_at": result.closed_at,
            }
            closed.append(row)
            if cycle_config:
                state = cycles.get(symbol)
                if state:
                    state["trade_count"] += 1
                    state["cycle_realized_pnl"] += float(result.pnl)
                    if result.exit_reason == "SL":
                        state["losses"] += abs(min(0.0, float(result.pnl)))
                        if meta.get("is_recovery"):
                            state["recovery_losses"] += 1
                        cap = cycle_config["max_recovery_trades"]
                        if cap is not None and state["recovery_entries"] >= cap:
                            state["status"] = "FAILED_LIMIT"
                            completed_cycles.append(dict(state))
                            cycles.pop(symbol, None)
                            cycle_counts["cycles_failed_limit"] += 1
                    elif result.exit_reason == "TP":
                        state["status"] = "RECOVERED" if state["recovery_entries"] else "BASE_TP"
                        state["cycle_closed_at"] = ts
                        state["cycle_net_pnl"] = state["cycle_realized_pnl"]
                        completed_cycles.append(dict(state))
                        cycles.pop(symbol, None)
                        if state["recovery_entries"]:
                            cycle_counts["cycles_recovered"] += 1

        entries = sorted(signals_at_time.get(ts, []), key=lambda x: x[0])
        for symbol, signal in entries:
            direction = str(signal["direction"]).upper()
            if not system.can_search_symbol(symbol):
                continue
            trend = btc_trends_at_time(btc_frames, ts)
            if trend["1d"] != direction or trend["4h"] != direction:
                continue
            levels = _levels(signal, atr_multiple, rr)
            if levels is None:
                skips.append({"symbol": symbol, "signal_ts": ts, "reason": "invalid_atr_or_levels"})
                continue
            stop, target = levels
            is_recovery = bool(cycle_config and symbol in cycles)
            risk_amount = None
            if cycle_config and not is_recovery:
                # A new independent cycle starts with the base trade.
                cycle_serial[symbol] = cycle_serial.get(symbol, 0) + 1
            if is_recovery:
                state = cycles[symbol]
                risk_amount = recovery_risk_amount(
                    state["losses"], state["base_risk_amount"],
                    state["base_target_profit"], rr,
                    cycle_config["objective"], cycle_config["sizing_method"],
                    state["recovery_entries"],
                )
                if risk_amount <= 0 or risk_amount >= system.execution.equity:
                    state["status"] = "FAILED_RISK_CAP"
                    completed_cycles.append(dict(state))
                    cycles.pop(symbol, None)
                    cycle_counts["cycles_sizing_failed"] += 1
                    continue
                risk_pct = risk_amount / system.execution.equity * 100.0
            else:
                risk_pct = RISK_PERCENT
            try:
                pos = system.open_trade(
                    symbol=symbol, direction=direction,
                    entry_price=float(signal["entry_price"]),
                    stop_loss=stop, take_profit=target,
                    risk_percent=risk_pct, timestamp=ts,
                )
                if pos is None:
                    continue
                if cycle_config and not is_recovery:
                    state = {
                        "symbol": symbol, "cycle_id": f"{symbol}-{cycle_serial[symbol]}",
                        "started_at": ts, "cycle_closed_at": None,
                        "base_risk_amount": float(pos.risk_amount),
                        "base_target_profit": float(pos.risk_amount) * rr,
                        "losses": 0.0, "recovery_losses": 0,
                        "recovery_entries": 0, "trade_count": 0,
                        "cycle_realized_pnl": 0.0, "cycle_net_pnl": None,
                        "status": "ACTIVE", "objective": cycle_config["objective"],
                        "sizing_method": cycle_config["sizing_method"],
                        "max_recovery_trades": cycle_config["max_recovery_trades"],
                    }
                    cycles[symbol] = state
                    cycle_counts["cycles_started"] += 1
                elif is_recovery:
                    cycles[symbol]["recovery_entries"] += 1
                active[symbol] = {
                    "atr_multiple": atr_multiple, "reward_ratio": rr,
                    "cycle_id": cycles[symbol]["cycle_id"] if cycle_config else None,
                    "cycle_trade_number": cycles[symbol]["trade_count"] + 1 if cycle_config else None,
                    "is_recovery": is_recovery,
                    "recovery_entry_number": cycles[symbol]["recovery_entries"] if is_recovery else 0,
                    "recovery_objective": cycle_config["objective"] if cycle_config else "none",
                    "recovery_sizing": cycle_config["sizing_method"] if cycle_config else "none",
                }
            except (ValueError, KeyError) as exc:
                skips.append({"symbol": symbol, "signal_ts": ts, "reason": "contract_or_size", "error": str(exc)})
        if progress and (idx % max(1, len(times)//20) == 0 or idx == len(times)):
            progress(idx, len(times), ts)
    for state in cycles.values():
        state["status"] = "UNRESOLVED_AT_END"
        completed_cycles.append(dict(state))
        cycle_counts["cycles_unresolved"] += 1
    trades = pd.DataFrame(closed)
    summary = _equity_metrics(closed, INITIAL_CAPITAL)
    summary.update({
        "universe": universe, "symbols": len(data),
        "atr_multiple": atr_multiple, "reward_ratio": rr,
        "filter_mode": FILTER_MODE, "risk_percent_base": RISK_PERCENT,
        "open_trades_at_end": len(system.execution.open_positions),
        "max_open_per_symbol": 1,
        **cycle_counts,
        "max_recovery_entries_observed": max((x["recovery_entries"] for x in completed_cycles), default=0),
        "max_cycle_loss": max((x["losses"] for x in completed_cycles), default=0.0),
        "max_cycle_net_exposure": max((x["base_risk_amount"] + x["losses"] for x in completed_cycles), default=0.0),
    })
    return summary, trades, pd.DataFrame(completed_cycles), pd.DataFrame(skips)


def run(data_root="Backtest/data/okx", output_dir="Backtest/results/run48"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    try:
        monitor.set_stage("dataset_loading", "loading 76-symbol cached OKX dataset")
        data = load_okx_data(data_root, expected_symbols=76)
        monitor.set_stage("contract_specs", "loading actual OKX linear SWAP lot/minimum/tick specifications")
        specs = OKXContractSpecs()
        specs.load()
        monitor.set_stage("signal_preparation", "preparing causal OB signals")
        signals, obs = prepare_signals(data, progress_monitor=monitor)
        signals = _attach_atr(signals, data)
        monitor.set_stage("btc_trends", "building causal BTC 1d/4h/1h trend frames")
        btc_frames = build_btc_trend_frames(data)
        cutoff = int(pd.Timestamp(CUTOFF_UTC).timestamp() * 1000)
        test_data, test_signals = _split_oos(data, signals, cutoff)
        sets = {
            "all_76": sorted(test_data),
            "selected_26_training_strong": [s for s in SELECTED_26 if s in test_data],
        }
        if len(sets["selected_26_training_strong"]) != 26:
            raise ValueError("Run #47 selected 26 symbol set does not match the cached dataset")
        configs = []
        for universe in sets:
            for atr in ATR_MULTIPLIERS:
                for rr in REWARD_RATIOS:
                    configs.append((universe, atr, rr, None))
                    for objective in OBJECTIVES:
                        for sizing in SIZING_METHODS:
                            for limit in RECOVERY_LIMITS:
                                configs.append((universe, atr, rr, {
                                    "objective": objective, "sizing_method": sizing,
                                    "max_recovery_trades": limit,
                                }))
        summaries = []
        started = time.monotonic()
        total_events = len(configs)
        for i, (universe, atr, rr, cycle_cfg) in enumerate(configs, 1):
            label = f"{universe}_atr{atr:g}_rr{rr:g}"
            if cycle_cfg:
                cap = "unlimited" if cycle_cfg["max_recovery_trades"] is None else str(cycle_cfg["max_recovery_trades"])
                label += f"_{cycle_cfg['objective']}_{cycle_cfg['sizing_method']}_cap{cap}"
            selected = sets[universe]
            subset_data = {s: test_data[s] for s in selected}
            subset_signals = {s: test_signals[s] for s in selected}
            def report(done, total, ts):
                percent = int(done*100/total) if total else 100
                month = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m")
                print(f"[PROGRESS] scenario {i}/{total_events} {label} | {percent}% | {done}/{total} events | month={month}", flush=True)
            summary, trades, cycles, skips = _simulate(
                subset_data, subset_signals, obs, specs, btc_frames,
                atr, rr, universe, cycle_cfg, report,
            )
            summary["scenario"] = label
            summary["recovery_objective"] = cycle_cfg["objective"] if cycle_cfg else "none"
            summary["recovery_sizing"] = cycle_cfg["sizing_method"] if cycle_cfg else "none"
            summary["max_recovery_trades"] = cycle_cfg["max_recovery_trades"] if cycle_cfg else 0
            summaries.append(summary)
            trades.to_csv(out / f"trades_{label}.csv", index=False)
            cycles.to_csv(out / f"cycles_{label}.csv", index=False)
            skips.to_csv(out / f"skips_{label}.csv", index=False)
            elapsed = time.monotonic() - started
            print(f"[SCENARIO COMPLETE] {i}/{total_events} {label} | trades={len(trades)} | elapsed={elapsed/60:.1f}m", flush=True)
        frame = pd.DataFrame(summaries)
        frame.to_csv(out / "run48_summary.csv", index=False)
        (out / "README.md").write_text(
            "# Run #48 — per-symbol one-position and recovery cycles\n\n"
            f"- OOS cutoff UTC: {CUTOFF_UTC}\n- BTC filter: {FILTER_MODE}\n"
            "- OOS data only; selected set comes from Run #47 training-strong classification.\n"
            "- Base risk: 0.5% of current equity; actual OKX contract sizing.\n"
            "- ATR stop multiples: 1.0, 1.5, 2.0; RR: 1.5, 2.0, 2.5.\n"
            "- Fees, slippage and funding: zero.\n"
            "- Portfolio caps retained: 6 total, 3 per direction; max one open position per symbol.\n"
            "- Recovery caps mean additional recovery entries after the initial trade; unlimited has no count cap.\n"
            "- Exact recovery aims to recover prior losses plus the initial trade TP target; breakeven aims only to recover prior losses.\n"
            "- Double-size is tested independently and is not guaranteed to recover losses.\n"
            "- A recovery risk request at or above current equity is rejected and the cycle is marked failed; OKX min/lot rules still apply.\n"
            "- Unresolved cycles at the end remain reported, not assumed recovered.\n",
            encoding="utf-8",
        )
        print(frame.to_string(index=False), flush=True)
        print(f"[COMPLETE] {len(configs)} scenarios saved to {out}", flush=True)
    finally:
        monitor.stop()


if __name__ == "__main__":
    run()
