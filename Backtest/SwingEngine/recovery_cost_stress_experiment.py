"""Cost, funding and compounding stress audit for Run #60's four leading scenarios."""
from __future__ import annotations

import json
import os
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
# Six base-risk positions at 0.5% each define the portfolio-wide open-risk ceiling.
MAX_PORTFOLIO_OPEN_RISK_PERCENT = 3.0
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


def _cycle_close_status(recovery_entries, cycle_net_pnl, base_target_profit):
    """Classify a closed cycle using its realized net PnL, not its exit label."""
    if not recovery_entries:
        return "BASE_TP"
    if cycle_net_pnl >= base_target_profit - 1e-9:
        return "RECOVERED_TARGET"
    if cycle_net_pnl >= -1e-9:
        return "RECOVERED_BREAKEVEN"
    return "CLOSED_SHORTFALL"


def _available_portfolio_risk(system):
    """Remaining cash risk before reaching the portfolio-wide open-risk cap."""
    open_risk = sum(
        float(position.risk_amount)
        for position in system.execution.open_positions.values()
    )
    risk_ceiling = (
        float(system.execution.equity)
        * MAX_PORTFOLIO_OPEN_RISK_PERCENT / 100.0
    )
    return max(0.0, risk_ceiling - open_risk), open_risk


def _update_open_risk_peaks(system, max_amount, max_percent):
    """Update peak open risk from one consistent portfolio/equity snapshot."""
    open_risk = sum(
        float(position.risk_amount)
        for position in system.execution.open_positions.values()
    )
    max_amount = max(float(max_amount), open_risk)
    equity = float(system.execution.equity)
    if equity > 0:
        max_percent = max(float(max_percent), open_risk / equity * 100.0)
    return max_amount, max_percent


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


def recovery_risk_amount(losses, base_risk, base_target_profit, rr, objective, sizing_method, recovery_entries, *, stop_distance=None, target_distance=None, base_quantity=None):
    """Return requested risk dollars for the next recovery entry.

    Exact recovery sizes risk from the cycle's outstanding cash target.
    Double-size sizes the next position at 2x, 4x, 8x... the initial
    position's base quantity; risk dollars therefore also depend on the
    current signal's stop distance.
    """
    if sizing_method == "exact_recovery":
        target = base_target_profit if objective == "initial_target_profit" else 0.0
        required_profit = float(losses) + float(target)
        if target_distance is not None:
            if stop_distance is None or float(stop_distance) <= 0 or float(target_distance) <= 0:
                raise ValueError("exact recovery requires positive rounded stop and target distances")
            # Use the actual tick-rounded reward/risk ratio so the required
            # contract count reaches the cash target after OKX price rounding.
            actual_rr = float(target_distance) / float(stop_distance)
            return required_profit / actual_rr
        return required_profit / float(rr)
    if sizing_method == "double_size":
        if stop_distance is None or base_quantity is None:
            raise ValueError("double_size requires stop_distance and base_quantity")
        if float(stop_distance) <= 0 or float(base_quantity) <= 0:
            raise ValueError("stop_distance and base_quantity must be positive")
        target_quantity = float(base_quantity) * (2 ** (int(recovery_entries) + 1))
        return target_quantity * float(stop_distance)
    raise ValueError("Unknown recovery sizing method")


def _simulate(data, signals, obs, specs, btc_frames, atr_multiple, rr,
              universe, cycle_config=None, progress=None, fee_rate=0.0,
              slippage_rate=0.0, funding_rate_8h=0.0):
    system = TradingSystem(
        initial_capital=INITIAL_CAPITAL, fee_rate=fee_rate, slippage_rate=slippage_rate,
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
                    "cycles_breakeven_met": 0, "cycles_initial_target_met": 0,
                    "cycles_recovered_breakeven": 0,
                    "cycles_recovered_target": 0,
                    "cycles_closed_shortfall": 0,
                    "cycles_failed_limit": 0, "cycles_unresolved": 0,
                    "cycles_sizing_failed": 0,
                    "cycles_portfolio_risk_failed": 0}
    max_open_risk_amount = 0.0
    max_open_risk_percent = 0.0
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
            # Conservative funding stress: charge the long/short position as a payer
            # at every crossed 8-hour settlement, using entry notional.
            funding_periods = max(0, ts // 28_800_000 - pos.opened_at // 28_800_000)
            funding_cost = (pos.entry_price * pos.quantity * funding_rate_8h
                            * funding_periods)
            if funding_cost:
                result.pnl -= funding_cost
                system.execution.equity -= funding_cost
                result.result = ("WIN" if result.pnl > 0 else
                                 "LOSS" if result.pnl < 0 else "BREAKEVEN")
            meta = active.pop(symbol, {})
            row = {
                **meta, "symbol": result.symbol, "direction": result.direction,
                "entry_price": result.entry_price, "exit_price": result.exit_price,
                "quantity": result.quantity, "contract_count": result.contract_count,
                "risk_percent": result.risk_percent, "risk_amount": pos.risk_amount,
                "pnl": result.pnl, "funding_cost": funding_cost, "result": result.result,
                "exit_reason": result.exit_reason, "opened_at": result.opened_at,
                "closed_at": result.closed_at,
            }
            closed.append(row)
            # A losing close can reduce equity while other positions remain open.
            max_open_risk_amount, max_open_risk_percent = _update_open_risk_peaks(
                system, max_open_risk_amount, max_open_risk_percent
            )
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
                            state["cycle_closed_at"] = ts
                            state["cycle_net_pnl"] = state["cycle_realized_pnl"]
                            completed_cycles.append(dict(state))
                            cycles.pop(symbol, None)
                            cycle_counts["cycles_failed_limit"] += 1
                    elif result.exit_reason == "TP":
                        state["cycle_closed_at"] = ts
                        state["cycle_net_pnl"] = state["cycle_realized_pnl"]
                        state["recovered_to_breakeven"] = state["cycle_net_pnl"] >= -1e-9
                        state["achieved_initial_target"] = state["cycle_net_pnl"] >= state["base_target_profit"] - 1e-9
                        state["status"] = _cycle_close_status(
                            state["recovery_entries"],
                            state["cycle_net_pnl"],
                            state["base_target_profit"],
                        )
                        completed_cycles.append(dict(state))
                        cycles.pop(symbol, None)
                        if state["recovery_entries"]:
                            cycle_counts["cycles_recovered"] += 1
                        if state["recovered_to_breakeven"]:
                            cycle_counts["cycles_breakeven_met"] += 1
                        if state["achieved_initial_target"]:
                            cycle_counts["cycles_initial_target_met"] += 1
                        if state["recovery_entries"]:
                            if state["status"] == "RECOVERED_TARGET":
                                cycle_counts["cycles_recovered_target"] += 1
                            elif state["status"] == "RECOVERED_BREAKEVEN":
                                cycle_counts["cycles_recovered_breakeven"] += 1
                            else:
                                cycle_counts["cycles_closed_shortfall"] += 1

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
            available_risk, open_risk = _available_portfolio_risk(system)
            if available_risk <= 0:
                if is_recovery:
                    state = cycles.pop(symbol)
                    state["status"] = "FAILED_PORTFOLIO_RISK_CAP"
                    state["cycle_closed_at"] = ts
                    state["cycle_net_pnl"] = state["cycle_realized_pnl"]
                    completed_cycles.append(dict(state))
                    cycle_counts["cycles_portfolio_risk_failed"] += 1
                skips.append({"symbol": symbol, "signal_ts": ts, "reason": "portfolio_open_risk_cap"})
                continue
            if is_recovery:
                state = cycles[symbol]
                entry_price = float(signal["entry_price"])
                spec = system.execution.contract_specs[symbol]
                rounded_stop, rounded_target = spec.round_exit_prices(
                    direction=direction,
                    stop_loss=stop,
                    take_profit=target,
                )
                actual_stop_distance = abs(entry_price - float(rounded_stop))
                actual_target_distance = abs(float(rounded_target) - entry_price)
                risk_amount = recovery_risk_amount(
                    state["losses"], state["base_risk_amount"],
                    state["base_target_profit"], rr,
                    cycle_config["objective"], cycle_config["sizing_method"],
                    state["recovery_entries"],
                    stop_distance=actual_stop_distance,
                    target_distance=actual_target_distance,
                    base_quantity=state["base_quantity"],
                )
                if risk_amount <= 0 or risk_amount > available_risk:
                    state["status"] = "FAILED_PORTFOLIO_RISK_CAP"
                    state["cycle_closed_at"] = ts
                    state["cycle_net_pnl"] = state["cycle_realized_pnl"]
                    completed_cycles.append(dict(state))
                    cycles.pop(symbol, None)
                    cycle_counts["cycles_portfolio_risk_failed"] += 1
                    skips.append({"symbol": symbol, "signal_ts": ts, "reason": "recovery_exceeds_available_portfolio_risk"})
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
                    round_contracts_up=is_recovery,
                    max_risk_amount=available_risk,
                )
                if pos is None:
                    continue
                max_open_risk_amount, max_open_risk_percent = _update_open_risk_peaks(
                    system, max_open_risk_amount, max_open_risk_percent
                )
                if cycle_config and not is_recovery:
                    state = {
                        "symbol": symbol, "cycle_id": f"{symbol}-{cycle_serial[symbol]}",
                        "started_at": ts, "cycle_closed_at": None,
                        "base_risk_amount": float(pos.risk_amount),
                        "base_quantity": float(pos.quantity),
                        "base_contract_count": float(pos.contract_count or 0.0),
                        "base_target_profit": max(0.0, abs(float(pos.take_profit) - float(pos.entry_price)) * float(pos.quantity)\n                            - (float(pos.entry_price) + float(pos.take_profit)) * float(pos.quantity) * (fee_rate + slippage_rate)),
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
                if is_recovery and "risk budget" in str(exc).lower():
                    state = cycles.pop(symbol, None)
                    if state:
                        state["status"] = "FAILED_PORTFOLIO_RISK_CAP"
                        state["cycle_closed_at"] = ts
                        state["cycle_net_pnl"] = state["cycle_realized_pnl"]
                        completed_cycles.append(dict(state))
                        cycle_counts["cycles_portfolio_risk_failed"] += 1
                skips.append({"symbol": symbol, "signal_ts": ts, "reason": "contract_or_size", "error": str(exc)})
        if progress and (idx % max(1, len(times)//20) == 0 or idx == len(times)):
            progress(idx, len(times), ts)
    for state in cycles.values():
        state["status"] = "UNRESOLVED_AT_END"
        state["cycle_net_pnl"] = state["cycle_realized_pnl"]
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
        "max_open_risk_amount": max_open_risk_amount,
        "max_open_risk_percent": max_open_risk_percent,
        "max_portfolio_open_risk_percent_limit": MAX_PORTFOLIO_OPEN_RISK_PERCENT,
    })
    return summary, trades, pd.DataFrame(completed_cycles), pd.DataFrame(skips)


def run(data_root="Backtest/data/okx", output_dir=None):
    output_dir = output_dir or os.environ.get("RECOVERY_OUTPUT_DIR", "Backtest/results/recovery_cost_stress")
    fee_rate = float(os.environ.get("RECOVERY_FEE_RATE", "0.0005"))
    slippage_rate = float(os.environ.get("RECOVERY_SLIPPAGE_RATE", "0.0002"))
    funding_rate_8h = float(os.environ.get("RECOVERY_FUNDING_RATE_8H", "0.0001"))
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
                    for limit in RECOVERY_LIMITS:
                        for objective in OBJECTIVES:
                            configs.append((universe, atr, rr, {
                                "objective": objective, "sizing_method": "exact_recovery",
                                "max_recovery_trades": limit,
                            }))
                        configs.append((universe, atr, rr, {
                            "objective": "both_evaluated", "sizing_method": "double_size",
                            "max_recovery_trades": limit,
                        }))
        if os.environ.get("RECOVERY_TOP_SCENARIOS_ONLY", "").lower() in ("1", "true", "yes"):
            # Recheck the two leading recovery configurations against their
            # matching no-recovery baselines using the corrected risk reporting.
            configs = [
                ("all_76", 2.0, 2.0, None),
                ("all_76", 2.0, 2.0, {
                    "objective": "initial_target_profit",
                    "sizing_method": "exact_recovery",
                    "max_recovery_trades": 5,
                }),
                ("selected_26_training_strong", 2.0, 2.0, None),
                ("selected_26_training_strong", 2.0, 2.0, {
                    "objective": "breakeven",
                    "sizing_method": "exact_recovery",
                    "max_recovery_trades": None,
                }),
            ]
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
                fee_rate=fee_rate, slippage_rate=slippage_rate,
                funding_rate_8h=funding_rate_8h,
            )
            summary["scenario"] = label
            summary["recovery_objective"] = cycle_cfg["objective"] if cycle_cfg else "none"
            summary["recovery_sizing"] = cycle_cfg["sizing_method"] if cycle_cfg else "none"
            summary["max_recovery_trades"] = cycle_cfg["max_recovery_trades"] if cycle_cfg else 0
            summary["fee_rate_per_side"] = fee_rate
            summary["slippage_rate_per_side"] = slippage_rate
            summary["funding_rate_per_8h_assumed_paid"] = funding_rate_8h
            summary["max_cycle_loss_pct_initial"] = summary["max_cycle_loss"] / INITIAL_CAPITAL * 100.0
            summaries.append(summary)
            suffix = f"_fee{fee_rate:g}_slip{slippage_rate:g}_fund{funding_rate_8h:g}"
            trades.to_csv(out / f"trades_{label}{suffix}.csv", index=False)
            cycles.to_csv(out / f"cycles_{label}{suffix}.csv", index=False)
            skips.to_csv(out / f"skips_{label}{suffix}.csv", index=False)
            elapsed = time.monotonic() - started
            print(f"[SCENARIO COMPLETE] {i}/{total_events} {label} | trades={len(trades)} | elapsed={elapsed/60:.1f}m", flush=True)
        frame = pd.DataFrame(summaries)
        frame.to_csv(out / "recovery_cost_stress_summary.csv", index=False)
        (out / "README.md").write_text(
            "# Recovery cost, funding and compounding stress audit\n\n"
            f"- OOS cutoff UTC: {CUTOFF_UTC}\n- BTC filter: {FILTER_MODE}\n"
            "- OOS data only; selected set comes from Run #47 training-strong classification.\n"
            "- Base risk: 0.5% of current equity; actual OKX contract sizing.\n"
            "- ATR stop multiples: 1.0, 1.5, 2.0; RR: 1.5, 2.0, 2.5.\n"
            f"- Fee per side: {fee_rate:.5%}; slippage per side: {slippage_rate:.5%}.\n"
            f"- Funding stress: always-paid {funding_rate_8h:.5%} per crossed 8-hour UTC settlement on entry notional.\n"
            "- Portfolio caps retained: 6 total, 3 per direction; max one open position per symbol.\n"
            f"- Aggregate open-risk ceiling: {MAX_PORTFOLIO_OPEN_RISK_PERCENT:.2f}% of current equity.\n"
            "- Recovery caps mean additional recovery entries after the initial trade; unlimited has no count cap.\n"
            "- Exact recovery aims to recover prior losses plus the initial trade TP target; breakeven aims only to recover prior losses.\n"
            "- Recovery lot rounding may round upward to the next lotSz/minSz step, but never beyond remaining aggregate open-risk capacity.\n"
            "- Closed recovery cycles are classified as RECOVERED_TARGET, RECOVERED_BREAKEVEN or CLOSED_SHORTFALL using realized cycle PnL.\n"
            "- Double-size recovery doubles actual base quantity (2x, 4x, 8x...) from the initial trade; requested risk varies with the new stop distance and OKX lot rounding applies.\n"
            "- Double-size runs are unique (not duplicated for objectives); every completed cycle is scored for both breakeven and the initial trade target profit.\n"
            "- A recovery request that exceeds remaining aggregate open-risk capacity is rejected and the cycle is marked failed; OKX min/lot rules still apply.\n"
            "- Unresolved cycles at the end remain reported, not assumed recovered.\n",
            encoding="utf-8",
        )
        print(frame.to_string(index=False), flush=True)
        print(f"[COMPLETE] {len(configs)} scenarios saved to {out}", flush=True)
    finally:
        monitor.stop()


if __name__ == "__main__":
    run()
