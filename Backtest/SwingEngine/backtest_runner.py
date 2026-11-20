"""Chronological, simulation-only 9-combination Swing Engine backtest runner."""
from __future__ import annotations

import argparse
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from data_loader import load_okx_data, dataset_report, TIMEFRAME_MS
from order_block import detect_order_blocks
from signal_engine import generate_entry_signals
from trading_system import TradingSystem
from okx_contract_specs import OKXContractSpecs


INITIAL_CAPITAL = 100.0
RISK_LEVELS = (0.05, 1.0, 1.5)
REWARD_RATIOS = (1.5, 2.0, 2.5)
METHOD_PRIORITY = {
    "DIRECT": 0,
    "ENGULFING": 1,
    "PIN_BAR": 2,
    "M15_STRUCTURE_BREAK": 3,
}


class ProgressMonitor:
    """Threaded heartbeat plus explicit stage progress for long preparation work."""

    def __init__(self, interval_seconds=30):
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage = "initializing"
        self.detail = "starting runner"
        self.last_emit = 0.0
        self.last_activity = self.started
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        self.set_stage("startup", "runner initialized")

    def set_stage(self, stage, detail=""):
        now = time.monotonic()
        with self._lock:
            changed = stage != self.stage
            if changed:
                self.stage_started = now
            self.stage = stage
            self.detail = detail
            self.last_activity = now
            should_emit = changed or now - self.last_emit >= 15
            if should_emit:
                self.last_emit = now
        if should_emit:
            print(f"[STAGE] {stage} | {detail}", flush=True)

    def progress(self, stage, completed, total, detail=""):
        percent = int(completed * 100 / total) if total else 100
        self.set_stage(stage, f"{completed}/{total} ({percent}%) {detail}")

    def _heartbeat(self):
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                stage, detail = self.stage, self.detail
                stage_elapsed = time.monotonic() - self.stage_started
                idle_elapsed = time.monotonic() - self.last_activity
            total_elapsed = time.monotonic() - self.started
            print(
                f"[HEARTBEAT] alive | stage={stage} | {detail} | "
                f"stage_elapsed={stage_elapsed / 60:.1f}m | "
                f"no_substep_update={idle_elapsed / 60:.1f}m | total_elapsed={total_elapsed / 60:.1f}m",
                flush=True,
            )
            if idle_elapsed >= 120:
                print(
                    f"[STALL WARNING] no progress callback for {idle_elapsed / 60:.1f}m "
                    f"| stage={stage} | {detail}",
                    flush=True,
                )

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def prepare_signals(symbol_data, methods=None, zone_type="body",
                    touch_rule="first", require_4h_alignment=False,
                    progress_monitor=None):
    """Precompute causal signals and their source 1h OB rows per symbol."""
    signals = {}
    order_blocks = {}
    symbols = sorted(symbol_data.items())
    for symbol_index, (symbol, item) in enumerate(symbols, start=1):
        if progress_monitor:
            progress_monitor.set_stage("order_block_detection", f"symbol {symbol_index}/{len(symbols)}: {symbol}")
        one_hour = item.frames["1h"]
        fifteen = item.frames["15m"]
        four_hour = item.frames["4h"]
        def ob_progress(done, total):
            if progress_monitor:
                progress_monitor.set_stage("order_block_detection", f"{symbol} 1h candles {done}/{total} ({int(done * 100 / total) if total else 100}%)")
        obs = detect_order_blocks(one_hour, progress_callback=ob_progress if progress_monitor else None)
        order_blocks[symbol] = obs
        if progress_monitor:
            progress_monitor.set_stage("signal_generation", f"{symbol}: analyzing 15m structure and scanning {len(obs)} order blocks")
        def signal_progress(stage, done, total):
            if progress_monitor:
                progress_monitor.set_stage("signal_generation", f"{symbol} {stage} {done}/{total} ({int(done * 100 / total) if total else 100}%)")
        found = generate_entry_signals(
            symbol=symbol,
            candles_15m=fifteen,
            order_blocks_1h=obs,
            candles_4h=four_hour,
            methods=methods or (
                "DIRECT", "ENGULFING", "PIN_BAR",
                "M15_STRUCTURE_BREAK",
            ),
            zone_type=zone_type,
            touch_rule=touch_rule,
            require_4h_alignment=require_4h_alignment,
            progress_callback=signal_progress if progress_monitor else None,
        )
        if not found.empty:
            # Different confirmation methods can fire for one OB. Use only
            # the earliest signal for that OB, with a stable method tie-break.
            found = found.copy()
            found["_priority"] = found["method"].map(METHOD_PRIORITY).fillna(99)
            found = found.sort_values(
                ["signal_ts", "ob_id", "_priority"],
                kind="stable",
            ).drop_duplicates(["ob_id"], keep="first")
            signals[symbol] = found.drop(columns=["_priority"]).reset_index(drop=True)
        else:
            signals[symbol] = found
        if progress_monitor:
            progress_monitor.progress("symbol_preparation", symbol_index, len(symbols), f"completed {symbol}; OBs={len(obs)}; signals={len(signals[symbol])}")
    return signals, order_blocks


def _stop_and_target(signal, ob, rr):
    entry = float(signal["entry_price"])
    direction = str(signal["direction"]).upper()
    if direction == "BUY":
        stop = float(ob["wick_low"])
        distance = entry - stop
        target = entry + distance * rr
    elif direction == "SELL":
        stop = float(ob["wick_high"])
        distance = stop - entry
        target = entry - distance * rr
    else:
        raise ValueError(f"Unsupported signal direction: {direction}")
    if distance <= 0 or target <= 0:
        return None
    return stop, target


def _equity_metrics(trades, initial_capital):
    equity = float(initial_capital)
    peak = equity
    max_drawdown = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    r_values = []
    # Aggregate simultaneous exits before measuring drawdown so arbitrary
    # symbol ordering cannot create a false intratimestamp equity dip.
    pnl_by_time = {}
    for trade in trades:
        pnl = float(trade["pnl"])
        pnl_by_time[int(trade["closed_at"])] = (
            pnl_by_time.get(int(trade["closed_at"]), 0.0) + pnl
        )
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)
        risk = float(trade["risk_amount"])
        if risk > 0:
            r_values.append(pnl / risk)
    for timestamp in sorted(pnl_by_time):
        equity += pnl_by_time[timestamp]
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)
    count = len(trades)
    wins = sum(float(row["pnl"]) > 0 for row in trades)
    losses = sum(float(row["pnl"]) < 0 for row in trades)
    return {
        "initial_capital": float(initial_capital),
        "final_equity": equity,
        "net_pnl": equity - initial_capital,
        "closed_trades": count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wins / count * 100 if count else 0.0,
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        ),
        "max_drawdown_pct": max_drawdown,
        "average_r": sum(r_values) / len(r_values) if r_values else 0.0,
    }


def simulate_scenario(symbol_data, signals_by_symbol, order_blocks_by_symbol,
                      risk_percent, reward_ratio, contract_specs=None,
                      initial_capital=INITIAL_CAPITAL, progress_context=None):
    """Replay all symbols on one global candle-close timeline."""
    system = TradingSystem(
        initial_capital=initial_capital,
        fee_rate=0.0,
        slippage_rate=0.0,
        max_total=6,
        max_buy=3,
        max_sell=3,
        contract_specs=contract_specs,
    )

    candles_at_time = {}
    monitor = progress_context.get("monitor") if progress_context else None
    if monitor:
        monitor.set_stage(
            "scenario_timeline",
            f"scenario {progress_context['scenario_index']}/{progress_context['scenario_total']}: indexing 15m candles",
        )
    for symbol_index, (symbol, item) in enumerate(sorted(symbol_data.items()), start=1):
        if monitor:
            monitor.progress("scenario_timeline", symbol_index, len(symbol_data), f"indexing candles for {symbol}")
        frame = item.frames["15m"]
        for row in frame.to_dict("records"):
            close_time = int(row["ts"]) + TIMEFRAME_MS["15m"]
            candles_at_time.setdefault(close_time, {})[symbol] = row

    if monitor:
        monitor.set_stage("scenario_signals", f"scenario {progress_context['scenario_index']}/{progress_context['scenario_total']}: indexing entry signals")
    signals_at_time = {}
    for symbol, frame in signals_by_symbol.items():
        if frame is None or frame.empty:
            continue
        for row in frame.to_dict("records"):
            signals_at_time.setdefault(int(row["signal_ts"]), []).append((symbol, row))

    # A trade is checked only on candles closing strictly after its entry.
    all_times = sorted(set(candles_at_time) | set(signals_at_time))
    skipped = {"portfolio_limit": 0, "invalid_stop_or_target": 0,
               "position_size_or_contract": 0, "missing_ob": 0}
    closed_records = []
    skipped_records = []
    active_trade_metadata = {}
    system._diagnostic_skips = skipped_records
    started = time.monotonic()
    month_labels = [
        datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m")
        for ts in all_times
    ]
    month_totals = {}
    for label in month_labels:
        month_totals[label] = month_totals.get(label, 0) + 1
    month_done = {}
    report_every = max(1, len(all_times) // 20)
    last_reported = 0

    for event_index, timestamp in enumerate(all_times, start=1):
        current_candles = candles_at_time.get(timestamp, {})
        month = month_labels[event_index - 1]
        month_done[month] = month_done.get(month, 0) + 1

        # Exits first. This releases capacity before same-close entries.
        for symbol in sorted(list(system.execution.open_positions)):
            candle = current_candles.get(symbol)
            if candle is None:
                continue
            position = system.execution.open_positions[symbol]
            if timestamp <= position.opened_at:
                continue
            result = system.check_exit(symbol, candle, timestamp)
            if result is not None:
                trade_metadata = active_trade_metadata.pop(symbol, {})
                closed_records.append({
                    **trade_metadata,
                    "symbol": result.symbol,
                    "direction": result.direction,
                    "entry_price": result.entry_price,
                    "exit_price": result.exit_price,
                    "quantity": result.quantity,
                    "contract_count": result.contract_count,
                    "risk_percent": result.risk_percent,
                    "risk_amount": position.risk_amount,
                    "pnl": result.pnl,
                    "result": result.result,
                    "exit_reason": result.exit_reason,
                    "opened_at": result.opened_at,
                    "closed_at": result.closed_at,
                })

        # Signals are filled at their confirmation candle close.
        entries = sorted(
            signals_at_time.get(timestamp, []),
            key=lambda pair: (pair[0], METHOD_PRIORITY.get(str(pair[1]["method"]), 99),
                              int(pair[1]["ob_id"])),
        )
        for symbol, signal in entries:
            direction = str(signal["direction"]).upper()
            signal_metadata = {
                "method": str(signal.get("method", "UNKNOWN")),
                "ob_id": int(signal.get("ob_id", -1)),
                "ob_ts": signal.get("ob_ts"),
                "bos_ts": signal.get("bos_ts"),
                "signal_ts": int(signal.get("signal_ts", timestamp)),
                "touch_number": int(signal.get("touch_number", 0)),
                "touch_score": float(signal.get("touch_score", 0.0)),
                "htf_trend": str(signal.get("htf_trend", "UNKNOWN")),
            }
            skip_base = {
                "symbol": symbol,
                "direction": direction,
                **signal_metadata,
                "risk_percent": float(risk_percent),
                "reward_ratio": float(reward_ratio),
            }
            if not system.can_search_symbol(symbol):
                skipped_records.append({**skip_base, "skip_reason": "symbol_already_open"})
                continue
            if not system.can_open_trade(symbol, direction):
                skipped["portfolio_limit"] += 1
                skipped_records.append({**skip_base, "skip_reason": "portfolio_limit"})
                continue

            obs = order_blocks_by_symbol.get(symbol)
            ob_id = int(signal["ob_id"])
            if obs is None or ob_id not in obs.index:
                skipped["missing_ob"] += 1
                skipped_records.append({**skip_base, "skip_reason": "missing_ob"})
                continue
            ob_row = obs.loc[ob_id]
            signal_metadata.update({
                "ob_direction": str(ob_row.get("direction", "UNKNOWN")),
                "ob_body_low": float(ob_row.get("body_low", 0.0)),
                "ob_body_high": float(ob_row.get("body_high", 0.0)),
                "ob_wick_low": float(ob_row.get("wick_low", 0.0)),
                "ob_wick_high": float(ob_row.get("wick_high", 0.0)),
            })
            skip_base.update({
                "ob_direction": signal_metadata["ob_direction"],
                "ob_body_low": signal_metadata["ob_body_low"],
                "ob_body_high": signal_metadata["ob_body_high"],
                "ob_wick_low": signal_metadata["ob_wick_low"],
                "ob_wick_high": signal_metadata["ob_wick_high"],
            })
            levels = _stop_and_target(signal, ob_row, reward_ratio)
            if levels is None:
                skipped["invalid_stop_or_target"] += 1
                skipped_records.append({**skip_base, "skip_reason": "invalid_stop_or_target"})
                continue
            stop, target = levels
            try:
                position = system.open_trade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=float(signal["entry_price"]),
                    stop_loss=stop,
                    take_profit=target,
                    risk_percent=float(risk_percent),
                    timestamp=timestamp,
                )
                if position is not None:
                    active_trade_metadata[symbol] = {
                        **signal_metadata,
                        "stop_loss": float(stop),
                        "take_profit": float(target),
                    }
            except (ValueError, KeyError) as exc:
                # Includes exchange minimum-size rejection. Keep a count;
                # do not silently enlarge risk to force a fill.
                skipped["position_size_or_contract"] += 1
                skipped_records.append({
                    **skip_base,
                    "skip_reason": "position_size_or_contract",
                    "error": str(exc),
                })

        local_percent = int(event_index * 100 / len(all_times)) if all_times else 100
        if progress_context:
            completed = (
                (progress_context["scenario_index"] - 1) * len(all_times) + event_index
            )
            total = progress_context["total_events"]
            percent = int(completed * 100 / total) if total else 100
            elapsed = time.monotonic() - progress_context["started"]
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            scenario_text = (
                f"scenario {progress_context['scenario_index']}/"
                f"{progress_context['scenario_total']}"
            )
        else:
            percent = local_percent
            elapsed = time.monotonic() - started
            rate = event_index / elapsed if elapsed > 0 else 0
            eta = (len(all_times) - event_index) / rate if rate > 0 else 0
            scenario_text = "single scenario"
        if percent >= last_reported + 5 or event_index == len(all_times):
            last_reported = percent
            print(
                f"[PROGRESS] {percent:3d}% | {scenario_text} | "
                f"{event_index}/{len(all_times)} events "
                f"| month {month}: {month_done[month]}/{month_totals[month]} "
                f"| elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m",
                flush=True,
            )

    summary = _equity_metrics(closed_records, initial_capital)
    summary.update({
        "risk_percent": float(risk_percent),
        "reward_ratio": float(reward_ratio),
        "open_trades_at_end": len(system.execution.open_positions),
        "skipped_portfolio_limit": skipped["portfolio_limit"],
        "skipped_invalid_stop_or_target": skipped["invalid_stop_or_target"],
        "skipped_position_size_or_contract": skipped["position_size_or_contract"],
        "skipped_missing_ob": skipped["missing_ob"],
    })
    system._active_trade_metadata = active_trade_metadata
    return summary, pd.DataFrame(closed_records), system


def run_matrix(symbol_data, signals_by_symbol, order_blocks_by_symbol,
               contract_specs=None, initial_capital=INITIAL_CAPITAL,
               progress_monitor=None, diagnostics_out=None):
    summaries = []
    trade_frames = {}
    scenario_total = len(RISK_LEVELS) * len(REWARD_RATIOS)
    timeline = set()
    for symbol_index, (symbol, item) in enumerate(sorted(symbol_data.items()), start=1):
        if progress_monitor:
            progress_monitor.progress("timeline_preparation", symbol_index, len(symbol_data), f"indexing 15m close times for {symbol}")
        for ts in item.frames["15m"]["ts"].tolist():
            timeline.add(int(ts) + TIMEFRAME_MS["15m"])
    matrix_started = time.monotonic()
    for risk in RISK_LEVELS:
        for rr in REWARD_RATIOS:
            scenario_index = len(summaries) + 1
            summary, trades, system = simulate_scenario(
                symbol_data,
                signals_by_symbol,
                order_blocks_by_symbol,
                risk_percent=risk,
                reward_ratio=rr,
                contract_specs=contract_specs,
                initial_capital=initial_capital,
                progress_context={
                    "scenario_index": scenario_index,
                    "scenario_total": scenario_total,
                    "total_events": len(timeline) * scenario_total,
                    "started": matrix_started,
                    "monitor": progress_monitor,
                },
            )
            summaries.append(summary)
            scenario_name = f"risk_{risk:g}_rr_{rr:g}"
            trade_frames[scenario_name] = trades
            if diagnostics_out is not None:
                diagnostics_out[scenario_name] = pd.DataFrame(
                    getattr(system, "_diagnostic_skips", []),
                    columns=[
                        "symbol", "direction", "method", "ob_id", "ob_ts",
                        "bos_ts", "signal_ts", "touch_number", "touch_score",
                        "htf_trend", "risk_percent", "reward_ratio",
                        "skip_reason", "error",
                    ],
                )
                diagnostics_out[f"open_{scenario_name}"] = pd.DataFrame([
                    {
                        "symbol": symbol,
                        **active_meta,
                        "entry_price": position.entry_price,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "opened_at": position.opened_at,
                        "risk_amount": position.risk_amount,
                        "quantity": position.quantity,
                        "contract_count": position.contract_count,
                        "risk_percent": position.risk_percent,
                    }
                    for symbol, position in system.execution.open_positions.items()
                    for active_meta in [system._active_trade_metadata.get(symbol, {})]
                ])
            print(
                f"[SCENARIO COMPLETE] risk={risk:g}% RR={rr:g} "
                f"| closed={len(trades)} | open={len(system.execution.open_positions)}",
                flush=True,
            )
            for symbol in sorted(symbol_data):
                closed = int((trades["symbol"] == symbol).sum()) if not trades.empty else 0
                opened = int(symbol in system.execution.open_positions)
                signal_count = len(signals_by_symbol.get(symbol, []))
                print(
                    f"[SYMBOL] {symbol}: signals={signal_count}, "
                    f"closed={closed}, open={opened}",
                    flush=True,
                )
    return pd.DataFrame(summaries), trade_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Backtest/data/okx")
    parser.add_argument("--output-dir", default="Backtest/results/swing_engine")
    parser.add_argument("--expected-symbols", type=int, default=76)
    parser.add_argument(
        "--legacy-sizing", action="store_true",
        help="Use theoretical base-quantity sizing instead of OKX lot/min-size rules.",
    )
    parser.add_argument(
        "--require-4h-alignment", action="store_true",
        help="Require 4h trend alignment for entries.",
    )
    args = parser.parse_args()

    monitor = ProgressMonitor(interval_seconds=30)
    monitor.start()
    try:
        monitor.set_stage("dataset_validation", f"reading and validating files from {args.data_root}")

        def load_progress(timeframe, symbol, index, total):
            if timeframe == "4h":
                monitor.progress("dataset_loading", index, total, f"loaded 15m/1h/4h for {symbol}")

        data = load_okx_data(
            args.data_root,
            expected_symbols=args.expected_symbols,
            progress_callback=load_progress,
        )
        monitor.set_stage("dataset_validation_complete", f"validated {len(data)} symbols across three timeframes")
        print(f"Validated {len(data)} symbols.", flush=True)
        print(dataset_report(data).groupby("timeframe")["candles"].sum().to_string(), flush=True)

        contract_specs = None
        if not args.legacy_sizing:
            monitor.set_stage("contract_specs", "requesting public OKX instrument specifications")
            contract_specs = OKXContractSpecs()
            contract_specs.load()
            print(f"Loaded {len(contract_specs.get_all())} live OKX USDT linear SWAP specifications.", flush=True)

        monitor.set_stage("signal_preparation", f"starting OB and signal preparation for {len(data)} symbols")
        signals, obs = prepare_signals(
            data,
            require_4h_alignment=args.require_4h_alignment,
            progress_monitor=monitor,
        )
        total_signals = sum(len(frame) for frame in signals.values())
        monitor.set_stage("signal_preparation_complete", f"prepared {total_signals} unique-OB entry signals")
        print(f"Prepared {total_signals} unique-OB entry signals.", flush=True)

        monitor.set_stage("backtest_matrix", "starting nine chronological risk/RR combinations")
        diagnostics = {}
        summary, trades = run_matrix(
            data, signals, obs, contract_specs=contract_specs,
            progress_monitor=monitor,
            diagnostics_out=diagnostics,
        )
        monitor.set_stage("saving_results", f"writing summary and nine trade ledgers to {args.output_dir}")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "summary_9_combinations.csv"
        summary.to_csv(summary_path, index=False)
        for index, (name, frame) in enumerate(trades.items(), start=1):
            frame.to_csv(output / f"trades_{name}.csv", index=False)
            monitor.progress("saving_trade_ledgers", index, len(trades), name)
        for name, frame in diagnostics.items():
            prefix = "skipped" if name.startswith("risk_") else "open"
            suffix = name.removeprefix("open_").removeprefix("risk_")
            frame.to_csv(output / f"{prefix}_{suffix}.csv", index=False)
        from analyze_results import generate_diagnostics
        diagnostic_paths = generate_diagnostics(output)
        print(f"Generated {len(diagnostic_paths)} diagnostic reports.", flush=True)
        (output / "summary_9_combinations.json").write_text(
            json.dumps(summary.to_dict("records"), indent=2, allow_nan=True),
            encoding="utf-8",
        )
        print("\\n" + summary.to_string(index=False), flush=True)
        print(f"\\nSaved summary and 9 trade ledgers to {output}", flush=True)
        monitor.set_stage("complete", "all nine scenarios and output files completed")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
