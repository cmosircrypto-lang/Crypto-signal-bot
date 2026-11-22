"""Walk-forward historical evaluator for causal strategy-family genomes.

Only 15m/1h/4h closed OKX candles are used. The final OOS tail is reserved and
never evaluated here. Trades are simulated at the next 15m open after a signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import pandas as pd

from .family_signals import evaluate_genome_signals
from .genome import StrategyGenome

DAY_MS = 86_400_000
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": DAY_MS}
SYMBOL_TIMEFRAMES = ("15m", "1h", "4h")


@dataclass(frozen=True)
class FoldResult:
    start_ms: int
    end_ms: int
    trade_count: int
    net_pnl: float
    average_r: float
    profit_factor: float
    max_drawdown_pct: float
    gross_profit: float
    gross_loss: float


def _frames_of(value: Any) -> Mapping[str, pd.DataFrame]:
    frames = value.frames if hasattr(value, "frames") else value
    if not isinstance(frames, Mapping):
        raise TypeError("symbol data must expose a timeframe-to-frame mapping")
    return frames


def _normalize(frame: Any) -> pd.DataFrame:
    df = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"OHLCV frame missing columns: {sorted(required-set(df.columns))}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def build_complete_btc_daily(hourly: Any) -> pd.DataFrame:
    """Aggregate only complete, contiguous UTC days from closed hourly candles."""
    frame = _normalize(hourly)
    rows = []
    for day, group in frame.groupby((frame["timestamp"].astype("int64") // DAY_MS) * DAY_MS, sort=True):
        group = group.sort_values("timestamp")
        stamps = group["timestamp"].to_numpy(dtype="int64")
        expected = int(day) + pd.RangeIndex(24).to_numpy(dtype="int64") * TF_MS["1h"]
        if len(stamps) != 24 or not (stamps == expected).all():
            continue
        rows.append({
            "timestamp": int(day), "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()), "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]), "volume": float(group["volume"].sum()),
        })
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _closed_slice(frame: pd.DataFrame, timeframe: str, decision_ms: int, limit: int = 500) -> pd.DataFrame:
    """Binary-search the last fully closed bar; avoid rescanning the whole history."""
    duration = TF_MS[timeframe]
    close_times = frame["timestamp"].to_numpy(dtype="int64") + duration
    end = int(close_times.searchsorted(int(decision_ms), side="right"))
    start = max(0, end-limit)
    return frame.iloc[start:end].reset_index(drop=True)


def _find_btc_frames(all_data: Mapping[str, Any] | None) -> dict[str, pd.DataFrame] | None:
    if not all_data:
        return None
    names = [name for name in all_data if "BTC" in name.upper() and "USDT" in name.upper() and "SWAP" in name.upper()]
    if len(names) != 1:
        return None
    frames = _frames_of(all_data[names[0]])
    if not all(tf in frames for tf in ("1h", "4h")):
        return None
    hourly, four_hour = _normalize(frames["1h"]), _normalize(frames["4h"])
    daily = build_complete_btc_daily(hourly)
    if daily.empty:
        return None
    return {"1h": hourly, "4h": four_hour, "1d": daily}


def _fold_ranges(length: int, *, oos_fraction: float, warmup_fraction: float, fold_count: int):
    if not 0.0 < oos_fraction < 0.5 or not 0.0 < warmup_fraction < 1.0:
        raise ValueError("invalid walk-forward fractions")
    development_end = int(length * (1.0 - oos_fraction))
    first = int(length * warmup_fraction)
    if development_end <= first or fold_count < 1:
        raise ValueError("not enough candles for requested walk-forward folds")
    edges = [first + (development_end - first) * i // fold_count for i in range(fold_count + 1)]
    return [(edges[i], edges[i + 1]) for i in range(fold_count) if edges[i + 1] > edges[i]], development_end


def _trade_exit(direction: str, candle: Mapping[str, Any], position: dict[str, float]):
    high, low = float(candle["high"]), float(candle["low"])
    if direction == "LONG":
        sl_hit, tp_hit = low <= position["stop"], high >= position["target"]
    else:
        sl_hit, tp_hit = high >= position["stop"], low <= position["target"]
    if sl_hit:
        return position["stop"], "SL"
    if tp_hit:
        return position["target"], "TP"
    return None


def _simulate_fold(
    symbol: str, raw_frames: Mapping[str, pd.DataFrame], btc_frames: Mapping[str, pd.DataFrame] | None,
    genome: StrategyGenome, direction: str, start_ms: int, end_ms: int, *,
    initial_equity: float, swing_signal_provider=None,
) -> FoldResult:
    fifteen = raw_frames["15m"]
    hourly = raw_frames["1h"]
    four_hour = raw_frames["4h"]
    # Use the finest timeframe required by the selected families. This preserves
    # 15m confirmations for EMA/Swing, while avoiding redundant 15m recalculation
    # for strategies whose decisions are strictly 1h or 4h based.
    families = set(genome.signal_families)
    if families & {"EMA_CROSS_PRO", "SWING_OB"}:
        event_tf = "15m"
    elif families == {"SMC_LIQUIDITY_SWEEP"}:
        event_tf = "4h"
    else:
        event_tf = "1h"
    decision_times = raw_frames[event_tf]["timestamp"].to_numpy(dtype="int64") + TF_MS[event_tf]
    event_signals: dict[int, dict[str, Any]] = {}
    seen_signatures: set[tuple] = set()

    for decision_ms in decision_times:
        decision_ms = int(decision_ms)
        if decision_ms < start_ms or decision_ms >= end_ms:
            continue
        frames = {
            "15m": _closed_slice(fifteen, "15m", decision_ms),
            "1h": _closed_slice(hourly, "1h", decision_ms),
            "4h": _closed_slice(four_hour, "4h", decision_ms),
        }
        if any(len(frames[tf]) < 40 for tf in ("15m", "1h")):
            continue
        # Do not let a data gap turn an old candle into a fresh signal.
        latest_1h_close = int(frames["1h"].iloc[-1]["timestamp"]) + TF_MS["1h"]
        if latest_1h_close > decision_ms or decision_ms-latest_1h_close >= TF_MS["1h"]:
            continue
        if int(frames["15m"].iloc[-1]["timestamp"]) + TF_MS["15m"] != decision_ms:
            continue
        needs_four_hour = bool(
            set(genome.signal_families) & {
                "EMA_CROSS_PRO", "BTC_ALIGNED_PULLBACK", "SMC_LIQUIDITY_SWEEP"
            }
        )
        if needs_four_hour and (
            frames["4h"].empty or
            decision_ms - (int(frames["4h"].iloc[-1]["timestamp"]) + TF_MS["4h"]) >= TF_MS["4h"]
        ):
            continue
        if btc_frames:
            btc = {
                "1h": _closed_slice(btc_frames["1h"], "1h", decision_ms),
                "4h": _closed_slice(btc_frames["4h"], "4h", decision_ms),
                "1d": _closed_slice(btc_frames["1d"], "1d", decision_ms),
            }
            if "BTC_ALIGNED_PULLBACK" in genome.signal_families:
                if (not btc["1h"].empty and
                    int(btc["1h"].iloc[-1]["timestamp"]) + TF_MS["1h"] <= decision_ms and
                    decision_ms-(int(btc["1h"].iloc[-1]["timestamp"])+TF_MS["1h"]) < TF_MS["1h"] and
                    not btc["4h"].empty and not btc["1d"].empty and
                    decision_ms-(int(btc["4h"].iloc[-1]["timestamp"])+TF_MS["4h"]) < TF_MS["4h"] and
                    decision_ms-(int(btc["1d"].iloc[-1]["timestamp"])+TF_MS["1d"]) < TF_MS["1d"]):
                    pass
                else:
                    btc = None
        else:
            btc = None
        swing_signal = None
        if callable(swing_signal_provider):
            swing_signal = swing_signal_provider(symbol, direction, decision_ms, frames)
        signal = evaluate_genome_signals(
            frames, genome, direction, btc_frames=btc, swing_signal=swing_signal,
        )
        if not signal:
            continue
        signature = (
            signal.get("method", ""), tuple(sorted(signal.get("family_signal_timestamps", {}).items())),
            tuple(signal.get("families_agreed", ())),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        event_signals[decision_ms] = signal

    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    r_values: list[float] = []
    position = None
    trades = 0
    last_candle = None

    window = fifteen.loc[
        (fifteen["timestamp"] >= start_ms) & (fifteen["timestamp"] < end_ms)
    ]
    for row in window.to_dict("records"):
        ts = int(row["timestamp"])
        last_candle = row
        if position is not None and ts > position["opened_at"]:
            exited = _trade_exit(direction, row, position)
            if exited:
                exit_price, _reason = exited
                signed = 1.0 if direction == "LONG" else -1.0
                pnl = (exit_price - position["entry"]) * signed * position["quantity"]
                equity += pnl
                gross_profit += max(0.0, pnl)
                gross_loss += max(0.0, -pnl)
                r_values.append(pnl / position["risk_amount"] if position["risk_amount"] else 0.0)
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak-equity)/peak*100 if peak > 0 else 100.0)
                trades += 1
                position = None

        signal = event_signals.get(ts)
        if position is None and signal is not None and equity > 0:
            source_entry = float(signal["entry"])
            risk_distance = float(signal["risk_distance"])
            reward_distance = float(signal["reward_distance"])
            entry = float(row["open"])
            if risk_distance > 0 and reward_distance > 0 and entry > 0:
                if direction == "LONG":
                    stop, target = entry-risk_distance, entry+reward_distance
                else:
                    stop, target = entry+risk_distance, entry-reward_distance
                if stop > 0 and target > 0:
                    risk_amount = equity * genome.risk_percent / 100.0
                    quantity = risk_amount / risk_distance
                    position = {
                        "entry": entry, "stop": stop, "target": target,
                        "risk_amount": risk_amount, "quantity": quantity,
                        "opened_at": ts, "source_entry": source_entry,
                    }
                    # Entry is at this 15m open; same-candle collision is SL-first.
                    exited = _trade_exit(direction, row, position)
                    if exited:
                        exit_price, _reason = exited
                        signed = 1.0 if direction == "LONG" else -1.0
                        pnl = (exit_price-entry)*signed*quantity
                        equity += pnl
                        gross_profit += max(0.0,pnl)
                        gross_loss += max(0.0,-pnl)
                        r_values.append(pnl/risk_amount if risk_amount else 0.0)
                        peak=max(peak,equity)
                        max_dd=max(max_dd,(peak-equity)/peak*100 if peak>0 else 100.0)
                        trades += 1
                        position=None

    if position is not None and last_candle is not None:
        exit_price=float(last_candle["close"])
        signed=1.0 if direction=="LONG" else -1.0
        pnl=(exit_price-position["entry"])*signed*position["quantity"]
        equity += pnl
        gross_profit += max(0.0,pnl)
        gross_loss += max(0.0,-pnl)
        r_values.append(pnl/position["risk_amount"] if position["risk_amount"] else 0.0)
        peak=max(peak,equity)
        max_dd=max(max_dd,(peak-equity)/peak*100 if peak>0 else 100.0)
        trades += 1

    pf = gross_profit/gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
    return FoldResult(
        start_ms=int(start_ms), end_ms=int(end_ms), trade_count=trades,
        net_pnl=equity-initial_equity,
        average_r=sum(r_values)/len(r_values) if r_values else 0.0,
        profit_factor=min(float(pf), 10.0), max_drawdown_pct=max_dd,
        gross_profit=gross_profit, gross_loss=gross_loss,
    )


def evaluate_candidate_walk_forward(
    all_symbol_data: Mapping[str, Any], symbol: str, genome: StrategyGenome, direction: str, *,
    initial_equity: float = 100.0, oos_fraction: float = 0.15,
    warmup_fraction: float = 0.50, fold_count: int = 3,
    swing_signal_provider=None,
) -> dict[str, float | int]:
    """Return historical fitness metrics from rolling validation folds only.

    The final oos_fraction of 15m candles is excluded from all fold ranges.
    Every decision uses only 15m/1h/4h candles whose close time is <= decision time.
    """
    genome.validate()
    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    symbol = symbol.strip().upper()
    if symbol not in all_symbol_data:
        raise KeyError(f"symbol not found in dataset: {symbol}")
    raw = _frames_of(all_symbol_data[symbol])
    if not all(tf in raw for tf in SYMBOL_TIMEFRAMES):
        raise ValueError("symbol data must contain 15m, 1h and 4h frames")
    normalized = {tf: _normalize(raw[tf]) for tf in SYMBOL_TIMEFRAMES}
    folds, development_end = _fold_ranges(
        len(normalized["15m"]), oos_fraction=oos_fraction,
        warmup_fraction=warmup_fraction, fold_count=fold_count,
    )
    btc = _find_btc_frames(all_symbol_data) if "BTC_ALIGNED_PULLBACK" in genome.signal_families else None
    fold_results = []
    for start_idx, end_idx in folds:
        start_ms = int(normalized["15m"].iloc[start_idx]["timestamp"])
        end_ms = int(normalized["15m"].iloc[end_idx]["timestamp"])
        fold_results.append(_simulate_fold(
            symbol, normalized, btc, genome, direction, start_ms, end_ms,
            initial_equity=initial_equity, swing_signal_provider=swing_signal_provider,
        ))
    trades = sum(f.trade_count for f in fold_results)
    total_profit = sum(f.gross_profit for f in fold_results)
    total_loss = sum(f.gross_loss for f in fold_results)
    pf = total_profit/total_loss if total_loss > 0 else (10.0 if total_profit > 0 else 0.0)
    r_weighted = sum(f.average_r*f.trade_count for f in fold_results)/trades if trades else 0.0
    positive_ratio = sum(f.net_pnl > 0 for f in fold_results)/len(fold_results) if fold_results else 0.0
    fold_scores=[]
    for f in fold_results:
        return_component=(math.tanh(f.net_pnl/max(initial_equity,1.0)*5.0)+1.0)/2.0
        pf_component=min(f.profit_factor/2.0,1.0)
        dd_component=max(0.0,1.0-f.max_drawdown_pct/30.0)
        fold_scores.append(0.45*return_component+0.30*pf_component+0.25*dd_component)
    validation_score=sum(fold_scores)/len(fold_scores) if fold_scores else 0.0
    complexity_penalty=min(1.0,0.04*len(genome.signal_families)+
                           (0.03 if genome.family_combination in ("ALL","VOTE") else 0.0))
    return {
        "average_r": float(r_weighted),
        "max_drawdown_pct": float(max((f.max_drawdown_pct for f in fold_results), default=0.0)),
        "profit_factor": float(min(pf,10.0)),
        "positive_window_ratio": float(positive_ratio),
        "trade_count": int(trades),
        "validation_score": float(validation_score),
        "complexity_penalty": float(complexity_penalty),
    }
