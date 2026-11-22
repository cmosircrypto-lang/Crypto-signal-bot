"""Causal, family-aware signal generation for research-only closed-candle data.

Input frames must already be truncated to fully closed candles at the evaluation
cutoff. This module never fetches data and never reads future rows.
"""
from __future__ import annotations

from typing import Any, Mapping
import math

import pandas as pd

from .genome import StrategyGenome

_REQUIRED = ("open", "high", "low", "close", "volume")


def _frame(frames: Mapping[str, Any], tf: str, minimum: int = 40) -> pd.DataFrame | None:
    value = frames.get(tf)
    if value is None:
        return None
    df = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if not set(_REQUIRED).issubset(df.columns) or len(df) < minimum:
        return None
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="stable")
        if df["timestamp"].duplicated().any():
            df = df.drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]
    out["ema9"] = close.ewm(span=9, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    out["rsi"] = 100 - 100 / (1 + rs)
    out.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    out.loc[(loss == 0) & (gain == 0), "rsi"] = 50.0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up, down = high.diff(), -low.diff()
    plus = up.where((up > down) & (up > 0), 0.0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus = down.where((down > up) & (down > 0), 0.0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    denom = (plus + minus).replace(0, float("nan"))
    dx = 100 * (plus - minus).abs() / denom
    out["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return out


def _valid(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _result(family: str, direction: str, df: pd.DataFrame, sl_distance: float,
            tp_distance: float, quality: float, reason: str = "") -> dict[str, Any] | None:
    last = df.iloc[-1]
    entry = float(last["close"])
    atr = float(last["atr"])
    if not all(_valid(x) for x in (entry, atr, sl_distance, tp_distance, quality)):
        return None
    if entry <= 0 or atr <= 0 or sl_distance <= 0 or tp_distance <= 0:
        return None
    if direction == "LONG":
        stop, target = entry - sl_distance, entry + tp_distance
    else:
        stop, target = entry + sl_distance, entry - tp_distance
    if stop <= 0 or target <= 0:
        return None
    ts = last.get("timestamp", len(df)-1)
    return {
        "family": family, "direction": direction, "signal_timestamp": str(ts),
        "entry": entry, "stop_loss": stop, "take_profit": target,
        "risk_distance": sl_distance, "reward_distance": tp_distance,
        "rr": tp_distance / sl_distance, "setup_quality": max(0.0, min(1.0, quality)),
        "reason": reason, "candle_closed": True,
    }


def _volume_ratio(df: pd.DataFrame, window: int = 20) -> float:
    avg = float(df["volume"].iloc[-window-1:-1].mean())
    return float(df["volume"].iloc[-1]) / avg if avg > 0 else 0.0


def _evaluate_one(family: str, frames: Mapping[str, Any], genome: StrategyGenome,
                  direction: str, btc_frames: Mapping[str, Any] | None = None,
                  swing_signal: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    if family == "SWING_OB":
        if not swing_signal:
            return None
        if str(swing_signal.get("direction", "")).upper() != direction:
            return None
        required = ("entry", "stop_loss", "take_profit")
        if not all(_valid(swing_signal.get(k)) for k in required):
            return None
        entry, stop, target = (float(swing_signal[k]) for k in required)
        risk = abs(entry-stop)
        reward = abs(target-entry)
        if risk <= 0 or reward <= 0 or (entry-stop if direction == "LONG" else stop-entry) <= 0:
            return None
        return {
            "family": family, "direction": direction,
            "signal_timestamp": str(swing_signal.get("signal_timestamp", "")),
            "entry": entry, "stop_loss": stop, "take_profit": target,
            "risk_distance": risk, "reward_distance": reward, "rr": reward/risk,
            "setup_quality": float(swing_signal.get("setup_quality", 0.5)),
            "reason": "SwingEngine closed-candle signal", "candle_closed": bool(swing_signal.get("candle_closed", False)),
        }

    tf = "4h" if family == "SMC_LIQUIDITY_SWEEP" else "1h"
    raw = _frame(frames, tf, 60 if tf == "4h" else 40)
    if raw is None:
        return None
    df = _indicators(raw)
    last, prev = df.iloc[-1], df.iloc[-2]
    close, atr = float(last["close"]), float(last["atr"])
    if not _valid(atr) or atr <= 0:
        return None
    vol = _volume_ratio(df)
    direction_sign = 1 if direction == "LONG" else -1

    if family == "EMA_CROSS_PRO":
        h4raw = _frame(frames, "4h", 60)
        h1raw = _frame(frames, "1h", 60)
        m15raw = _frame(frames, "15m", 40)
        if h4raw is None or h1raw is None or m15raw is None:
            return None
        h4, h1, m15 = _indicators(h4raw), _indicators(h1raw), _indicators(m15raw)
        age = None
        for offset in range(min(genome.ema_cross_lookback, len(h4)-1)):
            i = len(h4)-1-offset
            a, b = h4.iloc[i-1], h4.iloc[i]
            crossed = (a.ema9 <= a.ema21 and b.ema9 > b.ema21) if direction == "LONG" else (a.ema9 >= a.ema21 and b.ema9 < b.ema21)
            if crossed:
                age = offset
                break
        l4, l1, lm = h4.iloc[-1], h1.iloc[-1], m15.iloc[-1]
        aligned = (l4.ema9 > l4.ema21 and l1.ema9 > l1.ema21 and l4.ema21 > l4.ema50) if direction == "LONG" else (l4.ema9 < l4.ema21 and l1.ema9 < l1.ema21 and l4.ema21 < l4.ema50)
        slope = float(l4.ema9 - h4.ema9.iloc[-4])
        slope_ok = slope * direction_sign >= genome.ema_slope_atr * float(l4.atr)
        rsi_ok = (45 <= float(lm.rsi) <= 68) if direction == "LONG" else (32 <= float(lm.rsi) <= 55)
        if age is None or not aligned or not slope_ok or float(l4.adx) < genome.ema_min_adx_4h or float(l1.adx) < genome.ema_min_adx_1h or vol < genome.ema_volume_mult or not rsi_ok:
            return None
        q = 0.35 + 0.2*min(float(l4.adx)/35,1) + 0.2*min(vol/3,1) + 0.15*(1-age/genome.ema_cross_lookback) + 0.1
        return _result(family,direction,h1,genome.ema_sl_atr*float(l1.atr),genome.ema_tp_atr*float(l1.atr),q,"recent 4h EMA cross with 1h confirmation")

    if family == "BTC_ALIGNED_PULLBACK":
        btc_frames = btc_frames or {}
        b1raw, b4raw, bdraw = (_frame(btc_frames, x, 40) for x in ("1h","4h","1d"))
        if b1raw is None or b4raw is None or bdraw is None:
            return None
        b1,b4,bd = (_indicators(x) for x in (b1raw,b4raw,bdraw))
        bias4 = "LONG" if b4.ema9.iloc[-1] > b4.ema21.iloc[-1] else "SHORT"
        biasd = "LONG" if bd.ema9.iloc[-1] > bd.ema21.iloc[-1] else "SHORT"
        l4 = _indicators(_frame(frames,"4h",40)) if _frame(frames,"4h",40) is not None else None
        if bias4 != direction or biasd != direction or l4 is None:
            return None
        if (l4.ema9.iloc[-1] > l4.ema21.iloc[-1]) != (direction == "LONG"):
            return None
        gap = abs(float(l4.ema9.iloc[-1]-l4.ema21.iloc[-1])) / float(l4.atr.iloc[-1])
        if gap < genome.btc_gap_atr:
            return None
        ema = float(last["ema21"])
        touched = (float(last["low"]) <= ema+genome.btc_touch_atr*atr and close > ema and close > float(last["open"])) if direction=="LONG" else (float(last["high"]) >= ema-genome.btc_touch_atr*atr and close < ema and close < float(last["open"]))
        extension = abs(close-ema)/atr
        if not touched or extension > genome.btc_max_extension_atr or vol < genome.btc_pullback_volume:
            return None
        q = 0.45 + 0.2*min(gap/0.8,1) + 0.2*min(vol/2.5,1) + 0.15
        return _result(family,direction,df,genome.btc_pullback_sl_atr*atr,genome.btc_pullback_tp_atr*atr,q,"BTC 1d/4h bias and 1h EMA21 pullback")

    if family == "SMC_LIQUIDITY_SWEEP":
        n = genome.smc_sweep_lookback
        if len(df) < n+25:
            return None
        prior = df.iloc[-n-1:-1]
        if direction == "LONG":
            swept = float(last.low) < float(prior.low.min()) and close > float(prior.low.min()) and close > float(last.open)
            trend = float(last.ema21) > float(last.ema50)
            wick = min(float(last.open),close)-float(last.low)
            stop_dist = max(close-(float(last.low)-genome.smc_sl_buffer_atr*atr),genome.smc_min_sl_atr*atr)
        else:
            swept = float(last.high) > float(prior.high.max()) and close < float(prior.high.max()) and close < float(last.open)
            trend = float(last.ema21) < float(last.ema50)
            wick = float(last.high)-max(float(last.open),close)
            stop_dist = max((float(last.high)+genome.smc_sl_buffer_atr*atr)-close,genome.smc_min_sl_atr*atr)
        target_dist = genome.smc_tp_atr*atr
        if not swept or not trend or stop_dist > genome.smc_max_sl_atr*atr or target_dist/stop_dist < genome.smc_min_rr or vol < genome.smc_min_volume:
            return None
        q = 0.3+0.25*min(abs(float(last.ema21-last.ema50))/atr,1)+0.2*min(vol/2,1)+0.25*min(wick/atr,1)
        return _result(family,direction,df,stop_dist,target_dist,q,"4h liquidity sweep and trend confirmation")

    if family == "MOMENTUM_BREAKOUT":
        n=genome.momentum_lookback
        prior=df.iloc[-n-1:-1]
        buffer=genome.momentum_breakout_buffer_atr*atr
        breakout = close > float(prior.high.max())+buffer if direction=="LONG" else close < float(prior.low.min())-buffer
        if not breakout or float(last.adx)<genome.momentum_min_adx or vol<genome.momentum_volume_mult:
            return None
        q=0.4+0.25*min(float(last.adx)/40,1)+0.2*min(vol/2.5,1)+0.15
        return _result(family,direction,df,1.5*atr,3.0*atr,q,"closed-candle range breakout")

    if family == "MEAN_REVERSION":
        mean=df.close.rolling(20).mean().iloc[-1]
        std=df.close.rolling(20).std().iloc[-1]
        z=(close-float(mean))/float(std) if _valid(std) and float(std)>0 else 0.0
        rsi=float(last.rsi)
        if direction=="LONG":
            valid=z <= -genome.mean_reversion_zscore and rsi <= genome.mean_reversion_rsi_low and close > float(last.open)
        else:
            valid=z >= genome.mean_reversion_zscore and rsi >= genome.mean_reversion_rsi_high and close < float(last.open)
        if not valid:
            return None
        q=0.45+0.25*min(abs(z)/3,1)+0.2*min(abs(rsi-50)/40,1)+0.1
        return _result(family,direction,df,1.2*atr,1.8*atr,q,"RSI and price z-score exhaustion reversal")

    if family == "VOLATILITY_SQUEEZE":
        n=genome.squeeze_bb_period
        mid=df.close.rolling(n).mean()
        sd=df.close.rolling(n).std()
        width=(4*sd/mid.replace(0,float("nan"))).dropna()
        if len(width)<30:
            return None
        percentile=float((width.iloc[:-1] <= width.iloc[-1]).mean()*100)
        prior=df.iloc[-genome.momentum_lookback-1:-1]
        expansion=(close-float(prev.close))*direction_sign/atr
        breakout=close>float(prior.high.max()) if direction=="LONG" else close<float(prior.low.min())
        if percentile>genome.squeeze_width_percentile or expansion<genome.squeeze_expansion_atr or not breakout:
            return None
        q=0.4+0.3*max(0,1-percentile/100)+0.3*min(expansion/3,1)
        return _result(family,direction,df,1.5*atr,3.0*atr,q,"low-width squeeze followed by directional expansion")
    return None


def evaluate_genome_signals(
    frames: Mapping[str, Any], genome: StrategyGenome, direction: str, *,
    btc_frames: Mapping[str, Any] | None = None,
    swing_signal: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate all selected families and apply SINGLE/ANY/ALL/VOTE semantics.

    For ANY, the strongest family is used. ALL requires every family to agree.
    VOTE requires min_family_agreement votes. Hybrid levels use median risk and
    reward distances, avoiding a family-specific price-level cherry-pick.
    """
    genome.validate()
    direction=direction.upper()
    if direction not in ("LONG","SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    signals=[]
    for family in genome.signal_families:
        signal=_evaluate_one(family,frames,genome,direction,btc_frames,swing_signal)
        if signal and signal["candle_closed"]:
            signals.append(signal)
    count=len(signals)
    required = 1 if genome.family_combination in ("SINGLE","ANY") else (
        len(genome.signal_families) if genome.family_combination=="ALL" else genome.min_family_agreement
    )
    if count < required:
        return None
    signals.sort(key=lambda x:(-x["setup_quality"],x["family"]))
    chosen=signals if genome.family_combination in ("ALL","VOTE") else signals[:1]
    if genome.family_combination=="SINGLE" and len(chosen)!=1:
        return None
    if len(chosen)==1:
        winner=dict(chosen[0])
    else:
        risks=sorted(x["risk_distance"] for x in chosen)
        rewards=sorted(x["reward_distance"] for x in chosen)
        risk=risks[len(risks)//2]
        reward=rewards[len(rewards)//2]
        entry=float(chosen[0]["entry"])
        stop=entry-risk if direction=="LONG" else entry+risk
        target=entry+reward if direction=="LONG" else entry-reward
        winner={**chosen[0],"entry":entry,"stop_loss":stop,"take_profit":target,
                "risk_distance":risk,"reward_distance":reward,"rr":reward/risk,
                "setup_quality":sum(x["setup_quality"] for x in chosen)/len(chosen)}
    winner["method"]="+".join(x["family"] for x in chosen)
    winner["families_agreed"]=[x["family"] for x in chosen]
    winner["family_votes"]=count
    winner["direction"]=direction
    winner["candle_closed"]=all(x["candle_closed"] for x in chosen)
    return winner
